import numpy as np
import threading

try:
    from androguard.core.apk import APK
except ImportError:
    from androguard.core.bytecodes.apk import APK

try:
    from androguard.misc import AnalyzeAPK
    HAS_ANALYZE = True
except ImportError:
    HAS_ANALYZE = False

from fuzzywuzzy import fuzz


def _analyze_apk_with_timeout(apk_path, timeout=60):
    """
    Run AnalyzeAPK in a thread with a timeout.
    Returns (a, d, dx) or (None, None, None) if it times out.
    """
    result = [None, None, None]
    error  = [None]

    def target():
        try:
            from androguard.misc import AnalyzeAPK
            result[0], result[1], result[2] = AnalyzeAPK(apk_path)
        except Exception as e:
            error[0] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout)

    if t.is_alive():
        print(f"[feature_extractor] AnalyzeAPK timed out after {timeout}s — falling back to APK-only mode")
        return None, None, None

    if error[0]:
        print(f"[feature_extractor] AnalyzeAPK error: {error[0]} — falling back to APK-only mode")
        return None, None, None

    return result[0], result[1], result[2]


def extract_features(apk_path, most_relevant_features):
    """
    Extract features from an APK file and map them against the most relevant
    feature list used during model training.

    Strategy:
      1. Try AnalyzeAPK with a 60-second timeout to get DEX classes/methods.
      2. If it times out or fails, fall back to APK-only parsing.
      3. Match using: exact → substring → fuzzy fallback.
    """
    try:
        classes = []
        methods = []

        # ── Step 1: Try full DEX analysis (with timeout) ──────────────────────
        # Skip DEX analysis for large files — permissions/intents are enough
        import os
        file_size_mb = os.path.getsize(apk_path) / (1024 * 1024)
        if file_size_mb > 100:
            print(f"[feature_extractor] File is {file_size_mb:.1f} MB — skipping DEX analysis, using APK-only mode")
            a, d, dx = None, None, None
        else:
            a, d, dx = _analyze_apk_with_timeout(apk_path, timeout=30)

        if a is None:
            # Fallback: APK-only parse
            a = APK(apk_path)
        else:
            # Extract classes and methods from DEX
            dex_list = d if isinstance(d, list) else [d]
            for dex in dex_list:
                if dex is None:
                    continue
                try:
                    for dex_class in dex.get_classes():
                        name = dex_class.get_name()
                        if name:
                            classes.append(name)
                        for method in dex_class.get_methods():
                            mname = method.get_name()
                            if mname:
                                methods.append(mname)
                except Exception as e:
                    print(f"[feature_extractor] DEX parse warning: {e}")

        print(f"[feature_extractor] DEX classes: {len(classes)}, methods: {len(methods)}")
        # ─────────────────────────────────────────────────────────────────────

        # ── Step 2: APK-level features ────────────────────────────────────────
        permissions       = a.get_permissions()  or []
        hardware_software = a.get_features()     or []
        activities        = a.get_activities()   or []
        services          = a.get_services()     or []
        receivers         = a.get_receivers()    or []
        providers         = a.get_providers()    or []

        all_intents = []
        for component_type, components in [
            ("activity", activities),
            ("service",  services),
            ("receiver", receivers),
            ("provider", providers),
        ]:
            for comp in components:
                intent = a.get_intent_filters(component_type, comp)
                if intent:
                    for key in ("action", "category"):
                        val = intent.get(key)
                        if val is not None:
                            if isinstance(val, list):
                                all_intents.extend(v for v in val if v)
                            else:
                                all_intents.append(val)

        app_extracted_features = (
            permissions
            + hardware_software
            + activities
            + providers
            + receivers
            + services
            + all_intents
            + classes
            + methods
        )
        # ─────────────────────────────────────────────────────────────────────

        print(f"[feature_extractor] Total extracted features: {len(app_extracted_features)}")

        # ── Step 3: Match against trained feature list ────────────────────────
        app_features_set = {str(f) for f in app_extracted_features if f is not None}

        extraction_result = []
        matches = []

        for required_feature in most_relevant_features:
            req_str = str(required_feature) if required_feature is not None else ""

            # 1. Exact match — O(1)
            if req_str in app_features_set:
                extraction_result.append(1)
                matches.append((required_feature, req_str))
                continue

            # 2. Substring match
            sub_match = next(
                (f for f in app_features_set if req_str and req_str in f),
                None
            )
            if sub_match:
                extraction_result.append(1)
                matches.append((required_feature, sub_match))
                continue

            # 3. Fuzzy fallback — only when exact/substring both fail
            best_score = 0
            best_feat  = None
            for app_feat in app_features_set:
                score = fuzz.ratio(req_str, app_feat)
                if score > best_score:
                    best_score = score
                    best_feat  = app_feat
                if best_score >= 95:
                    break

            if best_score >= 90:
                extraction_result.append(1)
                matches.append((required_feature, best_feat))
            else:
                extraction_result.append(0)
        # ─────────────────────────────────────────────────────────────────────

        print(f"\n--- Matches found: {len(matches)} / {len(most_relevant_features)} features ---")
        for match in matches[:10]:
            print(f"  {match[0]}  →  {match[1]}")
        if len(matches) > 10:
            print(f"  ... and {len(matches) - 10} more")

        return np.array(extraction_result).reshape(1, -1), matches

    except Exception as e:
        print(f"Error in feature extraction: {e}")
        import traceback
        traceback.print_exc()
        return np.zeros((1, len(most_relevant_features))), []
