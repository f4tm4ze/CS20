import numpy as np
from androguard.core.bytecodes import apk, dvm
from androguard.misc import AnalyzeAPK
from fuzzywuzzy import fuzz

def extract_features(apk_path, most_relevant_features):
    """
    Function to extract features from an APK file.
    
    Input:
        APK file path
        most relevant features
    
    Output:
        result of the mapping between the extracted features and the most relevant features
    """
    
    try:
        a, d, dx = AnalyzeAPK(apk_path)
        apk_file = apk.APK(apk_path)

        # Extract classes and methods
        classes = []
        methods = []

        # For androguard 3.4.0a1, d is a DalvikVMFormat object
        if d is not None:
            for dex_class in d.get_classes():
                classes.append(dex_class.get_name())
                for method in dex_class.get_methods():
                    methods.append(method.get_name())

        # Extract activities
        activities = a.get_activities()

        # Extract intents from activities
        activities_intents = []
        for activity in activities:
            activity_intents = a.get_intent_filters("activity", activity)
            activities_intents.append(activity_intents)

        # Extract services
        services = a.get_services()

        # Extract intents from services
        services_intents = []
        for service in services:
            service_intents = a.get_intent_filters("service", service)
            services_intents.append(service_intents)

        # Extract receivers
        receivers = a.get_receivers()

        # Extract intents from receivers
        receivers_intents = []
        for receiver in receivers:
            receiver_intents = a.get_intent_filters("receiver", receiver)
            receivers_intents.append(receiver_intents)

        # Extract providers
        providers = a.get_providers()

        # Extract intents from providers
        providers_intents = []
        for provider in providers:
            provider_intents = a.get_intent_filters("provider", provider)
            providers_intents.append(provider_intents)

        intents = activities_intents + services_intents + receivers_intents + providers_intents

        all_intents_action_and_category = []
        for intent in intents:
            if intent is not None:
                action = intent.get("action")
                if action is not None:
                    if isinstance(action, list):
                        for ac in action:
                            if ac:
                                all_intents_action_and_category.append(ac)
                    else:
                        all_intents_action_and_category.append(action)
                        
                category = intent.get("category")
                if category is not None:
                    if isinstance(category, list):
                        for c in category:
                            if c:
                                all_intents_action_and_category.append(c)
                    else:
                        all_intents_action_and_category.append(category)

        # Extract permissions
        permissions = a.get_permissions()

        # Extract hardware and software features
        hardware_software_features = a.get_features()

        # Aggregate all extracted app features
        app_extracted_features = (permissions + hardware_software_features + activities +
                                 providers + receivers + services + all_intents_action_and_category +
                                 classes + methods)

        # Initialize empty list for results
        extraction_result = []
        matches = []

        # Iterate through features and permissions to check for matches
        for required_feature in most_relevant_features:
            match_found = False
            for app_feature in app_extracted_features:
                if app_feature is not None and required_feature is not None:
                    # Calculate the similarity score between the two strings
                    similarity_score = fuzz.ratio(str(required_feature), str(app_feature))
                    # Set a threshold for similarity
                    threshold = 90
                    # Check if the similarity score is above the threshold
                    if similarity_score >= threshold or str(required_feature) in str(app_feature):
                        extraction_result.append(1)
                        matches.append((required_feature, app_feature))
                        match_found = True
                        break
            if not match_found:
                extraction_result.append(0)

        print(f"\n---------- Matches found in feature mapping: {len(matches)} ----------\n")
        for match in matches[:10]:
            print(f"Match found: {match[0]} in {match[1]}")
        if len(matches) > 10:
            print(f"... and {len(matches) - 10} more matches")
        print("\n------------------------------------------------------\n")

        return np.array(extraction_result).reshape(1, -1), matches
    
    except Exception as e:
        print(f"Error in feature extraction: {e}")
        # Return zeros as fallback
        return np.zeros((1, len(most_relevant_features))), []