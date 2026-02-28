import pickle
import cloudpickle
import os
import streamlit as st
import numpy as np

@st.cache_resource
def load_all_models():
    """Load all trained models with caching"""
    models = {}
    model_paths = {
        'Hybrid (XGB+LGBM)': 'models/hybrid_results.pkl',
        'Random Forest': 'models/baseline_rf_results.pkl',
        'SVM': 'models/baseline_svm_results.pkl',
        'XGBoost': 'models/xgb_results.pkl',
        'LightGBM': 'models/lgbm_results.pkl'
    }
    
    features = None
    for name, path in model_paths.items():
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    # Try cloudpickle first, fall back to pickle
                    try:
                        data = cloudpickle.load(f)
                    except:
                        f.seek(0)
                        data = pickle.load(f)
                    
                    if 'classifier' in data:
                        models[name] = data['classifier']
                        if features is None and 'features' in data:
                            features = data['features']
            except Exception as e:
                st.warning(f"Could not load {name}: {e}")
    
    # Fix: Check if features is None or empty properly
    if features is None or (isinstance(features, (list, np.ndarray)) and len(features) == 0):
        # Create dummy features for testing
        features = [f"feature_{i}" for i in range(100)]
        st.info("Using dummy features - model predictions may not be accurate")
    
    return models, features