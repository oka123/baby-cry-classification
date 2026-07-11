# -*- coding: utf-8 -*-
"""
model_loader.py
================
Pemuatan artefak hasil training: registry.json, config.json,
label_encoder.joblib, dan 16 model (+ scaler / norm stats masing-masing)
dari folder `deployed_models/`.

Menggunakan st.cache_resource agar setiap artefak hanya dimuat sekali
selama sesi aplikasi berjalan (tidak reload setiap interaksi user).
"""

import os
import json
import warnings
import logging

import joblib
import streamlit as st

from config import MODEL_DIR


def check_models_available():
    """Mengecek apakah folder model sudah tersedia."""
    return os.path.isdir(MODEL_DIR)


@st.cache_resource(show_spinner=False)
def load_registry():
    """Memuat daftar model yang tersedia secara dinamis dari folder."""
    registry = {}
    
    traditional_dir = os.path.join(MODEL_DIR, "traditional")
    if os.path.isdir(traditional_dir):
        for f in os.listdir(traditional_dir):
            if f.endswith(".joblib"):
                key = f.replace(".joblib", "")
                algo = "SVM" if "svm" in f else "Random Forest"
                include_pitch = "mfcc_only" not in f
                
                registry[key] = {
                    "type": "traditional",
                    "algo": algo,
                    "include_pitch": include_pitch,
                    "model_file": os.path.join("traditional", f),
                    "is_noaug": "noaug" in f
                }
                
    dl_dir = os.path.join(MODEL_DIR, "deep_learning")
    if os.path.isdir(dl_dir):
        for f in os.listdir(dl_dir):
            if f.endswith(".keras"):
                key = f.replace(".keras", "")
                algo = "CNN-LSTM" if "lstm" in f else "CNN"
                
                registry[key] = {
                    "type": "deep_learning",
                    "algo": algo,
                    "include_pitch": False,
                    "model_file": os.path.join("deep_learning", f),
                    "is_noaug": "noaug" in f
                }
                
    return registry


@st.cache_resource(show_spinner=False)
def load_model_bundle(model_key):
    """
    Memuat 1 model beserta artefak pendukungnya berdasarkan `model_key`.
    """
    registry = load_registry()
    if model_key not in registry:
        raise ValueError(f"Model '{model_key}' tidak ditemukan pada registry")

    info = registry[model_key]
    bundle = {"info": info}

    if info["type"] == "traditional":
        data = joblib.load(os.path.join(MODEL_DIR, info["model_file"]))
        bundle["model"] = data["model"]
        bundle["scaler"] = data["scaler"]
        bundle["label_encoder"] = data["label_encoder"]
    else:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        
        import tensorflow as tf

        bundle["model"] = tf.keras.models.load_model(
            os.path.join(MODEL_DIR, info["model_file"])
        )
        enc_path = os.path.join(MODEL_DIR, "deep_learning", "dl_label_encoder.joblib")
        enc_data = joblib.load(enc_path)
        bundle["label_encoder"] = enc_data["label_encoder"]

    return bundle
