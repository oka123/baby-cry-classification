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

from config import MODEL_DIR, REGISTRY_FILE, CONFIG_FILE, LABEL_ENCODER_FILE


def check_models_available():
    """Mengecek apakah folder deployed_models & registry.json sudah tersedia."""
    return os.path.isdir(MODEL_DIR) and os.path.isfile(REGISTRY_FILE)


@st.cache_resource(show_spinner=False)
def load_registry():
    """Memuat metadata seluruh 16 model (algoritma, skema fitur, metrik, dst)."""
    with open(REGISTRY_FILE, "r") as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def load_global_config():
    """Memuat config.json (daftar kelas, sample rate, durasi segmen, dst)."""
    with open(CONFIG_FILE, "r") as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def load_label_encoder():
    """Memuat LabelEncoder yang dipakai bersama oleh seluruh 16 model."""
    return joblib.load(LABEL_ENCODER_FILE)


@st.cache_resource(show_spinner=False)
def load_model_bundle(model_key):
    """
    Memuat 1 model beserta artefak pendukungnya berdasarkan `model_key`
    pada registry.json.

    - Model "traditional" (SVM / Random Forest) -> memuat model (.joblib)
      dan StandardScaler (.joblib).
    - Model "deep_learning" (CNN / CNN-LSTM) -> memuat model (.keras)
      dan statistik normalisasi z-score (.json).
    """
    registry = load_registry()
    if model_key not in registry:
        raise ValueError(f"Model '{model_key}' tidak ditemukan pada registry.json")

    info = registry[model_key]
    bundle = {"info": info}

    if info["type"] == "traditional":
        bundle["model"] = joblib.load(os.path.join(MODEL_DIR, info["model_file"]))
        bundle["scaler"] = joblib.load(os.path.join(MODEL_DIR, info["scaler_file"]))
    else:
        # Import TensorFlow di sini (bukan di top-level) agar aplikasi tetap
        # bisa dibuka walau TensorFlow belum siap / model DL belum dipilih.
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        
        import tensorflow as tf

        bundle["model"] = tf.keras.models.load_model(
            os.path.join(MODEL_DIR, info["model_file"])
        )
        with open(os.path.join(MODEL_DIR, info["norm_file"]), "r") as f:
            bundle["norm_stats"] = json.load(f)

    return bundle
