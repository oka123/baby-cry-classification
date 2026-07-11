# -*- coding: utf-8 -*-
"""
config.py
=========
Konfigurasi global aplikasi Streamlit "Klasifikasi Tangisan Bayi".

Seluruh parameter di file ini HARUS SAMA dengan parameter yang dipakai
pada notebook training (penyelesaian-kasus-2.py), agar preprocessing dan
ekstraksi fitur saat inference identik dengan saat training.
"""

import os

# ------------------------------------------------------------------
# PATH
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Folder tempat model hasil training disimpan
MODEL_DIR = os.path.join(BASE_DIR, "../saved_models/saved_models")

# ------------------------------------------------------------------
# PARAMETER PREPROCESSING AUDIO (harus sama dengan notebook training)
# ------------------------------------------------------------------
TARGET_SR = 22050                 # sample rate target hasil resampling
SEGMENT_DURATION = 2.0            # durasi tiap segmen (detik)
SEGMENT_LENGTH = int(TARGET_SR * SEGMENT_DURATION)  # jumlah sample per segmen
TOP_DB = 25                       # ambang batas (dB) untuk trimming silence

# ------------------------------------------------------------------
# PARAMETER EKSTRAKSI FITUR (harus sama dengan notebook training)
# ------------------------------------------------------------------
N_MFCC = 13
FMIN_PITCH = 50
FMAX_PITCH = 1000

# ------------------------------------------------------------------
# LABEL KELAS
# ------------------------------------------------------------------
CLASSES = ["belly_pain", "burping", "discomfort", "hungry", "tired"]

# Terjemahan label ke Bahasa Indonesia (untuk tampilan UI)
CLASS_LABELS_ID = {
    "belly_pain": "Sakit Perut",
    "burping": "Perlu Bersendawa",
    "discomfort": "Tidak Nyaman (mis. popok basah)",
    "hungry": "Lapar",
    "tired": "Lelah / Mengantuk",
}

CLASS_ICONS = {
    "belly_pain": "🤕",
    "burping": "🤰",
    "discomfort": "😣",
    "hungry": "🍼",
    "tired": "😴",
}
