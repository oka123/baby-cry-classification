# -*- coding: utf-8 -*-
"""
feature_extraction.py
======================
Ekstraksi fitur audio & feature engineering untuk keperluan INFERENCE.

Seluruh fungsi di file ini SENGAJA disalin identik dari notebook training
(bagian "7. EKSTRAKSI FITUR" & "8. FEATURE ENGINEERING") supaya representasi
fitur yang dipakai untuk prediksi konsisten dengan representasi fitur yang
dipakai saat training model.
"""

import numpy as np
import librosa

from config import N_MFCC, FMIN_PITCH, FMAX_PITCH


def extract_features(y, sr, n_mfcc=N_MFCC):
    """
    Mengekstrak seluruh fitur audio dari 1 segmen sinyal (frame-level / time
    series), TANPA melakukan agregasi statistik. Identik dengan
    `extract_features()` pada notebook training.

    Fitur yang diekstraksi:
    - RMS Energy (domain waktu)
    - Zero Crossing Rate / ZCR (domain waktu)
    - Spectral Centroid (domain frekuensi)
    - Spectral Bandwidth (domain frekuensi)
    - Pitch / F0 via algoritma YIN
    - MFCC (n_mfcc koefisien)
    """
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]

    try:
        pitch = librosa.yin(y, fmin=FMIN_PITCH, fmax=FMAX_PITCH, sr=sr)
        pitch = np.nan_to_num(pitch, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception:
        pitch = np.zeros_like(rms)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    return {
        "rms": rms,
        "zcr": zcr,
        "spectral_centroid": spectral_centroid,
        "spectral_bandwidth": spectral_bandwidth,
        "pitch": pitch,
        "mfcc": mfcc,
    }


def aggregate_statistics(feature_array):
    """Meringkas array fitur per-frame (1D/2D) menjadi statistik: mean, std, min, max."""
    feature_array = np.array(feature_array)
    if feature_array.ndim == 1:
        return np.array([
            np.mean(feature_array), np.std(feature_array),
            np.min(feature_array), np.max(feature_array),
        ])
    mean_ = np.mean(feature_array, axis=1)
    std_ = np.std(feature_array, axis=1)
    min_ = np.min(feature_array, axis=1)
    max_ = np.max(feature_array, axis=1)
    return np.concatenate([mean_, std_, min_, max_])


def build_feature_vector(features_dict, use_pitch=True):
    """
    Menggabungkan seluruh fitur menjadi 1 vektor tetap, dipakai oleh model
    ML tradisional (SVM / Random Forest). Identik dengan
    `build_feature_vector()` pada notebook training.
    """
    parts = [
        aggregate_statistics(features_dict["rms"]),
        aggregate_statistics(features_dict["zcr"]),
        aggregate_statistics(features_dict["spectral_centroid"]),
        aggregate_statistics(features_dict["spectral_bandwidth"]),
        aggregate_statistics(features_dict["mfcc"]),
    ]
    if use_pitch:
        parts.append(aggregate_statistics(features_dict["pitch"]))
    return np.concatenate(parts)


def build_deep_feature_matrix(features_dict, use_pitch=True):
    """
    Membangun representasi 2D (time_steps, n_features) dipakai oleh model
    deep learning (CNN / CNN-LSTM). Identik dengan
    `build_deep_feature_matrix()` pada notebook training.
    """
    mfcc = features_dict["mfcc"]
    n_frames = mfcc.shape[1]
    if use_pitch:
        pitch = features_dict["pitch"]
        if len(pitch) != n_frames:
            pitch = np.interp(
                np.linspace(0, 1, n_frames),
                np.linspace(0, 1, len(pitch)),
                pitch,
            )
        combined = np.vstack([mfcc, pitch.reshape(1, -1)])
    else:
        combined = mfcc
    return combined.T


def pad_or_truncate(matrix, target_len):
    """
    Menyeragamkan panjang time-steps suatu matriks fitur (padding nol/truncating),
    identik dengan `pad_or_truncate()` pada notebook training. `target_len`
    diambil dari statistik normalisasi (`{key}_norm.json`) yang disimpan saat
    training, sehingga bentuk input persis sama dengan saat model dilatih.
    """
    current_len = matrix.shape[0]
    if current_len == target_len:
        return matrix
    elif current_len < target_len:
        pad = np.zeros((target_len - current_len, matrix.shape[1]))
        return np.vstack([matrix, pad])
    else:
        return matrix[:target_len, :]
