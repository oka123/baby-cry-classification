# -*- coding: utf-8 -*-
"""
feature_extraction.py
======================
Ekstraksi fitur audio & feature engineering untuk keperluan INFERENCE.

"""

import numpy as np
import librosa

from config import N_MFCC, FMIN_PITCH, FMAX_PITCH


def extract_features(y, sr, n_mfcc=N_MFCC):
    """
    Mengekstrak seluruh fitur audio dari 1 segmen sinyal (frame-level / time
    series), TANPA melakukan agregasi statistik.

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
        pitch, _, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr
        )
        pitch = pitch[~np.isnan(pitch)]
        if len(pitch) == 0:
            pitch = np.array([0.0])
    except Exception:
        pitch = np.array([0.0])

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    return {
        "rms": rms,
        "zcr": zcr,
        "spectral_centroid": spectral_centroid,
        "spectral_bandwidth": spectral_bandwidth,
        "pitch": pitch,
        "mfcc": mfcc,
    }


def summarize_stats(arr):
    """Ringkas array 1D jadi mean, std, min, max."""
    return [np.mean(arr), np.std(arr), np.min(arr), np.max(arr)]


def build_feature_vector(features_dict, use_pitch=True):
    """
    Menggabungkan seluruh fitur menjadi 1 vektor tetap, dipakai oleh model
    ML tradisional (SVM / Random Forest). Identik dengan
    `build_feature_vector()` pada notebook training.
    """
    row = []
    row.extend(summarize_stats(features_dict["rms"]))
    row.extend(summarize_stats(features_dict["zcr"]))
    row.extend(summarize_stats(features_dict["spectral_centroid"]))
    row.extend(summarize_stats(features_dict["spectral_bandwidth"]))
    for i in range(features_dict["mfcc"].shape[0]):
        row.extend(summarize_stats(features_dict["mfcc"][i]))
    if use_pitch:
        row.extend(summarize_stats(features_dict["pitch"]))
    return np.array(row)


def extract_melspectrogram(y, sr, n_mels=128):
    """
    Mengekstrak fitur Mel-Spectrogram (2D) untuk input CNN / CNN-LSTM.
    Identik dengan `extract_melspectrogram()` pada notebook training.
    """
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db


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
