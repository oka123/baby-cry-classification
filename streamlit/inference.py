# -*- coding: utf-8 -*-
"""
inference.py
=============
Orkestrasi seluruh alur inference: dari file audio mentah yang diunggah
pengguna sampai menghasilkan label klasifikasi akhir.

Alur (harus sejalan dengan notebook training, TANPA tahap augmentasi
karena augmentasi hanya relevan untuk memperbanyak data latih):

    1. Preprocessing   (resample, normalisasi, trim silence)
    2. Segmentasi      (potong menjadi klip 2 detik tanpa overlap)
    3. Ekstraksi fitur (RMS, ZCR, centroid, bandwidth, pitch, MFCC) / segmen
    4. Feature engineering (vektor statistik utk ML tradisional,
       ATAU matriks 2D ternormalisasi utk deep learning)
    5. Prediksi per segmen oleh model yang dipilih
    6. Agregasi probabilitas antar segmen -> label akhir
"""

import numpy as np

from preprocessing import load_and_preprocess, segment_audio
from feature_extraction import (
    extract_features,
    extract_melspectrogram,
    build_feature_vector,
)
from model_loader import load_model_bundle


def _predict_traditional(bundle, raw_features_list, use_pitch):
    """Prediksi probabilitas per segmen menggunakan model ML tradisional."""
    scaler = bundle["scaler"]
    model = bundle["model"]

    segment_probs = []
    for feats in raw_features_list:
        vec = build_feature_vector(feats, use_pitch=use_pitch).reshape(1, -1)
        vec_scaled = scaler.transform(vec)
        probs = model.predict_proba(vec_scaled)[0]
        segment_probs.append(probs)
    return np.array(segment_probs)


def _predict_deep_learning(bundle, segments, sr):
    """Prediksi probabilitas per segmen menggunakan model deep learning (Mel-Spectrogram)."""
    model = bundle["model"]

    matrices = [extract_melspectrogram(seg, sr) for seg in segments]
    X = np.stack(matrices)
    
    # Normalisasi min-max (per-batch/file inference) identik dengan notebook training
    X_norm = (X - X.min()) / (X.max() - X.min() + 1e-8)
    
    # Tambahkan dimensi channel
    X_norm = X_norm[..., np.newaxis]

    probs_all = model.predict(X_norm, verbose=0)
    return np.array(probs_all)


def predict_audio(uploaded_file, model_key):
    """
    Menjalankan seluruh alur inference untuk 1 file audio menggunakan
    model `model_key`.

    Returns
    -------
    dict berisi seluruh artefak antara (untuk keperluan visualisasi alur)
    beserta hasil akhir klasifikasi:
        - preprocessing      : dict hasil load_and_preprocess()
        - segments           : list segmen audio (2 detik)
        - sr                 : sample rate hasil preprocessing
        - raw_features_list  : list dict fitur mentah per segmen
        - segment_probs      : ndarray (n_segmen, n_kelas) probabilitas per segmen
        - segment_preds      : ndarray (n_segmen,) label prediksi per segmen
        - mean_probs         : ndarray (n_kelas,) rata-rata probabilitas antar segmen
        - final_label        : str label akhir (nama kelas asli, mis. "hungry")
        - classes            : list nama seluruh kelas sesuai urutan label encoder
        - model_info         : dict metadata model dari registry.json
    """
    bundle = load_model_bundle(model_key)
    info = bundle["info"]
    label_encoder = bundle["label_encoder"]
    use_pitch = info["include_pitch"]

    # 1 Preprocessing
    pre = load_and_preprocess(uploaded_file)
    y_processed, sr = pre["y_processed"], pre["sr_processed"]

    # 2 Segmentasi (2 detik, tanpa overlap)
    segments = segment_audio(y_processed)

    # 3 Ekstraksi fitur mentah per segmen
    raw_features_list = [extract_features(seg, sr) for seg in segments]

    # 4 & 5 Feature engineering + prediksi, sesuai tipe model
    if info["type"] == "traditional":
        segment_probs = _predict_traditional(bundle, raw_features_list, use_pitch)
    else:
        segment_probs = _predict_deep_learning(bundle, segments, sr)

    segment_preds = np.argmax(segment_probs, axis=1)

    # 6 Agregasi: rata-rata probabilitas antar segmen.
    #    Dipilih dibanding voting mayoritas murni karena tetap dapat
    #    membedakan hasil walau jumlah segmen sedikit atau seri (tie).
    mean_probs = segment_probs.mean(axis=0)
    final_pred_idx = int(np.argmax(mean_probs))
    final_label = label_encoder.inverse_transform([final_pred_idx])[0]

    return {
        "preprocessing": pre,
        "segments": segments,
        "sr": sr,
        "raw_features_list": raw_features_list,
        "segment_probs": segment_probs,
        "segment_preds": segment_preds,
        "mean_probs": mean_probs,
        "final_label": final_label,
        "classes": list(label_encoder.classes_),
        "model_info": info,
    }
