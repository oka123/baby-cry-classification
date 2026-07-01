# -*- coding: utf-8 -*-
"""
preprocessing.py
================
Preprocessing & segmentasi audio untuk keperluan INFERENCE.

Fungsi-fungsi di sini SENGAJA dibuat identik dengan fungsi `preprocess_audio()`
dan `segment_audio()` pada notebook training, agar representasi data yang
"dilihat" oleh model saat inference sama persis dengan saat training.

Catatan: augmentasi audio (time-stretch, pitch-shift, noise) TIDAK dipakai
di sini karena augmentasi hanya relevan untuk memperbanyak data LATIH,
bukan untuk data yang sedang diklasifikasikan.
"""

import os
import tempfile
from typing import Any
import numpy as np
import librosa

from config import TARGET_SR, TOP_DB, SEGMENT_LENGTH


def _reset_pointer(file_like):
    """Reset posisi baca file-like object (misal file upload Streamlit)."""
    if hasattr(file_like, "seek"):
        try:
            file_like.seek(0)
        except Exception:
            pass


def load_and_preprocess(file_like, target_sr=TARGET_SR, top_db=TOP_DB):
    """
    Melakukan tahap preprocessing pada 1 file audio:
    1. Load audio ASLI (sr asli) -> untuk visualisasi "sebelum preprocessing"
    2. Load ulang dengan resampling ke target_sr
    3. Normalisasi amplitudo ke rentang [-1, 1]
    4. Trimming silence di awal & akhir sinyal

    Parameters
    ----------
    file_like : str atau file-like object
        Path menuju file audio, atau objek hasil st.file_uploader.

    Returns
    -------
    dict berisi sinyal & sample rate sebelum/sesudah preprocessing.
    """
    temp_file_path = None
    audio_source: Any
    if not isinstance(file_like, (str, bytes, os.PathLike)) and hasattr(file_like, "read"):
        try:
            file_name = getattr(file_like, "name", "temp_audio.wav")
            _, ext = os.path.splitext(file_name)
            if not ext:
                ext = ".wav"
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
                _reset_pointer(file_like)
                temp_file.write(file_like.read())
                temp_file_path = temp_file.name
            audio_source = temp_file_path
        except Exception:
            audio_source = file_like
    else:
        audio_source = file_like

    try:
        # 1) Audio asli (tanpa resampling) -> untuk keperluan visualisasi
        _reset_pointer(audio_source)
        y_raw, sr_raw = librosa.load(audio_source, sr=None, mono=True)

        # 2) Load & resampling ke target_sr (identik dengan preprocess_audio())
        _reset_pointer(audio_source)
        y, sr = librosa.load(audio_source, sr=target_sr, mono=True)
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception:
                pass

    # 3) Normalisasi amplitudo: skala sinyal agar nilai absolut maksimum = 1
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    # 4) Trimming silence di awal & akhir sinyal
    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    if len(y_trimmed) == 0:
        y_trimmed = y

    return {
        "y_raw": y_raw,
        "sr_raw": sr_raw,
        "y_processed": y_trimmed,
        "sr_processed": sr,
    }


def segment_audio(y, segment_length=SEGMENT_LENGTH):
    """
    Memotong sinyal audio menjadi beberapa segmen berdurasi tetap TANPA overlap,
    identik dengan fungsi `segment_audio()` pada notebook training.

    - Jika audio lebih pendek dari 1 segmen -> di-padding nol.
    - Jika lebih panjang -> dipotong berurutan tanpa overlap.
    """
    if len(y) < segment_length:
        y_padded = np.pad(y, (0, segment_length - len(y)), mode="constant")
        return [y_padded]

    segments = []
    start = 0
    while start + segment_length <= len(y):
        segments.append(y[start:start + segment_length])
        start += segment_length

    return segments if segments else [y[:segment_length]]
