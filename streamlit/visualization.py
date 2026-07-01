# -*- coding: utf-8 -*-
"""
visualization.py
=================
Kumpulan fungsi visualisasi untuk menampilkan ALUR data audio di aplikasi
Streamlit: dari sinyal mentah -> preprocessing -> segmentasi -> fitur
(mel-spectrogram & MFCC) -> hasil prediksi (probabilitas per kelas & per segmen).

Semua fungsi langsung merender ke halaman Streamlit yang aktif (st.pyplot).
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import streamlit as st


def plot_waveform_before_after(y_raw, sr_raw, y_processed, sr_processed):
    """Membandingkan waveform sebelum & sesudah preprocessing."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 5))

    librosa.display.waveshow(y_raw, sr=sr_raw, ax=axes[0], color="gray")
    axes[0].set_title(
        f"Audio Asli (sr={sr_raw} Hz, durasi={len(y_raw) / sr_raw:.2f}s)"
    )

    librosa.display.waveshow(y_processed, sr=sr_processed, ax=axes[1], color="teal")
    axes[1].set_title(
        "Setelah Preprocessing — resample + normalisasi + trim silence "
        f"(sr={sr_processed} Hz, durasi={len(y_processed) / sr_processed:.2f}s)"
    )

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_segments(y_processed, sr, segments):
    """Menampilkan waveform penuh dengan pewarnaan area tiap segmen 2 detik."""
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y_processed, sr=sr, ax=ax, color="lightgray")

    seg_len = len(segments[0]) if segments else 0
    for i in range(len(segments)):
        start_t = i * seg_len / sr
        end_t = start_t + seg_len / sr
        ax.axvspan(start_t, end_t, alpha=0.25, color=plt.get_cmap("tab10")(i % 10))
        ax.text((start_t + end_t) / 2, ax.get_ylim()[1] * 0.85, f"S{i + 1}",
                ha="center", fontsize=8, fontweight="bold")

    ax.set_title(f"Segmentasi menjadi {len(segments)} segmen @ 2 detik (tanpa overlap)")
    ax.set_xlabel("Waktu (detik)")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_mel_spectrogram(y, sr, title="Mel-Spectrogram"):
    """Menampilkan mel-spectrogram dari 1 segmen audio."""
    fig, ax = plt.subplots(figsize=(7, 3))
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    img = librosa.display.specshow(mel_spec_db, sr=sr, x_axis="time", y_axis="mel", ax=ax)
    ax.set_title(title)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_mfcc(mfcc, sr, title="MFCC"):
    """Menampilkan koefisien MFCC dari 1 segmen audio."""
    fig, ax = plt.subplots(figsize=(7, 3))
    img = librosa.display.specshow(mfcc, sr=sr, x_axis="time", ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Koefisien MFCC")
    fig.colorbar(img, ax=ax)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_probability_bar(mean_probs, classes, class_labels_id=None):
    """Bar chart probabilitas akhir (rata-rata seluruh segmen) per kelas."""
    labels = [class_labels_id.get(c, c) if class_labels_id else c for c in classes]
    order = np.argsort(mean_probs)[::-1]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#E63946" if i == order[0] else "#2E86AB" for i in range(len(classes))]
    ax.barh([labels[i] for i in order], mean_probs[order] * 100,
            color=[colors[i] for i in order])
    ax.set_xlabel("Probabilitas (%)")
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.set_title("Probabilitas Prediksi Akhir (rata-rata seluruh segmen)")
    for i, idx in enumerate(order):
        ax.text(mean_probs[idx] * 100 + 1, i, f"{mean_probs[idx] * 100:.1f}%",
                 va="center", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_segment_predictions(segment_preds, segment_probs, classes):
    """Bar chart keyakinan (confidence) prediksi untuk setiap segmen audio."""
    n_seg = len(segment_preds)
    fig, ax = plt.subplots(figsize=(max(6, n_seg * 1.0), 3))

    confidences = [segment_probs[i, segment_preds[i]] * 100 for i in range(n_seg)]
    pred_labels = [classes[p] for p in segment_preds]

    bars = ax.bar(range(1, n_seg + 1), confidences, color="#457B9D")
    ax.set_xticks(range(1, n_seg + 1))
    ax.set_xlabel("Segmen ke-")
    ax.set_ylabel("Keyakinan (%)")
    ax.set_ylim(0, 115)
    ax.set_title("Prediksi & Tingkat Keyakinan per Segmen")
    for bar, lbl in zip(bars, pred_labels):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                 lbl, ha="center", fontsize=7, rotation=30)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
