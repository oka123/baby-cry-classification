# -*- coding: utf-8 -*-
"""
app.py
=======
Aplikasi Streamlit: Klasifikasi Jenis Tangisan Bayi.

Menggunakan salah satu dari 16 model (SVM, Random Forest, CNN, CNN-LSTM
x skema fitur MFCC saja / MFCC+Pitch x dengan/tanpa under-sampling) yang
telah dilatih pada notebook `penyelesaian-kasus-2.py` dan disimpan pada
folder `deployed_models/`.

Alur aplikasi (ditampilkan sebagai langkah 1-5 di halaman ini):
    1. Upload audio
    2. Preprocessing (resample, normalisasi, trim silence)
    3. Segmentasi (2 detik per segmen)
    4. Ekstraksi fitur (mel-spectrogram & MFCC per segmen)
    5. Hasil klasifikasi (probabilitas per kelas & per segmen)

Jalankan dengan:
    streamlit run app.py
"""

import os
import warnings

# Redam log TensorFlow & Keras
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
import streamlit as st

from config import CLASS_LABELS_ID, CLASS_ICONS
from model_loader import load_registry, check_models_available
from inference import predict_audio
from visualization import (
    plot_waveform_before_after,
    plot_segments,
    plot_mel_spectrogram,
    plot_mfcc,
    plot_probability_bar,
    plot_segment_predictions,
)

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Klasifikasi Tangisan Bayi",
    page_icon="👶",
    layout="wide",
)

st.title("👶 Klasifikasi Jenis Tangisan Bayi")
st.caption(
    "Menggunakan 16 model Machine Learning & Deep Learning (SVM, Random Forest, "
    "CNN, CNN-LSTM) yang dilatih pada dataset **Donate-a-Cry Corpus** "
    "(sumber data: [Kaggle Infant Cry Audio Corpus](https://www.kaggle.com/datasets/warcoder/infant-cry-audio-corpus)) "
    "untuk mengenali alasan bayi menangis dari rekaman audio."
)

if not check_models_available():
    st.error(
        "Folder `deployed_models/` beserta `registry.json` tidak ditemukan.\n\n"
        "Pastikan seluruh hasil training (16 model, scaler/norm stats, "
        "`label_encoder.joblib`, `config.json`, `registry.json`) sudah "
        "ditempatkan pada folder `deployed_models/` di direktori yang sama "
        "dengan `app.py`."
    )
    st.stop()

registry = load_registry()

# ============================================================
# SIDEBAR — PEMILIHAN MODEL
# ============================================================
st.sidebar.header("⚙️ Konfigurasi Model")

df_registry = pd.DataFrame([
    {
        "key": key,
        "Algoritma": info["algo"],
        "Skema Fitur": "MFCC + Pitch" if info["include_pitch"] else "MFCC saja",
        "Under-sampling": "Ya" if info["under_sampling"] else "Tidak",
        "Accuracy": info["accuracy"],
        "Precision": info["precision"],
        "Recall": info["recall"],
        "F1-Score": info["f1"],
    }
    for key, info in registry.items()
]).sort_values("F1-Score", ascending=False).reset_index(drop=True)

model_key = st.sidebar.selectbox(
    "Pilih salah satu dari 16 model:",
    options=df_registry["key"].tolist(),
    format_func=lambda k: (
        f"{registry[k]['algo']} | "
        f"{'MFCC+Pitch' if registry[k]['include_pitch'] else 'MFCC'} | "
        f"Undersampling={'Ya' if registry[k]['under_sampling'] else 'Tidak'}"
    ),
)

info_sel = registry[model_key]

# st.sidebar.metric("Accuracy (data uji)", f"{info_sel['accuracy'] * 100:.2f}%")
# c1, c2 = st.sidebar.columns(2)
# c1.metric("Precision", f"{info_sel['precision'] * 100:.1f}%")
# c2.metric("Recall", f"{info_sel['recall'] * 100:.1f}%")
# st.sidebar.metric("F1-Score (macro)", f"{info_sel['f1'] * 100:.2f}%")

# with st.sidebar.expander("📊 Bandingkan seluruh 16 model"):
#     st.dataframe(
#         df_registry.drop(columns="key").style.format(
#             {"Accuracy": "{:.2%}", "Precision": "{:.2%}", "Recall": "{:.2%}", "F1-Score": "{:.2%}"}
#         ),
#         width='stretch',
#         height=350,
st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ Kategori Tangisan Bayi")
st.sidebar.markdown(
    "Aplikasi ini dilatih dengan dataset **Donate-a-Cry Corpus** untuk mengklasifikasi 5 jenis tangisan bayi berikut:"
)
st.sidebar.markdown(
    "- 🍼 **Lapar (`hungry`)**\n"
    "- 🤕 **Sakit Perut (`belly_pain`)**\n"
    "- 😣 **Tidak Nyaman (`discomfort`)**\n"
    "- 😴 **Lelah / Mengantuk (`tired`)**\n"
    "- 🤰 **Perlu Bersendawa (`burping`)**\n"
)
st.sidebar.markdown(
    "Sumber Dataset: [Kaggle Infant Cry Audio Corpus](https://www.kaggle.com/datasets/warcoder/infant-cry-audio-corpus)"
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Pipeline preprocessing & ekstraksi fitur pada aplikasi ini identik dengan "
    "notebook training, kecuali tahap augmentasi (time-stretch, pitch-shift, "
    "noise) yang hanya relevan untuk memperbanyak data LATIH."
)

# ============================================================
# LANGKAH 1 — UPLOAD AUDIO
# ============================================================
st.header("1️⃣ Upload Rekaman Tangisan Bayi")
uploaded_file = st.file_uploader(
    "Unggah file audio (.wav, .mp3, .ogg, .flac, .m4a, .aac, .wma)",
    type=["wav", "mp3", "ogg", "flac", "m4a", "aac", "wma"]
)

if uploaded_file is None:
    st.info("Silakan unggah file audio untuk memulai klasifikasi.")
    st.stop()

# Inisialisasi session state untuk mempertahankan hasil prediksi
if "processed" not in st.session_state:
    st.session_state.processed = False
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None
if "processed_model" not in st.session_state:
    st.session_state.processed_model = None
if "result" not in st.session_state:
    st.session_state.result = None

# Reset status jika file atau model berubah
if uploaded_file != st.session_state.processed_file:
    st.session_state.processed = False
    st.session_state.result = None
    st.session_state.processed_file = uploaded_file

if model_key != st.session_state.processed_model:
    st.session_state.processed = False
    st.session_state.result = None
    st.session_state.processed_model = model_key

st.audio(uploaded_file.getvalue())
run = st.button("🔍 Proses & Klasifikasikan", type="primary")

if run:
    st.session_state.processed = True
    with st.spinner("Menjalankan preprocessing, segmentasi, ekstraksi fitur, dan prediksi..."):
        st.session_state.result = predict_audio(uploaded_file, model_key)

if not st.session_state.processed or st.session_state.result is None:
    st.stop()

result = st.session_state.result
pre = result["preprocessing"]
segments = result["segments"]

# ============================================================
# LANGKAH 2 — PREPROCESSING
# ============================================================
st.header("2️⃣ Preprocessing Audio")
st.markdown(
    "Audio di-**resample** ke 22.050 Hz, **dinormalisasi** amplitudonya ke "
    "rentang [-1, 1], lalu bagian **hening (silence)** di awal/akhir dipotong "
    "— identik dengan tahap preprocessing pada notebook training."
)
plot_waveform_before_after(
    pre["y_raw"], pre["sr_raw"], pre["y_processed"], pre["sr_processed"]
)

# ============================================================
# LANGKAH 3 — SEGMENTASI
# ============================================================
st.header("3️⃣ Segmentasi Audio (2 Detik per Segmen)")
st.markdown(
    f"Audio hasil preprocessing dipotong menjadi **{len(segments)} segmen** "
    "berdurasi 2 detik tanpa overlap (segmen terakhir di-padding nol bila "
    "kurang dari 2 detik)."
)
plot_segments(pre["y_processed"], result["sr"], segments)

# ============================================================
# LANGKAH 4 — EKSTRAKSI FITUR
# ============================================================
st.header("4️⃣ Ekstraksi Fitur")
st.markdown(
    "Dari setiap segmen diekstraksi fitur domain waktu (RMS, ZCR), domain "
    "frekuensi (Spectral Centroid & Bandwidth), Pitch (algoritma YIN), dan "
    "**MFCC** (13 koefisien) — fitur inilah yang diringkas (untuk ML "
    "tradisional) atau disusun sebagai matriks 2D (untuk deep learning) "
    "menjadi input model."
)

seg_choice = st.selectbox(
    "Pilih segmen untuk divisualisasikan:",
    options=list(range(1, len(segments) + 1)),
    format_func=lambda i: f"Segmen {i}",
)
col1, col2 = st.columns(2)
with col1:
    plot_mel_spectrogram(
        segments[seg_choice - 1], result["sr"],
        title=f"Mel-Spectrogram — Segmen {seg_choice}",
    )
with col2:
    plot_mfcc(
        result["raw_features_list"][seg_choice - 1]["mfcc"], result["sr"],
        title=f"MFCC — Segmen {seg_choice}",
    )

if len(segments) > 1:
    st.subheader("Prediksi & Keyakinan per Segmen")
    st.markdown(
        "Setiap segmen diklasifikasikan secara independen oleh model, lalu "
        "probabilitasnya **dirata-rata** untuk menghasilkan prediksi akhir "
        "yang lebih stabil."
    )
    plot_segment_predictions(
        result["segment_preds"], result["segment_probs"], result["classes"]
    )

# ============================================================
# LANGKAH 5 — HASIL KLASIFIKASI
# ============================================================
st.header("5️⃣ Hasil Klasifikasi")

final_label = result["final_label"]
final_label_id = CLASS_LABELS_ID.get(final_label, final_label)
final_icon = CLASS_ICONS.get(final_label, "👶")
confidence = result["mean_probs"][result["classes"].index(final_label)] * 100

st.success(
    f"### {final_icon} Prediksi: **{final_label_id}** (`{final_label}`)\n"
    f"Tingkat keyakinan rata-rata: **{confidence:.1f}%**"
)

plot_probability_bar(result["mean_probs"], result["classes"], CLASS_LABELS_ID)

st.info(
    f"Model yang digunakan: **{info_sel['algo']}** dengan skema fitur "
    f"**{'MFCC + Pitch' if info_sel['include_pitch'] else 'MFCC saja'}** "
    f"({'dengan' if info_sel['under_sampling'] else 'tanpa'} under-sampling)."
    # Commented out test accuracy display
    # f" Akurasi model ini pada data uji (saat training): "
    # f"**{info_sel['accuracy'] * 100:.2f}%**."
)
