# 👶 Klasifikasi Jenis Tangisan Bayi — Aplikasi Streamlit

Aplikasi berbasis web ini digunakan untuk mendeteksi penyebab bayi menangis dari berbagai format rekaman suara (seperti `.wav`, `.mp3`, `.ogg`, `.flac`, `.m4a`, dan lainnya). Proyek ini memanfaatkan model **Machine Learning (ML)** dan **Deep Learning (DL)** yang dilatih menggunakan dataset **Donate-a-Cry Corpus**.

Aplikasi ini dirancang dengan antarmuka yang ramah pengguna, visualisasi interaktif representasi audio (_waveform_, _Mel Spectrogram_, dan _MFCC_), serta alur pemrosesan sinyal digital (_Digital Signal Processing_ / DSP) yang transparan dan informatif.

---

## 🌟 Fitur Utama

- **10 Pilihan Model Klasifikasi**  
  Pengguna dapat memilih model berdasarkan algoritma (SVM, Random Forest, CNN, CNN-LSTM), skema fitur (MFCC / MFCC + Pitch / Mel-Spectrogram), serta strategi augmentasi data (dengan atau tanpa augmentasi).

- **Visualisasi Alur Pemrosesan (DSP)**
  - **Waveform Perbandingan**: Menampilkan sinyal audio sebelum dan sesudah preprocessing (_resampling_, normalisasi amplitudo, dan pemotongan bagian hening).
  - **Visualisasi Segmentasi**: Menampilkan pembagian audio menjadi segmen berdurasi 2 detik.
  - **Analisis Mel Spectrogram & MFCC**: Menampilkan representasi _Mel Spectrogram_ dan _Mel-Frequency Cepstral Coefficients_ (MFCC) dari segmen audio yang dipilih secara interaktif.

- **Agregasi Prediksi Segmen**  
  Setiap segmen berdurasi 2 detik diklasifikasikan secara independen. Nilai probabilitas seluruh segmen kemudian dirata-ratakan untuk menghasilkan prediksi akhir yang lebih stabil.

- **Tabel Perbandingan Performa**  
  Sidebar aplikasi menampilkan metrik akurasi, presisi, _recall_, dan _F1-score_ dari seluruh variasi model.

---

## 📂 Struktur Direktori Proyek

```text
.
├── deployed_models/
│   ├── registry.json              # Registri performa dan daftar model
│   ├── label_encoder.joblib       # Encoder label kelas
│   ├── config.json                # Konfigurasi model
│   └── ...                        # File model (.keras/.joblib) dan scaler
├── streamlit/
│   ├── app.py                     # Entry point aplikasi Streamlit
│   ├── config.py                  # Konfigurasi audio dan label
│   ├── preprocessing.py           # Preprocessing dan segmentasi audio
│   ├── feature_extraction.py      # Ekstraksi fitur audio
│   ├── model_loader.py            # Pemuatan model
│   ├── inference.py               # Pipeline inferensi
│   └── visualization.py           # Visualisasi audio dan hasil prediksi
├── requirements.txt
└── README.md
```

---

## ⚙️ Instalasi

### 1. Prasyarat

Pastikan telah menginstal **Python 3.9** atau versi yang lebih baru.

### 2. Kloning Repositori

```bash
git clone <url-repositori>
cd <nama-direktori-proyek>
```

### 3. Membuat Virtual Environment (Disarankan)

```bash
# Membuat virtual environment
python -m venv venv

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# macOS/Linux
source venv/bin/activate
```

### 4. Instal Dependensi

```bash
pip install -r requirements.txt
```

> [!NOTE]
> `librosa` menggunakan pustaka `soundfile` untuk membaca berbagai format audio. Apabila mengalami kendala saat memuat audio pada Linux atau macOS, pastikan _system library_ `libsndfile` telah terpasang.

---

## 🚀 Menjalankan Aplikasi

Pastikan berada pada direktori utama proyek, kemudian jalankan:

```bash
streamlit run streamlit/app.py
```

Browser akan terbuka secara otomatis pada alamat lokal (umumnya `http://localhost:8501`).

---

## 🔍 Alur Prediksi

1. **Upload Audio**  
   Unggah rekaman tangisan bayi dalam format `.wav`, `.mp3`, `.ogg`, `.flac`, `.m4a`, `.aac`, atau `.wma`.

2. **Preprocessing**  
   Audio di-_resample_ menjadi **22.050 Hz**, amplitudo dinormalisasi ke rentang **[-1, 1]**, kemudian bagian hening di awal dan akhir dipotong.

3. **Segmentasi**  
   Audio dibagi menjadi segmen sepanjang **2 detik** tanpa tumpang tindih (_non-overlapping_). Segmen terakhir akan diberi _zero-padding_ apabila durasinya kurang dari 2 detik.

4. **Ekstraksi Fitur**  
   Diekstraksi fitur RMS, ZCR, Spectral Centroid, Spectral Bandwidth, Pitch (algoritma YIN), dan 13 koefisien MFCC.

5. **Klasifikasi**  
   Model menghasilkan salah satu dari lima kategori berikut:
   - 🤕 **Sakit Perut** (`belly_pain`)
   - 🤰 **Perlu Bersendawa** (`burping`)
   - 😣 **Tidak Nyaman** (`discomfort`)
   - 🍼 **Lapar** (`hungry`)
   - 😴 **Lelah / Mengantuk** (`tired`)

---

## 📊 Detail Model Klasifikasi

Aplikasi mendukung **16 kombinasi model** yang dapat dipilih langsung melalui sidebar.

| Algoritma         | Skema Fitur         | Under-sampling |
| ----------------- | ------------------- | -------------- |
| **SVM**           | MFCC / MFCC + Pitch | Ya / Tidak     |
| **Random Forest** | MFCC / MFCC + Pitch | Ya / Tidak     |
| **CNN**           | MFCC / MFCC + Pitch | Ya / Tidak     |
| **CNN-LSTM**      | MFCC / MFCC + Pitch | Ya / Tidak     |

Seluruh informasi metrik performa setiap model dibaca secara otomatis dari berkas `registry.json` dan ditampilkan pada antarmuka aplikasi ketika model dipilih.
