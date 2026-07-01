# 👶 Klasifikasi Jenis Tangisan Bayi — Aplikasi Streamlit

Aplikasi berbasis web ini digunakan untuk mendeteksi alasan bayi menangis dari berbagai berkas rekaman suara (seperti `.wav`, `.mp3`, `.ogg`, `.flac`, `.m4a`, dsb.). Proyek ini memanfaatkan model Machine Learning (ML) tradisional dan Deep Learning (DL) yang dilatih menggunakan dataset **Donate-a-Cry Corpus**.

Aplikasi ini dirancang dengan antarmuka yang ramah pengguna, visualisasi interaktif dari representasi audio (*waveform*, *spectrogram*, *MFCC*), serta alur kerja pemrosesan sinyal digital (*Digital Signal Processing*) yang transparan dan informatif.

---

## 🌟 Fitur Utama

1. **Pilihan 16 Model Klasifikasi**: Pengguna dapat memilih model klasifikasi berdasarkan eksperimen algoritma (SVM, Random Forest, CNN, CNN-LSTM), skema fitur (MFCC saja atau MFCC + Pitch), dan teknik penanganan ketidakseimbangan kelas (dengan/tanpa *Undersampling*).
2. **Visualisasi Alur Pemrosesan (DSP)**:
   * **Waveform Perbandingan**: Melihat sinyal audio mentah sebelum dan setelah dilakukan preprocessing (*resampling*, normalisasi amplitudo, dan pemotongan bagian sunyi).
   * **Visualisasi Segmen**: Menandai pembagian audio ke dalam potongan-potongan kecil berdurasi 2 detik.
   * **Analisis Spektogram & MFCC**: Menampilkan *Mel-Spectrogram* dan *Mel-Frequency Cepstral Coefficients* (MFCC) untuk segmen audio yang dipilih pengguna secara interaktif.
3. **Agregasi Prediksi Segmen**: Model mengklasifikasikan setiap segmen 2 detik secara independen, kemudian merata-ratakan nilai probabilitas seluruh segmen untuk mendapatkan hasil klasifikasi akhir yang stabil.
4. **Tabel Perbandingan Performa**: Fitur sidebar untuk membandingkan metrik akurasi, presisi, recall, dan F1-score dari 16 variasi model pada data uji.

---

## 📂 Struktur Direktori Proyek

```markdown
├── deployed_models/            # Menyimpan model latih dan metadata pendukung
│   ├── registry.json           # Registri performa dan nama file 16 model
│   ├── label_encoder.joblib    # Encoder kelas target
│   ├── config.json             # Konfigurasi model global
│   └── [Model & Scaler files]  # File model (.keras / .joblib) & scaler
├── streamlit/
│   ├── app.py                      # File entri utama aplikasi Streamlit
│   ├── config.py                   # Parameter audio & konfigurasi label kelas
│   ├── preprocessing.py            # Modul pembersihan & segmentasi audio
│   ├── feature_extraction.py       # Modul ekstraksi fitur (MFCC, Pitch, RMS, ZCR, dsb.)
│   ├── model_loader.py             # Modul pemuatan model ter-cache
│   ├── inference.py                # Orkestrasi alur prediksi audio
│   └── visualization.py            # Modul visualisasi grafik audio & probabilitas
├── requirements.txt                # Dependensi pustaka Python
└── README.md                       # Dokumentasi proyek (file ini)
```

---

## ⚙️ Cara Instalasi & Persiapan

### 1. Prasyarat
Pastikan Anda telah menginstal **Python 3.9** atau versi di atasnya pada komputer Anda.

### 2. Kloning Repositori & Masuk ke Direktori
```bash
git clone <url-repositori-ini>
cd <nama-direktori-proyek>
```

### 3. Buat Lingkungan Virtual (Opsional tetapi Disarankan)
```bash
# Membuat environment virtual
python -m venv venv

# Mengaktifkan environment virtual
# Di Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Di macOS/Linux:
source venv/bin/activate
```

### 4. Instal Dependensi Pustaka
Instal semua pustaka yang tercantum pada berkas `requirements.txt`:
```bash
pip install -r requirements.txt
```

> [!NOTE]
> Pustaka `librosa` bergantung pada `soundfile` untuk membaca format audio. Jika Anda mengalami kendala saat memuat audio pada OS tertentu (misal Linux/macOS), pastikan *system library* seperti `libsndfile` sudah terpasang.

---

## 🚀 Cara Menjalankan Aplikasi

1. Pastikan Anda berada di dalam direktori proyek utama.
2. Jalankan perintah Streamlit dengan menunjuk ke berkas `app.py` di dalam folder `streamlit`:
   ```bash
   streamlit run streamlit/app.py
   ```
3. Browser Anda akan terbuka secara otomatis dan menampilkan alamat lokal aplikasi (biasanya `http://localhost:8501`).

---

## 🔍 Alur Prediksi 5 Langkah

1. **Upload Audio**: Unggah berkas audio tangisan bayi (format `.wav`, `.mp3`, `.ogg`, `.flac`, `.m4a`, `.aac`, atau `.wma`).
2. **Preprocessing**: Sinyal di-*resample* ke $22.050\text{ Hz}$, amplitudonya dinormalisasi ke skala $[-1, 1]$, dan bagian hening (*silence*) di awal/akhir dipotong.
3. **Segmentasi**: Audio dipotong menjadi segmen-segmen sepanjang 2 detik tanpa tumpang tindih (*non-overlapping*). Jika durasi sisa segmen terakhir kurang dari 2 detik, dilakukan *zero-padding*.
4. **Ekstraksi Fitur**: Diekstraksi fitur RMS, ZCR, Spectral Centroid, Spectral Bandwidth, Pitch (via algoritma YIN), dan 13 koefisien MFCC.
5. **Hasil Klasifikasi**: Menampilkan prediksi akhir berupa salah satu dari 5 kondisi bayi:
   * **Sakit Perut** (`belly_pain`) 🤕
   * **Perlu Bersendawa** (`burping`) 🤰
   * **Tidak Nyaman** (`discomfort`) 😣
   * **Lapar** (`hungry`) 🍼
   * **Lelah / Mengantuk** (`tired`) 😴

---

## 📊 Detail Model Klasifikasi

Proyek ini mendukung evaluasi langsung dengan **16 kombinasi model** terlatih yang dapat dipilih pada sidebar aplikasi:

| Algoritma | Skema Fitur | Under-Sampling |
| :--- | :--- | :--- |
| **SVM** | MFCC Saja / MFCC + Pitch | Ya / Tidak |
| **Random Forest** | MFCC Saja / MFCC + Pitch | Ya / Tidak |
| **CNN** | MFCC Saja / MFCC + Pitch | Ya / Tidak |
| **CNN-LSTM** | MFCC Saja / MFCC + Pitch | Ya / Tidak |

Semua informasi metrik performa masing-masing kombinasi model ditampilkan secara otomatis dari berkas `registry.json` pada saat Anda memilih model tersebut di antarmuka web.
