# 🍌 Banana Ripeness Classification

Aplikasi UAS Deep Learning berbasis **Python, Streamlit, TensorFlow/Keras, dan Ultralytics YOLOv8**. Seluruh inference dilakukan di laptop/server aplikasi. Tidak ada database, login, training ulang, atau API inference eksternal.

- **Klasifikasi Pisang**: upload/ambil foto, preview, tekan **Analisis Kematangan**, lalu lihat prediksi MobileNetV2, empat probabilitas softmax, dan saran penanganan.
- **Deteksi Banyak Pisang**: YOLOv8n COCO → box `banana` → crop dengan padding 3% → MobileNetV2 → anotasi. Hasil mencakup jumlah box, jumlah per kelas, dua jenis skor yang dibedakan, dan download PNG.
- **Contoh lokal** menggunakan foto dari `samples/`; tombol analisis tetap menjalankan model asli.

## Persyaratan

Gunakan **Python 3.11 atau 3.12 64-bit** (direkomendasikan 3.12), koneksi internet untuk instalasi dan unduhan bobot pertama, serta ruang disk untuk TensorFlow dan PyTorch. CPU cukup; aplikasi tidak memerlukan CUDA. Checkpoint ini dilatih dengan TensorFlow **2.21.0** dan disimpan dengan Keras **3.13.2**; versi tersebut dikunci di `requirements.txt`.

## Model asli

Lokasi yang dimuat aplikasi:

```text
models/mobilenet_model.keras
models/yolov8n.pt
```

Workspace awal berisi `models/selected_model.keras`, bukan `mobilenet_model.keras`. Checkpoint asli sudah disalin **tanpa perubahan isi** ke nama yang diminta; sumber tidak ditimpa. SHA-256 keduanya:

```text
e44c76ed0d49f1a9aa2fc3467912df65dc6c6c1fbf9333f4e21a4ca0e6873347
```

Model besar diabaikan Git. Setelah clone di laptop lain, salin checkpoint asli secara terpisah. Tidak ada fallback prediksi palsu jika checkpoint hilang atau tidak kompatibel.

## Instalasi Windows (PowerShell)

Buka terminal pada folder proyek:

```powershell
cd C:\Users\Mayceren\Documents\GitHub\deeplearning
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/download_yolo.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Perintah di atas tidak memerlukan aktivasi environment atau perubahan execution policy PowerShell. Jika memakai laptop berbeda, sesuaikan perintah `cd`. Jika `python` menunjuk versi lain, gunakan executable Python 3.12 atau `py -3.12` jika Python Launcher tersedia.

Environment `.venv` menggunakan paket terisolasi agar TensorFlow lama pada Python global tidak menimbulkan konflik.

## Instalasi macOS/Linux

Pastikan `python3.12` tersedia (atau ganti dengan `python3.11`):

```bash
cd /path/to/deeplearning
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/download_yolo.py
python -m streamlit run app.py
```

Linux minimal/server dapat membutuhkan library OpenCV berikut:

```bash
sudo apt-get update
sudo apt-get install -y libgl1 libglib2.0-0
```

TensorFlow 2.21 membutuhkan wheel yang sesuai platform. Untuk Mac Intel lama yang tidak memiliki wheel kompatibel, gunakan Linux x86-64/VM atau laptop Windows/Linux yang didukung; jangan menurunkan versi TensorFlow secara acak untuk checkpoint ini.

## Menjalankan dan memakai aplikasi

Buka **http://localhost:8501** pada browser laptop. Terminal harus tetap berjalan. Hentikan dengan `Ctrl+C`.

1. Pilih **Klasifikasi Pisang** atau **Deteksi Banyak Pisang**.
2. Pilih sumber foto: **Upload gambar**, **Kamera**, atau **Contoh lokal**.
3. Periksa preview; untuk deteksi, atur confidence threshold jika perlu (default `0.25`).
4. Tekan tombol analisis. Inference hanya berjalan ketika tombol ditekan.
5. Baca prediksi dan peringatan. Mode deteksi menyediakan download PNG.
6. Tekan **Ganti foto** / **Coba gambar lain** untuk mengosongkan hasil sesi tersebut.

Mengubah threshold tidak otomatis mengulang inference. Hasil sebelumnya tetap menampilkan threshold yang dipakai dan aplikasi meminta analisis ulang. Model menggunakan `st.cache_resource`; hasil foto hanya berada dalam `session_state`, bukan cache bersama antar pengguna.

## Membuka dari HP pada Wi-Fi yang sama

Konfigurasi sudah menggunakan `server.address = "0.0.0.0"` dan port `8501`.

1. Hubungkan HP dan laptop ke Wi-Fi yang sama.
2. Cari IPv4 laptop: jalankan `ipconfig` pada Windows, `ipconfig getifaddr en0` pada macOS (interface dapat berbeda), atau `hostname -I` pada Linux.
3. Misalnya IPv4 laptop `192.168.1.20`, buka **http://192.168.1.20:8501** dari HP. Gunakan IP laptop, bukan `localhost` pada HP.
4. Jika gagal, pastikan server masih berjalan, jaringan bukan guest Wi-Fi dengan client isolation, dan VPN tidak memisahkan jaringan.
5. Di Windows, izinkan Python pada **Private networks** melalui Windows Defender Firewall. Jika perlu, administrator dapat membuat aturan inbound TCP **8501** untuk profil Private saja melalui *Windows Defender Firewall with Advanced Security*. Jangan mematikan seluruh firewall.
6. Di Linux dengan UFW, administrator dapat mengizinkan port untuk subnet lokal, misalnya `sudo ufw allow from 192.168.1.0/24 to any port 8501 proto tcp` (sesuaikan subnet).

**Kamera:** `st.camera_input` mengambil foto sekali, bukan video real-time. Browser biasanya mengizinkan kamera di `localhost` atau HTTPS, tetapi memblokirnya pada `http://192.168...`. Upload dari galeri tetap tersedia. Untuk kamera HP, gunakan deployment HTTPS atau reverse proxy HTTPS dengan sertifikat yang dipercaya HP. Berikan izin kamera di browser. Jangan menonaktifkan perlindungan browser atau CORS/XSRF.

## Unduhan YOLO dan penggunaan offline

Unduh sekali dengan `python scripts/download_yolo.py`. Jika belum ada, analisis pertama Mode 2 juga mencoba mengunduhnya, dengan batas waktu koneksi. Mode 1 tidak memerlukan YOLO dan tetap tersedia bila unduhan gagal.

Jika internet/proxy memblokir unduhan otomatis, ambil bobot dari [rilis resmi Ultralytics](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt) melalui browser, lalu simpan sebagai `models/yolov8n.pt`. Jangan menggantinya dengan bobot klasifikasi/segmentasi. Kategori `banana` ditemukan dari `model.names`, bukan indeks yang ditulis tetap dalam kode. Setelah dependency dan kedua checkpoint tersedia, kedua mode dapat dijalankan tanpa internet.

## Kontrak preprocessing dan kelas

Referensi yang benar-benar tersedia: `references/Banana_Ripeness.ipynb` (sel `read_image` dan `build_model`).

1. Validasi JPEG/PNG, koreksi orientasi EXIF, konversi RGB. Transparansi PNG dikomposit ke latar putih.
2. `tf.image.resize(..., (224, 224), method="bilinear", antialias=False)` mengikuti pipeline `read_image` notebook, termasuk perilaku sampling TensorFlow.
3. `float32 / 255.0`, clip ke `[0,1]`, tambah batch → `(1,224,224,3)`.
4. `tf.keras.models.load_model(model_path, compile=False)`; layer model `Rescaling(2.0, offset=-1.0)` mengubah rentang secara internal.
5. **Tidak** menggunakan `MobileNetV2.preprocess_input` di luar model. Tidak ada augmentasi pada inference.

| Indeks | Label output tetap | Warna anotasi |
| --- | --- | --- |
| 0 | matang | oranye |
| 1 | mentah | hijau |
| 2 | setengah-matang | kuning |
| 3 | terlalu-matang | cokelat |

Notebook memiliki helper visualisasi `predict_one` yang menggunakan default resize Pillow; aplikasi mengikuti bilinear TensorFlow pada pipeline training `read_image`, sesuai spesifikasi tugas. Urutan empat kelas di atas ditulis eksplisit dan tidak diurutkan ulang.

Mode deteksi memakai koordinat `xyxy` dari YOLO, NMS bawaan dengan IoU `0.45`, maksimum 100 deteksi per foto, clipping ke ukuran gambar asli, dan padding crop 3%. Anotasi memakai koordinat gambar yang telah dikoreksi EXIF, bukan koordinat input YOLO yang sudah di-resize. Kedua mode memakai fungsi preprocessing klasifikasi yang sama. Inference CPU diserialisasi dengan lock per model karena model cache dibagikan antarsesi.

## Validasi dan privasi

- Maksimal **10 MB**, **24 megapiksel**, sisi **32–12.000 piksel**, format nyata JPEG/PNG. Foto kosong, rusak, APNG, dan dimensi ekstrem ditolak dengan pesan.
- Foto pengguna hanya diproses dalam memori. YOLO menggunakan `save=False`, `save_txt=False`, dan `save_crop=False`; tidak ada upload ke layanan eksternal. Reset sesi melepaskan foto/hasil aplikasi. Server Streamlit tetap memegang data selama sesi browser aktif.
- File di `test-results/` hanya dibuat oleh skrip pengujian dari foto sampel lokal, bukan dari upload pengguna.
- Probabilitas harus finite, berada di `[0,1]`, memiliki empat kelas, dan berjumlah mendekati satu; output tidak valid memunculkan error.

## Pengujian

Instal dependency pengembangan:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/smoke_test.py
```

Di macOS/Linux setelah aktivasi, ganti `.\.venv\Scripts\python.exe` dengan `python`.

`tests/` mencakup validasi upload, orientasi EXIF, kesetaraan numerik preprocessing dengan notebook, validasi softmax/urutan kelas, clipping/padding/anotasi, lookup metadata, pipeline deteksi kosong, serta integrasi dengan checkpoint/foto asli. AppTest memeriksa startup, pergantian mode, prediksi sampel, reset, dan penggunaan ulang hasil pada rerun. Kasus unit tertentu memakai model stub hanya untuk menguji kondisi error/koordinat; aplikasi dan smoke test selalu menggunakan checkpoint asli.

Jalankan `python -m pytest -m "not integration" -q` untuk memisahkan tes integrasi checkpoint; AppTest prediksi sampel tetap membutuhkan model asli. Tes integrasi di-skip secara eksplisit jika model atau sampel yang diperlukan tidak tersedia. Smoke test menyimpan laporan nyata dan PNG anotasi di `test-results/`.

Pengujian browser opsional (server harus sudah berjalan):

```powershell
.\.venv\Scripts\python.exe -m playwright install chromium
.\.venv\Scripts\python.exe scripts/browser_smoke.py
```

Laporan hasil sesi implementasi tersedia di `docs/verification.md` setelah pengujian selesai. Jangan memakai hasil smoke test sebagai metrik akurasi dataset.

## Pemecahan masalah

| Gejala | Tindakan |
| --- | --- |
| Model tidak ditemukan | Salin checkpoint asli ke `models/mobilenet_model.keras`; pastikan bukan nama ganda `.keras.keras`. |
| Error deserialisasi Keras | Gunakan Python 3.11/3.12, TensorFlow 2.21.0, Keras 3.13.2; instal ulang `requirements.txt` dalam venv baru. |
| `DLL load failed` TensorFlow di Windows | Gunakan Python 64-bit, instal Microsoft Visual C++ Redistributable 2015–2022 x64, lalu buka terminal baru. |
| NumPy ABI / `dtype size changed` | Buat venv terisolasi baru dan instal requirements. Jangan mencampurkan paket NumPy versi lama dari Python global. |
| YOLO gagal download | Unduh bobot resmi manual; Mode 1 tetap dapat digunakan. |
| Tidak ada pisang terdeteksi | Atur cahaya/jarak objek atau threshold. Tidak ada fallback klasifikasi seluruh foto. |
| Hasil menyatu pada satu tandan | Pisahkan buah untuk pengambilan foto. YOLO COCO bukan penghitung setiap buah dalam tandan. |
| Port 8501 dipakai aplikasi lain | `python -m streamlit run app.py --server.port 8502`; gunakan port yang sama pada URL HP dan firewall. |

## Deployment jika diperlukan nanti

Aplikasi ini memerlukan server Python dengan RAM memadai untuk TensorFlow dan PyTorch; hosting static HTML/Cloudflare Workers tidak menjalankan stack ini. Opsi: VM Linux, platform container, atau Streamlit Community Cloud dengan kapasitas yang mencukupi. Upload checkpoint asli secara terpisah karena `.gitignore` mengabaikan model. Sediakan bobot YOLO pada server untuk menghindari unduhan saat request pengguna.

Dockerfile disediakan untuk deployment container:

```bash
docker build -t banana-ripeness .
docker run --rm -p 8501:8501 banana-ripeness
```

Build memerlukan `models/mobilenet_model.keras` dan disarankan `models/yolov8n.pt` dalam direktori build. Container membuka port 8501. Tambahkan reverse proxy HTTPS (misalnya Caddy/Nginx) di server agar kamera HP dapat dipakai. Docker/deployment tidak dijalankan otomatis dalam tugas implementasi lokal ini.

## Batasan

Prediksi bersifat visual, bukan pemeriksaan bagian dalam buah. Skor softmax bukan kepastian atau akurasi. YOLO COCO dapat menghasilkan false positive, melewatkan objek, atau menggabungkan banyak pisang dalam satu box. Jumlah yang ditampilkan adalah jumlah **box valid yang dihasilkan**, bukan klaim seluruh buah terhitung akurat. Akurasi klasifikasi crop belum dievaluasi dan tidak dapat disamakan dengan hasil dataset utama. Tidak ada fine-tuning YOLO tanpa dataset anotasi dan instruksi tambahan.

Dokumentasi resmi: [TensorFlow 2.21.0](https://pypi.org/project/tensorflow/2.21.0/), [Streamlit](https://docs.streamlit.io/), [YOLOv8](https://docs.ultralytics.com/models/yolov8/), [parameter prediksi YOLO](https://docs.ultralytics.com/modes/predict/).
