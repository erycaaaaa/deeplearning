# Catatan verifikasi implementasi

Tanggal: **11 November 2026** (Asia/Jakarta). Pengujian dilakukan pada Windows dengan Python **3.12.2**, CPU, dan virtual environment proyek yang terisolasi.

## Environment

| Paket | Versi yang dijalankan |
| --- | --- |
| TensorFlow | 2.21.0 |
| Keras | 3.13.2 |
| Streamlit | 1.55.0 |
| Ultralytics | 8.4.31 |
| NumPy | 2.2.6 |
| Pillow | 11.3.0 |
| PyTorch | 2.15.0 |
| Torchvision | 0.29.2 |

`python -m pip check` menghasilkan **No broken requirements found**. Environment awal yang mewarisi paket Python global menimbulkan konflik dengan `tensorflow-intel 2.18.0`; environment akhir sudah terisolasi dan konflik tersebut tidak ada.

## Checkpoint dan notebook

- Model sumber: `models/selected_model.keras`, 23.689.141 byte. Sumber tetap utuh.
- Salinan identik untuk aplikasi: `models/mobilenet_model.keras`.
- SHA-256 kedua file: `e44c76ed0d49f1a9aa2fc3467912df65dc6c6c1fbf9333f4e21a4ca0e6873347`.
- Model berhasil dimuat menggunakan `tf.keras.models.load_model(..., compile=False)`.
- Input `(None,224,224,3)`, output `(None,4)`, softmax, dan `Rescaling(2,-1)` telah diperiksa.
- Referensi aktual: `references/Banana_Ripeness.ipynb`. Output notebook menyatakan `{'matang': 0, 'mentah': 1, 'setengah-matang': 2, 'terlalu-matang': 3}`.
- Preprocessing dibandingkan secara numerik dengan `tf.image.resize` bilinear pada `read_image` notebook. Unit test melewati pemeriksaan toleransi `1e-7`, dtype float32, rentang `[0,1]`, batch dimension, dan koreksi EXIF.
- Bobot resmi `models/yolov8n.pt` berhasil diunduh (6.549.796 byte). Tidak ada retraining atau fine-tuning.

## Unit, integrasi, dan smoke test

Seluruh **31 tes lulus** (`67,02` detik pada pengujian lengkap terakhir). JUnit tersimpan di `test-results/pytest.xml`. Setelah penyesuaian lebar pilihan mode, tes startup/pergantian mode dijalankan lagi dan lulus.

Pengujian mencakup validasi upload, file kosong/rusak/terpotong/terlalu besar, dimensi ekstrem, transparansi, EXIF, kelas dan probabilitas, koordinat/padding/anotasi, metadata banana, kegagalan download, hasil deteksi kosong, model asli, AppTest kedua mode, perubahan threshold tanpa inference otomatis, rerun tanpa inference ulang, serta reset foto.

Smoke test aktual dengan kedua checkpoint menghasilkan:

| Input | Hasil nyata |
| --- | --- |
| `samples/single_banana.jpeg` | `matang`, probabilitas `0,9966180921` (~99,66%) |
| `samples/multiple_bananas.jpeg` | 5 box valid; matang 1, mentah 1, setengah-matang 0, terlalu-matang 3 |
| Gambar putih 640 × 480 | 0 box; tanpa klasifikasi seluruh gambar |

Seluruh probabilitas finite, berada di `[0,1]`, dan berjumlah mendekati satu. Semua box berada dalam batas gambar asli. Jumlah hasil klasifikasi dan jumlah ringkasan sesuai jumlah box valid. Laporan lengkap dengan koordinat, confidence YOLO, serta probabilitas MobileNetV2 ada di `test-results/smoke_report.json`; anotasi sampel ada di `test-results/multiple_bananas_annotated.png`.

Foto sampel banyak pisang memiliki objek yang terlewat/tergabung oleh YOLO; **5 box bukan pernyataan jumlah semua buah pada foto**. Hasil ini tidak mengukur akurasi klasifikasi, deteksi, atau crop.

Pengujian menemukan bahwa Ultralytics mengganti `PIL.Image.open` dan mencoba plugin HEIF saat gambar rusak dibuka. Validasi aplikasi sekarang memakai referensi decoder Pillow asli. Kasus foto rusak sesudah YOLO dimuat sudah lulus tes dan tidak memerlukan plugin HEIF tambahan. Label anotasi juga sudah diperiksa secara visual dan dipisah agar tidak saling menimpa pada sampel.

## Startup dan browser

- Streamlit berhasil startup di port 8501. `/_stcore/health` memberikan **HTTP 200**, isi `ok`.
- Browser Chromium dijalankan melalui Playwright yang sudah tersedia pada environment pengujian (Python global, Playwright 1.63.0); server aplikasi dan inference tetap memakai `.venv` terisolasi.
- Alur browser nyata: upload JPG pada kedua mode → preview → analisis dengan model asli → hasil → reset.
- Download PNG berhasil dan tersimpan sebagai `test-results/browser_download.png`.
- Upload rusak setelah YOLO dimuat menampilkan pesan yang benar dan menonaktifkan analisis.
- PNG putih dianalisis melalui browser dan menghasilkan pesan tidak ada pisang terdeteksi.
- Layout diperiksa pada lebar **360, 390, 768, dan 1440 piksel**, termasuk hasil deteksi. Lebar dokumen sama dengan viewport dan tidak ditemukan tombol/gambar/pilihan mode yang melampaui batas horizontal.
- Screenshot desktop dan mobile diperiksa secara visual. Mode utama tersusun satu kolom di 360/390 piksel; gambar dan box mengikuti lebar container, dengan daftar objek untuk membaca label/skor lengkap.
- Tidak ada error JavaScript pada pengujian browser. Laporan rinci dan screenshot tersedia di `test-results/browser_report.json` dan `test-results/*.png`.

## Batas verifikasi

- Tidak memakai HP fisik; koneksi Wi-Fi, firewall laptop, izin kamera perangkat, dan HTTPS pada HP belum diuji langsung. Panduan tersedia pada README.
- Kamera telah diimplementasikan memakai `st.camera_input`; pengambilan foto kamera fisik belum diuji. Upload galeri menjadi alternatif yang telah diuji melalui browser.
- Instalasi native macOS/Linux, Docker build, dan deployment publik belum dijalankan.
- Tidak ada evaluasi akurasi baru, klaim kalibrasi probabilitas, atau pengujian video real-time.
- Artefak `test-results/` menggunakan sampel lokal/gambar putih pengujian dan diabaikan Git. Aplikasi tidak menyimpan upload pengguna ke disk.
