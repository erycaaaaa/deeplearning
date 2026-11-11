"""Banana Ripeness Classification — local Streamlit application."""
from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path

import streamlit as st

from services.classifier import (
    CLASS_ADVICE, CLASS_COLORS, CLASS_DESCRIPTIONS, CLASS_NAMES,
    DEFAULT_MODEL_PATH, ClassifierError, Prediction, load_classifier,
)
from services.detector import DetectionResult, DetectorError, load_detector
from utils.image_utils import MAX_BYTES, ImageValidationError, validate_image

ROOT = Path(__file__).resolve().parent
LOGGER = logging.getLogger(__name__)
MODES = ("Klasifikasi Pisang", "Deteksi Banyak Pisang")
VISUAL_NOTE = "Prediksi bersifat visual dan tidak menjamin kondisi bagian dalam buah. Periksa buah secara langsung."


def reset_photo(mode: str) -> None:
    for suffix in ("digest", "image", "result", "elapsed", "result_digest"):
        st.session_state.pop(f"{mode}_{suffix}", None)
    st.session_state[f"{mode}_revision"] = st.session_state.get(f"{mode}_revision", 0) + 1


def section_title(number: str, title: str, subtitle: str) -> None:
    st.html(f'<div class="section-heading"><span class="step">{number}</span>'
            f'<div><h2>{title}</h2><p>{subtitle}</p></div></div>')


def sample_path(mode: str) -> Path | None:
    stem = "single_banana" if mode == "single" else "multiple_bananas"
    return next((ROOT / "samples" / f"{stem}{suffix}" for suffix in (".jpg", ".jpeg", ".png")
                 if (ROOT / "samples" / f"{stem}{suffix}").is_file()), None)


def photo_input(mode: str):
    revision = st.session_state.get(f"{mode}_revision", 0)
    choices = ["Upload gambar", "Kamera"]
    example = sample_path(mode)
    if example:
        choices.append("Contoh lokal")
    source = st.radio("Sumber foto", choices, horizontal=True, width="stretch", key=f"{mode}_source_{revision}")
    data, filename = None, None
    if source == "Upload gambar":
        upload = st.file_uploader(
            "Pilih foto pisang", type=["jpg", "jpeg", "png"],
            help="JPG, JPEG, PNG · Maksimal 10 MB · Maksimal 24 megapiksel.",
            key=f"{mode}_upload_{revision}",
        )
        if upload is not None:
            if upload.size > MAX_BYTES:
                st.error("Ukuran foto melebihi 10 MB. Kompres foto lalu coba lagi.")
                return None
            data, filename = upload.getvalue(), upload.name
    elif source == "Kamera":
        st.caption("Ambil satu foto. Jika kamera tidak diizinkan, pilih Upload gambar dari galeri.")
        shot = st.camera_input("Ambil foto pisang", key=f"{mode}_camera_{revision}")
        if shot is not None:
            if shot.size > MAX_BYTES:
                st.error("Ukuran foto kamera melebihi 10 MB. Gunakan foto yang lebih kecil.")
                return None
            data, filename = shot.getvalue(), shot.name
        st.caption("Kamera pada HP biasanya memerlukan HTTPS. Upload galeri tetap dapat digunakan melalui Wi-Fi lokal.")
    elif example:
        data, filename = example.read_bytes(), example.name
        st.caption("Foto sampel workspace. Analisis tetap menggunakan model asli saat tombol ditekan.")

    if data is None:
        return None
    digest = hashlib.sha256(data).hexdigest()
    if st.session_state.get(f"{mode}_digest") != digest:
        st.session_state.pop(f"{mode}_result", None)
        st.session_state.pop(f"{mode}_image", None)
        st.session_state.pop(f"{mode}_digest", None)
        try:
            image = validate_image(data, filename)
        except ImageValidationError as exc:
            st.error(str(exc))
            return None
        st.session_state[f"{mode}_digest"] = digest
        st.session_state[f"{mode}_image"] = image
    image = st.session_state[f"{mode}_image"]
    st.image(image, caption=f"Preview foto · {image.width:,} × {image.height:,} piksel", width="stretch")
    return image


def empty_result(multi: bool) -> None:
    st.html('<div class="empty-result"><div class="scan-symbol" aria-hidden="true">⌗</div>'
            '<h3>Siap mengenali kematangan</h3><p>Masukkan foto, lalu tekan tombol analisis. '
            'Hasil akan muncul di sini.</p></div>')
    steps = ("Deteksi area pisang → klasifikasi setiap crop → hasil anotasi"
             if multi else "Foto pisang → MobileNetV2 → empat probabilitas kematangan")
    st.caption(steps)


def render_single_result(prediction: Prediction, elapsed: float) -> None:
    index = prediction.class_index
    st.html(f'<div class="prediction-card" style="--class-color:{CLASS_COLORS[index]}">'
            '<span class="eyebrow">HASIL KLASIFIKASI</span>'
            f'<h3>{prediction.label.capitalize()}</h3>'
            f'<p><strong>{prediction.confidence:.1%}</strong> probabilitas kelas terpilih</p></div>')
    st.write(CLASS_DESCRIPTIONS[index])
    st.markdown("**Probabilitas seluruh kelas**")
    for label, probability in zip(CLASS_NAMES, prediction.probabilities):
        st.progress(float(probability), text=f"{label.capitalize()} · {probability:.2%}")
    st.caption("Skor softmax MobileNetV2 menunjukkan keluaran model, bukan jaminan kebenaran prediksi.")
    st.html(f'<div class="advice"><strong>Saran penanganan</strong><p>{CLASS_ADVICE[index]}</p></div>')
    st.caption(f"Waktu proses: {elapsed:.2f} detik · Termasuk pemuatan model pada analisis pertama.")
    st.info(VISUAL_NOTE, icon="ℹ️")


def render_multi_result(result: DetectionResult, elapsed: float) -> None:
    count = len(result.detections)
    st.html(f'<div class="detection-total"><strong>{count}</strong><div>objek pisang terdeteksi'
            '<span>Berdasarkan bounding box YOLO pada foto ini</span></div></div>')
    st.image(result.annotated_image, caption="Hasil deteksi · Nomor pada kotak sesuai daftar objek", width="stretch")
    st.caption(f"Confidence threshold YOLO: {result.threshold:.2f} · NMS IoU: 0.45 · Waktu: {elapsed:.2f} detik")
    if not count:
        st.warning("Tidak ada pisang terdeteksi. Coba foto lebih dekat, pencahayaan lebih merata, "
                   "atau turunkan confidence threshold, lalu analisis kembali. Seluruh foto tidak diklasifikasikan.")
    else:
        cards = "".join(
            f'<div class="count-card"><span class="color-dot" style="background:{color}"></span>'
            f'<span>{label.capitalize()}</span><strong>{result.counts[label]}</strong></div>'
            for label, color in zip(CLASS_NAMES, CLASS_COLORS)
        )
        st.html(f'<div class="count-grid">{cards}</div>')
        st.markdown("**Prediksi per objek**")
        for item in result.detections:
            with st.expander(f"Objek {item.object_id} · {item.prediction.label.capitalize()} · "
                             f"MobileNetV2 {item.prediction.confidence:.1%}", expanded=count <= 3):
                st.write(f"**Confidence deteksi YOLO:** {item.detection_confidence:.2%}")
                st.write(f"**Probabilitas klasifikasi MobileNetV2:** {item.prediction.confidence:.2%}")
                for label, probability in zip(CLASS_NAMES, item.prediction.probabilities):
                    st.progress(probability, text=f"{label.capitalize()} · {probability:.2%}")
        st.warning("Pisang yang bertumpuk atau bergerombol dapat tergabung dalam satu kotak atau terlewat. "
                   "Jumlah kotak bukan jaminan jumlah seluruh pisang; klasifikasi crop gabungan bisa kurang akurat.")
        st.caption("Akurasi pada crop deteksi belum dievaluasi dan tidak dapat disamakan dengan evaluasi dataset utama.")
    st.download_button("Download hasil anotasi PNG", data=result.png_bytes,
                       file_name="banana_ripeness_result.png", mime="image/png",
                       on_click="ignore", width="stretch", key="download_annotation")
    st.info(VISUAL_NOTE, icon="ℹ️")


def workspace(multi: bool) -> None:
    mode = "multi" if multi else "single"
    left, right = st.columns([1, 1], gap="large")
    with left, st.container(key=f"input_{mode}", border=True):
        section_title("01", "Masukkan foto", "Gunakan foto yang terang dan tidak buram.")
        if not multi:
            st.caption("Untuk satu pisang atau satu objek utama. Gunakan mode deteksi untuk beberapa objek.")
        image = photo_input(mode)
        threshold = 0.25
        if multi:
            threshold = st.slider("Confidence threshold YOLO", 0.05, 0.95, 0.25, 0.05,
                                  key="detection_threshold",
                                  help="Nilai lebih rendah dapat menambah deteksi sekaligus meningkatkan deteksi keliru.")
            st.caption("Pengaturan baru diterapkan ketika tombol analisis ditekan.")
        missing_model = not DEFAULT_MODEL_PATH.is_file()
        if missing_model:
            st.error(f"Checkpoint belum ditemukan. Lokasi yang diharapkan: {DEFAULT_MODEL_PATH}")
        analyze = st.button("Deteksi & Analisis" if multi else "Analisis Kematangan",
                            type="primary", width="stretch", disabled=image is None or missing_model,
                            key=f"analyze_{mode}")
        if image is not None:
            st.button("Coba gambar lain" if multi else "Ganti foto", width="stretch",
                      key=f"reset_{mode}", on_click=reset_photo, args=(mode,))
        if analyze and image is not None:
            st.session_state.pop(f"{mode}_result", None)
            message = ("Memuat YOLO dan menganalisis setiap area pisang…" if multi
                       else "MobileNetV2 sedang menganalisis foto…")
            with st.spinner(message, show_time=True):
                start = time.perf_counter()
                try:
                    classifier = load_classifier()
                    if multi:
                        result = load_detector().detect_and_classify(image, classifier, threshold)
                    else:
                        result = classifier.classify(image)
                    st.session_state[f"{mode}_result"] = result
                    st.session_state[f"{mode}_result_digest"] = st.session_state[f"{mode}_digest"]
                    st.session_state[f"{mode}_elapsed"] = time.perf_counter() - start
                except (ClassifierError, DetectorError, ImageValidationError) as exc:
                    st.error(str(exc))
                except ImportError as exc:
                    st.error(f"Dependency belum lengkap: {exc}. Jalankan python -m pip install -r requirements.txt.")
                except Exception:
                    LOGGER.exception("Inference failed")
                    st.error("Analisis gagal. Periksa log terminal, dependency, dan checkpoint, lalu coba lagi. "
                             "Tidak ada prediksi yang dibuat.")
    with right, st.container(key=f"output_{mode}", border=True):
        section_title("02", "Hasil analisis", "Tingkat kematangan dan ringkasan foto Anda.")
        result = st.session_state.get(f"{mode}_result") if image is not None else None
        if result is None:
            empty_result(multi)
        elif st.session_state.get(f"{mode}_result_digest") == st.session_state.get(f"{mode}_digest"):
            if multi:
                if abs(result.threshold - threshold) > 1e-8:
                    st.info("Threshold telah berubah. Hasil di bawah memakai pengaturan sebelumnya. "
                            "Tekan Deteksi & Analisis untuk memperbarui.")
                render_multi_result(result, st.session_state[f"{mode}_elapsed"])
            else:
                render_single_result(result, st.session_state[f"{mode}_elapsed"])


def main() -> None:
    st.set_page_config(page_title="Banana Ripeness Classification", page_icon="🍌",
                       layout="wide", initial_sidebar_state="collapsed")
    st.html((ROOT / "styles" / "app.css").read_text(encoding="utf-8"))
    with st.container(key="shell"):
        st.html('<header class="brand-bar"><a class="brand" href="#" aria-label="Banana Ripeness beranda">'
                '<span class="brand-icon" aria-hidden="true">🍌</span><span>banana<span class="brand-light">vision</span></span></a>'
                '<span class="project-pill">UAS · DEEP LEARNING</span></header>')
        st.html('<section class="hero"><div class="hero-copy"><span class="eyebrow">DARI FOTO, KENALI KEMATANGANNYA</span>'
                '<h1>Banana Ripeness<br><span>Classification</span></h1>'
                '<p>Kenali tingkat kematangan pisang untuk membantu penanganan, '
                'konsumsi, dan prioritas penjualan. Cukup dengan satu foto.</p>'
                '<div class="hero-tags"><span>4 tingkat kematangan</span><span>Upload atau kamera</span>'
                '<span>Diproses lokal</span></div></div>'
                '<div class="hero-art" aria-hidden="true"><div class="art-orbit"></div>'
                '<span class="banana-art">🍌</span><span class="art-label">A little science.<br>A fresher perspective.</span></div></section>')
        st.html('<div class="mode-heading"><h2>Satu foto. Pilih cara analisisnya.</h2>'
                '<p>Klasifikasi satu objek atau deteksi beberapa area pisang sekaligus.</p></div>')
        mode = st.radio("Mode analisis", MODES, horizontal=True, width="stretch",
                        label_visibility="collapsed", key="mode_selector")
        multi = mode == MODES[1]
        st.caption("YOLOv8 menemukan area pisang; MobileNetV2 memprediksi kematangan setiap area."
                   if multi else "MobileNetV2 mengklasifikasikan kematangan objek utama pada foto.")
        workspace(multi)
        st.html('<section class="guide"><div><span class="eyebrow">FOTO YANG LEBIH BAIK</span>'
                '<h2>Bantu model melihat dengan jelas.</h2></div><div class="guide-tips">'
                '<p><strong>01 · Cahaya merata</strong><br>Hindari bayangan pekat dan pantulan berlebih.</p>'
                '<p><strong>02 · Objek terlihat utuh</strong><br>Dekatkan kamera dan gunakan latar sederhana.</p>'
                '<p><strong>03 · Beri jarak</strong><br>Pisahkan pisang agar deteksi lebih mudah.</p></div></section>')
        with st.expander("Tentang model & batasan penggunaan"):
            st.write("MobileNetV2 menggunakan checkpoint hasil transfer learning ImageNet dan augmentasi pencahayaan. "
                     "Tidak ada training ulang saat aplikasi dijalankan. Mode deteksi memakai YOLOv8n pretrained COCO.")
            st.write("Urutan output tetap: 0 = matang, 1 = mentah, 2 = setengah-matang, 3 = terlalu-matang. "
                     "Gambar dikoreksi orientasinya, diubah ke RGB, di-resize bilinear TensorFlow ke 224 × 224, "
                     "lalu dinormalisasi ke [0,1]. Rescaling ke [-1,1] sudah ada di dalam checkpoint.")
            st.write("Foto diproses dalam memori server lokal, tidak disimpan permanen, dan tidak dikirim ke API eksternal. "
                     "Unduhan pertama YOLO hanya mengambil bobot model publik. Kamera mengambil foto, bukan video real-time.")
            st.write("YOLO COCO dapat melewatkan pisang, menggabungkan beberapa buah dalam satu kotak, atau menghasilkan "
                     "deteksi keliru. Hasil crop belum memiliki evaluasi akurasi tersendiri. Tidak ada klaim akurasi untuk foto pengguna.")
            st.caption(VISUAL_NOTE)
        st.html('<footer class="footer"><span>🍌 <strong>Banana Ripeness Classification</strong></span>'
                '<span>Proyek UAS Deep Learning · Python, Streamlit & Computer Vision</span></footer>')


if __name__ == "__main__":
    main()
