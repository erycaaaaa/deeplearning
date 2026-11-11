from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_startup_and_mode_switch_without_inference():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=60).run()
    assert not app.exception
    assert app.button(key="analyze_single").disabled
    app.radio(key="mode_selector").set_value("Deteksi Banyak Pisang").run()
    assert not app.exception
    assert app.button(key="analyze_multi").disabled


def test_real_single_sample_and_reset():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=120).run()
    app.radio(key="single_source_0").set_value("Contoh lokal").run()
    app.button(key="analyze_single").click().run()
    assert not app.exception
    assert not app.error
    assert len(app.get("progress")) == 4
    original = app.session_state["single_result"]
    app.run()
    assert app.session_state["single_result"] is original  # reruns reuse the actual result
    app.button(key="reset_single").click().run()
    assert not app.exception
    assert app.button(key="analyze_single").disabled


def test_real_multi_sample_threshold_rerun_and_reset():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=180).run()
    app.radio(key="mode_selector").set_value("Deteksi Banyak Pisang").run()
    app.radio(key="multi_source_0").set_value("Contoh lokal").run()
    app.button(key="analyze_multi").click().run()
    assert not app.exception
    assert not app.error
    original = app.session_state["multi_result"]
    assert len(original.detections) == sum(original.counts.values()) > 0
    app.slider(key="detection_threshold").set_value(0.5).run()
    assert not app.exception
    assert app.session_state["multi_result"] is original
    assert any("Threshold telah berubah" in message.value for message in app.info)
    app.button(key="reset_multi").click().run()
    assert not app.exception
    assert app.button(key="analyze_multi").disabled


def test_no_detection_result_ui_has_clear_message_and_download():
    app = AppTest.from_string('''
from PIL import Image
from app import render_multi_result
from services.detector import DetectionResult
render_multi_result(DetectionResult([], Image.new("RGB", (640, 480), "white"), 0.25), 0.1)
''').run()
    assert not app.exception
    assert any("Tidak ada pisang terdeteksi" in message.value for message in app.warning)
    assert len(app.get("download_button")) == 1
