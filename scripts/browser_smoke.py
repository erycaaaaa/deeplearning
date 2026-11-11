"""Real browser uploads, real inference, and responsive overflow checks.

Start Streamlit first. Test images are the repository's local samples only.
"""
from __future__ import annotations

import json
import re
from io import BytesIO
from pathlib import Path

from PIL import Image
from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "test-results"


def verify_layout(page, name: str, width: int, height: int):
    page.set_viewport_size({"width": width, "height": height})
    page.wait_for_timeout(500)
    metrics = page.evaluate("""() => ({
        viewport: window.innerWidth,
        document: document.documentElement.scrollWidth,
        body: document.body.scrollWidth,
        overflowing: [...document.querySelectorAll('button, img, [role="radiogroup"]')]
            .filter(el => {
                const r = el.getBoundingClientRect();
                return r.width > 0 && (r.right > innerWidth + 1 || r.left < -1);
            }).map(el => ({tag: el.tagName, text: el.textContent.slice(0,70)}))
    })""")
    assert metrics["document"] <= width + 1, metrics
    assert not metrics["overflowing"], metrics
    # Streamlit scrolls its app panel, so select a useful viewport for screenshots.
    if name == "initial":
        page.locator(".brand-bar").scroll_into_view_if_needed()
    elif name == "single_result":
        page.locator(".prediction-card").scroll_into_view_if_needed()
    else:
        page.locator(".detection-total").scroll_into_view_if_needed()
    page.screenshot(path=str(OUTPUT / f"{name}_{width}.png"), full_page=True)
    return metrics


def main():
    OUTPUT.mkdir(exist_ok=True)
    errors = []
    report = {}
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1080}, device_scale_factor=1)
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto("http://localhost:8501", wait_until="networkidle", timeout=60_000)
        page.get_by_role("button", name="Analisis Kematangan", exact=True).wait_for()
        for width, height in [(360, 800), (390, 844), (768, 1024), (1440, 1080)]:
            report[f"initial_{width}"] = verify_layout(page, "initial", width, height)
        page.locator('input[type="file"]').set_input_files(str(ROOT / "samples" / "single_banana.jpeg"))
        page.get_by_text(re.compile(r"^Preview foto")).wait_for(timeout=30_000)
        page.get_by_role("button", name="Analisis Kematangan", exact=True).click()
        page.get_by_text("Probabilitas seluruh kelas", exact=True).wait_for(timeout=180_000)
        report["single_upload_inference"] = "passed"
        report["single_390"] = verify_layout(page, "single_result", 390, 844)
        report["single_1440"] = verify_layout(page, "single_result", 1440, 1080)
        page.get_by_role("button", name="Ganti foto", exact=True).click()
        expect(page.get_by_role("button", name="Analisis Kematangan", exact=True)).to_be_disabled(timeout=30_000)
        report["single_reset"] = "passed"
        page.get_by_text("Deteksi Banyak Pisang", exact=True).click()
        page.get_by_role("button", name="Deteksi & Analisis", exact=True).wait_for()
        page.locator('input[type="file"]').set_input_files(str(ROOT / "samples" / "multiple_bananas.jpeg"))
        page.get_by_text(re.compile(r"^Preview foto")).wait_for(timeout=30_000)
        page.get_by_role("button", name="Deteksi & Analisis", exact=True).click()
        page.get_by_text("Prediksi per objek", exact=True).wait_for(timeout=180_000)
        report["multi_upload_inference"] = "passed"
        for width, height in [(360, 800), (390, 844), (768, 1024), (1440, 1080)]:
            report[f"multi_{width}"] = verify_layout(page, "multi_result", width, height)
        with page.expect_download() as event:
            page.get_by_role("button", name="Download hasil anotasi PNG", exact=True).click()
        download = event.value
        download.save_as(str(OUTPUT / "browser_download.png"))
        assert download.failure() is None
        report["png_download"] = "passed"
        page.get_by_role("button", name="Coba gambar lain", exact=True).click()
        expect(page.get_by_role("button", name="Deteksi & Analisis", exact=True)).to_be_disabled(timeout=30_000)
        report["multi_reset"] = "passed"
        # Reproduce corrupt uploads after Ultralytics has been imported.
        page.locator('input[type="file"]').set_input_files({
            "name": "broken.jpg", "mimeType": "image/jpeg", "buffer": b"not a valid photo"
        })
        page.get_by_text("Foto rusak atau tidak dapat dibaca. Pilih file lain.", exact=True).wait_for()
        expect(page.get_by_role("button", name="Deteksi & Analisis", exact=True)).to_be_disabled(timeout=30_000)
        report["corrupt_upload_after_yolo"] = "passed"
        blank = BytesIO()
        Image.new("RGB", (640, 480), "white").save(blank, format="PNG")
        page.locator('input[type="file"]').set_input_files({
            "name": "blank.png", "mimeType": "image/png", "buffer": blank.getvalue()
        })
        page.get_by_text(re.compile(r"^Preview foto")).wait_for()
        page.get_by_role("button", name="Deteksi & Analisis", exact=True).click()
        page.get_by_text(re.compile(r"^Tidak ada pisang terdeteksi")).wait_for(timeout=60_000)
        report["real_blank_image_no_detection"] = "passed"
        report["blank_390"] = verify_layout(page, "multi_blank", 390, 844)
        report["javascript_errors"] = errors
        assert not errors, errors
        browser.close()
    (OUTPUT / "browser_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
