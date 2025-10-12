from flask import Flask, request, send_file, jsonify
import cv2
import pytesseract
from pytesseract import Output
import numpy as np
import io
import zipfile
import re

pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
app = Flask(__name__)

@app.route("/split", methods=["POST"])
def split_image():
    try:
        if "file" not in request.files:
            return jsonify({"error": "Không tìm thấy file trong request"}), 400

        file = request.files["file"]
        img_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blocks = []

        # Lọc contour hợp lệ (2 khung chính)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 80 and h > 80:
                blocks.append((x, y, w, h))

        # Sắp xếp từ trái qua phải
        blocks.sort(key=lambda b: b[0])
        if len(blocks) < 2:
            return jsonify({"error": "Không phát hiện đủ 2 khung để gộp"}), 400

        # ✅ Lấy 2 khung chính
        x1, y1, w1, h1 = blocks[0]
        x2, y2, w2, h2 = blocks[1]

        # Gộp 2 khung
        x_min = max(0, min(x1, x2) - 50)
        x_max = min(img.shape[1], max(x1 + w1, x2 + w2) + 50)
        y_top = max(0, min(y1, y2) - 100)

        # --- Tìm dòng PCS để xác định y_bottom ---
        ocr_data = pytesseract.image_to_data(
            gray, lang="eng", config="--psm 6", output_type=Output.DICT
        )

        pcs_y_bottom = None
        for i, text in enumerate(ocr_data["text"]):
            t = text.strip().upper()
            if "PCS" in t:
                pcs_y_bottom = ocr_data["top"][i] + ocr_data["height"][i]
        if pcs_y_bottom:
            y_bottom = min(pcs_y_bottom + 50, img.shape[0])  # đủ để không cắt mất dòng PCS
        else:
            y_bottom = max(y1 + h1, y2 + h2) + 150  # fallback nếu OCR không thấy PCS

        # Cắt vùng cuối cùng
        crop = img[y_top:y_bottom, x_min:x_max]

        # ✅ Kiểm tra xem vùng dưới có chứa “501M” hay “PCS” không
        check_text = pytesseract.image_to_string(
            crop, lang="eng", config="--psm 6"
        ).upper()
        if not ("PCS" in check_text or "501M" in check_text):
            # Nếu không có, mở rộng nhẹ xuống dưới
            extend = 100
            y_bottom = min(y_bottom + extend, img.shape[0])
            crop = img[y_top:y_bottom, x_min:x_max]

        # Ghi file zip
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
            _, enc = cv2.imencode(".jpg", crop)
            zipf.writestr("label_pair_with_pcs.jpg", enc.tobytes())

        zip_buffer.seek(0)
        return send_file(
            zip_buffer,
            as_attachment=True,
            download_name="care_blocks.zip",
            mimetype="application/zip"
        )

    except Exception as e:
        print("[ERROR]", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
