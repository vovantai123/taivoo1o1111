from flask import Flask, request, send_file, jsonify
import cv2
import pytesseract
import numpy as np
import io
import zipfile
import re

pytesseract.pytesseract_cmd = "/usr/bin/tesseract"
app = Flask(__name__)

@app.route("/split", methods=["POST"])
def split_image():
    try:
        if "file" not in request.files:
            return jsonify({"error": "Không có file"}), 400

        file = request.files["file"]
        img_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        h_img, w_img = gray.shape[:2]

        # --- OCR toàn ảnh để dò vị trí text ---
        data = pytesseract.image_to_data(
            gray, lang="eng", config="--psm 6", output_type=pytesseract.Output.DICT
        )

        y_code = None
        y_pcs = None
        x_min, x_max = w_img, 0

        for i, text in enumerate(data["text"]):
            t = text.strip().upper()
            if not t:
                continue

            # 📍 Mã sản phẩm
            if re.search(r"501M", t):
                y_code = min(y_code or h_img, data["top"][i])
                x_min = min(x_min, data["left"][i])
                x_max = max(x_max, data["left"][i] + data["width"][i])

            # 📍 PCS dòng dưới
            if "PCS" in t:
                y_pcs = max(y_pcs or 0, data["top"][i] + data["height"][i])
                x_min = min(x_min, data["left"][i])
                x_max = max(x_max, data["left"][i] + data["width"][i])

        # --- Nếu phát hiện được mã & PCS ---
        if y_code and y_pcs:
            margin_top = 700  # để bao cả 2 khung trên
            margin_bottom = 60

            y_start = max(0, y_code - margin_top)
            y_end = min(h_img, y_pcs + margin_bottom)
            x_min = max(0, x_min - 80)
            x_max = min(w_img, x_max + 80)

            crop = img[y_start:y_end, x_min:x_max]
            _, enc = cv2.imencode(".jpg", crop)

            # --- Tạo tên file ---
            filename = f"care_label_{int(y_start)}_{int(y_end)}.jpg"

            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
                zipf.writestr(filename, enc.tobytes())

            zip_buffer.seek(0)
            return send_file(
                zip_buffer,
                as_attachment=True,
                download_name="care_label.zip",
                mimetype="application/zip",
            )

        else:
            return jsonify({
                "error": "Không phát hiện được mã hoặc PCS trong ảnh",
                "debug": {"y_code": y_code, "y_pcs": y_pcs}
            }), 400

    except Exception as e:
        print("[ERROR]", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
