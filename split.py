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
        results = []

        # --- Lọc contour hợp lệ ---
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 80 and h > 80:
                results.append((y, x, w, h))
        results.sort(key=lambda r: (r[0], r[1]))

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:

            if len(results) == 2:
                y1, x1, w1, h1 = results[0]
                y2, x2, w2, h2 = results[1]

                y_top = max(0, min(y1, y2) - 200)
                x_min = max(0, min(x1, x2) - 60)
                x_max = min(img.shape[1], max(x1 + w1, x2 + w2) + 60)

                # --- Dò vùng dưới để tìm “PCS” chính xác ---
                h_img = img.shape[0]
                bottom_zone_y = int(h_img * 0.35)
                roi_bottom = gray[bottom_zone_y:h_img, :]
                text_bottom = pytesseract.image_to_string(
                    roi_bottom, lang="eng", config="--psm 6"
                )

                pcs_line = re.search(r"(\d[\d., ]*PCS)", text_bottom.upper())
                if pcs_line:
                    # Lấy toạ độ tương ứng của dòng “PCS”
                    data = pytesseract.image_to_data(
                        roi_bottom, lang="eng", config="--psm 6", output_type=Output.DICT
                    )
                    pcs_y_bottom = None
                    for i, word in enumerate(data["text"]):
                        if "PCS" in word.upper():
                            pcs_y_bottom = bottom_zone_y + data["top"][i] + data["height"][i]
                            break
                    if pcs_y_bottom:
                        y_bottom = min(pcs_y_bottom + 50, int(h_img * 0.9))
                    else:
                        y_bottom = int(h_img * 0.9)
                else:
                    # fallback an toàn
                    y_bottom = int(h_img * 0.9)

                # --- Cắt hình ---
                crop = img[y_top:y_bottom, x_min:x_max]
                _, enc = cv2.imencode(".jpg", crop)
                zipf.writestr("merged_two_blocks.jpg", enc.tobytes())
                print(f"[INFO] Cropped up to PCS line (y_bottom={y_bottom})")

            else:
                # Giữ logic cũ
                for i, (y, x, w, h) in enumerate(results):
                    crop = img[y:y + h, x:x + w]
                    _, enc = cv2.imencode(".jpg", crop)
                    zipf.writestr(f"block_{i + 1}.jpg", enc.tobytes())

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
