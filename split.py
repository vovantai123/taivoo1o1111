from flask import Flask, request, send_file, jsonify
import cv2
import pytesseract
from pytesseract import Output
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
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blocks = []

        # --- Lọc contour lớn ---
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 100 and h > 100:
                blocks.append((x, y, w, h))
        blocks.sort(key=lambda b: (b[1], b[0]))

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
            i = 0
            group_index = 0
            while i < len(blocks) - 1:
                x1, y1, w1, h1 = blocks[i]
                x2, y2, w2, h2 = blocks[i + 1]

                # Hai khung trên cùng hàng
                if abs(y1 - y2) < 150:
                    x_min = max(0, min(x1, x2) - 40)
                    x_max = min(img.shape[1], max(x1 + w1, x2 + w2) + 40)
                    y_top = max(0, min(y1, y2) - 80)
                    y_blocks_bottom = max(y1 + h1, y2 + h2)

                    # --- OCR phần dưới của cặp này ---
                    search_y1 = y_blocks_bottom
                    search_y2 = min(search_y1 + 800, img.shape[0])
                    roi = gray[search_y1:search_y2, x_min:x_max]

                    ocr = pytesseract.image_to_data(roi, lang="eng", config="--psm 6", output_type=Output.DICT)

                    y_care = y_code = y_pcs = None

                    for j, text in enumerate(ocr["text"]):
                        t = text.strip().upper()
                        if not t:
                            continue
                        if "CARE" in t and y_care is None:
                            y_care = ocr["top"][j]
                        if re.search(r"501M", t):
                            y_code = ocr["top"][j]
                        if "PCS" in t:
                            y_pcs = ocr["top"][j] + ocr["height"][j]

                    # --- Xác định vùng cắt ---
                    if y_pcs:
                        y_bottom = search_y1 + y_pcs + 60
                    elif y_code:
                        y_bottom = search_y1 + y_code + 250
                    else:
                        y_bottom = search_y1 + 600

                    y_bottom = min(y_bottom, img.shape[0])

                    # --- Cắt vùng từ y_top tới dòng PCS ---
                    crop = img[y_top:y_bottom, x_min:x_max]

                    # --- Đặt tên file ---
                    roi_text = pytesseract.image_to_string(crop, lang="eng", config="--psm 6").upper()
                    code_match = re.search(r"(501M[A-Z0-9./-]*)", roi_text)
                    pcs_match = re.search(r"[\d.,]+\s*PCS", roi_text)
                    care_match = re.search(r"(CARE\s*\d+)", roi_text)

                    parts = []
                    if care_match:
                        parts.append(care_match.group(1).replace(" ", "_"))
                    if code_match:
                        parts.append(code_match.group(1))
                    if pcs_match:
                        parts.append(pcs_match.group(0).replace(" ", "_"))

                    filename = "_".join(parts) if parts else f"block_{group_index+1}"
                    _, enc = cv2.imencode(".jpg", crop)
                    zipf.writestr(f"{filename}.jpg", enc.tobytes())

                    print(f"[INFO] Saved: {filename}")
                    group_index += 1
                    i += 2
                    continue

                i += 1

        zip_buffer.seek(0)
        return send_file(zip_buffer, as_attachment=True, download_name="care_blocks.zip", mimetype="application/zip")

    except Exception as e:
        print("[ERROR]", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
