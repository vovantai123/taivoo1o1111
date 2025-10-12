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

        # Threshold rõ nét để lấy contour
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

        # --- Lấy contour và lọc khung lớn (2 nhãn mỗi hàng) ---
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 100 and h > 100:
                boxes.append((x, y, w, h))

        if len(boxes) < 2:
            return jsonify({"error": "Không phát hiện đủ khung"}), 400

        # Sắp xếp theo y (từ trên xuống)
        boxes.sort(key=lambda b: (b[1], b[0]))

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
            i = 0
            label_index = 1

            while i < len(boxes) - 1:
                x1, y1, w1, h1 = boxes[i]
                x2, y2, w2, h2 = boxes[i + 1]

                # Nếu 2 khung nằm gần cùng hàng (tức là 1 cặp)
                if abs(y1 - y2) < 150:
                    # Lấy vùng bao quanh 2 khung
                    x_min = max(0, min(x1, x2) - 40)
                    x_max = min(img.shape[1], max(x1 + w1, x2 + w2) + 40)
                    y_top = max(0, min(y1, y2) - 100)
                    y_block_bottom = max(y1 + h1, y2 + h2)

                    # Tìm phần text bên dưới để lấy CARE/MÃ/PCS
                    search_y1 = y_block_bottom
                    search_y2 = min(search_y1 + 800, img.shape[0])
                    roi = gray[search_y1:search_y2, x_min:x_max]

                    ocr = pytesseract.image_to_data(
                        roi, lang="eng", config="--psm 6", output_type=Output.DICT
                    )

                    y_care = y_code = y_pcs = None
                    for j, text in enumerate(ocr["text"]):
                        t = text.strip().upper()
                        if not t:
                            continue
                        if "CARE" in t and y_care is None:
                            y_care = ocr["top"][j]
                        if re.search(r"501M", t) and y_code is None:
                            y_code = ocr["top"][j]
                        if "PCS" in t:
                            y_pcs = ocr["top"][j] + ocr["height"][j]

                    # --- Xác định điểm cắt dưới ---
                    if y_pcs:
                        y_bottom = search_y1 + y_pcs + 60
                    elif y_code:
                        y_bottom = search_y1 + y_code + 200
                    else:
                        y_bottom = search_y1 + 500  # fallback

                    y_bottom = min(y_bottom, img.shape[0])

                    # Cắt ra vùng hoàn chỉnh (2 khung + phần dưới)
                    crop = img[y_top:y_bottom, x_min:x_max]

                    # Đặt tên file theo text
                    text_crop = pytesseract.image_to_string(crop, lang="eng", config="--psm 6").upper()
                    care = re.search(r"(CARE\s*\d+)", text_crop)
                    code = re.search(r"(501M[A-Z0-9./-]*)", text_crop)
                    pcs = re.search(r"[\d.,]+\s*PCS", text_crop)
                    name_parts = []
                    if care: name_parts.append(care.group(1).replace(" ", "_"))
                    if code: name_parts.append(code.group(1))
                    if pcs: name_parts.append(pcs.group(0).replace(" ", "_"))
                    filename = "_".join(name_parts) if name_parts else f"label_{label_index}"

                    _, enc = cv2.imencode(".jpg", crop)
                    zipf.writestr(f"{filename}.jpg", enc.tobytes())

                    print(f"[INFO] Saved {filename}")
                    label_index += 1

                    # Bỏ qua 2 khung này → sang nhóm kế tiếp
                    i += 2
                    continue

                i += 1

        zip_buffer.seek(0)
        return send_file(zip_buffer, as_attachment=True, download_name="labels_split.zip", mimetype="application/zip")

    except Exception as e:
        print("[ERROR]", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
