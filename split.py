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
            return jsonify({"error": "Không tìm thấy file trong request"}), 400

        file = request.files["file"]
        img_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []

        # --- Lưu lại vị trí và text từng khung ---
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 80 and h > 80:
                roi = gray[y:y + h, x:x + w]
                roi = cv2.resize(roi, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
                roi = cv2.adaptiveThreshold(
                    roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 31, 9
                )
                text = pytesseract.image_to_string(
                    roi, lang="eng+fra+spa", config="--oem 3 --psm 6"
                )
                boxes.append((y, x, w, h, text.strip()))

        # --- Sắp xếp từ trên xuống dưới, trái sang phải ---
        boxes.sort(key=lambda r: (r[0], r[1]))

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
            i = 0
            block_index = 1
            while i < len(boxes):
                y_start = boxes[i][0]
                x_min = boxes[i][1]
                x_max = boxes[i][1] + boxes[i][2]
                y_bottom = boxes[i][0] + boxes[i][3]

                found_pcs = False

                # Gom dần xuống cho đến khi gặp “PCS”
                j = i
                while j < len(boxes):
                    y, x, w, h, text = boxes[j]
                    y_bottom = max(y_bottom, y + h)
                    x_min = min(x_min, x)
                    x_max = max(x_max, x + w)

                    if re.search(r"p\s*\.?\s*c\s*\.?\s*s", text, re.IGNORECASE):
                        found_pcs = True
                        break
                    j += 1

                if found_pcs:
                    # Mở rộng biên một chút để không cắt mất chữ
                    pad_top = 20
                    pad_bottom = 40
                    pad_left = 30
                    pad_right = 30

                    y1 = max(y_start - pad_top, 0)
                    y2 = min(y_bottom + pad_bottom, img.shape[0])
                    x1 = max(x_min - pad_left, 0)
                    x2 = min(x_max + pad_right, img.shape[1])

                    crop = img[y1:y2, x1:x2]
                    _, enc = cv2.imencode(".jpg", crop)

                    zipf.writestr(f"block_{block_index:02d}.jpg", enc.tobytes())
                    print(f"[OK] Cắt block {block_index:02d} (PCS found at box {j})")

                    block_index += 1
                    i = j + 1  # bắt đầu block mới từ khung sau PCS
                else:
                    # nếu không có PCS nữa thì bỏ qua phần còn lại
                    print(f"[SKIP] Không tìm thấy PCS sau khung {i}")
                    break

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

