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

        # Làm sạch ảnh
        gray = cv2.medianBlur(gray, 3)
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 80 and h > 80:
                boxes.append((y, x, w, h))

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
                j = i

                # --- Gom dần xuống, OCR kiểm tra PCS toàn vùng ---
                while j < len(boxes):
                    y, x, w, h = boxes[j]
                    y_bottom = max(y_bottom, y + h)
                    x_min = min(x_min, x)
                    x_max = max(x_max, x + w)

                    region = gray[y_start:y_bottom, x_min:x_max]
                    text = pytesseract.image_to_string(region, lang="eng", config="--psm 6")

                    # Regex “mềm” để bắt PCS bị OCR sai
                    if re.search(r"p[\s\.\-_/]*c[\s\.\-_/]*s", text, re.IGNORECASE):
                        found_pcs = True
                        break
                    j += 1

                if found_pcs:
                    pad_top, pad_bottom, pad_left, pad_right = 15, 35, 25, 25
                    y1 = max(y_start - pad_top, 0)
                    y2 = min(y_bottom + pad_bottom, img.shape[0])
                    x1 = max(x_min - pad_left, 0)
                    x2 = min(x_max + pad_right, img.shape[1])

                    crop = img[y1:y2, x1:x2]
                    _, enc = cv2.imencode(".jpg", crop)
                    zipf.writestr(f"block_{block_index:02d}.jpg", enc.tobytes())
                    print(f"[OK] Block {block_index} có PCS ✅")

                    block_index += 1
                    i = j + 1
                else:
                    print("[SKIP] Không tìm thấy PCS, dừng lại.")
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
