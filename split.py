from flask import Flask, request, send_file, jsonify
import cv2
import numpy as np
import pytesseract
import io
import zipfile

app = Flask(__name__)
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

@app.route("/split", methods=["POST"])
def split_image():
    try:
        if "file" not in request.files:
            return jsonify({"error": "Không tìm thấy file trong request"}), 400

        file = request.files["file"]
        img_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY_INV)

        # Morphological closing để nối liền khung và chữ bên dưới
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 15))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blocks = []

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w > 300 and h > 200:
                blocks.append((x, y, w, h))

        # Sắp xếp theo vị trí trên – dưới
        blocks.sort(key=lambda b: b[1])

        merged = []
        skip_next = False
        for i in range(len(blocks)):
            if skip_next:
                skip_next = False
                continue

            x, y, w, h = blocks[i]
            # Nếu có block kế tiếp nằm gần theo chiều ngang (cặp TRƯỚC + SAU)
            if i + 1 < len(blocks):
                x2, y2, w2, h2 = blocks[i + 1]
                if abs(y - y2) < 100:  # cùng hàng
                    # Gộp 2 block thành 1 vùng
                    x_min = min(x, x2)
                    x_max = max(x + w, x2 + w2)
                    y_top = min(y, y2)
                    y_bottom = max(y + h, y2 + h2) + 250  # lấy thêm vùng mã + PCS
                    y_bottom = min(y_bottom, img.shape[0])
                    merged.append((x_min, y_top, x_max - x_min, y_bottom - y_top))
                    skip_next = True
                    continue

        # Lưu ZIP các block
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
            for i, (x, y, w, h) in enumerate(merged):
                crop = img[y:y+h, x:x+w]
                _, enc = cv2.imencode(".jpg", crop)
                zipf.writestr(f"block_{i+1}.jpg", enc.tobytes())

        zip_buffer.seek(0)
        return send_file(zip_buffer, as_attachment=True, download_name="care_blocks.zip")

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
