from flask import Flask, request, send_file, jsonify
import cv2
import numpy as np
import pytesseract, io, zipfile

app = Flask(__name__)
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

@app.route("/split", methods=["POST"])
def split_image():
    try:
        if "file" not in request.files:
            return jsonify({"error": "Không có file"}), 400

        file = request.files["file"]
        img_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        # Dò đường kẻ dọc + ngang
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
        vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_v, iterations=2)
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_h, iterations=2)
        grid = cv2.add(vertical, horizontal)

        contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        blocks = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w > 300 and h > 200:
                blocks.append((x, y, w, h))

        # Sắp xếp từ trên xuống, trái qua phải
        blocks = sorted(blocks, key=lambda b: (b[1], b[0]))

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for i, (x, y, w, h) in enumerate(blocks):
                crop = img[y:y+h, x:x+w]
                _, enc = cv2.imencode(".jpg", crop)
                zf.writestr(f"block_{i+1}.jpg", enc.tobytes())

        zip_buffer.seek(0)
        return send_file(zip_buffer, as_attachment=True, download_name="care_blocks.zip")

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
