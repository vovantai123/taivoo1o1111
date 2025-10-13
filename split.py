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
        gray = cv2.medianBlur(gray, 3)

        # OCR full ảnh, lấy toạ độ từng dòng
        data = pytesseract.image_to_data(
            gray, lang="eng", config="--psm 6", output_type=pytesseract.Output.DICT
        )

        pcs_lines = []
        for i, text in enumerate(data["text"]):
            if re.search(r"p[\s\.\-_/]*[c0gq][\s\.\-_/]*[s5\$]", text, re.IGNORECASE):
                top = data["top"][i]
                height = data["height"][i]
                pcs_lines.append(top + height)

        if not pcs_lines:
            return jsonify({"error": "Không tìm thấy dòng PCS nào trong ảnh"}), 404

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
            prev_y = 0
            block_index = 1

            for pcs_y in pcs_lines:
                # Cắt từ phần trên (prev_y) đến dòng PCS (pcs_y)
                y1 = max(prev_y - 30, 0)
                y2 = min(pcs_y + 40, img.shape[0])

                crop = img[y1:y2, :]
                _, enc = cv2.imencode(".jpg", crop)
                zipf.writestr(f"block_{block_index:02d}.jpg", enc.tobytes())

                print(f"[OK] Block {block_index}: cắt đến dòng PCS tại y={pcs_y}")
                block_index += 1
                prev_y = pcs_y

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
