from flask import Flask, request, send_file, jsonify
import cv2
import pytesseract
import numpy as np
import io
import zipfile
import re

# Render dùng Linux, Tesseract cài trong /usr/bin/
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

        # --- OCR từng khung ---
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 80 and h > 80:
                roi = gray[y:y + h, x:x + w]
                roi = cv2.resize(roi, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                roi = cv2.adaptiveThreshold(
                    roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 31, 9
                )
                text = pytesseract.image_to_string(
                    roi, lang="eng+fra+spa", config="--oem 3 --psm 4"
                )

                replacements = {"&": "À", "¢": "ç", "|": "l", "¢¢": "é"}
                for wrong, right in replacements.items():
                    text = text.replace(wrong, right)

                results.append((y, x, w, h, text.strip()))

        # --- Sắp xếp các block ---
        results.sort(key=lambda r: (r[0], r[1]))

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
            block_index = 0
            i = 0
            while i < len(results):
                y, x, w, h, text1 = results[i]

                if i + 1 < len(results):
                    y2, x2, w2, h2, text2 = results[i + 1]
                    if abs(y - y2) < 100:  # cùng hàng
                        # Xác định vùng gộp chung
                        x_min = min(x, x2)
                        x_max = max(x + w, x2 + w2)
                        y_top = min(y, y2)
                        y_bottom = max(y + h, y2 + h2) + 200  # tạm lấy thêm để quét “PCS”

                        region = gray[y_top:y_bottom, x_min:x_max]
                        text_region = pytesseract.image_to_string(
                            region, lang="eng", config="--psm 6"
                        )

                        # Nếu tìm thấy chữ PCS thì cắt đến đó thôi
                        pcs_match = re.search(r"PCS", text_region, re.IGNORECASE)
                        if pcs_match:
                            # Tìm vị trí xuất hiện của PCS trong hình (theo dòng OCR)
                            roi_lines = text_region.splitlines()
                            line_index = next((idx for idx, l in enumerate(roi_lines) if "PCS" in l.upper()), None)
                            if line_index is not None:
                                # cắt đến dòng PCS thôi
                                h_per_line = region.shape[0] // max(len(roi_lines), 1)
                                y_bottom = y_top + h_per_line * (line_index + 1)

                        crop = img[y_top:y_bottom, x_min:x_max]
                        _, enc = cv2.imencode(".jpg", crop)

                        # Đọc phần mã CARE
                        roi_code = gray[y_bottom - 150:y_bottom, x_min:x_max]
                        code_text = pytesseract.image_to_string(
                            roi_code, lang="eng", config="--psm 6"
                        ).strip()

                        match = re.search(r"(CARE\s*\d+)", code_text.upper())
                        filename = match.group(1).replace(" ", "_") + ".jpg" if match else f"care_block_{block_index + 1}.jpg"

                        zipf.writestr(filename, enc.tobytes())

                        print(f"[INFO] CARE block {block_index + 1}: {filename}")
                        block_index += 1
                        i += 2
                        continue
                i += 1

        zip_buffer.seek(0)
        return send_file(
            zip_buffer,
            as_attachment=True,
            download_name="care_blocks.zip",
            mimetype="application/zip"
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
