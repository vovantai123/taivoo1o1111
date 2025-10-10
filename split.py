from flask import Flask, request, send_file, jsonify
import cv2
import pytesseract
from pytesseract import Output
import numpy as np
import io
import zipfile
import re

# ⚙️ Cấu hình Tesseract
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

        # --- OCR từng khung phát hiện ---
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

                # Chuẩn hóa lỗi OCR phổ biến
                replacements = {"&": "À", "¢": "ç", "|": "l", "¢¢": "é"}
                for wrong, right in replacements.items():
                    text = text.replace(wrong, right)

                results.append((y, x, w, h, text.strip()))

        # --- Sắp xếp block theo thứ tự trên - dưới, trái - phải ---
        results.sort(key=lambda r: (r[0], r[1]))

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
            block_index = 0
            i = 0

            while i < len(results):
                y, x, w, h, text1 = results[i]

                if i + 1 < len(results):
                    y2, x2, w2, h2, text2 = results[i + 1]
                    # Hai contour cùng hàng (trước / sau)
                    if abs(y - y2) < 100:
                        # --- Vùng cơ bản ---
                        y_top = min(y, y2)
                        y_bottom = max(y + h, y2 + h2) + 200

                        # --- Tính vùng ghép chính giữa ---
                        x_min_raw = min(x, x2)
                        x_max_raw = max(x + w, x2 + w2)

                        # Căn giữa hai nhãn để tránh lệch
                        mid_x = (x_min_raw + x_max_raw) // 2
                        half_width = (x_max_raw - x_min_raw) // 2

                        x_min = max(0, mid_x - half_width)
                        x_max = min(img.shape[1], mid_x + half_width)

                        # --- Nới nhẹ sang trái (5% chiều rộng, tránh mất viền trái) ---
                        shift_left = int((x_max - x_min) * 0.05)
                        x_min = max(x_min - shift_left, 0)

                        # --- Cắt vùng ---
                        region = gray[y_top:y_bottom, x_min:x_max]

                        # --- OCR dò chữ CODE / PCS ---
                        ocr_data = pytesseract.image_to_data(
                            region, lang="eng", config="--psm 6", output_type=Output.DICT
                        )

                        pcs_y_bottom = None
                        code_y_bottom = None
                        care_y_bottom = None

                        for j, word in enumerate(ocr_data["text"]):
                            textw = word.strip().upper()
                            top = ocr_data["top"][j]
                            height_word = ocr_data["height"][j]

                            if "PCS" in textw:
                                pcs_y_bottom = y_top + top + height_word + 10
                            elif "CARE" in textw:
                                care_y_bottom = y_top + top + height_word + 20
                            elif re.match(r"^[A-Z0-9/.\-]{5,}$", textw) and "PCS" not in textw:
                                code_y_bottom = y_top + top + height_word + 80

                        # --- Xác định vị trí cắt dưới ---
                        candidates = [v for v in [care_y_bottom, code_y_bottom, pcs_y_bottom] if v]
                        if candidates:
                            y_bottom = min(max(candidates) + 50, img.shape[0])
                        else:
                            y_bottom = min(y_bottom + 150, img.shape[0])

                        # --- Crop ---
                        crop = img[y_top:y_bottom, x_min:x_max]

                        # --- Encode ảnh ---
                        _, enc = cv2.imencode(".jpg", crop)

                        # --- Tìm CARE CODE để đặt tên ---
                        roi_code = gray[max(y_bottom - 200, 0):y_bottom, x_min:x_max]
                        code_text = pytesseract.image_to_string(
                            roi_code, lang="eng", config="--psm 6"
                        ).strip()

                        match = re.search(r"(CARE\s*\d+)", code_text.upper())
                        if match:
                            filename = match.group(1).replace(" ", "_") + ".jpg"
                        else:
                            filename = f"block_{block_index + 1}.jpg"

                        zipf.writestr(filename, enc.tobytes())
                        print(f"[INFO] Saved block {block_index + 1}: {filename}")

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
        print("[ERROR]", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
