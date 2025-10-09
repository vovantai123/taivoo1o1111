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

        # Chuyển grayscale và nhị phân
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
                    if abs(y - y2) < 100:  # cùng hàng (TRƯỚC / SAU)
                        # --- Xác định vùng gộp ---
                        x_min = min(x, x2)
                        x_max = max(x + w, x2 + w2)
                        y_top = min(y, y2)
                        y_bottom = max(y + h, y2 + h2) + 200  # quét dư xuống để dò chữ

                        region = gray[y_top:y_bottom, x_min:x_max]

                        # --- OCR dò vị trí chữ PCS ---
                        ocr_data = pytesseract.image_to_data(
                            region, lang="eng", config="--psm 6", output_type=Output.DICT
                        )

                        pcs_y_bottom = None
                        code_y_bottom = None

                        # 🔍 tìm vị trí chữ "PCS"
                        for j, word in enumerate(ocr_data["text"]):
                            if "PCS" in word.upper():
                                top = ocr_data["top"][j]
                                height_word = ocr_data["height"][j]
                                pcs_y_bottom = y_top + top + height_word + 10  # dừng ngay sau chữ PCS
                                break

                        # 🔍 tìm dòng mã (chữ + số + /)
                        for j, word in enumerate(ocr_data["text"]):
                            text = word.strip().upper()
                            if re.match(r"^[A-Z0-9/.\-]{5,}$", text) and "PCS" not in text:
                                top = ocr_data["top"][j]
                                height_word = ocr_data["height"][j]
                                code_y_bottom = y_top + top + height_word + 80
                                break

                        # --- Quyết định điểm cắt dưới ---
                        # --- Quyết định điểm cắt dưới (xử lý cả CARE - CODE - PCS) ---
                        care_y_bottom = None

                        for j, word in enumerate(ocr_data["text"]):
                            if "CARE" in word.upper():
                                top = ocr_data["top"][j]
                                height_word = ocr_data["height"][j]
                                care_y_bottom = y_top + top + height_word + 20
                                break

                        # Lấy vị trí thấp nhất trong 3 loại (CARE, CODE, PCS)
                        candidates = [v for v in [care_y_bottom, code_y_bottom, pcs_y_bottom] if v]
                        if candidates:
                            y_bottom = min(max(candidates) + 50, img.shape[0])  # +50 để lấy trọn dòng PCS
                        else:
                            y_bottom = min(y_bottom + 150, img.shape[0])


                        # --- Dịch sang trái để có khoảng trống ---
                        shift_left = 50  # pixel cần thụt sang trái
                        x_min = max(x_min - shift_left, 0)

                        # --- Cắt block ra ---
                        crop = img[y_top:y_bottom, x_min:x_max]

                        # --- Mã hóa ảnh ---
                        _, enc = cv2.imencode(".jpg", crop)

                        # --- Lấy tên file theo CARE CODE (nếu có) ---
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
