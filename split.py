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

        # --- Lọc contour hợp lệ ---
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

        # --- Sắp xếp contour ---
        results.sort(key=lambda r: (r[0], r[1]))

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:

            # ✅ Nếu có đúng 2 khung → gộp toàn bộ và dò text ở vùng trên
            if len(results) == 2:
                y1, x1, w1, h1, _ = results[0]
                y2, x2, w2, h2, _ = results[1]

                # Bước 1: Gộp vùng cơ bản của 2 khung
                y_top = max(0, min(y1, y2) - 250)
                y_bottom = min(img.shape[0], max(y1 + h1, y2 + h2) + 300)
                x_min = max(0, min(x1, x2) - 80)
                x_max = min(img.shape[1], max(x1 + w1, x2 + w2) + 80)

                # Bước 2: Dò toàn ảnh để tìm dòng chữ trên cùng
                ocr_all = pytesseract.image_to_data(
                    gray, lang="eng", config="--psm 6", output_type=Output.DICT
                )
                all_tops = []
                for i, t in enumerate(ocr_all["text"]):
                    if t.strip():
                        all_tops.append(ocr_all["top"][i])
                if all_tops:
                    top_text = min(all_tops)
                    # Nếu dòng chữ trên cùng nằm cao hơn vùng cắt hiện tại >100px → mở rộng thêm
                    if top_text < y_top:
                        y_top = max(0, top_text - 100)

                # Bước 3: Cắt vùng mới (bao luôn phần text trên đầu)
                crop = img[y_top:y_bottom, x_min:x_max]
                _, enc = cv2.imencode(".jpg", crop)
                zipf.writestr("merged_two_blocks.jpg", enc.tobytes())
                print(f"[INFO] 2 blocks merged fully with top text preserved. y_top={y_top}")

            # ✅ Các trường hợp khác giữ nguyên logic cũ
            else:
                block_index = 0
                i = 0
                while i < len(results):
                    y, x, w, h, text1 = results[i]

                    if i + 1 < len(results):
                        y2, x2, w2, h2, text2 = results[i + 1]
                        if abs(y - y2) < 100:
                            y_top = min(y, y2)
                            y_bottom = max(y + h, y2 + h2) + 200
                            x_min_raw = min(x, x2)
                            x_max_raw = max(x + w, x2 + w2)
                            mid_x = (x_min_raw + x_max_raw) // 2
                            half_width = (x_max_raw - x_min_raw) // 2
                            x_min = max(0, mid_x - half_width)
                            x_max = min(img.shape[1], mid_x + half_width)
                            shift_left = int((x_max - x_min) * 0.05)
                            x_min = max(x_min - shift_left, 0)

                            crop = img[y_top:y_bottom, x_min:x_max]
                            _, enc = cv2.imencode(".jpg", crop)

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
