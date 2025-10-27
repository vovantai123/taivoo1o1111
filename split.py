from flask import Flask, request, send_file, jsonify
import cv2
import numpy as np
import io
import zipfile

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
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY_INV)

        # Nối khung + chữ thành block hoàn chỉnh
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 15))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blocks = []

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w > 300 and h > 200:
                blocks.append((x, y, w, h))

        # Sắp xếp top→bottom, left→right
        blocks.sort(key=lambda b: (b[1], b[0]))

        merged = []
        row_blocks = []

        for b in blocks:
            x, y, w, h = b
            if not row_blocks:
                row_blocks.append(b)
                continue

            prev_x, prev_y, prev_w, prev_h = row_blocks[-1]

            # Nếu cùng hàng
            if abs(y - prev_y) < 100:
                # Nếu gần nhau theo trục X → cùng CARE (Trước/Sau)
                if abs(x - (prev_x + prev_w)) < 200:
                    row_blocks.append(b)
                else:
                    # CARE mới
                    merged.append(row_blocks)
                    row_blocks = [b]
            else:
                merged.append(row_blocks)
                row_blocks = [b]

        if row_blocks:
            merged.append(row_blocks)

        # --- Xuất từng CARE block ---
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for i, group in enumerate(merged):
                xs = [x for x, _, w, _ in group]
                xe = [x + w for x, _, w, _ in group]
                ys = [y for _, y, _, _ in group]
                ye = [y + h for _, y, _, h in group]

                x_min, x_max = min(xs), max(xe)
                y_min, y_max = min(ys), max(ye) + 250  # CARE + PCS dưới
                y_max = min(y_max, img.shape[0])

                crop = img[y_min:y_max, x_min:x_max]
                _, enc = cv2.imencode(".jpg", crop)
                zf.writestr(f"care_{i+1}.jpg", enc.tobytes())

        zip_buffer.seek(0)
        return send_file(zip_buffer, as_attachment=True, download_name="care_blocks.zip")

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
