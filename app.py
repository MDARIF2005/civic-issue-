from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
import cv2
import os
from datetime import datetime
import google.generativeai as genai
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["OUTPUT_FOLDER"] = "static/output"

# === Gemini ===
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# === Load model ===
model = tf.keras.models.load_model("model/mobilenetv2_multilabel.h5")

# === Class names ===
class_names = [
    "bulk_garbage", "cracks", "open_manhole", "pothole", "trash",
    "aerosol_cans", "aluminum_food_cans", "aluminum_soda_cans", "cardboard_boxes",
    "disposable_plastic_cutlery", "food_waste", "glass_beverage_bottles",
    "paper_cups", "plastic_cup_lids", "plastic_shopping_bags",
    "plastic_trash_bags", "plastic_water_bottles", "styrofoam_food_containers"
]

COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0)
]

description_cache = {}

# === Function to generate CAM for multiple classes ===
def generate_multi_cam(image_path, results):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_array = np.expand_dims(img_resized.astype("float32") / 255.0, axis=0)

    last_conv_layer = model.get_layer("Conv_1")
    grad_model = tf.keras.models.Model([model.inputs], [last_conv_layer.output, model.output])

    overlay = img.copy()

    for idx, (cls, conf) in enumerate(results):
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model([img_array])
            loss = predictions[0][class_names.index(cls)]
        grads = tape.gradient(loss, conv_outputs)
        weights = tf.reduce_mean(grads, axis=(0, 1, 2))
        cam = np.dot(conv_outputs[0], weights.numpy())

        # Normalize CAM
        cam = np.maximum(cam, 0)
        cam = cam / cam.max()
        cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
        heatmap = (cam * 255).astype(np.uint8)

        color = COLORS[idx % len(COLORS)]
        mask = cv2.GaussianBlur(heatmap, (15, 15), 0)

        # Binary mask for contours
        _, binary = cv2.threshold(mask, 160, 255, cv2.THRESH_BINARY)
        binary = cv2.dilate(binary, np.ones((5, 5), np.uint8), iterations=2)
        binary = cv2.erode(binary, np.ones((3, 3), np.uint8), iterations=1)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 400:  # ignore small noise
                # Smooth the curve
                epsilon = 0.02 * cv2.arcLength(cnt, True)
                smooth_cnt = cv2.approxPolyDP(cnt, epsilon, True)

                # Draw curved outline
                cv2.polylines(overlay, [smooth_cnt], True, color, 3)

                # Find centroid for label placement
                M = cv2.moments(smooth_cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    continue

                # Label text
                label = f"{cls} ({conf*100:.1f}%)"
                (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

                # Calculate label position (shifted slightly upward)
                label_x = max(10, cx - text_w // 2)
                label_y = max(25, cy - text_h - 15)

                # Draw semi-transparent rectangle
                overlay_rect = overlay.copy()
                cv2.rectangle(
                    overlay_rect,
                    (label_x - 5, label_y - text_h - 5),
                    (label_x + text_w + 5, label_y + 10),
                    color,
                    -1
                )
                # Blend rectangle
                overlay = cv2.addWeighted(overlay_rect, 0.4, overlay, 0.6, 0)

                # Draw label text (bright color for visibility)
                cv2.putText(
                    overlay,
                    label,
                    (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

    output_path = os.path.join(app.config["OUTPUT_FOLDER"], os.path.basename(image_path))
    cv2.imwrite(output_path, overlay)
    return output_path


# === Prediction ===
def predict_image(img_path, threshold=0.8):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_array = np.expand_dims(img_resized.astype("float32") / 255.0, axis=0)
    preds = model.predict(img_array)[0]
    return [(class_names[i], float(preds[i])) for i in range(len(preds)) if preds[i] >= threshold]

# === Gemini Description ===
def get_description_from_gemini(issue_name, loc):
    if issue_name in description_cache:
        return description_cache[issue_name]
    try:
        model_g = genai.GenerativeModel("models/gemini-2.5-flash")
        prompt = f"Describe the civic issue '{issue_name}' and about its location {loc} in 1-2 lines: its impact and why it matters."
        response = model_g.generate_content(prompt)
        text = response.text.strip() if hasattr(response, "text") and response.text else "No description available."
        description_cache[issue_name] = text
        return text
    except Exception as e:
        print(f"⚠️ Gemini API Error: {e}")
        return "Description not available (API error)."

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    results = predict_image(filepath, threshold=0.8)
    output_img_path = generate_multi_cam(filepath, results)

    lat = request.form.get("latitude")
    lon = request.form.get("longitude")
    maps_link = f"https://www.google.com/maps?q={lat},{lon}" if lat and lon else None

    enriched_predictions = []
    for cls, conf in results:
        desc = get_description_from_gemini(cls,maps_link)
        enriched_predictions.append({
            "class": cls,
            "confidence": f"{conf*100:.2f}%",
            "description": desc
        })

    response = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "location": {"latitude": lat, "longitude": lon, "google_maps": maps_link},
        "predictions": enriched_predictions,
        "annotated_image": output_img_path
    }
    return jsonify(response)

if __name__ == "__main__":
    os.makedirs("static/uploads", exist_ok=True)
    os.makedirs("static/output", exist_ok=True)
    app.run(debug=True)
