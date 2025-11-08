# 🧠 Civic Issue Detection using AI (MobileNetV2 + Flask + Gemini)

An AI-powered web application that detects **civic issues** such as potholes, open manholes, and garbage from uploaded images.  
It automatically labels issues, generates short descriptions using **Google Gemini AI**, and adds real-time **Google Maps** location links.

---

## 🚀 Features
- 🔍 Detects multiple civic issues in one image (multi-label classification)
- 🧩 Highlights detected regions using heatmaps and bounding curves
- 🧠 Auto-generates issue descriptions using **Gemini 2.5 Flash**
- 🌍 Captures live GPS location and adds **Google Maps** link
- 💻 Web interface built with **Flask**, **TensorFlow**, and **OpenCV**
- 📷 Supports image upload & instant visualization of detected issues

---

## 🏗️ Project Structure
📦 edunet-project/
┣ 📂 model/
┃ ┗ mobilenetv2_multilabel.h5
┣ 📂 static/
┃ ┣ 📂 uploads/ # Uploaded images
┃ ┗ 📂 output/ # Annotated output images
┣ 📂 templates/
┃ ┗ index.html # Frontend UI
┣ app.py # Flask backend
┣ requirements.txt # Dependencies
┗ README.md # Project documentation




---

## ⚙️ Installation Guide

### 1️⃣ Clone the Repository

git clone https://github.com/MDARIF2005/civic-issue-.git
cd civic-issue-detection
2️⃣ Install Required Packages


pip install -r requirements.txt
3️⃣ Configure Google Gemini API Key
Obtain your API key from Google AI Studio
Then set it as an environment variable:

🪟 For Windows (PowerShell)


setx GEMINI_API_KEY "YOUR_API_KEY_HERE"
🐧 For macOS/Linux


export GEMINI_API_KEY="YOUR_API_KEY_HERE"
Or add it directly inside app.py:



genai.configure(api_key="YOUR_API_KEY")
▶️ Running the Flask App


python app.py
Then open your browser and visit:
👉 http://127.0.0.1:5000

🧠 How It Works
Upload a photo of a street or area.

The app uses MobileNetV2 to detect objects like potholes, garbage, or open manholes.

OpenCV draws focus heatmaps/contours around detected regions.

Gemini AI generates a short description of each issue.

The app displays:

✅ Detected issue names

💯 Confidence scores

📝 AI-generated descriptions

🌐 Clickable Google Maps location

🌍 Example Output
Uploaded Image	Annotated Output

Predictions:



{
  "time": "2025-11-08 16:10:42",
  "location": {
    "latitude": "17.443",
    "longitude": "78.391",
    "google_maps": "https://www.google.com/maps?q=17.443,78.391"
  },
  "predictions": [
    {
      "class": "pothole",
      "confidence": "92.15%",
      "description": "A pothole is a road defect that causes traffic risk and vehicle damage."
    },
    {
      "class": "garbage",
      "confidence": "85.43%",
      "description": "Garbage accumulation causes pollution and blocks drainage systems."
    }
  ],
  "annotated_image": "static/output/test_image.png"
}
🧩 Tech Stack
Component	Technology
Frontend	HTML5, CSS3, JavaScript
Backend	Flask (Python)
Model	TensorFlow MobileNetV2
Visualization	OpenCV (Bounding curves / heatmaps)
AI Descriptions	Google Gemini 2.5 Flash
Deployment	Gunicorn + Render / Railway

🔑 Environment Variables
Variable	Description
GEMINI_API_KEY	Your Google Gemini API key
UPLOAD_FOLDER	Path for uploaded files
OUTPUT_FOLDER	Path for annotated images

🧰 Dependencies
See requirements.txt:


Flask==3.0.3
Werkzeug==3.0.3
tensorflow==2.15.0
opencv-python==4.10.0.84
numpy==1.26.4
google-generativeai==0.7.2
Pillow==10.4.0
matplotlib==3.9.2
gunicorn==23.0.0
Install them all:


pip install -r requirements.txt
☁️ Deployment (Optional)
For hosting on Render / Railway:



web: gunicorn app:app
runtime.txt


python-3.11.9
🧪 Test Images
You can generate civic issue test data using Gemini or upload:

pothole.png

open_manhole.png

garbage.png

👨‍💻 Author
MD Arif
🎓 B.Tech (AI & ML), HITAM College, Hyderabad
🚀 Student Startup Coordinator | Social Media Manager | AI Developer
💼 LinkedIn | 🌐 Portfolio

