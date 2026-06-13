from deep_translator import GoogleTranslator
from flask import Flask, request, jsonify
from waitress import serve
import base64
import numpy as np
import cv2
import joblib
import time
import threading
from flask_cors import CORS 

# ==== Import your detection engine ====
import sign_detection as sd

app = Flask(__name__)
CORS(app) 

# ==== Load Models ====
print("🔵 Loading STATIC pipeline...")
pipe = joblib.load("asl_STATIC_OLD_SAFE_v3.joblib")
sd.STATIC_NORM_STEP = pipe.named_steps["norm"]
sd.STATIC_CLF = pipe.named_steps["clf"]

label_encoder = joblib.load("le_OLD_SAFE.joblib")
sd.STATIC_LABELS[:] = list(label_encoder.classes_)

print("🟣 Loading DYNAMIC model...")
from tensorflow.keras.models import load_model
sd.DYNAMIC_MODEL = load_model("asl_dynamic_final.h5")
sd.DYNAMIC_LABELS[:] = ['hello','thanks','go','family','happy','how','yes','no','morning']

print("✅ All models loaded.")

# thread safety for Mediapipe
mp_lock = threading.Lock()


@app.route("/api/detect", methods=["POST"])
def detect():
    try:
        frame_data = request.json["frame"]
        img_bytes = base64.b64decode(frame_data.split(",")[1])
    except:
        return jsonify({"error": "bad_frame"}), 400

    np_img = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # ===== VISUAL PROCESSING (Mediapipe + motion + annotated image) =====
    with mp_lock:
        kp, motion, annotated = sd.process_vision_only(frame)

    _, buff = cv2.imencode(".jpg", annotated)
    annotated_b64 = base64.b64encode(buff).decode()

    # ===== ML INFERENCE =====
    # Passes keypoints (kp) and motion to detector
    word, conf = sd.detect_full_inference(kp, motion)

    # 💡 FIX: Convert NumPy float32 to Python float before JSON serialization
    return jsonify({
        "image_base64": annotated_b64,
        "word": word,
        "conf": float(conf) 
    })


@app.route("/api/new_sentence", methods=["POST"])
def new_sentence():
    sd.reset_detection_state() 
    return jsonify({"status": "ok"})

@app.route("/api/translate", methods=["POST"])
def translate_text():
    try:
        data = request.json

        text = data.get("text", "")
        target = data.get("target", "hi")

        translated = GoogleTranslator(
            source="auto",
            target=target
        ).translate(text)

        return jsonify({
            "translated": translated
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500
print(app.url_map)

@app.route("/api/analyze", methods=["POST"])
def analyze():

    sentence = request.json.get("sentence","").lower()

    intent = "General Communication"
    priority = "Low"
    actions = []
    emotion = ""

    if "happy" in sentence:
      emotion = "Happy"

    elif "mad" in sentence:
       emotion = "Frustrated"

    if "doctor" in sentence:
        intent = "Medical Assistance"
        priority = "High"
        actions = [
            "Contact Doctor",
            "Find Nearby Hospital",
            "Notify Caregiver"
        ]

    elif "help" in sentence:
        intent = "Emergency Request"
        priority = "Critical"
        actions = [
            "Call Emergency Contact",
            "Notify Family",
            "Locate Assistance"
        ]

    elif "food" in sentence:
      intent = "Food Request"
      priority = "Low"
      actions = [
          "Provide Food",
           "Show Meal Options",
            "Notify Caregiver if Request Repeats"
        ]

    elif "water" in sentence:
       intent = "Water Request"
       priority = "Low"
       actions = [
          "Provide Water",
           "Check Hydration Needs",
            "Notify Caregiver if Request Repeats"
        ]
       
    elif "house" in sentence:
       intent = "Location Inquiry"
       priority = "Medium"
       actions = [
          "Show Home Location",
           "Open Maps",
            "Provide Directions"
        ]

    elif "happy" in sentence:
       intent = "Positive Emotion"
       priority = "Low"
       actions = [
          "Continue Conversation",
           "Record Positive Feedback"
           "Share Positive Mood"
        ]
       
    elif "mad" in sentence:
      intent = "Negative Emotion"
      priority = "Medium"
      actions = [
          "Offer Assistance",
          "Provide Support",
           "Ask Follow-up Questions"
        ]

    return jsonify({
    "intent": intent,
    "priority": priority,
    "actions": actions,
    "emotion": emotion
})

@app.route("/api/knowledge", methods=["POST"])
def knowledge():

    sentence = request.json.get("sentence","").lower()

    if "food" in sentence:
        return jsonify({
            "knowledge":"Food is a basic human need and proper nutrition is important for health."
        })

    elif "water" in sentence:
        return jsonify({
            "knowledge":"Hydration is essential for maintaining body functions."
        })

    elif "house" in sentence:
        return jsonify({
            "knowledge":"A house provides shelter, safety and security."
        })
    
    elif "mad" in sentence:
       return jsonify({
          "knowledge":
           "The user may be experiencing frustration or distress and may need assistance."
    })

    elif "happy" in sentence:
        return jsonify({
           "knowledge":
            "The user is expressing a positive emotion."
    })

    return jsonify({
        "knowledge":"General communication detected."
    })


if __name__ == "__main__":
    print("🔥 ASL SERVER RUNNING at http://localhost:5000")
    serve(app, host="0.0.0.0", port=5000, threads=8)

