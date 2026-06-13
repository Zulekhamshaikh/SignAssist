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


if __name__ == "__main__":
    print("🔥 ASL SERVER RUNNING at http://localhost:5000")
    serve(app, host="0.0.0.0", port=5000, threads=8)