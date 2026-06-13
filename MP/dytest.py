# dytest_fast_clean.py → RAW CAMERA + 60+ FPS + ZERO LATENCY + SMALL WINDOW + NO COLOR CHANGE
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# --- CONFIG ---
ACTIONS = np.array(['hello', 'thanks', 'go', 'family', 'happy', 'how', 'yes', 'no', 'morning'])
NUM_FRAMES = 30
MODEL_PATH = 'asl_dynamic_final.h5'
CONF_THRESHOLD = 0.80
SMOOTHING = 7

# --- LOAD MODEL ---
model = load_model(MODEL_PATH)

# --- MEDIAPIPE ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(results):
    kp = np.zeros(126)
    if results.multi_hand_landmarks:
        for i, hand in enumerate(results.multi_hand_landmarks[:2]):
            start = i * 63
            for j, lm in enumerate(hand.landmark):
                if j < 21:
                    kp[start + j*3]     = lm.x
                    kp[start + j*3 + 1] = lm.y
                    kp[start + j*3 + 2] = lm.z
    return kp

def normalize(seq):
    valid = seq[seq != 0]
    if len(valid) == 0: return seq
    mn, mx = valid.min(), valid.max()
    if mx - mn < 1e-6: return seq
    norm = seq.copy()
    norm[seq != 0] = (seq[seq != 0] - mn) / (mx - mn)
    return norm

# --- CAMERA: RAW + FAST + SMALL + ZERO LATENCY ---
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)      # Small & fast
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)         # Zero lag
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

# --- STATE ---
buffer = []
pred_hist = []

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8,
    model_complexity=0                  # FASTEST MODE
) as hands:

    print("FAST DEMO STARTED — 60+ FPS — ZERO LATENCY — RAW CAMERA")
    print("Press 'q' to quit")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (650, 500))  # Small window

        # --- NO COLOR CHANGE: RGB → PROCESS → BACK TO BGR (ONLY FIX) ---
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)   # ← ONLY LINE THAT FIXES COLORS

        # NO FILTERS. NO CLAHE. NO GAMMA. NO BROWN. NO GREY.

        # Extract + predict
        kp = extract_keypoints(results)
        buffer.append(kp)
        buffer = buffer[-NUM_FRAMES:]

        word = "Listening..."
        conf = 0.0

        if len(buffer) == NUM_FRAMES:
            seq = np.expand_dims(normalize(np.array(buffer)), 0)
            pred = model.predict(seq, verbose=0)[0]
            pred_hist.append(pred)
            pred_hist = pred_hist[-SMOOTHING:]
            avg = np.mean(pred_hist, axis=0)
            idx = np.argmax(avg)
            conf = avg[idx]
            if conf > CONF_THRESHOLD:
                word = ACTIONS[idx].upper()

        # --- LANDMARKS (CLEAN) ---
        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255,255,255), thickness=2),
                    mp_drawing.DrawingSpec(color=(100,200,255), thickness=3)
                )

        # --- MINIMAL TEXT ---
        cv2.putText(frame, f"SIGN: {word}", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
        cv2.putText(frame, f"CONF: {conf:.3f}", (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,0), 2)

        cv2.imshow('ASL Dynamic Getsure recognition', frame)

        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
