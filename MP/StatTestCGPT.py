import cv2
import mediapipe as mp
import numpy as np
import joblib

PIPELINE_FILE = 'asl_STATIC_OLD_SAFE_v3.joblib'
ENCODER_FILE = 'le_OLD_SAFE.joblib'

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.85, min_tracking_confidence=0.85)
mp_drawing = mp.solutions.drawing_utils

def extract_raw_126(results):
    vec = np.zeros(126, dtype=np.float32)
    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            handed = results.multi_handedness[idx].classification[0].label  # 'Right' or 'Left'
            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
            if handed == 'Right':
                vec[:63] = coords
            else:
                vec[63:] = coords
    return vec.reshape(1, -1)

print("Loading model...")
pipeline = joblib.load(PIPELINE_FILE)
label_encoder = joblib.load(ENCODER_FILE)

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for lm in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

        X = extract_raw_126(results)
        probs = pipeline.predict_proba(X)[0]
        idx = int(np.argmax(probs))
        conf = probs[idx]

        if conf < 0.60:
            text = "Uncertain"
        else:
            text = f"{label_encoder.inverse_transform([idx])[0]} ({conf*100:.1f}%)"
    else:
        text = "No Hand"

    cv2.putText(frame, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 3)
    cv2.putText(frame, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)

    cv2.imshow("SignAssist Live", frame)
    if (cv2.waitKey(5) & 0xFF) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
