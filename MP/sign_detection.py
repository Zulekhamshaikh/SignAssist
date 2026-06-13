# sign_detection.py
import numpy as np
import mediapipe as mp
import cv2
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
from collections import deque
from tensorflow.keras.models import load_model
import time
import sys # For better error printing

# ====== GLOBALS FILLED BY app.py ======
STATIC_LABELS = []
DYNAMIC_LABELS = []
STATIC_NORM_STEP = None
STATIC_CLF = None
DYNAMIC_MODEL = None

# 💡 NEW/UPDATED GLOBALS FOR GAP CONTROL
LAST_DETECTION_TIME = 0.0  # Time (seconds) when the last sign was successfully detected
LAST_DETECTED_WORD = ""    # Last word detected to prevent immediate repetition
INTER_GESTURE_GAP = 4.0    # The required time gap in seconds

# ====== CONFIG ======
DYNAMIC_BUFFER_SIZE = 30 # Matches your data collection script
FRAME_WIDTH = 480 # Increased size for better stability in vision processing
FRAME_HEIGHT = 360

# ====== Normalizer (Needed for joblib pipeline loading) ======
class HandNormalizer132(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def _norm_hand(self, hand):
        if np.all(hand == 0):
            return hand
        wrist = hand[0]
        if np.var(wrist) < 1e-6 or np.all(wrist == 0):
            valid = hand[np.any(hand != 0, axis=1)]
            anchor = valid.mean(axis=0) if len(valid) > 0 else hand.mean(axis=0)
        else:
            anchor = wrist

        centered = hand - anchor
        scale = np.ptp(centered[:, :2], axis=0).max()
        scale = max(scale, 1e-6)

        norm = centered / scale
        norm[:, 2] = np.clip(norm[:, 2], -1.5, 1.5)
        return norm

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        landmarks = X[:, :126].reshape(-1, 2, 21, 3)
        relative = X[:, 126:]
        for i in range(2):
            for j in range(landmarks.shape[0]):
                landmarks[j, i] = self._norm_hand(landmarks[j, i])
        flat = landmarks.reshape(-1, 126)
        return np.hstack([flat, relative])

# ====== Mediapipe ======
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
# FIX: Lowered confidence for better detection in low-quality cameras
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.6) 

prev_kp = None
motion_history = deque(maxlen=15)

# 💡 CRITICAL FIX: Dynamic buffer definition as a deque
dynamic_buffer = deque(maxlen=30) 


def extract_keypoints(results):
    """
    EXTRACTS keypoints by DETECTION ORDER (0, 1) to match original training data.
    """
    kp = np.zeros(126)
    
    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
            start = i * 63
            for j, lm in enumerate(hand_landmarks.landmark):
                kp[start + j*3] = lm.x
                kp[start + j*3 + 1] = lm.y
                kp[start + j*3 + 2] = lm.z
                
    return kp


def get_motion(kp):
    global prev_kp, motion_history
    if np.all(kp == 0):
        motion_history.append(0.0)
        prev_kp = None
        return 0.0
    if prev_kp is None or np.all(prev_kp == 0):
        prev_kp = kp.copy()
        motion_history.append(0.0)
        return 0.0

    diff = np.abs(kp - prev_kp)
    both = np.logical_and(kp != 0, prev_kp != 0)
    valid = diff[both]
    motion = np.mean(valid) if len(valid) > 0 else 0.0

    motion_history.append(motion)
    prev_kp = kp.copy()

    return np.mean(motion_history)


def draw_landmarks(frame, results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=2),
                mp_drawing.DrawingSpec(color=(255,0,255), thickness=2)
            )
    return frame


def normalize_sequence(seq):
    valid = seq[seq != 0]
    if valid.size == 0:
        return seq
    mn, mx = valid.min(), valid.max()
    if mx - mn < 1e-6:
        return seq
    norm = seq.copy()
    mask = seq != 0
    norm[mask] = (seq[mask] - mn) / (mx - mn)
    return norm


def process_vision_only(frame):
    """FAST lane: Mediapipe + drawing + motion."""
    ori_shape = frame.shape
    small = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT)) 
    
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    annotated = draw_landmarks(small.copy(), results)
    annotated = cv2.resize(annotated, (ori_shape[1], ori_shape[0]))

    kp = extract_keypoints(results)
    motion = get_motion(kp)

    return kp, motion, annotated

def reset_detection_state():
    """Resets the timing and word state, typically called when starting a new sentence."""
    global LAST_DETECTION_TIME, LAST_DETECTED_WORD
    LAST_DETECTION_TIME = 0.0
    LAST_DETECTED_WORD = ""
    dynamic_buffer.clear()


def detect_full_inference(kp, current_motion):
    """STATIC + DYNAMIC detection with crucial GAP enforcement."""
    global LAST_DETECTION_TIME, LAST_DETECTED_WORD
    
    detected = None
    conf = 0.0
    now = time.time()
    
    # CRITICAL GAP CHECK (Backend enforced)
    is_gap_clear = LAST_DETECTION_TIME == 0.0 or (now - LAST_DETECTION_TIME >= INTER_GESTURE_GAP)

    if np.any(kp != 0) and is_gap_clear:
        # --- 1. STATIC GESTURE (Uses DETECTION ORDER KEYPOINTS) ---
        if current_motion < 0.020:
            try:
                kp_full = np.pad(kp, (0, 6), 'constant').reshape(1, -1)
                kp_norm = STATIC_NORM_STEP.transform(kp_full)
                kp_clf = kp_norm[:, :126]
                prob = STATIC_CLF.predict_proba(kp_clf)[0]
                
                if np.max(prob) > 0.85:
                    candidate_word = STATIC_LABELS[np.argmax(prob)]
                    
                    if candidate_word.lower() != LAST_DETECTED_WORD.lower():
                        detected = candidate_word
                        conf = np.max(prob)

            except Exception as e:
                print(f"Static detection error: {e}", file=sys.stderr)
                pass
        
        # --- 2. DYNAMIC GESTURE (Uses DETECTION ORDER KEYPOINTS) ---
        # Only run dynamic prediction if no static word was detected
        if not detected:
            # 💡 CRITICAL: Append the raw (unflipped) keypoints to the deque
            dynamic_buffer.append(kp)
            
            # Dynamic prediction only runs when the buffer is full (30 frames)
            if len(dynamic_buffer) == DYNAMIC_BUFFER_SIZE:
                
                # Check for motion over the whole sequence:
                # Count frames in the buffer that actually contain non-zero keypoints
                active_frames = np.sum([np.any(f != 0) for f in dynamic_buffer]) 
                # If more than 50% of the frames have keypoints, assume motion/active gesture
                if active_frames > DYNAMIC_BUFFER_SIZE * 0.5: 
                    try:
                        # 💡 CRITICAL: Convert deque to numpy array first
                        seq_array = np.array(dynamic_buffer)
                        # Normalize the 30-frame sequence before prediction
                        seq = normalize_sequence(seq_array).reshape(1, 30, 126)
                        pred = DYNAMIC_MODEL.predict(seq, verbose=0)[0]
                        
                        if np.max(pred) > 0.88:
                            candidate_word = DYNAMIC_LABELS[np.argmax(pred)]
                            
                            if candidate_word.lower() != LAST_DETECTED_WORD.lower():
                                detected = candidate_word
                                conf = np.max(pred)
                                
                    except Exception as e:
                        # Print detailed traceback for TensorFlow errors
                        print(f"Dynamic detection error: {e}", file=sys.stderr)
                        pass
                
                # Clear buffer if prediction failed (motion too low, or confidence too low)
                if not detected:
                    dynamic_buffer.clear()

    # --- FINAL UPDATE OF STATE ---
    if detected:
        # Clear buffer upon successful detection
        dynamic_buffer.clear()
        
        LAST_DETECTION_TIME = now
        LAST_DETECTED_WORD = detected
    
    return detected, conf