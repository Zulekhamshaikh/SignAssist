import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from collections import defaultdict

# --- Configuration ---
DATASET_PATH = 'rashmi.csv'
TARGET_FRAMES = 1000  # Number of frames to capture per gesture

# Map keyboard keys to your specific gestures
GESTURE_MAP = {
    ord('k'): "help",
    ord('e'): "stop",
    ord('w'): "water",
    ord('f'): "food",
    ord('h'): "house",
    ord('r'): "friend",
    ord('d'): "doctor",
    ord('a'): "mad",
    ord('c'): "what",
    ord('o'): "where",
    ord('g'): "this",
    ord('y'): "that",
    ord('i'): "I",
    ord('t'): "you"
}

# --- Setup MediaPipe ---
mp_hands = mp.solutions.hands
# Use high confidence for cleaner data collection
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.85, 
    min_tracking_confidence=0.85
)
mp_drawing = mp.solutions.drawing_utils


# --- Core Function: Feature Extraction (Unified 126 features/row) ---
def extract_features(results):
    """
    Extracts and normalizes 126 landmark features (63 per hand) into a single vector.
    Fills with zeros if a hand is not detected, ensuring a fixed-length row.
    """
    feature_vector = np.zeros(126, dtype=np.float32)

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # MediaPipe's handedness is critical here: 'Right' or 'Left'
            handedness_info = results.multi_handedness[idx].classification[0].label
            
            # Flatten 21 landmarks * 3 coords (x, y, z) = 63 features
            # Note: Features are already normalized by MediaPipe to 0-1 range relative to the frame size.
            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

            if handedness_info == 'Right':
                # Right hand features (0-62)
                feature_vector[0:63] = coords
            elif handedness_info == 'Left':
                # Left hand features (63-125)
                feature_vector[63:126] = coords

    return feature_vector


def create_header():
    """Generates the CSV header row (label, R_X1...R_Z21, L_X1...L_Z21)."""
    headers = ['label']
    for hand in ['R', 'L']:
        for i in range(1, 22):
            for coord in ['X', 'Y', 'Z']:
                headers.append(f'{hand}_{coord}{i}')
    return headers


def collect_data():
    """Main loop for camera feed and key-triggered data collection."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Check/Create CSV file
    if not os.path.exists(DATASET_PATH):
        with open(DATASET_PATH, 'w') as f:
            f.write(','.join(create_header()) + '\n')
        print(f"Created new dataset file: {DATASET_PATH}")
    else:
        print(f"Appending data to existing dataset: {DATASET_PATH}")
    
    # State variables
    data_to_save = []
    current_gesture = None
    collected_count = 0
    total_saved_counts = defaultdict(int)

    # Initial CSV check for existing counts (useful when continuing a session)
    try:
        if os.path.getsize(DATASET_PATH) > 0:
            existing_df = pd.read_csv(DATASET_PATH)
            for label in existing_df['label'].unique():
                total_saved_counts[label] = len(existing_df[existing_df['label'] == label])
    except pd.errors.EmptyDataError:
        pass # File exists but is empty, continue
    except Exception as e:
        print(f"Error reading existing CSV: {e}")
        pass
    
    # Map for display purposes
    key_display = ", ".join([f"'{chr(k)}'={label.upper()}" for k, label in GESTURE_MAP.items()])

    print(f"\nAvailable Gestures: {key_display}")
    print("Press the corresponding key to START recording, or 'q' to QUIT and save.")

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image = cv2.flip(image, 1) # Flip image horizontally for natural view
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image with MediaPipe Hands
        results = hands.process(image_rgb)
        
        # Draw landmarks if detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                )

        # --- Data Collection Logic ---
        if current_gesture:
            # Check for hand detection before adding data
            if results.multi_hand_landmarks:
                feature_vector = extract_features(results)
                
                # Append label followed by 126 features
                data_row = [current_gesture] + feature_vector.tolist()
                data_to_save.append(data_row)
                collected_count += 1
                
                # Check if target is met
                if collected_count >= TARGET_FRAMES:
                    print(f"[{current_gesture.upper()}] Target reached: {collected_count}/{TARGET_FRAMES}")
                    
                    # Save collected batch to CSV
                    with open(DATASET_PATH, 'a') as f:
                        for row in data_to_save:
                            f.write(','.join(map(str, row)) + '\n')
                    
                    total_saved_counts[current_gesture] += collected_count
                    
                    # Reset state for next gesture
                    print(f"Data saved for {current_gesture}. Ready for next gesture.")
                    data_to_save = []
                    current_gesture = None
                    collected_count = 0
            else:
                 # Display warning if collecting but no hand is seen
                 cv2.putText(image, "NO HAND DETECTED - HOLD POSE STEADY!", (10, 150), 
                             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)


        # --- UI Display ---
        status_text = f"Status: {'RECORDING' if current_gesture else 'READY'}"
        gesture_text = f"GESTURE: {current_gesture.upper() if current_gesture else 'NONE (Press Key)'}"
        count_text = f"Collected: {collected_count}/{TARGET_FRAMES}"

        cv2.putText(image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, gesture_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
        if current_gesture:
             cv2.putText(image, count_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imshow('SignAssist Data Collector', image)

        # --- Key Listeners ---
        key = cv2.waitKey(5) & 0xFF
        
        if key == ord('q'):
            # --- CRITICAL: Safe Shutdown Save ---
            if data_to_save:
                print(f"Saving {len(data_to_save)} incomplete frames for {current_gesture.upper()} before quitting.")
                with open(DATASET_PATH, 'a') as f:
                    for row in data_to_save:
                        f.write(','.join(map(str, row)) + '\n')
                total_saved_counts[current_gesture] += len(data_to_save)
            # -----------------------------------
            break

        # Check for start recording key press
        if key in GESTURE_MAP and not current_gesture:
            label = GESTURE_MAP[key]
            current_gesture = label
            collected_count = 0
            data_to_save = []
            print(f"--- STARTING COLLECTION for: {current_gesture.upper()} ---")

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    print("\n--- Data Collection Session Ended ---")
    print("Total frames saved per gesture in current file:")
    for label, count in total_saved_counts.items():
        print(f"- {label.upper()}: {count} frames")


if __name__ == '__main__':
    collect_data()
