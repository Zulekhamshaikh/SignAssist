import cv2
import mediapipe as mp
import numpy as np
import os
import time
import re

# --- 1. CONFIGURATION ---
DATA_PATH = os.path.join('Dynamic_ASL_Data')
# Define your dynamic actions here. Example set:
ACTIONS = ['hello', 'thanks', 'go', 'family', 'happy', 'how', 'yes', 'no', 'morning']
NUM_EXAMPLES = 50                                     # Total examples (sequences) to collect in THIS run
NUM_FRAMES = 30                                       # Frames in one sequence (time-step length) - REVERTED TO 30 FOR CONSISTENCY
COUNTDOWN_SECONDS = 3                                # Time to prepare the gesture
FEATURES_PER_HAND = 63                                # 21 hand landmarks * 3 coords (X, Y, Z)
FEATURES_PER_FRAME = FEATURES_PER_HAND * 2            # both hands → 126

# Create necessary action folders
for action in ACTIONS:
    os.makedirs(os.path.join(DATA_PATH, action), exist_ok=True)

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def extract_keypoints(results):
    """
    Extracts keypoints for both hands in a fixed order (Hand 1 then Hand 2).
    Fills missing hand data with zeros (126 features total).
    """
    # Initialize empty array for a fixed size (126 features)
    full_keypoints = np.zeros(FEATURES_PER_FRAME)

    if results.multi_hand_landmarks:
        # Iterate through detected hands
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if i >= 2: # Only process up to 2 hands
                break

            start_index = i * FEATURES_PER_HAND
            current_hand_keypoints = []
            for landmark in hand_landmarks.landmark:
                # Append X, Y, Z coordinates
                current_hand_keypoints.extend([landmark.x, landmark.y, landmark.z])

            full_keypoints[start_index : start_index + FEATURES_PER_HAND] = current_hand_keypoints

    return full_keypoints


def get_next_index(action):
    """Finds the next available sequence index for a given action by scanning existing files."""
    action_path = os.path.join(DATA_PATH, action)
    max_index = -1
    
    # Regex to match the filename format (e.g., 'hello_5.npy')
    pattern = re.compile(rf'^{re.escape(action)}_(\d+)\.npy$')
    
    for filename in os.listdir(action_path):
        match = pattern.match(filename)
        if match:
            try:
                index = int(match.group(1))
                if index > max_index:
                    max_index = index
            except ValueError:
                continue # Ignore if index is not an integer
    return max_index + 1


# --- 2. MAIN COLLECTION LOOP ---
cap = cv2.VideoCapture(0)

# Main Action Loop
for action in ACTIONS:
    # Determine starting index based on existing files
    starting_index = get_next_index(action)
    current_examples_collected = 0
    total_examples = starting_index + NUM_EXAMPLES

    print(f"\n===== READY TO START COLLECTION FOR: {action.upper()} =====")
    print(f"Current index count: {starting_index}. Collecting {NUM_EXAMPLES} new examples.")

    # Wait for initial SPACEBAR press to start the set, or 'S' to skip the entire action
    key = 0 
    while True:
        ret, frame = cap.read()
        if not ret: break

        # Display status
        cv2.putText(frame, f'ACTION: {action} (Index: {starting_index})',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, 'PRESS SPACE TO BEGIN | SHIFT+S TO SKIP ACTION',
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('SignAssist Data Collector', frame)

        key = cv2.waitKey(10) & 0xFF
        if key == ord(' '):
            break
        elif key == ord('S'): # Use capital 'S' to skip the entire action
            print(f"--- ⏩ SKIPPING ENTIRE ACTION: {action.upper()} ---")
            break # Break out of this initial wait loop
        elif key == ord('q'):
            cap.release(); cv2.destroyAllWindows(); exit()

    # If the key was 'S', we skip the rest of the code in the 'action' loop and continue to the next action.
    if key == ord('S'):
        continue # Go to the next action in the ACTIONS list


    # Example Loop: Runs until NUM_EXAMPLES new sequences are collected
    while current_examples_collected < NUM_EXAMPLES:
        # Calculate the unique index for the current example
        example_num = starting_index + current_examples_collected
        sequence = []

        # --- COUNTDOWN PHASE ---
        for i in range(COUNTDOWN_SECONDS, 0, -1):
            ret, frame = cap.read()
            if not ret: break

            # Display countdown and action name
            cv2.putText(frame, f'GET READY: {action.upper()} - {i}',
                        (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
            cv2.putText(frame, f'NEW EXAMPLE {current_examples_collected + 1}/{NUM_EXAMPLES}',
                        (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow('SignAssist Data Collector', frame)
            
            # Allow quitting during countdown
            if cv2.waitKey(1000) & 0xFF == ord('q'):
                cap.release(); cv2.destroyAllWindows(); exit()


        print(f"--- Capturing {action} Index {example_num} ({current_examples_collected + 1}/{NUM_EXAMPLES} in this run) ---")

        # Initialize mediapipe fresh for each example
        with mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_hands=2
        ) as hands:

            # --- CAPTURE PHASE ---
            frame_count = 0
            while cap.isOpened() and frame_count < NUM_FRAMES:
                ret, frame = cap.read()
                if not ret: break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)

                # Extract and Append Keypoints
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                frame_count += 1

                # UI Feedback
                cv2.putText(frame, f'ACTION: {action} (Index: {example_num})',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(frame, f'RECORDING: Frame {frame_count}/{NUM_FRAMES}',
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Draw landmarks
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                cv2.imshow('SignAssist Data Collector', frame)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    cap.release(); cv2.destroyAllWindows(); exit()

        # --- SAVE AND HANDS-FREE PAUSE PHASE ---
        if len(sequence) == NUM_FRAMES:
            sequence_array = np.array(sequence)
            save_path = os.path.join(DATA_PATH, action, f'{action}_{example_num}.npy')
            np.save(save_path, sequence_array)
            print(f"✅ Saved: {save_path} | Shape: {sequence_array.shape}")
            
            current_examples_collected += 1 # Only count successful saves
            
            # Hands-Free Pause: Wait before starting the next example automatically
            if current_examples_collected < NUM_EXAMPLES:
                print(f"--- Auto-starting next example in 1 second... ---")
                end_time = time.time() + 1.0 # 1 second pause
                while time.time() < end_time:
                    ret, frame = cap.read()
                    if not ret: break
                    
                    # Display countdown to the next capture
                    cv2.putText(frame, f'ACTION: {action} SAVED (Index: {example_num})',
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f'AUTO-NEXT IN {round(end_time - time.time(), 1)}s...',
                                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    cv2.imshow('SignAssist Data Collector', frame)
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        cap.release(); cv2.destroyAllWindows(); exit()
            
        else:
            print(f"❌ Failed to capture {NUM_FRAMES} frames for {action} Index {example_num}. Trying next example.")


print("\n--- 🎉 DATA COLLECTION COMPLETE ---")
cap.release()
cv2.destroyAllWindows()