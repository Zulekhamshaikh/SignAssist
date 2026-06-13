import pandas as pd
import numpy as np

TARGET_GESTURES = ["that", "you","stop", "help", "where", "this"]
DATA_FILE = "master_static_gesture_dataNT.csv"
OUTPUT_FILE = "master_static_gesture_dataNT.csv"  # overwrite final dataset

df = pd.read_csv(DATA_FILE)

print("Before cleaning:", df.shape)

def clean_direction(df_gesture, gesture_name, target_count=6000):
    # wrist = R_1, index fingertip = R_9 (MediaPipe standard)
    wrist = df_gesture[['R_X1','R_Y1']].values
    tip   = df_gesture[['R_X9','R_Y9']].values

    vec = tip - wrist
    angles = np.degrees(np.arctan2(vec[:,1], vec[:,0]))

    mean = np.mean(angles)
    std = np.std(angles)

    # keep only angles within ±2 standard deviations (removes sloppy variations)
    mask = (angles > mean - 2*std) & (angles < mean + 2*std)
    cleaned = df_gesture[mask]

    # Normalize counts back to ~6000 total (or less if not available)
    if len(cleaned) > target_count:
        cleaned = cleaned.sample(target_count, random_state=42)

    print(f"{gesture_name}: before={len(df_gesture)}, after={len(cleaned)}")
    return cleaned

df_clean_parts = []

for g in TARGET_GESTURES:
    df_g = df[df.label == g]
    df_clean_parts.append(clean_direction(df_g, g))

# keep all other gestures untouched
df_others = df[~df.label.isin(TARGET_GESTURES)]
df_clean_parts.append(df_others)

# rebuild + shuffle
df_final = pd.concat(df_clean_parts, ignore_index=True)
df_final = df_final.sample(frac=1, random_state=42)

df_final.to_csv(OUTPUT_FILE, index=False)
print("✅ Cleaning complete. After cleaning:", df_final.shape)