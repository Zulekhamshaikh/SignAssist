import pandas as pd
import os
import glob
from collections import defaultdict

# --- 1. CONFIGURATION ---
# Define the directory where ALL your group's CSV files are stored.
# IMPORTANT: Put ALL your group's CSV files into this folder.
CSV_DIRECTORY = 'Static_Gesture_CSVs' 
OUTPUT_FILENAME = 'master_static_gesture_data.csv'

# Define the expected number of landmark features (126 + 1 'label' column)
EXPECTED_COLUMNS = 127 
LABEL_COLUMN = 'label'

def create_expected_header():
    """Generates the expected CSV header row (label, R_X1...R_Z21, L_X1...L_Z21)."""
    headers = [LABEL_COLUMN]
    for hand in ['R', 'L']:
        for i in range(1, 22):
            for coord in ['X', 'Y', 'Z']:
                headers.append(f'{hand}_{coord}{i}')
    return headers

def load_and_merge_csvs(directory):
    """
    Loads all CSV files in the specified directory, validates their structure,
    merges them, and adds a 'signer' column.
    """
    all_files = glob.glob(os.path.join(directory, "*.csv"))
    if not all_files:
        print(f"❌ Error: No CSV files found in directory: {directory}")
        return None

    print(f"🔍 Found {len(all_files)} CSV files to merge.")
    
    list_of_dfs = []
    
    # Generate the header exactly as your collection script produces it
    expected_headers = create_expected_header()

    for filename in all_files:
        base_name = os.path.basename(filename)
        
        try:
            # Load the CSV, forcing the first row to be the header (even if it's missing)
            df = pd.read_csv(filename)

            # --- Validation Checks ---
            if df.columns.tolist() != expected_headers:
                print(f"⚠️ Warning: Header mismatch in {base_name}. Dropping header row and re-assigning.")
                # Assuming the first row is corrupted data, load again skipping header
                df = pd.read_csv(filename, header=None)
                if df.shape[1] != EXPECTED_COLUMNS:
                    print(f"❌ Error: {base_name} has {df.shape[1]} columns, expected {EXPECTED_COLUMNS}. Skipping.")
                    continue
                # Assign the correct header
                df.columns = expected_headers
            
            # --- Signer Identification (Essential for generalization analysis) ---
            # Extracts the signer's name from the filename (e.g., 'rashmi_data.csv' -> 'rashmi')
            signer_id = base_name.split('_')[0]
            df['signer'] = signer_id
            
            print(f"Loaded {base_name} ({len(df)} rows). Signer: {signer_id}")
            list_of_dfs.append(df)
            
        except pd.errors.EmptyDataError:
            print(f"⚠️ Warning: {base_name} is empty. Skipping.")
        except Exception as e:
            print(f"❌ Fatal Error reading {base_name}: {e}. Skipping file.")
            
    if not list_of_dfs:
        print("❌ All files failed validation or load. Aborting merge.")
        return None
        
    # Concatenate all DataFrames into one master DataFrame
    master_df = pd.concat(list_of_dfs, axis=0, ignore_index=True)
    return master_df

# --- 2. MAIN EXECUTION ---
if __name__ == "__main__":
    
    # Ensure directory exists (optional, but good practice)
    os.makedirs(CSV_DIRECTORY, exist_ok=True)
    
    master_data = load_and_merge_csvs(CSV_DIRECTORY)

    if master_data is not None:
        # --- 3. FINAL CLEANUP AND INSPECTION ---
        
        # Drop any rows that have missing values (shouldn't happen with your collection script)
        master_data.dropna(inplace=True) 

        # Shuffle the data to mix up signers and gestures before training
        master_data = master_data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # --- Data Report ---
        gesture_counts = master_data[LABEL_COLUMN].value_counts()
        signer_counts = master_data['signer'].value_counts()

        print("\n📊 MASTER DATASET REPORT:")
        print(f"Total Valid Rows Merged: {len(master_data)}")
        print(f"Unique Signers: {master_data['signer'].nunique()}")
        print("-" * 30)
        
        print("Frames per Gesture (Combined):")
        print(gesture_counts)
        print("-" * 30)
        
        print("Frames per Signer:")
        print(signer_counts)
        print("-" * 30)

        # --- 4. SAVE THE MASTER FILE ---
        # Note: We save ALL columns including the 'signer' column for tracking
        master_data.to_csv(OUTPUT_FILENAME, index=False)
        print(f"\n✅ Successfully saved merged data to: {OUTPUT_FILENAME}")