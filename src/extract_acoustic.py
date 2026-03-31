import os
import glob
import pandas as pd
import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call
import warnings

# Ignore harmless librosa warnings to keep the console clean
warnings.filterwarnings('ignore')

DATA_DIR = "./data"
OUTPUT_CSV = "acoustic_and_prosodic_features.csv"

def extract_features():
    print("Starting classical acoustic and prosodic feature extraction...")
    
    mp3_files = glob.glob(os.path.join(DATA_DIR, "**/*.mp3"), recursive=True)
    print(f"Found a total of {len(mp3_files)} .mp3 files. Starting...\n")

    dataset_rows = []

    for i, file_path in enumerate(mp3_files):
        print(f"[{i+1}/{len(mp3_files)}] Processing: {os.path.basename(file_path)}", end="\r")
        
        # --- Labeling based on your original folder logic ---
        if "dementia-audio" in file_path:
            label = 1
        elif "control-audio" in file_path:
            label = 0
        else:
            continue

        try:
            # 1. Load audio with Librosa (returns numpy array and sample rate)
            y, sr = librosa.load(file_path, sr=16000, mono=True)
            
            filename = os.path.basename(file_path)
            row_data = {"filename": filename, "label": label}

            # ==========================================
            # PART A: LIBROSA FEATURES (Acoustic)
            # ==========================================
            
            # 1. MFCC (20 coefficients)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            for j in range(20):
                row_data[f"mfcc_{j}_mean"] = np.mean(mfccs[j])
                row_data[f"mfcc_{j}_std"] = np.std(mfccs[j])

            # 2. Zero-Crossing Rate (ZCR)
            zcr = librosa.feature.zero_crossing_rate(y)
            row_data["zcr_mean"] = np.mean(zcr)
            row_data["zcr_std"] = np.std(zcr)

            # 3. Spectral Centroid
            centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            row_data["centroid_mean"] = np.mean(centroid)
            row_data["centroid_std"] = np.std(centroid)

            # 4. Spectral Bandwidth
            bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            row_data["bandwidth_mean"] = np.mean(bandwidth)
            row_data["bandwidth_std"] = np.std(bandwidth)

            # 5. RMS Energy
            rms = librosa.feature.rms(y=y)
            row_data["rms_mean"] = np.mean(rms)
            row_data["rms_std"] = np.std(rms)

            # ==========================================
            # PART B: PARSELMOUTH (Praat) FEATURES (Prosodic)
            # ==========================================
            
            # Create a Parselmouth Sound object directly from the librosa array
            # This saves disk I/O time because we don't load the mp3 twice
            snd = parselmouth.Sound(y, sampling_frequency=sr)

            # 1. Pitch (F0) Extraction
            pitch = snd.to_pitch()
            pitch_values = pitch.selected_array['frequency']
            # Filter out unvoiced frames (where frequency is 0)
            pitch_values = pitch_values[pitch_values > 0] 
            
            if len(pitch_values) > 0:
                row_data["f0_mean"] = np.mean(pitch_values)
                row_data["f0_std"] = np.std(pitch_values)
            else:
                row_data["f0_mean"] = 0
                row_data["f0_std"] = 0

            # 2. Jitter and Shimmer Extraction (Voice Pathology Features)
            # We need a PointProcess object (glottal pulses) to calculate jitter/shimmer
            point_process = call([snd, pitch], "To PointProcess (cc)")
            
            # Jitter (local) - standard Praat parameters
            jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            # Handle potential NaN values (if speech is entirely unvoiced)
            row_data["jitter_local"] = jitter if not np.isnan(jitter) else 0

            # Shimmer (local) - standard Praat parameters
            shimmer = call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            row_data["shimmer_local"] = shimmer if not np.isnan(shimmer) else 0

            # Append the completed row to our dataset
            dataset_rows.append(row_data)

        except Exception as e:
            # Line break so it doesn't overwrite the progress bar
            print(f"\nError processing file {file_path}: {e}")

    # Create and save DataFrame
    df = pd.DataFrame(dataset_rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n\nSuccess! Acoustic and prosodic features extracted from {len(df)} files.")
    print(f"Saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    extract_features()