# A kód célja: Az ECAPA-TDNN modell segítségével jellemzőket (embeddingeket) kinyerni a hangfájlokból, majd ezeket egy CSV fájlba menteni, ahol minden sor egy fájlt reprezentál, és oszlopokban vannak az embedding értékek és a címkék (Demencia vagy Kontroll).
import logging
import os
import glob
import pandas as pd
import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier

DATA_DIR = "./data"
OUTPUT_CSV = "./csv_output/ecapa_embeddings.csv"

def extract_features():
    print("Loading ECAPA-TDNN model from SpeechBrain...")
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb", 
        savedir="./tmp_model",
        run_opts={"device":"cuda" if torch.cuda.is_available() else "cpu"}
    )

    mp3_files = glob.glob(os.path.join(DATA_DIR, "**/*.mp3"), recursive=True)
    print(f"Found {len(mp3_files)} .mp3 files. Starting extraction...")

    dataset_rows = []

    for i, file_path in enumerate(mp3_files):
        print(f"[{i+1}/{len(mp3_files)}] Processing: {os.path.basename(file_path)}", end="\r")
        # --- NEW LABEL LOGIC BASED ON YOUR FOLDERS ---
        if "dementia-audio" in file_path:
            label = 1
        elif "control-audio" in file_path:
            label = 0
       # elif "WLS-audio" in file_path:
            # WLS is usually a separate dataset (Wisconsin Longitudinal Study)
            # You can change this to 0, 1, or skip it entirely depending on your goal.
       #     label = 2 
        else:
            print(f"Skipping {file_path} - unknown category.")
            continue

        try:
            signal, fs = torchaudio.load(file_path)

            if signal.shape[0] > 1:
                signal = signal.mean(dim=0, keepdim=True)

            if fs != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)
                signal = resampler(signal)

            with torch.no_grad():
                embeddings = classifier.encode_batch(signal)
                emb_flat = embeddings.squeeze().cpu().numpy()

            filename = os.path.basename(file_path)
            row_data = {"filename": filename, "label": label}

            for j, val in enumerate(emb_flat):
                row_data[f"e_{j}"] = val

            dataset_rows.append(row_data)

            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(mp3_files)} files...")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    df = pd.DataFrame(dataset_rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSuccess! Extracted features for {len(df)} files.")
    print(f"Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    extract_features()
