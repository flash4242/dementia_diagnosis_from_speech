import os
import glob
import pandas as pd
import numpy as np
import torchaudio
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = "./data"

def analyze_audio_lengths():
    print("Analyzing 1920 audio files (this will take a minute)...")
    mp3_files = glob.glob(os.path.join(DATA_DIR, "**/*.mp3"), recursive=True)
    
    records = []
    for f in mp3_files:
        try:
            # torchaudio.info is lightning fast because it only reads metadata, not the whole file
            info = torchaudio.info(f)
            duration = info.num_frames / info.sample_rate
            
            # Figure out which class this belongs to based on folder name
            label = "Dementia" if "dementia-audio" in f else "Control" if "control-audio" in f else "Other"
            
            records.append({"Filename": os.path.basename(f), "Label": label, "Duration_Sec": duration})
        except Exception as e:
            print(f"Could not read {f}: {e}")
            
    df_audio = pd.DataFrame(records)
    
    print("\n" + "="*40)
    print("AUDIO LENGTH STATISTICS (Weeks 3-4)")
    print("="*40)
    
    # Calculate Min, Max, Mean, Std, Median overall and per group
    stats = df_audio.groupby("Label")["Duration_Sec"].agg(['min', 'max', 'mean', 'median', 'std', 'count'])
    print(stats)
    
    # Generate a plot for your 20-page report!
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df_audio, x="Duration_Sec", hue="Label", bins=50, kde=True, element="step")
    plt.title("Distribution of Audio Lengths: Control vs Dementia")
    plt.xlabel("Length in Seconds")
    plt.ylabel("Number of Recordings")
    plt.savefig("audio_length_distribution.png", dpi=300) # High quality for Word/LaTeX docs
    print("\n[✔] Saved 'audio_length_distribution.png' for your report!")

def analyze_metadata():
    print("\n" + "="*40)
    print("CLINICAL METADATA STATISTICS")
    print("="*40)
    
    excel_path = os.path.join(DATA_DIR, "Pitt-data.xlsx")
    if not os.path.exists(excel_path):
        print(f"Could not find {excel_path}")
        return
        
    df = pd.read_excel(excel_path)
    
    print("\n1. Missing Data Check:")
    missing = df.isnull().sum()
    print(missing[missing > 0]) # Only print columns that actually have missing data
    
    print("\n2. Basic Demographics:")
    # We will print the raw summary. (You will need to look at the exact column names!)
    print(df.describe(include='all'))

if __name__ == "__main__":
    analyze_audio_lengths()
    analyze_metadata()
