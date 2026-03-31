import os
import glob
import warnings
import pandas as pd
import numpy as np
import torchaudio
import matplotlib.pyplot as plt
import seaborn as sns


DATA_DIR = "./data"

def perform_eda():
    print("="*50)
    print("1. PARSING AUDIO FILES & EXTRACTING IDs")
    print("="*50)
    
    mp3_files = glob.glob(os.path.join(DATA_DIR, "**/*.mp3"), recursive=True)
    audio_records = []
    
    for f in mp3_files:
        if "WLS" in f: continue # Skip the WLS files as they are self-reported and not part of the main clinical dataset
        
        filename = os.path.basename(f)
        try:
            name_parts = filename.replace('.mp3', '').split('-')
            patient_id = int(name_parts[0]) 
            visit_num = int(name_parts[1]) if len(name_parts) > 1 else 0
            
            info = torchaudio.info(f)
            duration = info.num_frames / info.sample_rate
            
            audio_records.append({
                "Filename": filename, 
                "id": patient_id, 
                "Visit": visit_num, 
                "Duration_Sec": duration
            })
        except Exception as e:
            pass

    df_audio = pd.DataFrame(audio_records)
    print(f"Loaded {len(df_audio)} Pitt audio files.")
    print(f"Unique Patients in Audio: {df_audio['id'].nunique()}")

    print("\n" + "="*50)
    print("2. LOADING CLINICAL METADATA")
    print("="*50)
    
    excel_path = os.path.join(DATA_DIR, "Pitt-data.xlsx")
    df_meta = pd.read_excel(excel_path, sheet_name="data", skiprows=2) 
    df_meta.columns = df_meta.columns.str.strip()
    
    print(f"Total Rows in Metadata: {len(df_meta)}")
    print(f"Unique Patient IDs in Metadata: {df_meta['id'].nunique()}")

    # Check for duplicate rows
    duplicates = df_meta.duplicated(subset=['id', 'idate']).sum()
    print(f"Duplicate entries found (same ID and Date): {duplicates}")

    # Modern Pandas: Create new columns as a dictionary and concat to avoid fragmentation warning
    def map_diagnosis(code):
        if pd.isna(code): return "Unknown"
        code = int(code)
        if code in [1, 100, 2, 200, 3, 300, 4, 420, 430]: return "Dementia"
        elif code in [8, 800, 851]: return "Control"
        elif code in [6, 7, 610, 611, 720, 740, 730, 770]: return "MCI / Other"
        else: return "Other"
        
    sex_col = [c for c in df_meta.columns if 'sex' in c.lower()][0]
    
    new_cols = pd.DataFrame({
        'Diagnosis_Group': df_meta['basedx'].apply(map_diagnosis),
        'Gender': df_meta[sex_col].map({1: 'Male', 0: 'Female'})
    })
    # Concat the new columns to the original dataframe
    df_meta = pd.concat([df_meta, new_cols], axis=1)

    print("\nMissing Data in Metadata (Descending):")
    missing = df_meta.isnull().sum()
    missing_sorted = missing[missing > 0].sort_values(ascending=False)
    print(missing_sorted)

    print("\n" + "="*50)
    print("3. MERGING & GENERATING REPORT STATISTICS")
    print("="*50)
    
    df_merged = pd.merge(df_audio, df_meta, on='id', how='left')
    
    print("\n--- Audio Length by Diagnosis (Min/Max/Avg/Std/Median) ---")
    stats = df_merged.groupby('Diagnosis_Group')['Duration_Sec'].agg(['min', 'max', 'mean', 'median', 'std', 'count'])
    print(stats)
    
    print("\n--- Gender Ratio by Diagnosis ---")
    print(pd.crosstab(df_merged['Diagnosis_Group'], df_merged['Gender']))

    # --- PLOTTING ---
    sns.set_theme(style="whitegrid")
    
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df_merged, x="Diagnosis_Group", hue="Diagnosis_Group", legend=False, palette="Set2")
    plt.title("Distribution of Diagnoses in Pitt Corpus")
    plt.savefig("plot_1_diagnoses.png", dpi=300, bbox_inches='tight')
    plt.clf()
    
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df_merged, x="Diagnosis_Group", hue="Gender", palette="Pastel1")
    plt.title("Gender Distribution Across Diagnoses")
    plt.savefig("plot_2_gender_ratio.png", dpi=300, bbox_inches='tight')
    plt.clf()
    
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df_merged, x="Duration_Sec", hue="Diagnosis_Group", bins=40, kde=True, element="step")
    plt.title("Distribution of Audio Lengths by Diagnosis")
    plt.xlabel("Length in Seconds")
    plt.savefig("plot_3_audio_lengths.png", dpi=300, bbox_inches='tight')
    
    age_col = [c for c in df_merged.columns if 'age' in c.lower()]
    mmse_col = [c for c in df_merged.columns if 'mms' in c.lower()]
    
    if age_col:
        plt.clf()
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df_merged, x="Diagnosis_Group", y=age_col[0], hue="Diagnosis_Group", legend=False, palette="Set3")
        plt.title("Age Distribution by Diagnosis")
        plt.savefig("plot_4_age_matching.png", dpi=300, bbox_inches='tight')
        
    if mmse_col:
        plt.clf()
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df_merged, x="Diagnosis_Group", y=mmse_col[0], hue="Diagnosis_Group", legend=False, palette="Set1")
        plt.title("Mini-Mental State Exam (MMSE) Scores by Diagnosis")
        plt.savefig("plot_5_mmse_scores.png", dpi=300, bbox_inches='tight')

    print("\n[✔] SUCCESS! Generated plots 1-5.")

if __name__ == "__main__":
    perform_eda()