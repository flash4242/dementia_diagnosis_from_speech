import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torchaudio
import librosa
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings

# Ignore warnings for cleaner console output
warnings.filterwarnings('ignore')

# Global Configuration Paths
DATA_DIR = "./data"
ECAPA_CSV_PATH = "ecapa_embeddings.csv"
OUTPUT_RHYTHM_CSV = "speech_pause_features.csv"
OUTPUT_REDUCED_CSV = "ecapa_pca_reduced.csv"

def run_eda_and_visualization(df):
    """Generates class distribution and embedding visualization (t-SNE)."""
    print("\n" + "="*50)
    print("STEP 1: EXPLORATORY DATA ANALYSIS & VISUALIZATION")
    print("="*50)
    
    # --- 1. Class Distribution ---
    print("Analyzing class distribution...")
    class_counts = df['label'].value_counts()
    
    label_map = {0: 'Control (0)', 1: 'Dementia (1)'}
    color_map = {0: '#66b3ff', 1: '#ff9999'}
    
    actual_labels = [label_map.get(idx, f"Unknown ({idx})") for idx in class_counts.index]
    actual_colors = [color_map.get(idx, '#cccccc') for idx in class_counts.index]
    
    for idx, count in class_counts.items():
        print(f"  - {label_map.get(idx, f'Unknown ({idx})')}: {count} samples")
    
    plt.figure(figsize=(6, 6))
    plt.pie(class_counts, labels=actual_labels, autopct='%1.1f%%', 
            colors=actual_colors, startangle=90)
    plt.title('Dataset Class Distribution')
    plt.savefig('class_distribution.png')
    plt.close()
    print("  -> Saved 'class_distribution.png'")

    # --- 2. Audio Length Distribution ---
    print("\nExtracting audio lengths (using Torchaudio metadata)...")
    mp3_files = glob.glob(os.path.join(DATA_DIR, "**/*.mp3"), recursive=True)
    durations = []
    
    for file_path in mp3_files:
        try:
            info = torchaudio.info(file_path)
            duration_sec = info.num_frames / info.sample_rate
            label = "Dementia" if "dementia-audio" in file_path else "Control"
            durations.append({"length_sec": duration_sec, "group": label})
        except Exception:
            continue
            
    df_durations = pd.DataFrame(durations)
    
    plt.figure(figsize=(10, 5))
    sns.histplot(data=df_durations, x='length_sec', hue='group', kde=True, bins=30, palette=['#ff9999', '#66b3ff'])
    plt.title('Distribution of Audio File Lengths (Seconds)')
    plt.xlabel('Length (seconds)')
    plt.ylabel('Number of Files')
    plt.savefig('audio_lengths_histogram.png')
    plt.close()
    print("  -> Saved 'audio_lengths_histogram.png'")

    # --- 3. Embedding Visualization (t-SNE) ---
    print("\nRunning Dimensionality Reduction for Visualization (t-SNE)...")
    X = df.drop(columns=['filename', 'label'])
    y = df['label']
    n_samples = X.shape[0]
    
    if n_samples < 3:
        print("Not enough data points for t-SNE visualization (minimum 3 required).")
    else:
        # Dynamic parameter settings to avoid crash on small subsets
        n_components_pca = min(50, n_samples)
        pca_vis = PCA(n_components=n_components_pca, random_state=42)
        X_pca_vis = pca_vis.fit_transform(X)
        
        tsne_perplexity = min(30, max(1, n_samples - 1))
        tsne = TSNE(n_components=2, perplexity=tsne_perplexity, random_state=42)
        X_tsne = tsne.fit_transform(X_pca_vis)
        
        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            x=X_tsne[:, 0], y=X_tsne[:, 1], 
            hue=y.map({0: 'Control', 1: 'Dementia'}), 
            palette=['#66b3ff', '#ff9999'], 
            alpha=0.8, edgecolor='k', s=80
        )
        plt.title(f'ECAPA-TDNN Embeddings Visualization (t-SNE, n={n_samples})')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.legend(title='Class')
        plt.savefig('tsne_visualization.png')
        plt.close()
        print("  -> Saved 'tsne_visualization.png'")


def extract_rhythm_features():
    """Extracts pause and speech ratio features using Librosa."""
    print("\n" + "="*50)
    print("STEP 2: RHYTHM & VOICE ACTIVITY DETECTION (VAD)")
    print("="*50)
    
    mp3_files = glob.glob(os.path.join(DATA_DIR, "**/*.mp3"), recursive=True)
    print(f"Found {len(mp3_files)} .mp3 files. Starting rhythm extraction...")

    dataset_rows = []

    for i, file_path in enumerate(mp3_files):
        print(f"  [{i+1}/{len(mp3_files)}] Processing: {os.path.basename(file_path)}", end="\r")
        
        if "dementia-audio" in file_path:
            label = 1
        elif "control-audio" in file_path:
            label = 0
        else:
            continue

        try:
            y, sr = librosa.load(file_path, sr=16000, mono=True)
            total_duration = librosa.get_duration(y=y, sr=sr)
            
            # Split signal based on energy (threshold 30dB below peak)
            non_mute_intervals = librosa.effects.split(y, top_db=30)
            
            speech_duration = 0.0
            for interval in non_mute_intervals:
                speech_duration += (interval[1] - interval[0]) / sr
                
            silence_duration = total_duration - speech_duration
            pause_count = len(non_mute_intervals) - 1 if len(non_mute_intervals) > 0 else 0
            avg_pause_duration = silence_duration / pause_count if pause_count > 0 else 0.0
            speech_ratio = speech_duration / total_duration if total_duration > 0 else 0.0

            dataset_rows.append({
                "filename": os.path.basename(file_path),
                "label": label,
                "total_duration_sec": total_duration,
                "speech_duration_sec": speech_duration,
                "silence_duration_sec": silence_duration,
                "pause_count": pause_count,
                "avg_pause_duration_sec": avg_pause_duration,
                "speech_ratio": speech_ratio
            })

        except Exception as e:
            print(f"\nError processing {file_path}: {e}")

    df_rhythm = pd.DataFrame(dataset_rows)
    df_rhythm.to_csv(OUTPUT_RHYTHM_CSV, index=False)
    print(f"\n  -> Success! Extracted rhythm features for {len(df_rhythm)} files.")
    print(f"  -> Saved to '{OUTPUT_RHYTHM_CSV}'")


def analyze_and_reduce_features(df):
    """Analyzes feature correlations and performs PCA dimensionality reduction."""
    print("\n" + "="*50)
    print("STEP 3: FEATURE CORRELATION & DIMENSIONALITY REDUCTION")
    print("="*50)
    
    n_samples = len(df)
    if n_samples < 50:
        print(f"WARNING: Only {n_samples} rows found. PCA reduction requires more data.")
        return

    filenames = df['filename']
    y = df['label']
    X = df.drop(columns=['filename', 'label'])

    print(f"Original feature space dimension: {X.shape[1]} (Samples: {n_samples})")

    # --- 1. Target Correlation ---
    print("\nCalculating feature correlation with the target variable...")
    correlations = X.apply(lambda col: col.corr(y)).abs()
    top_features = correlations.sort_values(ascending=False).head(20)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_features.index, y=top_features.values, palette='viridis')
    plt.title('Top 20 ECAPA Features Correlated with Dementia (Absolute Value)')
    plt.ylabel('Pearson Correlation')
    plt.xlabel('ECAPA Feature Name')
    plt.xticks(rotation=45)
    plt.savefig('top_correlated_features.png', bbox_inches='tight')
    plt.close()
    print("  -> Saved 'top_correlated_features.png'")

    # --- 2. PCA Reduction ---
    print("\nRunning Principal Component Analysis (PCA)...")
    
    # Standardization is mandatory before PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca_full = PCA()
    pca_full.fit(X_scaled)
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    
    # Find how many components are needed to retain 95% of the information
    n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
    plt.axhline(y=0.95, color='r', linestyle='-', label='95% Variance Retained')
    plt.axvline(x=n_components_95, color='g', linestyle='-', label=f'{n_components_95} Components')
    plt.title('PCA Explained Variance (Elbow Method)')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.legend()
    plt.grid(True)
    plt.savefig('pca_elbow_curve.png')
    plt.close()
    print(f"  -> Saved 'pca_elbow_curve.png'. Reduction possible to {n_components_95} dimensions.")

    print(f"Transforming dataset from {X.shape[1]} to {n_components_95} orthogonal dimensions...")
    pca_final = PCA(n_components=n_components_95, random_state=42)
    X_reduced = pca_final.fit_transform(X_scaled)

    reduced_df = pd.DataFrame(X_reduced, columns=[f"pca_{i+1}" for i in range(n_components_95)])
    reduced_df.insert(0, 'label', y.values)
    reduced_df.insert(0, 'filename', filenames.values)

    reduced_df.to_csv(OUTPUT_REDUCED_CSV, index=False)
    print(f"  -> Saved the reduced dataset to '{OUTPUT_REDUCED_CSV}'")


def main():
    print("Loading initial ECAPA embeddings dataset...")
    try:
        df_ecapa = pd.read_csv(ECAPA_CSV_PATH)
    except FileNotFoundError:
        print(f"Error: Could not find {ECAPA_CSV_PATH}. Please run the extraction script first.")
        return

    # Run the pipeline sequentially
    run_eda_and_visualization(df_ecapa)
    extract_rhythm_features()
    analyze_and_reduce_features(df_ecapa)
    
    print("\n" + "="*50)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("="*50)

if __name__ == "__main__":
    main()