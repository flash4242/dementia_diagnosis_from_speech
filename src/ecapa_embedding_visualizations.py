# A kód célja: Két dimenziós vizualizációkat (PCA és t-SNE módszerekkel) készíteni az ECAPA jellemzőkről, hogy megértsük, hogyan különülnek el a Demencia és Kontroll osztályok a jellemzőtérben. Ez segít megérteni, hogy mennyire jól választják szét a jellemzők a két osztályt.
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

CSV_PATH = "./csv_output/ecapa_embeddings.csv"
OUTPUT_DIR = "./plots"

def visualize_embeddings():
    print("1. Adatok beolvasása és előkészítése...")
    df = pd.read_csv(CSV_PATH)
    
    # Csak a tiszta 0 (Control) és 1 (Dementia) osztályokat tartjuk meg a vizualizációhoz
    df_filtered = df[df['label'].isin([0, 1])].copy()
    
    # Jellemzők (X) és Címkék (y) szétválasztása
    feature_cols = [col for col in df_filtered.columns if col.startswith('e_')]
    X = df_filtered[feature_cols].values
    y = df_filtered['label'].map({0: 'Control', 1: 'Dementia'}).values
    
    print("2. Adatok skálázása (StandardScaler)...")
    # A PCA és t-SNE érzékeny a skálázásra, ezért standardizáljuk az adatokat (átlag=0, szórás=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    print("3. Főkomponens-elemzés (PCA) futtatása...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette=['#1f77b4', '#d62728'], alpha=0.7)
    plt.title(f"ECAPA Embeddings PCA Vizualizáció\n(Megmagyarázott variancia: {sum(pca.explained_variance_ratio_)*100:.1f}%)")
    plt.xlabel("1. Főkomponens (PCA1)")
    plt.ylabel("2. Főkomponens (PCA2)")
    plt.savefig(os.path.join(OUTPUT_DIR, "ecapa_pca_plot.png"), dpi=300, bbox_inches='tight')
    plt.clf()
    
    print("4. t-SNE dimenziócsökkentés futtatása...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette=['#1f77b4', '#d62728'], alpha=0.7)
    plt.title("ECAPA Embeddings t-SNE Vizualizáció")
    plt.xlabel("t-SNE 1. dimenzió")
    plt.ylabel("t-SNE 2. dimenzió")
    plt.savefig(os.path.join(OUTPUT_DIR, "ecapa_tsne_plot.png"), dpi=300, bbox_inches='tight')
    
    print(f"\n[✔] Kész! Az ábrák elmentve a '{OUTPUT_DIR}' mappába.")

if __name__ == "__main__":
    visualize_embeddings()