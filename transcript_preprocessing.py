import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from scipy.stats import shapiro, ttest_ind, mannwhitneyu
from statsmodels.stats.multitest import multipletests

import warnings
warnings.filterwarnings("ignore")

INPUT_CSV = "bert_embeddings.csv"
PCA_OUTPUT = "bert_pca_reduced.csv"
SIGNIFICANCE_OUTPUT = "bert_significance_tests.csv"
SELECTED_OUTPUT = "bert_selected_features.csv"


# --------------------------------------------------
# STEP 1 — BASIC DATASET EDA
# --------------------------------------------------
def run_eda(df):

    print("\nSTEP 1: DATASET OVERVIEW")

    counts = df["label"].value_counts()

    plt.figure(figsize=(6,6))
    plt.pie(
        counts,
        labels=["Control","Dementia"],
        autopct="%1.1f%%",
        colors=["#66b3ff","#ff9999"]
    )
    plt.title("Transcript Dataset Class Distribution")
    plt.savefig("bert_class_distribution.png")
    plt.close()

    print("Saved bert_class_distribution.png")


# --------------------------------------------------
# STEP 2 — EMBEDDING VISUALIZATION
# --------------------------------------------------
def visualize_embeddings(df):

    print("\nSTEP 2: t-SNE Visualization")

    X = df.drop(columns=["patient_id","label"])
    y = df["label"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA pre-reduction (important for BERT)
    pca = PCA(n_components=min(50, len(df)))
    X_pca = pca.fit_transform(X_scaled)

    tsne = TSNE(
        n_components=2,
        perplexity=min(30, len(df)-1),
        random_state=42
    )

    X_tsne = tsne.fit_transform(X_pca)

    plt.figure(figsize=(10,8))
    sns.scatterplot(
        x=X_tsne[:,0],
        y=X_tsne[:,1],
        hue=y.map({0:"Control",1:"Dementia"}),
        palette=["#66b3ff","#ff9999"],
        s=80
    )

    plt.title("BERT Transcript Embeddings (t-SNE)")
    plt.savefig("bert_tsne.png")
    plt.close()

    print("Saved bert_tsne.png")


# --------------------------------------------------
# STEP 3 — STATISTICAL SIGNIFICANCE
# --------------------------------------------------
def significance_testing(df):

    print("\nSTEP 3: Statistical Feature Testing")

    X = df.drop(columns=["patient_id","label"])
    y = df["label"]

    control = X[y==0]
    dementia = X[y==1]

    results = []

    for col in X.columns:

        try:
            _, p1 = shapiro(control[col])
            _, p2 = shapiro(dementia[col])
            normal = p1>0.05 and p2>0.05
        except:
            normal = False

        if normal:
            _, p = ttest_ind(control[col], dementia[col])
            test="t-test"
        else:
            _, p = mannwhitneyu(control[col], dementia[col])
            test="Mann-Whitney"

        results.append({"feature":col,"p_value":p,"test":test})

    stats_df = pd.DataFrame(results)

    stats_df["adj_p"] = multipletests(
        stats_df["p_value"].fillna(1),
        method="fdr_bh"
    )[1]

    stats_df.to_csv(SIGNIFICANCE_OUTPUT,index=False)
    print("Saved statistical report.")

    significant = stats_df[stats_df["adj_p"]<0.05]["feature"].tolist()

    print(f"Significant features: {len(significant)}")

    selected_df = df[["patient_id","label"]+significant]
    selected_df.to_csv(SELECTED_OUTPUT,index=False)

    return significant


# --------------------------------------------------
# STEP 4 — PCA REDUCTION
# --------------------------------------------------
def run_pca(df, features):

    print("\nSTEP 4: PCA Reduction")

    X = df[features]
    y = df["label"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca_full = PCA()
    pca_full.fit(X_scaled)

    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n95 = np.argmax(cumvar>=0.95)+1

    print(f"Reducing to {n95} components (95% variance)")

    plt.figure(figsize=(10,6))
    plt.plot(cumvar)
    plt.axhline(0.95,color="red")
    plt.savefig("bert_pca_elbow.png")
    plt.close()

    pca = PCA(n_components=n95)
    X_red = pca.fit_transform(X_scaled)

    out = pd.DataFrame(
        X_red,
        columns=[f"pca_{i+1}" for i in range(n95)]
    )

    out.insert(0,"label",y.values)
    out.insert(0,"patient_id",df["patient_id"].values)

    out.to_csv(PCA_OUTPUT,index=False)
    print("Saved PCA dataset.")


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------
def main():
    print("Script started")
    
    df = pd.read_csv(INPUT_CSV)

    run_eda(df)
    visualize_embeddings(df)

    features = significance_testing(df)

    if features:
        run_pca(df, features)

    print("\nBERT ANALYSIS PIPELINE COMPLETE")


if __name__ == "__main__":
    main()