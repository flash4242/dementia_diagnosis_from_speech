# A kód célja: Statisztikai teszteket futtatni az egyes ECAPA jellemzőkön, hogy megvizsgáljuk, melyek különböznek szignifikánsan a Demencia és Kontroll osztályok között.
# Ez segít megérteni, hogy mely jellemzők lehetnek a legerősebb prediktorok.
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.stats.multitest as mt

CSV_PATH = "./csv_output/ecapa_embeddings.csv"
OUTPUT_DIR = "./csv_output"

def test_feature_significance():
    print("1. Adatok beolvasása az ecapa_embeddings.csv fájlból...")
    df = pd.read_csv(CSV_PATH)
    
    # Feature oszlopok kiválasztása (e_0 ... e_191)
    feature_cols = [col for col in df.columns if col.startswith('e_')]
    
    # Szétválasztjuk a két osztályt
    control_mask = df['label'] == 0
    dementia_mask = df['label'] == 1
    
    results = []
    
    print(f"2. Normalitásvizsgálat (Shapiro-Wilk) és statisztikai tesztek futtatása {len(feature_cols)} feature-ön...")
    
    for feat in feature_cols:
        control_vals = df.loc[control_mask, feat]
        dementia_vals = df.loc[dementia_mask, feat]
        
        # 1. Lépés: Normalitás tesztelése mindkét csoportra (Shapiro-Wilk)
        # Ha p > 0.05, akkor az eloszlás normálisnak tekinthető
        _, p_norm_control = stats.shapiro(control_vals)
        _, p_norm_dementia = stats.shapiro(dementia_vals)
        
        is_normal = (p_norm_control > 0.05) and (p_norm_dementia > 0.05)
        
        # 2. Lépés: Megfelelő teszt kiválasztása és futtatása
        if is_normal:
            test_used = "Welch's t-test"
            # Welch-teszt (nem feltételezünk egyenlő szórásokat)
            stat_val, p_val = stats.ttest_ind(control_vals, dementia_vals, equal_var=False)
        else:
            test_used = "Mann-Whitney U"
            # Mann-Whitney U teszt nem-normál adatokra
            stat_val, p_val = stats.mannwhitneyu(control_vals, dementia_vals, alternative='two-sided')
            
        results.append({
            'Feature': feat,
            'Is_Normal': is_normal,
            'Test_Used': test_used,
            'Statistic': stat_val,
            'P_Value_Raw': p_val
        })
        
    results_df = pd.DataFrame(results)
    
    print("3. P-értékek korrigálása (Benjamini-Hochberg FDR)...")
    # Multiple testing korrekció
    reject, pvals_corrected, _, _ = mt.multipletests(results_df['P_Value_Raw'], alpha=0.05, method='fdr_bh')
    results_df['P_Value_Adj'] = pvals_corrected
    results_df['Significant'] = reject
    
    # Rendezzük a legszignifikánsabbak (legkisebb korrigált p-érték) szerint
    results_df = results_df.sort_values(by='P_Value_Adj', ascending=True).reset_index(drop=True)
    
    # Statisztikák kiírása
    sig_count = results_df['Significant'].sum()
    welch_count = (results_df['Test_Used'] == "Welch's t-test").sum()
    mwu_count = (results_df['Test_Used'] == "Mann-Whitney U").sum()
    
    print("\n" + "="*60)
    print("STATISZTIKAI TESZTEK EREDMÉNYE")
    print("="*60)
    print(f"Vizsgált feature-ök száma: {len(feature_cols)}")
    print(f"Normál eloszlású (Welch-teszt): {welch_count} db")
    print(f"Nem normál eloszlású (Mann-Whitney U): {mwu_count} db")
    print(f"Statisztikailag szignifikáns (FDR korrigált p < 0.05): {sig_count} db")
    print("="*60)
    
    print("\nTOP 10 LEGSZIGNIFIKÁNSABB FEATURE:")
    print(results_df[['Feature', 'Test_Used', 'P_Value_Raw', 'P_Value_Adj', 'Significant']].head(10))
    
    # Mentés CSV-be
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stats_csv = os.path.join(OUTPUT_DIR, "ecapa_feature_significance_results.csv")
    results_df.to_csv(stats_csv, index=False)
    print(f"\nA teljes statisztikai riport elmentve ide: {stats_csv}")

if __name__ == "__main__":
    test_feature_significance()