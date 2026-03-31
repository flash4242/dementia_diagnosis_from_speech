# A kód célja: Megvizsgálni, hogy az egyes ECAPA jellemzők önmagukban mennyire képesek megkülönböztetni a Demencia és Kontroll osztályokat.
# Ez egy fontos lépés annak megértéséhez, hogy mely jellemzők a legerősebbek, és hogy a későbbi Gépi Tanuló modellünknek milyen teljesítményt kell megütnie ahhoz, hogy értelmes legyen.
import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score

CSV_PATH = "./csv_output/ecapa_embeddings.csv"
OUTPUT_CSV = "./csv_output/single_feature_performance.csv"

def evaluate_single_features():
    print("1. Adatok beolvasása...")
    df = pd.read_csv(CSV_PATH)
    
    # Csak a Demencia (1) és Kontroll (0) osztályokat vizsgáljuk
    df_filtered = df[df['label'].isin([0, 1])].copy()
    y = df_filtered['label'].values
    
    feature_cols = [col for col in df_filtered.columns if col.startswith('e_')]
    results = []
    
    print(f"2. Prediktív erő (AUC és Maximális Pontosság) számítása {len(feature_cols)} jellemzőre...\n")
    
    for feat in feature_cols:
        X_feat = df_filtered[feat].values
        
        # ROC-AUC számítás
        auc = roc_auc_score(y, X_feat)
        
        # Ha az AUC < 0.5, az azt jelenti, hogy a jellemző inverz irányban korrelál
        # (pl. minél kisebb az érték, annál inkább demens a páciens).
        # Hogy a jóságot egységesen mérjük, átfordítjuk az irányt.
        if auc < 0.5:
            auc_corrected = 1.0 - auc
            X_feat_directional = -X_feat # Inverz irány a küszöbkereséshez
        else:
            auc_corrected = auc
            X_feat_directional = X_feat
            
        # ROC görbe pontjainak (lehetséges metszéspontok/küszöbök) lekérése
        fpr, tpr, thresholds = roc_curve(y, X_feat_directional)
        
        max_acc = 0
        best_thresh = 0
        best_prec = 0
        
        # Megkeressük azt a küszöbértéket, ahol a legmagasabb a pontosság (Accuracy)
        for thresh in thresholds:
            preds = (X_feat_directional >= thresh).astype(int)
            acc = accuracy_score(y, preds)
            if acc > max_acc:
                max_acc = acc
                best_thresh = thresh
                best_prec = precision_score(y, preds, zero_division=0)
                
        results.append({
            'Feature': feat,
            'AUC': auc_corrected,
            'Max_Accuracy': max_acc,
            'Precision_at_Max_Acc': best_prec,
            'Best_Threshold': best_thresh
        })
        
    results_df = pd.DataFrame(results)
    
    # Rendezzük a legmagasabb Pontosság (és AUC) szerint csökkenő sorrendbe
    results_df = results_df.sort_values(by=['Max_Accuracy', 'AUC'], ascending=[False, False]).reset_index(drop=True)
    
    print("="*60)
    print("TOP 10 LEGERŐSEBB ECAPA JELLEMZŐ (ÖNMAGÁBAN)")
    print("="*60)
    print(f"{'Feature':<10} | {'ROC-AUC':<10} | {'Max Accuracy':<15} | {'Precision':<10} | {'Optimal threshold':<15}")
    print("-" * 75)
    
    top10 = results_df.head(10)
    for idx, row in top10.iterrows():
        print(f"{row['Feature']:<10} | {row['AUC']:.4f}  | {row['Max_Accuracy']:.2%}   | {row['Precision_at_Max_Acc']:.2%}   | {row['Best_Threshold']:.4f}")
    
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[✔] A teljes riport elmentve ide: {OUTPUT_CSV}")
    
    # Gyors ellenőrzés a konzulensnek:
    best_overall_acc = results_df.iloc[0]['Max_Accuracy']
    print(f"\nKONKLÚZIÓ: A legerősebb egyetlen ECAPA jellemző önmagában {best_overall_acc:.2%}-os pontosságot ér el.")
    print("A végleges modellünknek ezt kell túllépnie. Ha a modellünk")
    print(f"pontossága nem éri el a {best_overall_acc:.2%}-ot, akkor a modell underfittel vagy zajos.")

if __name__ == "__main__":
    evaluate_single_features()