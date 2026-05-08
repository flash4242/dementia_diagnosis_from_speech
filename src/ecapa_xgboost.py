import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
# MÓDOSÍTÁS 1: StratifiedGroupKFold importálása
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.metrics import (accuracy_score, roc_auc_score, classification_report, 
                             confusion_matrix, roc_curve, balanced_accuracy_score)
import warnings, os

# Figyelmeztetések kikapcsolása
warnings.filterwarnings('ignore')

# --- KONFIGURÁCIÓ ---
CSV_DIR = "csv_output" 
ECAPA_CSV = CSV_DIR + "/ecapa_embeddings.csv"
RHYTHM_CSV = CSV_DIR + "/speech_pause_features.csv"
PLOT_DIR = "./plots"
os.makedirs(PLOT_DIR, exist_ok=True)

def load_and_merge_data():
    print("1. Lépés: Adatok betöltése és fúziója...")
    try:
        df_ecapa = pd.read_csv(ECAPA_CSV)
        df_rhythm = pd.read_csv(RHYTHM_CSV)
    except FileNotFoundError as e:
        print(f"\nHiba: Nem található egy bemeneti fájl! Részletek: {e}")
        return None

    df_merged = pd.merge(df_ecapa, df_rhythm, on=['filename', 'label'], how='inner')
    print(f"  -> SIKERES FÚZIÓ: {df_merged.shape[0]} közös minta, összesen {df_merged.shape[1]-2} prediktor.\n")
    return df_merged

def train_and_evaluate_nested_cv(df):
    print("2. Lépés: Nested Cross-Validation (Beágyazott CV) indítása...")
    
    y = df['label']
    
    # MÓDOSÍTÁS 2: Páciens ID-k (csoportok) kinyerése a filename-ből (pl. "192-2.mp3" -> "192")
    groups = df['filename'].apply(lambda x: str(x).split('-')[0])
    
    X = df.drop(columns=['filename', 'label'])
    pos_weight = (y == 0).sum() / (y == 1).sum()

    # MÓDOSÍTÁS 3: KÜLSŐ CV lecserélése StratifiedGroupKFold-ra
    outer_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    # Hiperparaméter háló
    param_grid = {
        'max_depth': [2, 3],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [50, 100, 200],
        'subsample': [0.7, 0.9],
        'colsample_bytree': [0.5, 0.8]
    }

    # Tömb a "sosem látott" out-of-fold valószínűségek gyűjtéséhez
    out_of_fold_probs = np.zeros(len(y))

    print("  -> Modellek tanítása és hangolása a hajtásokon (Inner + Outer CV)...")

    # MÓDOSÍTÁS 4: groups paraméter átadása a split-nek
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups=groups)):
        print(f"     Hajtás {fold+1}/5 folyamatban...")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Kinyerjük a belső CV számára a tréning halmazba került csoportokat
        groups_train = groups.iloc[train_idx]
        
        # MÓDOSÍTÁS 5: BELSŐ CV lecserélése StratifiedGroupKFold-ra
        inner_cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
        
        xgb_clf = xgb.XGBClassifier(
            objective='binary:logistic', eval_metric='auc',
            scale_pos_weight=pos_weight, random_state=42
        )
        
        grid_search = GridSearchCV(
            estimator=xgb_clf, param_grid=param_grid, 
            scoring='roc_auc', cv=inner_cv, n_jobs=-1, verbose=0
        )
        
        # MÓDOSÍTÁS 6: groups paraméter átadása a betanításkor (GridSearchCV számára)
        grid_search.fit(X_train, y_train, groups=groups_train)
        best_fold_model = grid_search.best_estimator_
        
        # Predikció a külső teszthalmazon
        probs = best_fold_model.predict_proba(X_test)[:, 1]
        out_of_fold_probs[test_idx] = probs

    # --- 3. Lépés: Klinikai (Szenzitivitás-vezérelt) Küszöb keresése ---
    print("\n3. Lépés: Klinikai osztályozási küszöbérték meghatározása...")
    fpr, tpr, thresholds = roc_curve(y, out_of_fold_probs)
    
    # ---------------------------------------------------------
    # ORVOSI SZABÁLY: A Recall (TPR) legyen legalább 85% (0.85)
    # Ezzel kikényszerítjük, hogy minél kevesebb beteget mondjunk egészségesnek!
    # ---------------------------------------------------------
    min_required_recall = 0.85
    
    # Megkeressük az összes olyan küszöböt, ahol megvan a 85% Recall
    valid_thresholds = np.where(tpr >= min_required_recall)[0]
    
    if len(valid_thresholds) > 0:
        # Ezek közül azt választjuk, ahol a legkevesebb egészségest áldozzuk be (legkisebb FPR)
        best_idx = valid_thresholds[np.argmin(fpr[valid_thresholds])]
        optimal_threshold = thresholds[best_idx]
    else:
        # Biztonsági háló, ha a modell nem tudna 85%-ot (nagyon ritka)
        print("  Figyelem: A modell nem tudja elérni a 85%-os Recall-t. Visszatérés a maximális Recall-hoz.")
        optimal_threshold = thresholds[np.argmax(tpr)]

    print(f"  -> Alapértelmezett küszöb: 0.5000")
    print(f"  -> BEMENETI ELVÁRÁS: Legalább {min_required_recall*100}%-os Szenzitivitás (Recall)")
    print(f"  -> SZÁMÍTOTT OPTIMÁLIS küszöb ehhez: {optimal_threshold:.4f}")

    # Új predikciók osztályokra bontása az orvosi küszöbbel
    final_preds = (out_of_fold_probs >= optimal_threshold).astype(int)

    # --- Predikciós Valószínűségek Eloszlása ---
    plt.figure(figsize=(10, 6))
    # Egészségesek görbéje
    sns.kdeplot(out_of_fold_probs[y == 0], color='blue', fill=True, label='Valós Control (0)', alpha=0.4)
    # Demensek görbéje
    sns.kdeplot(out_of_fold_probs[y == 1], color='red', fill=True, label='Valós Dementia (1)', alpha=0.4)
    
    # Döntési küszöb vonala
    plt.axvline(x=optimal_threshold, color='black', linestyle='--', lw=2, 
                label=f'Klinikai Küszöb ({optimal_threshold:.4f})')
    
    plt.title('Modell által becsült valószínűségek eloszlása')
    plt.xlabel('Prediktált valószínűség a Demenciára (0.0 - 1.0)')
    plt.ylabel('Sűrűség (Minta aránya)')
    plt.legend()
    plt.savefig(os.path.join(PLOT_DIR, 'ecapa_probability_distribution.png'), bbox_inches='tight')
    plt.close()

    # --- Precision-Recall Görbe ---
    from sklearn.metrics import precision_recall_curve, average_precision_score
    precisions, recalls, pr_thresholds = precision_recall_curve(y, out_of_fold_probs)
    ap_score = average_precision_score(y, out_of_fold_probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, color='purple', lw=2, label=f'PR-görbe (AP = {ap_score:.4f})')
    
    plt.xlabel('Recall (Szenzitivitás)')
    plt.ylabel('Precision (Precízió)')
    plt.title('Precision-Recall Görbe')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(PLOT_DIR, 'ecapa_precision_recall_curve.png'), bbox_inches='tight')
    plt.close()

    # --- 4. Lépés: Végső Kiértékelés ---
    print("\n4. Lépés: Végső kiértékelés az optimális küszöbbel...")
    
    auc_score = roc_auc_score(y, out_of_fold_probs)
    acc_score = accuracy_score(y, final_preds)
    uar_score = balanced_accuracy_score(y, final_preds) # Unweighted Accuracy

    # --- ROC Görbe rajzolása ---
    from sklearn.metrics import auc
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC-görbe (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Véletlen tipp')
    
    # Bejelöljük rajta az általad választott klinikai küszöböt!
    plt.scatter([fpr[best_idx]], [tpr[best_idx]], color='red', s=100, zorder=5, 
                label=f'Kiválasztott Küszöb (Recall >= 85%)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Fals Pozitív Ráta (1 - Specificitás)')
    plt.ylabel('Valós Pozitív Ráta (Szenzitivitás)')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(PLOT_DIR, 'ecapa_roc_curve.png'), bbox_inches='tight')
    plt.close()

    print("\n" + "="*55)
    print("VÉGSŐ EREDMÉNYEK (Nested CV - Out-of-Fold Predicts)")
    print("="*55)
    print(f"ROC-AUC (Elsődleges metrika): {auc_score:.4f}")
    print(f"Unweighted Accuracy (UAR):    {uar_score:.4f}")
    print(f"Súlyozatlan Accuracy:         {acc_score:.4f}")
    print("\nRészletes metrikák (Precision, Recall, F1):")
    # A riportban a Recall = Sensitivity a Dementia osztályra!
    print(classification_report(y, final_preds, target_names=['Control (0)', 'Dementia (1)']))

    # --- 5. Lépés: Tévesztési Mátrix (Confusion Matrix) ---
    print("\n5. Lépés: Tévesztési mátrix és Feature Importance generálása...")
    cm = confusion_matrix(y, final_preds)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Prediktált Control', 'Prediktált Dementia'],
                yticklabels=['Valós Control', 'Valós Dementia'],
                annot_kws={"size": 16})
    plt.title(f'Tévesztési Mátrix (Optimális Küszöb: {optimal_threshold:.4f})', fontsize=14)
    plt.savefig(os.path.join(PLOT_DIR, 'ecapa_confusion_matrix.png'), bbox_inches='tight')
    plt.close()

    # --- Feature Importance (Production Modell) ---
    # MÓDOSÍTÁS 7: Végső grid search-ben is StratifiedGroupKFold alkalmazása
    final_grid = GridSearchCV(
        estimator=xgb.XGBClassifier(objective='binary:logistic', eval_metric='auc', scale_pos_weight=pos_weight, random_state=42),
        param_grid=param_grid, scoring='roc_auc', 
        cv=StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42), n_jobs=-1
    )
    # MÓDOSÍTÁS 8: groups paraméter átadása a végső betanításkor
    final_grid.fit(X, y, groups=groups)
    production_model = final_grid.best_estimator_

    plt.figure(figsize=(12, 8))
    xgb.plot_importance(production_model, max_num_features=20, importance_type='gain', 
                        title='XGBoost - Top 20 Jellemző (Teljes adaton tanítva)', 
                        xlabel='Információ nyereség (Gain)', ylabel='Jellemző neve', height=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'ecapa_feature_importance.png'), bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    df_merged = load_and_merge_data()
    if df_merged is not None:
        train_and_evaluate_nested_cv(df_merged)