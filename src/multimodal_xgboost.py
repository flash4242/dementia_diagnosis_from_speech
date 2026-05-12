import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.metrics import (accuracy_score, roc_auc_score, classification_report, 
                             confusion_matrix, roc_curve, balanced_accuracy_score,
                             precision_recall_curve, average_precision_score)
import os
import wandb
import joblib




# --- KONFIGURÁCIÓ ---
CSV_DIR = "./csv_output" 
ECAPA_CSV = os.path.join(CSV_DIR, "ecapa_embeddings.csv")
BERT_CSV = os.path.join(CSV_DIR, "bert_embeddings.csv")
PLOT_DIR = "./plots"
os.makedirs(PLOT_DIR, exist_ok=True)

def load_and_merge_data():
    print("1. Lépés: Adatok betöltése és fúziója (ECAPA + BERT)...")
    try:
        df_ecapa = pd.read_csv(ECAPA_CSV)
        df_bert = pd.read_csv(BERT_CSV)
    except FileNotFoundError as e:
        print(f"\nHiba: Nem található egy bemeneti fájl! Részletek: {e}")
        return None

    # 1. Oszlopnevek normalizálása (Minden legyen kisbetűs, hogy ne okozzon gondot a Label vs label)
    df_ecapa.columns = df_ecapa.columns.str.lower()
    df_bert.columns = df_bert.columns.str.lower()

    # 2. Defragmentáció a PerformanceWarning elkerülésére
    df_ecapa = df_ecapa.copy()
    df_bert = df_bert.copy()

    # 3. KÖZÖS KULCS LÉTREHOZÁSA (base_id)
    # ECAPA: filename-ből (pl. 001-0.mp3) levágjuk a kiterjesztést -> 001-0
    df_ecapa['base_id'] = df_ecapa['filename'].astype(str).apply(lambda x: os.path.splitext(x)[0])
    
    # BERT: patient_id-ből csinálunk base_id-t (biztos ami biztos, itt is levágjuk, ha lenne .cha kiterjesztés)
    df_bert['base_id'] = df_bert['patient_id'].astype(str).apply(lambda x: os.path.splitext(x)[0])

    # 4. Felesleges (régi) oszlopok eldobása fúzió előtt
    df_ecapa = df_ecapa.drop(columns=['filename'])
    df_bert = df_bert.drop(columns=['patient_id'])

    df_ecapa.columns = [f"{col}_ecapa" if col.startswith('e_') else col for col in df_ecapa.columns]
    df_bert.columns = [f"{col}_bert" if col.startswith('e_') else col for col in df_bert.columns]
    # 5. Fúzió a base_id és label alapján, suffixekkel
    df_merged = pd.merge(df_ecapa, df_bert, on=['base_id', 'label'], how='inner')
    
    print(f"  -> SIKERES FÚZIÓ: {df_merged.shape[0]} közös minta, összesen {df_merged.shape[1]-2} prediktor.\n")
    return df_merged

def train_and_evaluate_nested_cv(df):
    print("2. Lépés: Nested Cross-Validation indítása...")
    
    y = df['label']
    groups = df['base_id'].apply(lambda x: x.split('-')[0]) 
    
    X = df.drop(columns=['base_id', 'label']) 
    pos_weight = (y == 0).sum() / (y == 1).sum()

    # --- KÜLSŐ CV - StratifiedGroupKFold használata, hogy ugyanazon páciens ne kerülhessen test és train fold-ba is egyszerre (hiszen egy páciensnek van több felvétele is)
    outer_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    # Hiperparaméter háló (Az XGBoost megbirkózik a ~960 dimenzióval)
    param_grid = {
        'max_depth': [2, 3],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [50, 100, 200],
        'subsample': [0.7, 0.9],
        'colsample_bytree': [0.5, 0.8]
    }

    out_of_fold_probs = np.zeros(len(y))

    print("  -> Modellek tanítása és hangolása a hajtásokon (Inner + Outer CV)...")

    # A split metódusnak meg kell adni a groups változót!
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups=groups)):
        print(f"     Hajtás {fold+1}/5 folyamatban...")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Kinyerjük az aktuális hajtás tanító csoportjait (ez kell a belső CV-nek)
        groups_train = groups.iloc[train_idx]
        
        # --- BELSŐ CV (StratifiedGroupKFold) ---
        inner_cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
        
        xgb_clf = xgb.XGBClassifier(
            objective='binary:logistic', eval_metric='auc',
            scale_pos_weight=pos_weight, random_state=42
        )
        
        grid_search = GridSearchCV(
            estimator=xgb_clf, param_grid=param_grid, 
            scoring='roc_auc', cv=inner_cv, n_jobs=-1, verbose=0
        )
        
        # Tanítás és hangolás a belső CV-vel (itt adjuk át a csoportokat!)
        grid_search.fit(X_train, y_train, groups=groups_train)
        
        # --- ÚJ RÉSZ: Tanulási görbék kinyerése fánként (Boosting Rounds) ---
        best_params = grid_search.best_params_
        
        # Létrehozunk egy modellt a GridSearch által talált legjobb paraméterekkel
        fold_model = xgb.XGBClassifier(
            objective='binary:logistic', 
            eval_metric=['auc', 'logloss'], # Két metrikát is követünk!
            scale_pos_weight=pos_weight, 
            random_state=42,
            **best_params
        )
        
        # Betanítjuk az eval_set megadásával (ez utasítja az XGBoost-ot, hogy lépésenként logoljon)
        fold_model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=False
        )
        
        # Predikció a külső teszthalmazon a küszöbkereséshez
        probs = fold_model.predict_proba(X_test)[:, 1]
        out_of_fold_probs[test_idx] = probs

        # --- W&B FOLYAMATOS NAPLÓZÁS ---
        evals_result = fold_model.evals_result()
        num_trees = len(evals_result['validation_0']['auc'])
        
        # Végigmegyünk az összes fán (epoch), és beküldjük a W&B-be
        for i in range(num_trees):
            wandb.log({
                f"Fold_{fold+1}/Train_AUC": evals_result['validation_0']['auc'][i],
                f"Fold_{fold+1}/Val_AUC": evals_result['validation_1']['auc'][i],
                f"Fold_{fold+1}/Train_LogLoss": evals_result['validation_0']['logloss'][i],
                f"Fold_{fold+1}/Val_LogLoss": evals_result['validation_1']['logloss'][i],
                "boosting_round": i
            })
        

    # --- 3. Lépés: Klinikai (Szenzitivitás-vezérelt) Küszöb keresése ---
    print("\n3. Lépés: Klinikai osztályozási küszöbérték meghatározása...")
    fpr, tpr, thresholds = roc_curve(y, out_of_fold_probs)
    
    min_required_recall = 0.85
    valid_thresholds = np.where(tpr >= min_required_recall)[0]
    
    if len(valid_thresholds) > 0:
        best_idx = valid_thresholds[np.argmin(fpr[valid_thresholds])]
        optimal_threshold = thresholds[best_idx]
    else:
        print("  Figyelem: A modell nem tudja elérni a 85%-os Recall-t. Visszatérés a maximális Recall-hoz.")
        optimal_threshold = thresholds[np.argmax(tpr)]

    print(f"  -> BEMENETI ELVÁRÁS: Legalább {min_required_recall*100}%-os Szenzitivitás (Recall)")
    print(f"  -> SZÁMÍTOTT OPTIMÁLIS küszöb ehhez: {optimal_threshold:.4f}")

    final_preds = (out_of_fold_probs >= optimal_threshold).astype(int)

    # --- ÁBRÁK GENERÁLÁSA ---
    sns.set_theme(style="whitegrid")
    
    plt.figure(figsize=(10, 6))
    sns.kdeplot(out_of_fold_probs[y == 0], color='blue', fill=True, label='Valós Control (0)', alpha=0.4)
    sns.kdeplot(out_of_fold_probs[y == 1], color='red', fill=True, label='Valós Dementia (1)', alpha=0.4)
    plt.axvline(x=optimal_threshold, color='black', linestyle='--', lw=2, label=f'Klinikai Küszöb ({optimal_threshold:.4f})')
    plt.title('Modell által becsült valószínűségek eloszlása (ECAPA + BERT)')
    plt.xlabel('Prediktált valószínűség a Demenciára')
    plt.ylabel('Sűrűség (Minta aránya)')
    plt.legend()
    plt.savefig(os.path.join(PLOT_DIR, 'multimodal_probability_distribution.png'), bbox_inches='tight')
    plt.close()

    # Precision-Recall Görbe
    precisions, recalls, pr_thresholds = precision_recall_curve(y, out_of_fold_probs)
    ap_score = average_precision_score(y, out_of_fold_probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, color='purple', lw=2, label=f'PR-görbe (AP = {ap_score:.4f})')
    plt.xlabel('Recall (Szenzitivitás)')
    plt.ylabel('Precision (Precízió)')
    plt.title('Precision-Recall Görbe (ECAPA + BERT)')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(PLOT_DIR, 'multimodal_precision_recall_curve.png'), bbox_inches='tight')
    plt.close()

    # --- 4. Lépés: Végső Kiértékelés ---
    print("\n4. Lépés: Végső kiértékelés az optimális küszöbbel...")
    
    auc_score = roc_auc_score(y, out_of_fold_probs)
    acc_score = accuracy_score(y, final_preds)
    uar_score = balanced_accuracy_score(y, final_preds)

    # ROC Görbe rajzolása
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC-görbe (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.scatter([fpr[best_idx]], [tpr[best_idx]], color='red', s=100, zorder=5, label=f'Kiválasztott Küszöb (Recall >= 85%)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Fals Pozitív Ráta (1 - Specificitás)')
    plt.ylabel('Valós Pozitív Ráta (Szenzitivitás)')
    plt.title('Receiver Operating Characteristic (ROC) - Fused Model')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(PLOT_DIR, 'multimodal_roc_curve.png'), bbox_inches='tight')
    plt.close()

    print("\n" + "="*55)
    print("VÉGSŐ EREDMÉNYEK (Multimodális XGBoost Baseline)")
    print("="*55)
    print(f"ROC-AUC (Elsődleges metrika): {auc_score:.4f}")
    print(f"Unweighted Accuracy (UAR):    {uar_score:.4f}")
    print(f"Súlyozatlan Accuracy:         {acc_score:.4f}")
    print("\nRészletes metrikák (Precision, Recall, F1):")
    print(classification_report(y, final_preds, target_names=['Control (0)', 'Dementia (1)']))

    # --- 5. Lépés: Tévesztési Mátrix ---
    cm = confusion_matrix(y, final_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Prediktált Control', 'Prediktált Dementia'],
                yticklabels=['Valós Control', 'Valós Dementia'],
                annot_kws={"size": 16})
    plt.title(f'Tévesztési Mátrix (Optimális Küszöb: {optimal_threshold:.4f})', fontsize=14)
    plt.savefig(os.path.join(PLOT_DIR, 'multimodal_confusion_matrix.png'), bbox_inches='tight')
    plt.close()

    # --- Feature Importance (Production Modell) ---
    print("5. Lépés: Feature Importance generálása a teljes adaton...")
    final_grid = GridSearchCV(
        estimator=xgb.XGBClassifier(objective='binary:logistic', eval_metric='auc', scale_pos_weight=pos_weight, random_state=42),
        param_grid=param_grid, 
        scoring='roc_auc', 
        cv=StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42), # <--- CSERÉLVE
        n_jobs=-1
    )
    final_grid.fit(X, y, groups=groups) # <--- CSERÉLVE (groups átadása)
    production_model = final_grid.best_estimator_

    plt.figure(figsize=(12, 8))
    xgb.plot_importance(production_model, max_num_features=20, importance_type='gain', 
                        title='XGBoost - Top 20 Jellemző (ECAPA + BERT)', 
                        xlabel='Információ nyereség (Gain)', ylabel='Jellemző neve', height=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'multimodal_xgboost_feature_importance.png'))
    plt.close()
    print(f"\n[✔] Minden ábra sikeresen mentve a '{PLOT_DIR}' mappába.")

    # MODELL MENTÉSE INFERENCE-hez
    MODEL_DIR = "./saved_models"
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(production_model, os.path.join(MODEL_DIR, 'multimodal_xgboost.pkl'))
    
    # A klinikai küszöböt is el kell mentenünk!
    with open(os.path.join(MODEL_DIR, 'optimal_threshold.txt'), 'w') as f:
        f.write(str(optimal_threshold))
        
    print(f"  -> Modell és küszöb elmentve a {MODEL_DIR} mappába.")

    # --- 6. Lépés: Weights & Biases Naplózás ---
    print("\n6. Lépés: Eredmények és ábrák feltöltése a W&B-be...")
    
    # A legjobb hiperparaméterek naplózása a production modellből
    wandb.config.update(final_grid.best_params_)
    
    # Metrikák és ábrák feltöltése
    wandb.log({
        "test_roc_auc": auc_score,
        "test_uar": uar_score,
        "test_accuracy": acc_score,
        "optimal_threshold": optimal_threshold,
        "probability_distribution": wandb.Image(os.path.join(PLOT_DIR, 'multimodal_probability_distribution.png')),
        "precision_recall_curve": wandb.Image(os.path.join(PLOT_DIR, 'multimodal_precision_recall_curve.png')),
        "roc_curve": wandb.Image(os.path.join(PLOT_DIR, 'multimodal_roc_curve.png')),
        "confusion_matrix": wandb.Image(os.path.join(PLOT_DIR, 'multimodal_confusion_matrix.png')),
        "feature_importance": wandb.Image(os.path.join(PLOT_DIR, 'multimodal_xgboost_feature_importance.png'))
    })
    print("  -> W&B feltöltés sikeres!")

if __name__ == "__main__":
    # W&B inicializálása
    wandb.init(
        project="dementia-diagnosis", 
        name="multimodal-xgboost-baseline",
        notes="Nested CV XGBoost ECAPA és BERT fúzióval, 85% Recall orvosi küszöbbel."
    )
    
    df_merged = load_and_merge_data()
    if df_merged is not None:
        train_and_evaluate_nested_cv(df_merged)
        
    wandb.finish()