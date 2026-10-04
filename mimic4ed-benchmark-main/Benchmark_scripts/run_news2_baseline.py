import os
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

def main():
    print("-" * 50)
    print("Task 2 - Critical Outcome Prediction (NEWS2 Baseline)")
    print("-" * 50)

    # Load test set (as evaluated by GRU-D and other baselines)
    df_test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    
    target_col = 'outcome_critical'
    
    if 'score_NEWS2' not in df_test.columns:
        print("NEWS2 score not found in the dataset.")
        return
    
    y_test = df_test[target_col].astype(int).values
    
    # NEWS2 score is already calculated in the dataset
    # We use it directly as a probability-like score (higher score = higher risk)
    y_prob = df_test['score_NEWS2'].fillna(0).values
    
    auroc = roc_auc_score(y_test, y_prob)
    auprc = average_precision_score(y_test, y_prob)
    
    print(f"AUROC: {auroc:.4f}")
    print(f"AUPRC: {auprc:.4f}")
    
    # Append to baseline_results.csv
    results_csv = os.path.join(DATA_DIR, "baseline_results.csv")
    if os.path.exists(results_csv):
        df_results = pd.read_csv(results_csv)
        new_row = {
            'task': 'Task 2 - Critical Outcome',
            'target': target_col,
            'model': 'NEWS2 (Clinical Baseline)',
            'auroc': round(auroc, 4),
            'auprc': round(auprc, 4),
            'n_train': 0,
            'n_test': len(df_test),
            'pos_rate_test': round(np.mean(y_test), 4),
            'category': 'Clinical Baselines'
        }
        df_results = pd.concat([df_results, pd.DataFrame([new_row])], ignore_index=True)
        df_results.to_csv(results_csv, index=False)
        print("Added NEWS2 results to data/baseline_results.csv")

if __name__ == '__main__':
    main()
