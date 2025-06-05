import matplotlib.pyplot as plt
import os
import pandas as pd
from typing import List, Dict, Optional

def compare_models(
    results: Optional[List[Dict]] = None,
    csv_dir: Optional[str] = None,
    model_names: Optional[List[str]] = None
):
    """
    Compare models' training and testing results.
    Provide either a list of results dicts or a directory with CSV files.
    """
    if results is None and csv_dir is None:
        print("Provide either 'results' or 'csv_dir'.")
        return

    if csv_dir:
        csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
        if not csv_files:
            print(f"No CSV files in {csv_dir}")
            return
        results = []
        for f in csv_files:
            df = pd.read_csv(os.path.join(csv_dir, f))
            results.append({col: df[col].tolist() for col in df.columns})
        if model_names is None:
            model_names = [os.path.splitext(f)[0] for f in csv_files]
    else:
        if not results:
            print("No results to compare.")
            return
        if model_names is None:
            model_names = [f"Model {i+1}" for i in range(len(results))]

    metrics = [
        ("train_loss", "Train Loss"),
        ("train_acc", "Train Accuracy"),
        ("test_loss", "Test Loss"),
        ("test_acc", "Test Accuracy"),
    ]

    plt.figure(figsize=(12, 8))
    for idx, (key, ylabel) in enumerate(metrics, 1):
        plt.subplot(2, 2, idx)
        for i, res in enumerate(results):
            if key in res:
                plt.plot(res[key], label=model_names[i])
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
