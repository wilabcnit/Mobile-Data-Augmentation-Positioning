"""
Computational complexity benchmark of the proposed pipeline (KDE spatial augmentation +
1NN radio augmentation + wKNN positioning), using the repository's own models and data.

Each step is timed over N_RUNS repetitions (after one warm-up run) with time.perf_counter;
the table reports mean ± standard deviation. Results are printed and saved to
benchmark/complexity_results.md.

Usage: python benchmark/complexity.py
"""
import sys
import os
import time
import platform
import joblib
import numpy as np
import pandas as pd
import sklearn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from utils.data_loader import load_mdt_data
from generate.full_pipeline import knn_radio
from positioning.train import build_wknn

N_RUNS = 20
N_SYNTHETIC = 2000

def timeit(fn, n_runs=N_RUNS):
    fn()  # warm-up
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return np.mean(times), np.std(times)

def main():
    kde = joblib.load('models/spatial/kde.pkl')
    knn = joblib.load('models/radio/knn.pkl')
    train_df = load_mdt_data('data/train.csv', mode='all')
    test_df = load_mdt_data('data/test.csv', mode='all')
    rsrp_cols = [c for c in train_df.columns if 'RSRP' in c]
    X_test = test_df[rsrp_cols].values

    coords = kde.sample(N_SYNTHETIC, random_state=0)
    rsrp = knn_radio(knn, coords)
    X_aug = np.vstack([train_df[rsrp_cols].values, rsrp])
    y_aug = np.vstack([train_df[['LATITUDE', 'LONGITUDE']].values, coords])
    wknn = build_wknn().fit(X_aug, y_aug)

    def end_to_end():
        c = kde.sample(N_SYNTHETIC, random_state=0)
        r = knn_radio(knn, c)
        model = build_wknn().fit(np.vstack([train_df[rsrp_cols].values, r]),
                                 np.vstack([train_df[['LATITUDE', 'LONGITUDE']].values, c]))
        model.predict(X_test)

    steps = [
        (f"Spatial augmentation: KDE sampling of {N_SYNTHETIC} positions", lambda: kde.sample(N_SYNTHETIC, random_state=0), N_SYNTHETIC),
        (f"Radio augmentation: 1NN fingerprints for {N_SYNTHETIC} positions", lambda: knn_radio(knn, coords), N_SYNTHETIC),
        (f"wKNN training on {len(X_aug)} real + synthetic fingerprints", lambda: build_wknn().fit(X_aug, y_aug), len(X_aug)),
        (f"Positioning inference: wKNN on {len(X_test)} test fingerprints", lambda: wknn.predict(X_test), len(X_test)),
        ("End-to-end: augmentation + wKNN training + inference", end_to_end, len(X_test)),
    ]

    rows = []
    for name, fn, n in steps:
        mean, std = timeit(fn)
        rows.append({"Step": name,
                     "Total time (ms)": f"{1e3 * mean:.2f} ± {1e3 * std:.2f}",
                     "Per sample (µs)": f"{1e6 * mean / n:.2f}"})
    table = pd.DataFrame(rows)

    hardware = (f"{platform.processor() or platform.machine()}, {os.cpu_count()} logical CPUs, "
                f"Python {platform.python_version()}, scikit-learn {sklearn.__version__}")
    header = (f"Hardware: {hardware}\n"
              f"Data: {len(train_df)} real training fingerprints, {len(rsrp_cols)} cells; mean ± std over {N_RUNS} runs.\n")
    markdown = header + "\n" + table.to_markdown(index=False) + "\n"
    print(markdown)
    with open(os.path.join(os.path.dirname(__file__), 'complexity_results.md'), 'w') as f:
        f.write(markdown)

if __name__ == "__main__":
    main()
