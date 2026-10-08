"""
Elapsed time for synthetic data generation across high augmentation rates (Table IX).

Augmentation rate A = number of synthetic samples / number of real training samples,
so A = 10, 20, 30 generate 10x, 20x, 30x the real training set size.

Each cell is the elapsed time of generate/full_pipeline.synthesize(): KDE spatial
sampling + radio model inference for all synthetic samples. The models are loaded
once beforehand and nothing is written to disk, so the time covers the generation
itself only. Each cell is the median of N_RUNS runs.

Usage: python benchmark/generation_time.py
Results are printed and saved to benchmark/generation_time_results.md.
"""
import sys
import os
import time
import platform
import numpy as np
import pandas as pd
import torch
import sklearn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from generate.full_pipeline import synthesize, load_spatial_model, load_radio_model

AUGMENTATION_RATES = (10, 20, 30)
RADIO_MODELS = [('knn', 'KDE-1NN (Proposed)'), ('rf', 'KDE-RF'), ('gpr_rq', 'KDE-GPR (RQ)'),
                ('mlp', 'KDE-MLP'), ('cgan', 'KDE-CGAN'), ('rec_flow', 'KDE-ReFlow'), ('nf', 'KDE-NF')]
N_RUNS = 5

def cpu_model():
    try:
        with open('/proc/cpuinfo') as f:
            for line in f:
                if line.startswith('model name'):
                    return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()

def main():
    with open('data/train.csv') as f:
        n_real = sum(1 for _ in f) - 1
    spatial = load_spatial_model('kde')

    rows = []
    for radio_algo, label in RADIO_MODELS:
        radio = load_radio_model(radio_algo)
        row = {"Model": label}
        for a in AUGMENTATION_RATES:
            n = a * n_real
            times = []
            for _ in range(N_RUNS):
                start = time.perf_counter()
                synthesize('kde', radio_algo, n, spatial=spatial, radio=radio)
                times.append(time.perf_counter() - start)
            row[f"A = {a} ({n:,} samples)"] = f"{np.median(times):.3f} s"
            print(f"{label:<20} A={a:<3} {np.median(times):8.3f} s  (runs: {', '.join(f'{t:.3f}' for t in times)})", flush=True)
        rows.append(row)

    hardware = (f"{cpu_model()}, {os.cpu_count()} logical CPUs, {torch.get_num_threads()} PyTorch threads, "
                f"Python {platform.python_version()}, PyTorch {torch.__version__}, scikit-learn {sklearn.__version__}")
    header = (f"Hardware: {hardware}\n"
              f"Real training samples: {n_real:,}; median of {N_RUNS} runs; time = KDE sampling + radio "
              f"model inference (models preloaded, no file I/O).\n")
    markdown = header + "\n" + pd.DataFrame(rows).to_markdown(index=False) + "\n"
    print("\n" + markdown)
    with open(os.path.join(os.path.dirname(__file__), 'generation_time_results.md'), 'w') as f:
        f.write(markdown)

if __name__ == "__main__":
    main()
