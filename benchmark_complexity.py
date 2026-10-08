"""
Benchmarks elapsed wall-clock time for synthetic data generation
(spatial KDE sampling + radio feature prediction), reproducing the
measurement described in Sec. VI.E ("Empirical Benchmarks") of the
manuscript: median of 5 runs, excluding model loading and file I/O.

Spatial generator is fixed to KDE (the proposed spatial method).
Radio generator is varied over: knn (1NN), rf, gpr_rq, mlp, cgan, nf, rec_flow.
Augmentation rate A: A x m synthetic samples, m = rows in data/train.csv.

Usage:
    python benchmark_complexity.py                       # all 7 models
    python benchmark_complexity.py knn,rf                # just a subset
    N_RUNS=3 python benchmark_complexity.py gpr_rq       # fewer reps (gpr_rq/rec_flow are slow)
"""
import sys, os, time, statistics
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import torch
import joblib

from generate.full_pipeline import load_radio_model, knn_radio, batched_predict

torch.set_num_threads(1)  # keep single-threaded for fair, reproducible per-call timing

# ---------- Load models once (excluded from timing) ----------
spatial_model = joblib.load('models/spatial/kde.pkl')

RADIO_ALGOS = sys.argv[1].split(',') if len(sys.argv) > 1 else \
    ['knn', 'rf', 'gpr_rq', 'mlp', 'cgan', 'rec_flow', 'nf']
LABELS = {
    'knn': 'KDE-1NN (Proposed)', 'rf': 'KDE-RF', 'gpr_rq': 'KDE-GPR (RQ)', 'gpr_se': 'KDE-GPR (SE)',
    'mlp': 'KDE-MLP', 'cgan': 'KDE-CGAN', 'rec_flow': 'KDE-ReFlow', 'nf': 'KDE-NF',
}

radio_models = {algo: load_radio_model(algo) for algo in RADIO_ALGOS}   # knn = models/radio/knn.pkl (K = 1)

with open('data/train.csv') as f:
    m = sum(1 for _ in f) - 1  # size of the reference database (rows in data/train.csv)
A_VALUES = [10, 20, 30]
N_RUNS = int(os.environ.get('N_RUNS', '5'))

def run_once(radio_algo, n_samples):
    radio_model, r_ckpt = radio_models[radio_algo]
    t0 = time.perf_counter()

    coords = spatial_model.sample(n_samples)
    coords_tensor = torch.FloatTensor(coords)

    sklearn_models = ['rf', 'gpr_rq', 'gpr_se']
    if radio_algo == 'knn':
        # 1NN: one neighbour search, the nearest real fingerprint (with its -160 cells) is copied
        rsrp = knn_radio(radio_model, coords)
    elif radio_algo in sklearn_models:
        # GPR.predict() builds the full (n_samples x m) kernel matrix in memory; chunk the
        # INPUT only (not the math) so large A doesn't blow up RAM
        rsrp = batched_predict(radio_model.predict, coords)
    else:
        mean_key = 'mean_c' if 'mean_c' in r_ckpt else 'mean_in'
        std_key = 'std_c' if 'std_c' in r_ckpt else 'std_in'
        mean_in = torch.as_tensor(r_ckpt[mean_key]).float()
        std_in = torch.as_tensor(r_ckpt[std_key]).float()
        coords_norm = (coords_tensor - mean_in) / std_in
        out_m_key = 'mean_x' if 'mean_x' in r_ckpt else 'mean_out'
        out_s_key = 'std_x' if 'std_x' in r_ckpt else 'std_out'
        n_cells = len(r_ckpt[out_m_key])
        with torch.no_grad():
            if radio_algo == 'mlp':
                rsrp_norm = radio_model(coords_norm)
            elif radio_algo == 'cgan':
                z = torch.randn(n_samples, r_ckpt['latent_dim'])
                rsrp_norm = radio_model(z, coords_norm)
            elif radio_algo == 'nf':
                z = torch.randn(n_samples, n_cells)
                rsrp_norm = radio_model.inverse(z, coords_norm)
            elif radio_algo == 'rec_flow':
                xt = torch.randn(n_samples, n_cells)
                n_steps = 25
                dt = 1.0 / n_steps
                for i in range(n_steps):
                    t_val = torch.ones(n_samples, 1) * (i / n_steps)
                    xt = xt + radio_model(xt, t_val, coords_norm) * dt
                rsrp_norm = xt
        m_out = np.asarray(torch.as_tensor(r_ckpt[out_m_key]).cpu())
        s_out = np.asarray(torch.as_tensor(r_ckpt[out_s_key]).cpu())
        rsrp = rsrp_norm.numpy() * s_out + m_out

    return time.perf_counter() - t0, rsrp.shape[0]

print(f"m = {m} real training samples, median of {N_RUNS} runs")
print(f"{'Model':30s} " + " ".join(f"A={a:<9d}" for a in A_VALUES))
results = {}
for algo in RADIO_ALGOS:
    row = []
    for A in A_VALUES:
        n_samples = A * m
        run_once(algo, n_samples)  # warm-up, not timed
        times = [run_once(algo, n_samples)[0] for _ in range(N_RUNS)]
        row.append(statistics.median(times))
    results[algo] = row
    print(f"{LABELS[algo]:30s} " + " ".join(f"{t:8.3f}s " for t in row), flush=True)

print()
print("Raw results dict:", results)
