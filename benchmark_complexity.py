"""
Benchmarks elapsed wall-clock time for synthetic data generation
(spatial KDE sampling + radio feature prediction), reproducing the
measurement described in Sec. VI.E ("Empirical Benchmarks") of the
manuscript: median of N_RUNS runs, excluding model training/loading and file I/O,
all CPU cores.

The radio generators follow the paper implementation, not the mock pipeline:
one model per PCI (each predicts the RSRP of one cell) with the paper's
architectures and generation steps (chunks, Euler steps, noise, rounding,
clipping to -140 dBm). The proposed 1NN copies the fingerprint of the nearest
real measurement (one neighbour search for all PCIs) and adds shadowing noise to
the served cells.

Spatial generator is fixed to KDE (the proposed spatial method, models/spatial/kde.pkl).
Radio generator is varied over: knn (1NN), rf, gpr_rq, mlp, cgan, nf, rec_flow.
Augmentation rate A: A x m synthetic samples, m = rows in data/train.csv.

The per-PCI models are fitted on data/train.csv before timing. Inference time
does not depend on the learned weights, so the neural networks are trained for
TRAIN_EPOCHS epochs only and the GPR kernel hyperparameters are kept at their
initial values (optimizer=None). GPR predicts the mean only (the predictive
standard deviation is not used by the generator). All per-PCI GPRs have the same
size (same training points and kernel), so one PCI is timed and multiplied by the
number of PCIs.

Usage:
    python benchmark_complexity.py                       # all 7 models
    python benchmark_complexity.py knn,rf                # just a subset
    N_RUNS=3 python benchmark_complexity.py gpr_rq       # fewer reps (gpr_rq/rec_flow are slow)
"""
import sys, os, time, statistics, warnings
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic

warnings.filterwarnings("ignore")
torch.set_num_threads(os.cpu_count())  # all CPU cores, for every model

A_VALUES = [10, 20, 30]
N_RUNS = int(os.environ.get('N_RUNS', '5'))
TRAIN_EPOCHS = int(os.environ.get('TRAIN_EPOCHS', '2'))
UNSERVED_RSRP = -160
MIN_RSRP = -140
SHADOW = 8          # 1NN noise, City center: np.random.randint(-8.2, 8.2) -> [-8, 7]
Z_DIM = 16          # cGAN noise dimension
REFLOW_STEPS = 20   # Euler steps
GPR_CHUNK = 4000    # GPR predicts chunk x m kernel matrices

# ---------- Paper models (one per PCI) ----------
class PlainMLP(nn.Module):
    def __init__(self, xdim=2, hidden=(256, 128), dropout=0.1):
        super().__init__()
        layers, prev = [], xdim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class Generator(nn.Module):
    def __init__(self, z_dim=Z_DIM, c_dim=2, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim + c_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1))

    def forward(self, z, c):
        return self.net(torch.cat([z, c], dim=1))

def _prepare_rqs_params(params, K, B, min_bin=1e-3, min_deriv=1e-3):
    w_raw = params[:, :K]; h_raw = params[:, K:2*K]; d_raw = params[:, 2*K:2*K + (K+1)]
    widths = torch.softmax(w_raw, dim=1) * (2*B - K*min_bin) + min_bin
    heights = torch.softmax(h_raw, dim=1) * (2*B - K*min_bin) + min_bin
    derivs = torch.nn.functional.softplus(d_raw) + min_deriv
    cumwidths = torch.cumsum(widths, dim=1)
    cumwidths = torch.cat([torch.zeros_like(cumwidths[:, :1]), cumwidths], dim=1) + (-B)
    cumheights = torch.cumsum(heights, dim=1)
    cumheights = torch.cat([torch.zeros_like(cumheights[:, :1]), cumheights], dim=1) + (-B)
    return widths, heights, derivs, cumwidths, cumheights

def _rqs_forward(x, widths, heights, derivs, cumwidths, cumheights, B, eps=1e-6):
    x = x.view(-1); N, K = widths.shape
    bin_idx = (torch.sum((x[:, None] >= cumwidths[:, :-1]), dim=1) - 1).clamp(0, K-1)
    gather = lambda t: t.gather(1, bin_idx[:, None]).squeeze(1)
    xk, yk = gather(cumwidths[:, :-1]), gather(cumheights[:, :-1])
    wk, hk = gather(widths), gather(heights)
    dk, dk1 = gather(derivs[:, :-1]), gather(derivs[:, 1:])
    theta = ((x - xk) / wk).clamp(eps, 1 - eps)
    sk = hk / wk
    numerator = hk * (sk * theta**2 + dk * theta * (1 - theta))
    denom = sk + (dk + dk1 - 2*sk) * theta * (1 - theta)
    return (yk + numerator / denom).view(-1, 1)

class ConditionalRQSFlow(nn.Module):
    def __init__(self, xdim=2, K=8, hidden=(128, 128), B=200.0):
        super().__init__()
        self.K = K; self.B = float(B)
        layers, prev = [], xdim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.GELU()]
            prev = h
        layers += [nn.Linear(prev, 3*K + 1)]
        self.cond = nn.Sequential(*layers)

    def forward(self, z, x):
        params = self.cond(x)
        widths, heights, derivs, cumw, cumh = _prepare_rqs_params(params, self.K, self.B)
        return _rqs_forward(z, widths, heights, derivs, cumw, cumh, self.B)

def time_embedding(t, dim=32):
    half = dim // 2
    freqs = torch.exp(-np.log(10000) * torch.arange(0, half, device=t.device).float() / half)
    args = t * freqs
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb

class VelocityNet(nn.Module):
    def __init__(self, xdim=2, tdim=32, hidden=(256, 128)):
        super().__init__()
        layers, prev = [], 1 + xdim + tdim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.GELU()]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)
        self.tdim = tdim

    def forward(self, y, x, t):
        return self.net(torch.cat([y, x, time_embedding(t, dim=self.tdim)], dim=1))

# ---------- Fit the per-PCI models (excluded from timing) ----------
def short_train(model, loss_fn, X, y, batch_size=256, lr=1e-3):
    """TRAIN_EPOCHS epochs; loss_fn(model, xb, yb) -> scalar."""
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(TRAIN_EPOCHS):
        perm = torch.randperm(len(X_t))
        for i in range(0, len(X_t), batch_size):
            idx = perm[i:i+batch_size]
            loss = loss_fn(model, X_t[idx], y_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return model.eval()

def mlp_loss(model, xb, yb):
    return nn.functional.smooth_l1_loss(model(xb), yb)

def cgan_loss(model, xb, yb):
    # Generator step only: inference cost does not depend on the weights
    return nn.functional.mse_loss(model(torch.randn(len(xb), Z_DIM), xb), yb)

def nf_loss(model, xb, yb):
    return nn.functional.mse_loss(model(torch.randn(len(xb), 1), xb), yb)

def reflow_loss(model, xb, yb):
    z = torch.randn_like(yb); t = torch.rand(len(xb), 1)
    return nn.functional.smooth_l1_loss(model((1.0 - t) * z + t * yb, xb, t), yb - z)

def build_models(algo, X, Y):
    """List of per-PCI models (one per RSRP column)."""
    if algo == 'knn':
        return NearestNeighbors(n_neighbors=1).fit(X)
    if algo == 'gpr_rq':
        # One PCI is timed (all per-PCI GPRs have the same cost)
        return [GaussianProcessRegressor(RationalQuadratic(length_scale=1.0, alpha=1.0), alpha=1e-1,
                                         normalize_y=True, optimizer=None).fit(X, Y[:, 0])]
    if algo == 'rf':
        return [RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1).fit(X, Y[:, j]) for j in range(Y.shape[1])]
    make = {'mlp': (PlainMLP, mlp_loss), 'cgan': (Generator, cgan_loss),
            'nf': (ConditionalRQSFlow, nf_loss), 'rec_flow': (VelocityNet, reflow_loss)}[algo]
    return [short_train(make[0](), make[1], X, Y[:, j]) for j in range(Y.shape[1])]

# ---------- Generation, as in the paper scripts ----------
@torch.no_grad()
def torch_predict(algo, model, coords_t):
    chunk = 10000 if algo == 'rec_flow' else 20000
    preds = []
    for i in range(0, len(coords_t), chunk):
        c = coords_t[i:i+chunk]; n = c.size(0)
        if algo == 'mlp':
            p = model(c)
        elif algo == 'cgan':
            p = model(torch.randn(n, Z_DIM), c)
        elif algo == 'nf':
            p = model(torch.randn(n, 1), c)
        elif algo == 'rec_flow':
            p = torch.randn(n, 1); dt = 1.0 / REFLOW_STEPS
            for s in range(REFLOW_STEPS):
                p = p + model(p, c, torch.full((n, 1), s * dt)) * dt
        preds.append(p.numpy().flatten())
    return np.concatenate(preds)

NOISE = {'mlp': (-5, 6), 'cgan': (-3, 4), 'gpr_rq': (-SHADOW, SHADOW + 1)}   # np.random.randint bounds

def postprocess(algo, vals):
    if algo in NOISE:
        vals = vals + np.random.randint(*NOISE[algo], size=vals.shape)
    return np.maximum(np.round(vals).astype(int), MIN_RSRP)

def run_once(algo, n_samples):
    model = models[algo]
    t0 = time.perf_counter()
    coords = spatial_model.sample(n_samples)
    if algo == 'knn':
        # 1NN: one neighbour search, copy the nearest fingerprint, shadowing noise on served cells
        idx = model.kneighbors(coords, return_distance=False)[:, 0]
        rsrp = Y[idx]
        rsrp = np.where(rsrp == UNSERVED_RSRP, rsrp, rsrp + np.random.randint(-SHADOW, SHADOW, size=rsrp.shape))
        return time.perf_counter() - t0
    t_kde = time.perf_counter() - t0
    t1 = time.perf_counter()
    coords_t = torch.tensor(coords, dtype=torch.float32)
    for m in model:
        if algo == 'gpr_rq':
            vals = np.concatenate([m.predict(coords[i:i+GPR_CHUNK])
                                   for i in range(0, len(coords), GPR_CHUNK)])
        elif algo == 'rf':
            vals = m.predict(coords)
        else:
            vals = torch_predict(algo, m, coords_t)
        postprocess(algo, vals)
    t_radio = time.perf_counter() - t1
    if algo == 'gpr_rq':
        t_radio *= n_pci   # one PCI timed -> all PCIs
    return t_kde + t_radio

RADIO_ALGOS = sys.argv[1].split(',') if len(sys.argv) > 1 else \
    ['knn', 'rf', 'gpr_rq', 'mlp', 'cgan', 'rec_flow', 'nf']
LABELS = {
    'knn': 'KDE-1NN (Proposed)', 'rf': 'KDE-RF', 'gpr_rq': 'KDE-GPR (RQ)',
    'mlp': 'KDE-MLP', 'cgan': 'KDE-CGAN', 'rec_flow': 'KDE-ReFlow', 'nf': 'KDE-NF',
}

if __name__ == "__main__":
    np.random.seed(42); torch.manual_seed(42)
    spatial_model = joblib.load('models/spatial/kde.pkl')
    df = pd.read_csv('data/train.csv')
    X = df[['LATITUDE', 'LONGITUDE']].values
    Y = df.filter(like='RSRP').values
    m, n_pci = len(df), Y.shape[1]

    models = {}
    for algo in RADIO_ALGOS:
        t = time.perf_counter()
        models[algo] = build_models(algo, X, Y)
        print(f"Fitted {LABELS[algo]} per-PCI models in {time.perf_counter() - t:.1f} s (not timed)", flush=True)

    print(f"\nm = {m} real training samples, {n_pci} PCIs, median of {N_RUNS} runs")
    print(f"{'Model':30s} " + " ".join(f"A={a:<9d}" for a in A_VALUES))
    results = {}
    for algo in RADIO_ALGOS:
        row = []
        for A in A_VALUES:
            n_samples = A * m
            run_once(algo, n_samples)  # warm-up, not timed
            times = [run_once(algo, n_samples) for _ in range(N_RUNS)]
            row.append(statistics.median(times))
        results[algo] = row
        print(f"{LABELS[algo]:30s} " + " ".join(f"{t:8.3f}s " for t in row), flush=True)

    print()
    print("Raw results dict:", results)
