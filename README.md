# MDT Augmentation for Improved Positioning in Mobile Networks

This repository contains the code for the research paper: **"Improving Outdoor Multi-cell Fingerprinting-based Positioning via Mobile Data Augmentation."** https://arxiv.org/abs/2509.19405

The real MDT dataset used in the paper cannot be shared. This repository runs the full pipeline on a **mock MDT dataset** produced by `generate_mock_data.py`; the numbers below are for that mock data and illustrate the pipeline, while the results reported in the paper come from the real MDT data.

---

## Quick start (end-to-end reproduction)

```bash
pip install torch numpy pandas scikit-learn scipy joblib tabulate
./run_pipeline.sh
```

`run_pipeline.sh` runs every step in order: mock data generation → spatial and radio model training (incl. the KNN K ablation) → model evaluation → synthetic data generation (all 32 combinations) → positioning training and evaluation → complexity benchmark. Each step can also be run on its own (see below).

---

## Codebase Structure

```
├── data/                       # MDT datasets (70/10/20 split)
│   ├── train.csv
│   ├── validation.csv          # hyperparameter selection only
│   └── test.csv
├── augm_data/                  # Synthetic MDT datasets (one per spatial x radio combination)
├── models/
│   ├── spatial/                # KDE, GMM, GAN, NF models
│   ├── radio/                  # KNN, RF, GPR, MLP, cGAN, NF, ReFlow models
│   └── positioning/            # wKNN positioning models
├── train/                      # Training scripts
│   ├── spatial/
│   └── radio/
│       └── knn_k_ablation.py   # radio KNN, K = 1..5 on the validation set
├── test/                       # Model evaluation on the test set
│   ├── spatial/                # 1-D and 2-D (Fasano-Franceschini) KS tests
│   └── radio/                  # radio MAE; knn_k_positioning.py: K = 1..5 on the test set
├── generate/
│   └── full_pipeline.py        # synthetic MDT generation (spatial model -> radio model)
├── positioning/
│   ├── train.py                # wKNN training (real + synthetic, or synthetic only)
│   └── test.py                 # median positioning error + paired significance test
├── benchmark/
│   └── complexity.py           # computational complexity benchmark
├── utils/
│   ├── data_loader.py
│   ├── geo.py                  # haversine distance, lat/lon -> metres projection
│   ├── metrics.py              # radio MAE, spatial 1-D / 2-D KS tests
│   ├── ks2d_ff.py              # 2-D Fasano-Franceschini two-sample KS test
│   ├── gpr.py                  # GPR fitting helper
│   └── stats_compare.py        # paired / unpaired model comparison
├── generate_mock_data.py
└── run_pipeline.sh
```

---

## Dependencies

Python 3.8+ with:
* `torch` [2.7.1+cpu]
* `numpy` [2.0.2]
* `pandas` [2.3.0]
* `scikit-learn` [1.6.1]
* `scipy` [1.15.3]
* `joblib` [1.5.1]
* `tabulate` (results tables)

---

## Mock MDT data

```bash
python generate_mock_data.py
```

One urban scenario (city center around Bologna), 10,000 MDT measurements, 35 cells:

* **Coordinates:** valid WGS84 latitude/longitude (users around hotspots, offsets from a reference point converted to degrees).
* **Pathloss:** 3GPP TR 38.901 UMa NLOS (Table 7.4.1-1), carrier 1.8 GHz.
* **Shadow fading:** log-normal, σ = 7.8 dB, spatially correlated (exponential autocorrelation, 50 m decorrelation distance, TR 38.901 Table 7.5-6, UMa NLOS).
* **MDT reporting:** RSRP in the reporting range [-140, -44] dBm (TS 36.133), 1 dB resolution; each measurement reports only the detected cells (RSRP ≥ -124 dBm), at most the 8 strongest. All other cells are **not served** and stored as **-160**.
* **Split:** 70 % train (7,000) / 10 % validation (1,000) / 20 % test (2,000). The validation set is used only for hyperparameter selection (KDE bandwidth, radio KNN K); the test set only for the final evaluation.

---

## Methodology

### 1. Spatial Augmentation

**Goal:** learn the spatial distribution of users and generate new valid locations.

* **KDE:** Kernel Density Estimation (Gaussian kernel); bandwidth selected by log-likelihood on the validation set, over a grid spanning 0.1 %–100 % of the coordinate spread.
* **GMM:** Gaussian Mixture Model; number of components (1–40) selected by BIC.
* **GAN:** Generative Adversarial Network.
* **NF:** Normalizing Flow (RealNVP).

### 2. Radio Augmentation

**Goal:** learn the radio fingerprint at each location and generate new valid fingerprints.

* **KNN:** K-Nearest Neighbors regressor, **K = 1**, neighbours searched by distance in metres. For each synthetic position the fingerprint of the nearest real measurement is reused, **including its not-served cells (-160)**: a cell that is not served at the nearest original position is also not served at the synthetic position. `train/radio/knn_k_ablation.py` evaluates K = 1..5 on the validation set (see results).
* **RF:** Random Forest regressor.
* **GPR-RQ:** Gaussian Process, kernel = amplitude × Rational Quadratic + White noise.
* **GPR-SE:** Gaussian Process, kernel = amplitude × Squared Exponential + White noise.
* **MLP:** Multi-Layer Perceptron.
* **cGAN:** Conditional GAN trained with the least-squares (LSGAN) objective.
* **NF:** Conditional Normalizing Flow.
* **ReFlow:** Rectified Flow.

GPR inputs are positions in metres; targets are normalized (`normalize_y=True`), and the kernel hyperparameters (amplitude, length scale, noise level) are learned by maximizing the marginal likelihood on a 2,000-point subset, then the GPR is refit on all training points. The fitted GPR models are large, so they are not stored in the repository: `python train/radio/gpr_rq.py` and `python train/radio/gpr_se.py` re-create them. The synthetic datasets (`augm_data/`) and the positioning models (`models/positioning/`) are also not stored (several hundred MB); `python generate/full_pipeline.py all` and `python positioning/train.py all` re-create them.

The stochastic models (cGAN, NF, ReFlow) are scored both on a single draw (what the augmentation uses) and on the mean of 100 draws (their radio-map estimate, comparable to the deterministic regressors).

**_Implementation Note:_** For the mock data provided here, we use a single "one-shot" training model. With real MDT data, models should be trained per-PCI (as done in the paper) to capture cell-specific signal characteristics.

### 3. Positioning

A **weighted KNN (wKNN)** positioning model (K = 3, weights = inverse **Manhattan** distance between RSRP fingerprints) is trained on the **real training data + synthetic data** and tested on the real test set. Models trained on synthetic data only are also reported (`wknn_<spatial>_<radio>_synth`).

The **positioning error** is the great-circle (**haversine**) distance in metres between the estimated and the true WGS84 position. The haversine distance is used only for this error; the wKNN compares fingerprints with the Manhattan distance.

`positioning/test.py` reports the **median** positioning error. A median is marked significantly different from the original data when the paired bootstrap 95 % confidence interval of the median difference (same test points, 2,000 resamples) excludes 0.

### 4. Running the steps individually

```bash
python generate/full_pipeline.py kde knn     # one combination (2 x real training size) -> augm_data/synthetic_kde_knn.csv
python generate/full_pipeline.py all         # all 32 combinations
python positioning/train.py all              # baseline + all combinations (real+synthetic and synthetic-only)
python positioning/test.py                   # median positioning error tables
python benchmark/complexity.py               # complexity benchmark -> benchmark/complexity_results.md
```

---

## Results on the mock data (test set)

All numbers below are on the **test set** (2,000 measurements), produced by `run_pipeline.sh` with fixed seeds. Each synthetic dataset has **2 × the real training set = 14,000 samples** (`SYNTHETIC_RATIO` in `generate/full_pipeline.py`).

### Radio KNN: K ablation (validation set)

| K | Validation radio MAE (dB) | Validation median positioning error (m) |
| :--- | :--- | :--- |
| **1** | **2.54** | **5.26** |
| 2 | 2.72 | 5.46 |
| 3 | 2.93 | 5.54 |
| 4 | 3.11 | 5.59 |
| 5 | 3.28 | 5.58 |

K = 1 gives the lowest radio error and the lowest downstream positioning error: copying the nearest real fingerprint keeps its exact set of served / not-served cells, whereas averaging K > 1 neighbours mixes RSRP values with the -160 not-served value.

### Radio KNN: K = 1..5 positioning on the test set

`python test/radio/knn_k_positioning.py` — KDE positions + radio KNN with K = 1..5, wKNN trained on real + synthetic data, median positioning error on the test set (K itself is selected on the validation set above).

| Radio KNN | Median error (m) | vs original data (5.153 m), 95 % CI | K = 1 minus this K, 95 % CI |
| :--- | :--- | :--- | :--- |
| **K = 1** | **4.921** | **-0.232 [-0.522, -0.027]** | – |
| K = 2 | 5.063 | -0.090 [-0.318, 0.085] | -0.142 [-0.374, 0.041] |
| K = 3 | 5.285 | +0.132 [-0.065, 0.301] | **-0.364 [-0.618, -0.167]** |
| K = 4 | 5.359 | +0.206 [0.003, 0.382] | **-0.438 [-0.720, -0.196]** |
| K = 5 | 5.422 | +0.269 [0.056, 0.457] | **-0.501 [-0.807, -0.240]** |

K = 1 gives the lowest median error and is the only K that significantly improves on the original data; it is significantly better than K = 3, 4, 5 (bold: CI excludes 0).

### Spatial models: two-sample KS tests (generated vs test positions)

| Model | 2-D KS (Fasano-Franceschini) D | p (1,000 permutations) | 1-D KS lat / lon |
| :--- | :--- | :--- | :--- |
| KDE | 0.033 | 0.747 | 0.022 / 0.021 |
| NF | 0.049 | 0.142 | 0.021 / 0.038 |
| GMM | 0.092 | 0.001 | 0.074 / 0.061 |
| GAN | 0.216 | 0.001 | 0.188 / 0.099 |

### Radio models: MAE on the test set

| Model | MAE (dB) | MAE, mean of 100 draws (dB) |
| :--- | :--- | :--- |
| KNN (K=1) | 2.53 | – |
| GPR-RQ | 3.17 | – |
| RF | 3.76 | – |
| GPR-SE | 5.09 | – |
| MLP | 8.98 | – |
| ReFlow | 12.71 (single draw) | 10.95 |
| cGAN | 12.59 (single draw) | 11.29 |
| NF | 24.51 (single draw) | 13.57 |

### Positioning: median error (m)

`*` = median significantly different from the original data (paired bootstrap 95 % CI of the median difference excludes 0).

**wKNN trained on real + synthetic data (augmentation):**

| Spatial \ Radio | KNN | MLP | CGAN | GPR_RQ | GPR_SE | NF | REC_FLOW | RF |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Original data** | 5.153m | 5.153m | 5.153m | 5.153m | 5.153m | 5.153m | 5.153m | 5.153m |
| **KDE** | 4.921m* | 5.171m | 5.201m | 5.076m | 5.146m | 5.153m | 5.518m* | 5.183m |
| **NF** | 4.984m | 5.136m | 5.218m | 5.008m* | 5.187m | 5.136m | 5.535m* | 5.122m |
| **GAN** | 4.920m* | 5.131m | 5.157m | 5.064m | 5.147m | 5.163m | 5.503m* | 5.178m |
| **GMM** | 4.898m* | 5.129m | 5.237m | 5.099m | 5.141m | 5.145m | 5.481m* | 5.138m |

**wKNN trained on synthetic data only:**

| Spatial \ Radio | KNN | MLP | CGAN | GPR_RQ | GPR_SE | NF | REC_FLOW | RF |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Original data** | 5.153m | 5.153m | 5.153m | 5.153m | 5.153m | 5.153m | 5.153m | 5.153m |
| **KDE** | 5.680m* | 20.249m* | 94.848m* | 7.590m* | 9.710m* | 113.285m* | 64.555m* | 8.558m* |
| **NF** | 5.751m* | 20.826m* | 99.912m* | 7.569m* | 9.485m* | 121.908m* | 65.294m* | 8.850m* |
| **GAN** | 14.684m* | 35.886m* | 105.534m* | 21.856m* | 23.556m* | 128.640m* | 85.386m* | 23.687m* |
| **GMM** | 6.443m* | 20.279m* | 106.655m* | 8.642m* | 10.326m* | 128.506m* | 69.661m* | 9.890m* |

### Summary

* **Real + synthetic data:** KDE + KNN reduces the median error from 5.153 m to 4.921 m (paired bootstrap 95 % CI of the difference [-0.52, -0.03] m). KNN is the best radio model for every spatial model. KDE + KNN is significantly better than KDE + MLP, KDE + GPR-SE and KDE + RF; the KNN-based combinations GMM + KNN (4.898 m), GAN + KNN (4.920 m) and NF + KNN (4.984 m) are not significantly different from it.
* **Synthetic data only:** KDE + KNN gives the lowest median error of all 32 combinations (5.680 m); it is significantly better than GMM + KNN, KDE + GPR-RQ and KDE + RF, and not significantly different from NF + KNN (5.751 m).

### Computational complexity

`python benchmark/complexity.py` (results in `benchmark/complexity_results.md`):

Hardware: Intel(R) Xeon(R) Processor @ 2.80GHz, 4 logical CPUs, Python 3.13.16, scikit-learn 1.9.1
Data: 7000 real training fingerprints, 35 cells; 50 timed runs per step.

| Step                                                     |   Median time (ms) | Mean ± std (ms)   |   Median per sample (µs) |
|:---------------------------------------------------------|-------------------:|:------------------|-------------------------:|
| Spatial augmentation: KDE sampling of 14000 positions    |               1.24 | 1.26 ± 0.08       |                     0.09 |
| Radio augmentation: 1NN fingerprints for 14000 positions |              46.53 | 47.81 ± 4.49      |                     3.32 |
| wKNN training on 21000 real + synthetic fingerprints     |               0.2  | 0.21 ± 0.03       |                     0.01 |
| Positioning inference: wKNN on 2000 test fingerprints    |            1203.58 | 1645.09 ± 709.86  |                   601.79 |
| End-to-end: augmentation + wKNN training + inference     |            1883.64 | 2094.77 ± 966.47  |                   941.82 |

## Citation

If you use this code for your research, please cite:
```
@article{chahoud2025improving,
  title={Improving Outdoor Multi-cell Fingerprinting-based Positioning via Mobile Data Augmentation},
  author={Chahoud, Tony and Amorosa, Lorenzo Mario and Marini, Riccardo and De Nardis, Luca},
  journal={arXiv preprint arXiv:2509.19405},
  year={2025}
}
```
