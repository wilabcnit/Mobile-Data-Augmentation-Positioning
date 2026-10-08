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
│   └── radio/                  # radio MAE
├── generate/
│   └── full_pipeline.py        # synthetic MDT generation (spatial model -> radio model)
├── positioning/
│   ├── train.py                # wKNN training (real + synthetic, or synthetic only)
│   └── test.py                 # median / mean positioning error + paired significance test
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

GPR inputs are positions in metres; targets are normalized (`normalize_y=True`), and the kernel hyperparameters (amplitude, length scale, noise level) are learned by maximizing the marginal likelihood on a 2,000-point subset, then the GPR is refit on all training points. The fitted GPR models are large, so they are not stored in the repository: `python train/radio/gpr_rq.py` and `python train/radio/gpr_se.py` re-create them.

The stochastic models (cGAN, NF, ReFlow) are scored both on a single draw (what the augmentation uses) and on the mean of 100 draws (their radio-map estimate, comparable to the deterministic regressors).

**_Implementation Note:_** For the mock data provided here, we use a single "one-shot" training model. With real MDT data, models should be trained per-PCI (as done in the paper) to capture cell-specific signal characteristics.

### 3. Positioning

A **weighted KNN (wKNN)** positioning model (K = 3, weights = inverse **Manhattan** distance between RSRP fingerprints) is trained on the **real training data + synthetic data** and tested on the real test set. Models trained on synthetic data only are also reported (`wknn_<spatial>_<radio>_synth`).

The **positioning error** is the great-circle (**haversine**) distance in metres between the estimated and the true WGS84 position. The haversine distance is used only for this error; the wKNN compares fingerprints with the Manhattan distance.

`positioning/test.py` reports the **median** and the **mean** positioning error. A median is marked significantly different from the original data when the paired bootstrap 95 % confidence interval of the median difference (same test points, 2,000 resamples) excludes 0.

### 4. Running the steps individually

```bash
python generate/full_pipeline.py kde knn     # one combination -> augm_data/synthetic_kde_knn.csv
python generate/full_pipeline.py all         # all 32 combinations
python positioning/train.py all              # baseline + all combinations (real+synthetic and synthetic-only)
python positioning/test.py                   # results tables
python benchmark/complexity.py               # complexity benchmark -> benchmark/complexity_results.md
```

---

<!-- RESULTS -->

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
