### **README.md**

# MDT Augmentation for Improved Positioning in Mobile Networks

This repository contains the code for the research paper: **"Improving Outdoor Multi-cell Fingerprinting-based Positioning via Mobile Data Augmentation."** https://arxiv.org/abs/2509.19405

---

## Codebase Structure

```
mdt_augmentation/
├── data/                       # MDT datasets
│   ├── train.csv
│   ├── validation.csv          # hyperparameter selection only
│   └── test.csv
├── augm_data/                  # Output directory for synthetic MDT datasets
├── models/                     # Saved models
│   ├── spatial/                # KDE, GMM, GAN, NF models
│   ├── radio/                  # KNN, RF, GPR, MLP, cGAN, NF, ReFlow models
│   └── positioning/            # wKNN positioning models
├── train/                      # Training scripts
│   ├── spatial/
│   ├── radio/
├── test/                       # Evaluation scripts(MAE)
│   ├── spatial/
│   ├── radio/
│   │   └── knn_k_ablation.py   # sweeps K = 1..5 for radio KNN (validation set)
├── generate/                   # Synthetic data generator using trained modules
│   ├── full_pipeline.py
├── positioning/                # Positioning algorithm
│   ├── train.py
│   └── test.py
├── utils/
│   ├── data_loader.py
│   ├── metrics.py              # radio MAE, spatial 1-D / 2-D KS tests
│   ├── ks2d_ff.py              # 2-D Fasano-Franceschini KS test
│   ├── gpr.py                  # GPR fitting helper
│   └── stats_compare.py        # paired / unpaired model comparison
└── generate_mock_data.py


```

---

## Dependencies

The project requires Python 3.8+ and the following libraries:
* `torch` [2.7.1+cpu]
* `numpy`[2.0.2]
* `pandas`[2.3.0]
* `scikit-learn`[1.6.1]
* `scipy`[1.15.3]
* `joblib`[1.5.1]
* `tabulate` (results tables in `positioning/test.py`)

If you do not have real MDT traces yet, generate a realistic mock dataset:
```bash
python generate_mock_data.py

```

This creates `data/train.csv`, `data/validation.csv` and `data/test.csv`. The validation set contains new users drawn from the same scenario (same hotspots and cell towers) and does not overlap with train/test; it is used only to select hyperparameters (KDE bandwidth, KNN K), never the test set.

---

## Methodology

The augmentation pipeline consists of two independent modules:

### 1. Spatial Augmentation

**Goal:** Learn the spatial distribution of users and generate new valid locations.

* **KDE:** Kernel Density Estimation (Gaussian Kernel), bandwidth selected by validation log-likelihood
* **GMM:** Gaussian Mixture Models, number of components selected by BIC
* **GAN:** Generative Adversarial Network
* **NF:** Normalizing Flows (RealNVP architecture)

### 2. Radio Augmentation

**Goal:** Learn the radio fingerprint of users and generate new valid radio fingerprints.

* **KNN:** K-Nearest Neighbors Regressor (inverse-distance weighted, **K=10**). K is selected by `knn_k_ablation.py`, which reports, for K = 1, 2, 3, 5, 10, 20, the radio MAE and the downstream median positioning error on the **validation** set (wKNN trained on real + KDE/KNN synthetic data), and keeps the K with the lowest validation positioning error. On the mock data the validation errors are 52.1 m (K=1), 49.5 (K=2), 48.8 (K=3), 48.2 (K=5), **47.3 (K=10)** and 48.1 m (K=20).
* **RF:** Random Forest Regressor
* **GPR-RQ:** Gaussian Process (Amplitude × Rational Quadratic + White noise kernel)
* **GPR-SE:** Gaussian Process (Amplitude × Squared Exponential + White noise kernel)
* **MLP:** Multi-Layer Perceptron
* **cGAN:** Conditional GAN trained with the least-squares (LSGAN) objective
* **NF:** Normalizing Flow
* **ReFlow:** Rectified Flow (Diffusion-based model)

GPR kernel hyperparameters (amplitude, length scale, noise level) are learned by maximizing the marginal likelihood on a 2000-point subset, then the GPR is refit on all training points. The fitted GPR models are ~500 MB each, so they are not stored in the repository: re-create them with `python train/radio/gpr_rq.py` and `python train/radio/gpr_se.py`.

The stochastic models (cGAN, NF, ReFlow) are evaluated both on a single draw (what the augmentation uses) and on the mean of 100 draws (their radio-map estimate, comparable to the deterministic regressors).

**_Implementation Note:_** For the mock data provided here, we use a single "one-shot" training model. However, when dealing with real MDT data, models should be trained per-PCI (as done in the referenced paper) to accurately capture cell-specific signal characteristics.

### 3. Positioning

The final utility is assessed by training a **Weighted KNN (wKNN, K=3, inverse Manhattan distance in RSRP space)** positioning model on the augmented data (real training data + synthetic data) and testing it on the real test data. Models trained on the synthetic data only are also reported (`wknn_<spatial>_<radio>_synth`).


### 4. Generating Augmented Data

To generate a specific synthetic dataset (e.g. KDE + KNN), or all 32 combinations:

```
python generate/full_pipeline.py kde knn
python generate/full_pipeline.py all

```

*Output is saved to `augm_data/`.*

### 5. Training Positioning on Augmented Data

To assess if the augmented data improves positioning:

1. Generate data (Step 4).
2. Train wKNN on the new CSV (`all` trains the baseline and every available combination, real + synthetic and synthetic-only):
```
python positioning/train.py all
```

3. Test on test set (median and mean positioning error, with a paired significance test against the original data):
```
python positioning/test.py
```

---
## Performance Evaluation
Synthetic data was generated with the `generate_mock_data.py` script and the positioning performance was tested across all model combinations (2000 synthetic samples per combination, fixed seeds). The tables report the **median** positioning error (Euclidean distance between estimated and true position) on the real test set (2000 points). `*` marks a significant difference from the original data (paired Wilcoxon signed-rank test, p < 0.05).

**wKNN trained on real + synthetic data (augmentation):**

| Spatial \ Radio | KNN | MLP | CGAN | GPR_RQ | GPR_SE | NF | REC_FLOW | RF |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Original data** | 49.697m | 49.697m | 49.697m | 49.697m | 49.697m | 49.697m | 49.697m | 49.697m |
| **KDE** | 46.250m* | 48.901m | 50.460m* | 48.536m* | 48.539m* | 50.326m | 49.595m | 46.339m* |
| **NF** | 46.805m* | 49.156m | 50.465m* | 48.565m | 48.646m | 49.708m | 49.575m | 47.588m* |
| **GAN** | 51.057m* | 49.366m* | 51.254m* | 49.176m* | 49.051m* | 49.830m | 50.576m | 51.592m* |
| **GMM** | 46.033m* | 48.328m* | 50.536m* | 48.525m* | 48.445m* | 49.401m* | 49.659m | 46.340m* |

**wKNN trained on synthetic data only:**

| Spatial \ Radio | KNN | MLP | CGAN | GPR_RQ | GPR_SE | NF | REC_FLOW | RF |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Original data** | 49.697m | 49.697m | 49.697m | 49.697m | 49.697m | 49.697m | 49.697m | 49.697m |
| **KDE** | 45.971m* | 53.532m* | 64.143m* | 51.563m | 51.674m | 54.812m* | 51.506m* | 45.996m* |
| **NF** | 46.013m* | 53.658m* | 63.750m* | 51.652m | 51.589m | 52.977m* | 51.181m* | 46.905m* |
| **GAN** | 63.348m* | 57.725m* | 80.693m* | 55.721m* | 55.760m* | 75.874m* | 75.235m* | 66.107m* |
| **GMM** | 45.603m* | 52.410m* | 65.415m* | 50.584m | 50.996m | 52.081m* | 51.803m* | 45.803m* |

With augmentation, KDE + KNN reduces the median error from 49.70 m to 46.25 m (mean: 54.96 m → 51.13 m; paired Wilcoxon p < 0.001). The other best combinations (GMM + KNN 46.03 m, KDE + RF 46.34 m, GMM + RF 46.34 m) are not significantly different from KDE + KNN (paired Wilcoxon p ≥ 0.05; 95% bootstrap CIs on the median difference include 0). `python positioning/test.py` also prints the mean errors.

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
