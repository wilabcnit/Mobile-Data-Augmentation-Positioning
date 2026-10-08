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

* **KNN:** K-Nearest Neighbors Regressor (inverse-distance weighted, **K=10**). K is selected by `knn_k_ablation.py`, which reports, for K = 1, 2, 3, 5, 10, 20, the radio MAE and the downstream median positioning error on the **validation** set (wKNN trained on real + KDE/KNN synthetic data), and keeps the K with the lowest validation positioning error. On the mock data the validation errors are 51.7 m (K=1), 48.0 (K=2), 49.0 (K=3), 47.0 (K=5), **46.4 (K=10)** and 47.3 m (K=20).
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

The final utility is assessed by training a **Weighted KNN (wKNN)** positioning model on the augmented data (real training data + synthetic data) and testing it on the real test data. Models trained on the synthetic data only are also reported (`wknn_<spatial>_<radio>_synth`).


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
Synthetic data was generated with the `generate_mock_data.py` script and the positioning performance was tested across all model combinations (2000 synthetic samples per combination, fixed seeds). The tables report the **median** positioning error on the real test set (2000 points). `*` marks a significant difference from the original data (paired Wilcoxon signed-rank test, p < 0.05).

**wKNN trained on real + synthetic data (augmentation):**

| Spatial \ Radio | KNN | MLP | CGAN | GPR_RQ | GPR_SE | NF | REC_FLOW | RF |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Original data** | 49.991m | 49.991m | 49.991m | 49.991m | 49.991m | 49.991m | 49.991m | 49.991m |
| **KDE** | 45.825m* | 48.030m* | 50.787m* | 45.838m* | 45.838m* | 49.685m | 49.136m | 46.724m* |
| **NF** | 45.902m* | 47.226m* | 50.884m* | 46.011m* | 45.864m* | 49.989m | 49.628m | 44.738m* |
| **GAN** | 50.873m* | 48.129m* | 51.334m* | 48.258m* | 48.257m* | 50.324m* | 51.266m | 50.741m* |
| **GMM** | 44.807m* | 47.606m* | 50.727m* | 46.009m* | 46.217m* | 49.048m* | 48.818m | 45.029m* |

**wKNN trained on synthetic data only:**

| Spatial \ Radio | KNN | MLP | CGAN | GPR_RQ | GPR_SE | NF | REC_FLOW | RF |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Original data** | 49.991m | 49.991m | 49.991m | 49.991m | 49.991m | 49.991m | 49.991m | 49.991m |
| **KDE** | 45.920m* | 50.490m | 62.604m* | 46.226m* | 46.113m* | 53.801m* | 51.622m | 46.063m* |
| **NF** | 45.940m* | 50.017m | 62.204m* | 46.072m* | 46.214m* | 53.429m* | 52.731m* | 45.016m* |
| **GAN** | 62.932m* | 55.152m* | 80.539m* | 52.520m* | 52.726m* | 75.591m* | 75.605m* | 67.278m* |
| **GMM** | 44.528m* | 49.478m* | 64.690m* | 46.008m* | 46.146m* | 51.234m* | 50.505m | 44.578m* |

With augmentation, KDE + KNN reduces the median error from 49.99 m to 45.83 m (mean: 56.06 m → 50.83 m). The best combinations (NF + RF 44.74 m, GMM + KNN 44.81 m, GMM + RF 45.03 m, KDE + KNN 45.83 m) are not significantly different from KDE + KNN (paired Wilcoxon p ≥ 0.05; 95% bootstrap CIs on the median difference include 0). `python positioning/test.py` also prints the mean errors.

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
