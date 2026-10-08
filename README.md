# MDT Augmentation for Improved Positioning in Mobile Networks

This repository contains the code for the research paper: **"Improving Outdoor Multi-cell Fingerprinting-based Positioning via Mobile Data Augmentation."** https://arxiv.org/abs/2509.19405

The real MDT dataset of the paper cannot be shared; the pipeline runs here on a mock MDT dataset (`generate_mock_data.py`).

---

## Codebase Structure

```
mdt_augmentation/
├── data/                       # MDT datasets (70/10/20 split)
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
│   └── radio/                  # knn.py trains K = 1 (pipeline) and K = 2..5 (ablation)
├── test/                       # Evaluation scripts
│   ├── spatial/                # 1-D and 2-D (Fasano-Franceschini) KS tests
│   └── radio/                  # radio MAE (knn.py: K = 1..5)
├── generate/                   # Synthetic data generator using trained modules
│   └── full_pipeline.py
├── positioning/                # Positioning algorithm
│   ├── train.py
│   ├── test.py
│   └── knn_k_ablation.py       # radio KNN K = 1..5: validation / test positioning
├── benchmark/
│   └── generation_time.py      # synthetic data generation time
├── utils/
│   ├── data_loader.py
│   ├── geo.py                  # haversine distance, lat/lon -> metres
│   ├── gpr.py
│   ├── ks2d_ff.py              # 2-D Fasano-Franceschini KS test
│   ├── metrics.py
│   └── stats_compare.py
├── generate_mock_data.py
└── run_pipeline.sh             # full workflow, end to end
```

---

## Dependencies

Python 3.10+ and the versions used to train the stored models (`pip install -r requirements.txt`):
* `torch` [2.14.1]
* `numpy` [2.5.3]
* `pandas` [3.0.5]
* `scikit-learn` [1.9.1]
* `scipy` [1.18.1]
* `joblib` [1.6.0]
* `tabulate`

Generate the mock MDT dataset:
```bash
python generate_mock_data.py
```

This creates `data/train.csv` (70 %), `data/validation.csv` (10 %) and `data/test.csv` (20 %): 10,000 MDT measurements in a city-center scenario (WGS84 coordinates around Bologna, ~2 km²), 35 cells, 3GPP TR 38.901 UMa NLOS pathloss, 7.8 dB spatially correlated shadow fading (50 m decorrelation distance). Each measurement reports at most the 8 strongest detected cells; the other cells are not served (RSRP = -160).

To run the whole workflow (data → models → synthetic data → positioning):
```bash
./run_pipeline.sh
```

---

## Methodology

The augmentation pipeline consists of two independent modules:

### 1. Spatial Augmentation

**Goal:** Learn the spatial distribution of users and generate new valid locations.

* **KDE:** Kernel Density Estimation (Gaussian Kernel), bandwidth selected on the validation set
* **GMM:** Gaussian Mixture Models, number of components selected by BIC
* **GAN:** Generative Adversarial Network
* **NF:** Normalizing Flows (RealNVP architecture)

### 2. Radio Augmentation

**Goal:** Learn the radio fingerprint of users and generate new valid radio fingerprints.

* **KNN:** K-Nearest Neighbors Regressor **(K=1)**: the synthetic fingerprint is the fingerprint of the nearest real measurement, including its not-served cells (-160)
* **RF:** Random Forest Regressor
* **GPR-RQ:** Gaussian Process (Rational Quadratic Kernel + white noise)
* **GPR-SE:** Gaussian Process (Squared Exponential Kernel + white noise)
* **MLP:** Multi-Layer Perceptron
* **cGAN:** Conditional GAN (least-squares objective)
* **NF:** Normalizing Flow
* **ReFlow:** Rectified Flow (Diffusion-based model)

The GPR models are large, so they are not stored: `python train/radio/gpr_rq.py` and `python train/radio/gpr_se.py` re-create them.

**_Implementation Note:_** For the mock data provided here, we use a single "one-shot" training model. However, when dealing with real MDT data, models should be trained per-PCI (as done in the referenced paper) to accurately capture cell-specific signal characteristics.

### 3. Positioning

The final utility is assessed by training a **Weighted KNN (wKNN, K=3, inverse Manhattan distance between RSRP fingerprints)** positioning model on the augmented data (real training data + synthetic data) and testing it on the real test data. The positioning error is the haversine distance (m) between the estimated and true position.

### 4. Generating Augmented Data

Each synthetic dataset has 2 × the real training size (14,000 samples). To generate a specific synthetic dataset, or all 32 combinations:

```
python generate/full_pipeline.py kde knn
python generate/full_pipeline.py all
```

*Output is saved to `augm_data/`.*

### 5. Training Positioning on Augmented Data

1. Generate data (Step 4).
2. Train wKNN (baseline + all combinations):
```
python positioning/train.py all
```

3. Test on test set:
```
python positioning/test.py
```

---
## Performance Evaluation

Median positioning error on the real test set (2,000 measurements), wKNN trained on real + synthetic data. `*` = significantly different from the original data (paired bootstrap 95 % confidence interval of the median difference excludes 0).

| Spatial \ Radio | KNN | MLP | CGAN | GPR_RQ | GPR_SE | NF | REC_FLOW | RF |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Original data** | 5.153m | 5.153m | 5.153m | 5.153m | 5.153m | 5.153m | 5.153m | 5.153m |
| **KDE** | 4.921m* | 5.171m | 5.201m | 5.076m | 5.146m | 5.153m | 5.518m* | 5.183m |
| **NF** | 4.984m | 5.136m | 5.218m | 5.008m* | 5.187m | 5.136m | 5.535m* | 5.122m |
| **GAN** | 4.920m* | 5.131m | 5.157m | 5.064m | 5.147m | 5.163m | 5.503m* | 5.178m |
| **GMM** | 4.898m* | 5.129m | 5.237m | 5.099m | 5.141m | 5.145m | 5.481m* | 5.138m |

KDE + KNN reduces the median error from 5.153 m to 4.921 m, and KNN is the best radio model for every spatial model.

### Radio KNN: K ablation

`python train/radio/knn.py` trains the radio KNN for K = 1..5 and `python test/radio/knn.py` reports their radio MAE. `python positioning/knn_k_ablation.py` reports the median positioning error (KDE positions, real + synthetic data) on the validation set, where K is selected, and on the test set.

| K | Validation (m) | Test (m) |
| :--- | :--- | :--- |
| **1** | **5.26** | **4.92** |
| 2 | 5.46 | 5.06 |
| 3 | 5.54 | 5.29 |
| 4 | 5.59 | 5.36 |
| 5 | 5.58 | 5.42 |

K = 1 copies the nearest real fingerprint with its exact set of served / not-served cells, whereas K > 1 averages RSRP values with the -160 not-served value.

### Synthetic data generation time

`python benchmark/generation_time.py`: KDE sampling + radio model inference for A × the real training size (A = augmentation rate), median of 5 runs (Intel Xeon 2.8 GHz, 4 CPUs).

| Model | A = 10 | A = 20 | A = 30 |
| :--- | :--- | :--- | :--- |
| KDE-1NN (Proposed) | 0.120 s | 0.294 s | 0.346 s |
| KDE-RF | 0.125 s | 0.357 s | 0.476 s |
| KDE-GPR (RQ) | 22.424 s | 46.162 s | 68.371 s |
| KDE-MLP | 0.109 s | 0.264 s | 0.398 s |
| KDE-CGAN | 0.091 s | 0.222 s | 0.307 s |
| KDE-ReFlow | 3.093 s | 5.921 s | 8.319 s |
| KDE-NF | 0.146 s | 0.625 s | 0.862 s |

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
