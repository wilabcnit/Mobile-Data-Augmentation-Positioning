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
├── augm_data/                  # Synthetic MDT datasets (32 combinations + radio KNN K = 2..5)
├── models/                     # Saved models
│   ├── spatial/                # KDE, GMM, GAN, NF models
│   ├── radio/                  # KNN, RF, GPR, MLP, cGAN, NF, ReFlow models
│   └── positioning/            # wKNN positioning models
├── train/                      # Training scripts
│   ├── spatial/
│   └── radio/                  # knn.py trains K = 1 (pipeline) and K = 2..5 (ablation)
├── test/                       # Evaluation scripts
│   ├── spatial/                # 2-D (Fasano-Franceschini) KS test
│   └── radio/                  # radio MAE (knn.py: K = 1..5)
├── generate/                   # Synthetic data generator using trained modules
│   └── full_pipeline.py
├── positioning/                # Positioning algorithm
│   ├── train.py
│   ├── test.py
│   └── knn_k_ablation.py       # radio KNN K = 1..5: validation / test positioning
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

This creates `data/train.csv` (70 %), `data/validation.csv` (10 %) and `data/test.csv` (20 %): 10,000 MDT measurements in a city-center scenario (WGS84 coordinates around Bologna, ~2 km²), 35 cells, 3GPP TR 38.901 UMa NLOS pathloss, 7.8 dB spatially correlated shadow fading. Each measurement reports at most the 10 strongest detected cells; the other cells are not served (RSRP = -160).

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

The GPR models are stored for mean prediction only (without the N × N Cholesky factor, which is needed only for the predictive standard deviation).

**_Implementation Note:_** For the mock data provided here, we use a single "one-shot" training model. However, when dealing with real MDT data, models should be trained per-PCI (as done in the referenced paper) to accurately capture cell-specific signal characteristics.

### 3. Positioning

The final utility is assessed by training a **Weighted KNN (wKNN, K=3, inverse Manhattan distance between RSRP fingerprints)** positioning model on the augmented data (real training data + synthetic data) and testing it on the real test data. The positioning error is the haversine distance (m) between the estimated and true position.

### 4. Generating Augmented Data

Each synthetic dataset has 5 × the real training size (35,000 samples). To generate a specific synthetic dataset, or all of them (the 32 spatial × radio combinations and, for the K ablation, the radio KNN with K = 2..5: `synthetic_<spatial>_knn_k<K>.csv`):

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

All results are on the real **test set** (2,000 measurements). `*` =  different from the original data (paired bootstrap 95 % confidence interval of the median difference excludes 0).

### Radio models: mean absolute error (dB)

RSRP predicted at the test positions vs real RSRP (`python test/radio/<model>.py`).

| | **KNN K=1** | KNN K=2 | KNN K=3 | KNN K=4 | KNN K=5 | GPR-RQ | RF | GPR-SE | MLP | cGAN | ReFlow | NF |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MAE (dB)** | **2.911** | 3.159 | 3.382 | 3.585 | 3.779 | 3.659 | 4.252 | 5.720 | 9.675 | 15.068 | 14.421 | 24.566 |

### Spatial models: 2-D Kolmogorov-Smirnov test

Generated vs. real test positions (`python test/spatial/<model>.py`). Since MAE is undefined for unpaired positions, distributions are compared using the 2-D Fasano–Franceschini KS test (1,000 permutations). Results are Means over 10 independent draws; lower statistics and higher p-values indicate better agreement.


| | **KDE** | NF | GMM | GAN |
| :--- | :--- | :--- | :--- | :--- |
| **2-D KS (p)** | **0.040 (0.422)** | 0.054 (0.068) | 0.090 (0.001) | 0.208 (0.001) |

### Positioning: median error (m)

The KNN radio model is reported for K = 1 (proposed, `python positioning/test.py`) and K = 2..5 (`python positioning/knn_k_ablation.py`).
The positioning performance was then tested across all model combinations. The results below represent the Median Positioning Error achieved during this test run:

| Spatial \ Radio | KNN K=1 | KNN K=2 | KNN K=3 | KNN K=4 | KNN K=5 | MLP | CGAN | GPR_RQ | GPR_SE | NF | REC_FLOW | RF |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Original data** | 4.965m | 4.965m | 4.965m | 4.965m | 4.965m | 4.965m | 4.965m | 4.965m | 4.965m | 4.965m | 4.965m | 4.965m |
| **KDE** | 4.867m | **4.570m\*** | 4.657m\* | 4.742m\* | 4.842m | 4.887m\* | 5.040m\* | 4.589m\* | 4.894m | 4.943m | 5.473m\* | 4.887m |
| **NF** | 4.915m | 4.625m\* | 4.783m | 4.796m | 4.848m | 4.898m | 5.007m | 4.576m\* | 4.968m | 4.951m | 5.435m\* | 4.914m |
| **GAN** | 4.911m | 4.858m | 4.885m | 4.883m | 4.896m | 4.903m | 5.019m | 4.742m\* | 4.898m | 4.951m | 5.435m\* | 4.880m |
| **GMM** | 4.930m | 4.591m\* | 4.674m\* | 4.665m\* | 4.704m\* | 4.898m\* | 5.023m\* | 4.588m\* | 4.887m | 4.934m | 5.267m\* | 4.903m |

KDE + 1NN reduces the median error from 4.965 m to 4.867 m, but the difference is not significant. The lowest errors are obtained with the radio KNN with K = 2 (KDE: 4.570 m, GMM: 4.591 m, NF: 4.625 m) and with GPR-RQ (4.576–4.742 m), both significantly better than the original data for most spatial models.

K selected on the **validation set** (median positioning error, `python positioning/knn_k_ablation.py`): K = 4 for KDE, K = 3 for NF and GMM, K = 1 for GAN.

Validation set, median positioning error (m), radio KNN:

| Spatial \ Radio KNN | K=1 | K=2 | K=3 | K=4 | K=5 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Original data** | 5.396m | 5.396m | 5.396m | 5.396m | 5.396m |
| **KDE** | 5.185m | 4.960m | 4.895m | **4.864m** | 5.017m |
| **NF** | 4.949m | 4.889m | **4.769m** | 4.853m | 4.973m |
| **GAN** | **5.109m** | 5.221m | 5.286m | 5.293m | 5.320m |
| **GMM** | 5.059m | 5.056m | **4.895m** | 4.981m | 5.036m |


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
