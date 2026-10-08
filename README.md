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

All results are on the real **test set** (2,000 measurements). `*` = significantly different from the original data (paired bootstrap 95 % confidence interval of the median difference excludes 0).

### Positioning: median error (m)

wKNN trained on real + synthetic data. The KNN radio model is reported for K = 1 (proposed, `python positioning/test.py`) and K = 2..5 (`python positioning/knn_k_ablation.py`).

| Spatial \ Radio | KNN K=1 | KNN K=2 | KNN K=3 | KNN K=4 | KNN K=5 | MLP | CGAN | GPR_RQ | GPR_SE | NF | REC_FLOW | RF |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Original data** | 5.153m | 5.153m | 5.153m | 5.153m | 5.153m | 5.153m | 5.153m | 5.153m | 5.153m | 5.153m | 5.153m | 5.153m |
| **KDE** | **4.921m*** | 5.063m | 5.285m | 5.359m* | 5.422m* | 5.171m | 5.201m | 5.076m | 5.146m | 5.153m | 5.518m* | 5.183m |
| **NF** | **4.984m** | 5.175m | 5.242m | 5.109m | 5.202m | 5.136m | 5.218m | 5.008m* | 5.187m | 5.136m | 5.535m* | 5.122m |
| **GAN** | **4.920m*** | 5.133m | 5.291m | 5.352m* | 5.332m* | 5.131m | 5.157m | 5.064m | 5.147m | 5.163m | 5.503m* | 5.178m |
| **GMM** | **4.898m*** | 5.142m | 5.253m | 5.331m | 5.365m | 5.129m | 5.237m | 5.099m | 5.141m | 5.145m | 5.481m* | 5.138m |

KDE + 1NN reduces the median error from 5.153 m to 4.921 m. KNN with K = 1 is the best radio model for every spatial model; K > 1 averages RSRP values with the -160 not-served value, while K = 1 copies the nearest real fingerprint with its exact set of served / not-served cells.

K is selected on the **validation set** (median positioning error, `python positioning/knn_k_ablation.py`): K = 1 is the best for every spatial model.

| Spatial \ Radio KNN | K=1 | K=2 | K=3 | K=4 | K=5 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Original data** | 5.510m | 5.510m | 5.510m | 5.510m | 5.510m |
| **KDE** | **5.258m** | 5.464m | 5.543m | 5.587m | 5.584m |
| **NF** | **5.262m** | 5.349m | 5.545m | 5.721m | 5.762m |
| **GAN** | **5.243m** | 5.363m | 5.253m | 5.452m | 5.564m |
| **GMM** | **5.067m** | 5.373m | 5.696m | 5.694m | 5.751m |

### Radio models: mean absolute error (dB)

RSRP predicted at the test positions vs real RSRP (`python test/radio/<model>.py`). For the stochastic models (cGAN, NF, ReFlow): single draw and mean of 100 draws.

| Radio model | MAE (dB) | MAE, mean of 100 draws (dB) |
| :--- | :--- | :--- |
| **KNN K=1** | **2.527** | – |
| KNN K=2 | 2.763 | – |
| KNN K=3 | 2.965 | – |
| KNN K=4 | 3.158 | – |
| KNN K=5 | 3.324 | – |
| GPR-RQ | 3.165 | – |
| RF | 3.759 | – |
| GPR-SE | 5.094 | – |
| MLP | 8.979 | – |
| cGAN | 12.781 | 11.298 |
| ReFlow | 12.764 | 10.962 |
| NF | 24.352 | 13.581 |

### Spatial models: Kolmogorov-Smirnov tests

Generated vs real test positions (`python test/spatial/<model>.py`). A mean absolute error is not defined for the spatial models (a generated position has no paired true position), so their distributions are compared with the two-sample KS test: 1-D on latitude and longitude, and 2-D Fasano-Franceschini (p-value from 1,000 permutations). Lower statistic / higher p = closer to the real distribution.

| Spatial model | KS latitude (p) | KS longitude (p) | 2-D KS (p) |
| :--- | :--- | :--- | :--- |
| **KDE** | 0.032 (0.258) | 0.035 (0.173) | 0.049 (0.126) |
| NF | 0.026 (0.534) | 0.044 (0.042) | 0.047 (0.151) |
| GMM | 0.074 (<0.001) | 0.061 (0.001) | 0.092 (0.001) |
| GAN | 0.183 (<0.001) | 0.101 (<0.001) | 0.207 (0.001) |

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
