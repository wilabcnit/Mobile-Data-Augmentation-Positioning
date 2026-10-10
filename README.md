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

This creates `data/train.csv` (70 %), `data/validation.csv` (10 %) and `data/test.csv` (20 %): 10,000 MDT measurements in a city-center scenario (WGS84 coordinates around Bologna, ~2 km²), 35 cells, 3GPP TR 38.901 UMa NLOS pathloss, 7.8 dB spatially correlated shadow fading. Each measurement reports at most the 8 strongest detected cells; the other cells are not served (RSRP = -160).

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
* **GMM:** Gaussian Mixture Models
* **GAN:** Generative Adversarial Network
* **NF:** Normalizing Flows

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

Each synthetic dataset has 2 × the real training size (14,000 samples). Generated RSRP values below -160 dBm (not served) are set to -160. To generate a specific synthetic dataset, or all of them (the 32 spatial × radio combinations and, for the K ablation, the radio KNN with K = 2..5: `synthetic_<spatial>_knn_k<K>.csv`):

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
| **MAE (dB)** | **2.527** | 2.763 | 2.965 | 3.158 | 3.324 | 3.165 | 3.759 | 5.094 | 8.979 | 12.781 | 12.764 | 24.352 |

### Spatial models: 2-D Kolmogorov-Smirnov test

Generated vs. real test positions (`python test/spatial/<model>.py`). Since MAE is undefined for unpaired positions, distributions are compared using the 2-D Fasano–Franceschini KS test (1,000 permutations). Results are Means over 10 independent draws; lower statistics and higher p-values indicate better agreement.


| | **KDE** | NF | GMM | GAN |
| :--- | :--- | :--- | :--- | :--- |
| **2-D KS (p)** | **0.040 (0.422)** | 0.054 (0.068) | 0.091 (0.001) | 0.208 (0.001) |

### Positioning: median error (m)

The KNN radio model is reported for K = 1 (proposed, `python positioning/test.py`) and K = 2..5 (`python positioning/knn_k_ablation.py`).
The positioning performance was then tested across all model combinations. The results below represent the Median Positioning Error achieved during this test run:

| Spatial \ Radio | KNN K=1 | KNN K=2 | KNN K=3 | KNN K=4 | KNN K=5 | MLP | CGAN | GPR_RQ | GPR_SE | NF | REC_FLOW | RF |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Original data** | 5.153m | 5.153m | 5.153m | 5.153m | 5.153m | 5.153m | 5.153m | 5.153m | 5.153m | 5.153m | 5.153m | 5.153m |
| **KDE** | **4.921m\*** | 5.063m | 5.285m | 5.359m\* | 5.422m\* | 5.129m | 5.296m\* | 5.069m | 5.107m | 5.163m | 5.982m\* | 5.183m |
| **NF** | 4.984m | 5.175m | 5.242m | 5.109m | 5.202m | 5.147m | 5.278m\* | 4.975m\* | 5.104m | 5.146m | 5.780m\* | 5.122m |
| **GAN** | 4.920m\* | 5.133m | 5.291m | 5.352m\* | 5.332m\* | 5.141m | 5.278m\* | 5.038m\* | 5.191m | 5.173m | 5.713m\* | 5.178m |
| **GMM** | 5.031m | 5.080m | 5.356m\* | 5.299m | 5.414m\* | 5.136m | 5.275m\* | 5.067m\* | 5.153m | 5.141m | 5.782m\* | 5.160m |

KDE + 1NN reduces the median error from 5.153 m to 4.921 m. With the 1NN radio model, the four spatial models are not significantly different from each other (paired bootstrap 95 % CI of the median difference, KDE vs GMM: -0.110 m [-0.356, 0.078]; vs NF: -0.063 m [-0.290, 0.094]; vs GAN: 0.001 m [-0.284, 0.213]); KDE is preferred for its closer spatial fit (2-D KS). KNN with K = 1 is the best radio model for every spatial model; K > 1 averages RSRP values with the -160 not-served value, while K = 1 copies the nearest real fingerprint with its exact set of served / not-served cells.

K is selected on the **validation set** (median positioning error, `python positioning/knn_k_ablation.py`): K = 1 is the best for every spatial model.


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
