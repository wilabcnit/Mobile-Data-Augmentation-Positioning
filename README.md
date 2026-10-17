# MDT Augmentation for Improved Positioning in Mobile Networks

This repository contains the code for the paper **"Improving Outdoor Multi-cell Fingerprinting-based Positioning via Mobile Data Augmentation"** ([arXiv:2509.19405](https://arxiv.org/abs/2509.19405)).

The real minimization of drive-tests (MDT) dataset used in the paper cannot be disclosed due to user privacy concerns. This repository therefore includes a mock MDT dataset designed to reproduce the main characteristics of real MDT data, so that the pipeline can be run end to end and applied to any MDT dataset with the same structure.

---

## Codebase Structure

```
mdt_augmentation/
├── data/                       # MDT datasets (70/10/20 train/validation/test split)
├── augm_data/                  # Augmented (synthetic) MDT datasets
├── models/                     # Trained models
│   ├── spatial/                # KDE, GMM, GAN, NF
│   ├── radio/                  # KNN, RF, GPR, MLP, cGAN, NF, ReFlow
│   └── positioning/            # wKNN positioning models
├── train/                      # Training scripts (spatial/, radio/)
├── test/                       # Evaluation scripts (spatial/, radio/)
├── generate/
│   └── full_pipeline.py        # Synthetic data generation
├── positioning/
│   ├── train.py
│   ├── test.py
│   └── knn_k_ablation.py       # Radio KNN with K = 1..5
├── utils/                      
├── generate_mock_data.py       # Mock MDT dataset
└── run_pipeline.sh             # Full workflow
```

---

## Getting Started

Python 3.12+ is required. Install the dependencies (versions used for the stored models) for venv.:

```bash
pip install -r requirements.txt
```

Generate the MDT dataset:

```bash
python generate_mock_data.py
```

This creates a mock MDT dataset that mimics a real dense urban deployment: 10,000 measurements over 35 cells, with realistic propagation (3GPP TR 38.901) and MDT reporting (8 reported cells per measurement). It is split into train (70 %), validation (10 %) and test (20 %).

Run the full workflow (data → models → synthetic data → positioning):

```bash
./run_pipeline.sh
```

---

## Methodology

The augmentation pipeline consists of two independent modules.

### 1. Spatial Augmentation

Learns the spatial distribution of the users and generates new locations.

* **KDE:** Kernel Density Estimation (Gaussian kernel)
* **GMM:** Gaussian Mixture Model
* **GAN:** Generative Adversarial Network
* **NF:** Normalizing Flow

### 2. Radio Augmentation

Learns the radio fingerprint as a function of location and generates the fingerprints of the new locations.

* **KNN:** K-Nearest Neighbors *(K = 1 in the proposed model)*
* **RF:** Random Forest
* **GPR-RQ:** Gaussian Process (Rational Quadratic kernel)
* **GPR-SE:** Gaussian Process (Squared Exponential kernel)
* **MLP:** Multi-Layer Perceptron
* **cGAN:** Conditional GAN
* **NF:** Normalizing Flow
* **ReFlow:** Rectified Flow

> **Note:** We provide a single multi-output model. Alternatively, with real MDT data, the radio models can be trained per PCI, to capture cell-specific signal characteristics.

### 3. Positioning

A weighted KNN (wKNN, _K = 3_) positioning model is trained on the real training data together with the synthetic data, and evaluated on the real test data.

---

## Usage

**Generate synthetic data.** Each synthetic dataset has twice the size **(X2)** of the real training set.

```bash
python generate/full_pipeline.py kde knn    # one spatial / radio combination
python generate/full_pipeline.py all        # all combinations
```

Output is saved to `augm_data/`.

**Train and test the positioning models:**

```bash
python positioning/train.py all
python positioning/test.py
python positioning/knn_k_ablation.py
```

---

## Results

All results are on the original (non-augmented) test set. The KDE bandwidth and the radio KNN K are selected on the validation set.

### Radio models: mean absolute error (dB)

| | **KNN K=1** | KNN K=2 | KNN K=3 | KNN K=4 | KNN K=5 | GPR-RQ | RF | GPR-SE | MLP | cGAN | ReFlow | NF |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MAE (dB)** | **2.53** | 2.76 | 2.97 | 3.16 | 3.32 | 3.17 | 3.76 | 5.09 | 8.98 | 12.78 | 12.76 | 24.35 |

### Spatial models: 2-D Kolmogorov–Smirnov test

Generated vs. real test locations, 2-D Fasano–Franceschini KS test, mean over 10 independent draws. Lower statistic and higher p-value indicate better agreement.

| | **KDE** | NF | GMM | GAN |
| :--- | :--- | :--- | :--- | :--- |
| **2-D KS (p)** | **0.04 (0.42)** | 0.05 (0.07) | 0.09 (<0.01) | 0.21 (<0.01) |

### Positioning: median error (m)

| Spatial \ Radio | **KNN K=1** | KNN K=2 | KNN K=3 | KNN K=4 | KNN K=5 | MLP | cGAN | GPR-RQ | GPR-SE | NF | ReFlow | RF |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Original data** | 5.15 | 5.15 | 5.15 | 5.15 | 5.15 | 5.15 | 5.15 | 5.15 | 5.15 | 5.15 | 5.15 | 5.15 |
| **KDE** | **4.92\*** | 5.06 | 5.29 | 5.36\* | 5.42\* | 5.13 | 5.30\* | 5.07 | 5.11 | 5.16 | 5.98\* | 5.18 |
| **NF** | 4.98 | 5.18 | 5.24 | 5.11 | 5.20 | 5.15 | 5.28\* | 4.98\* | 5.10 | 5.15 | 5.78\* | 5.12 |
| **GAN** | 4.92\* | 5.13 | 5.29 | 5.35\* | 5.33\* | 5.14 | 5.28\* | 5.04\* | 5.19 | 5.17 | 5.71\* | 5.18 |
| **GMM** | 5.03 | 5.08 | 5.36\* | 5.30 | 5.41\* | 5.14 | 5.27\* | 5.07\* | 5.15 | 5.14 | 5.78\* | 5.16 |

*`*` = statistically differs from the original data at 95 % confidence(utils/stats_compare.py).*

> **Note:** All results above are obtained on the fully synthetic mock MDT dataset, not on the real MDT data of the paper. They illustrate the pipeline and are not intended to reproduce the values reported in the paper.

---

## Citation

If you use this code in your research, please cite:

```
@article{chahoud2025improving,
  title={Improving Outdoor Multi-cell Fingerprinting-based Positioning via Mobile Data Augmentation},
  author={Chahoud, Tony and Amorosa, Lorenzo Mario and Marini, Riccardo and De Nardis, Luca},
  journal={arXiv preprint arXiv:2509.19405},
  year={2025}
}
```
