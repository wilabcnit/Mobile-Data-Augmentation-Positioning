### **README.md**

# MDT Augmentation for Improved Positioning in Mobile Networks

This repository contains the code for the research paper: **"Improving Outdoor Multi-cell Fingerprinting-based Positioning via Mobile Data Augmentation."**

---

## Codebase Structure

```
mdt_augmentation/
├── data/                       # MDT datasets
│   ├── train.csv               
│   └── test.csv                
├── augm_data/                  # Output directory for synthetic MDT datasets
├── models/                     # Saved models
│   ├── spatial/                # KDE, GMM, GAN, NF models
│   ├── radio/                  # KNN, RF, GPR, MLP, cGAN, NF, ReFlow models
│   └── positioning/            # wKNN positioning models
├── train/                      # Training scripts
│   ├── spatial/                
│   ├── radio/                  
├── test/                       # Evaluation scripts
│   ├── spatial/                
│   ├── radio/                  
├── generate/                   # Synthetic data generator using trained modules
│   ├── full_pipeline.py        
├── positioning/                # Positioning algorithm
│   ├── train.py                
│   └── test.py                 
├── utils/                      
│   ├── data_loader.py          
│   ├── metrics.py              
│   └── stats_compare.py        
└── generate_mock_data.py       


```

---

## Dependencies

The project requires Python 3.8+ and the following libraries:
* `torch` (PyTorch)[2.7.1+cpu]
* `numpy`[2.0.2]
* `pandas`[2.3.0]
* `scikit-learn`[1.6.1]
* `scipy`[1.15.3]
* `joblib`[1.5.1]


If you do not have real MDT traces yet, generate a realistic mock dataset:
```bash
python generate_mock_data.py

```

This creates `data/train.csv` and `data/test.csv`.

---

## Methodology

The augmentation pipeline consists of two independent modules:

### 1. Spatial Augmentation

**Goal:** Learn the spatial distribution of users and generate new valid locations.

* **KDE:** Kernel Density Estimation (Gaussian Kernel)
* **GMM:** Gaussian Mixture Models
* **GAN:** Generative Adversarial Network
* **NF:** Normalizing Flows (RealNVP architecture)

### 2. Radio Augmentation

**Goal:** Learn the radio fingerprint of users and generate new valid radio fingerprints.

* **KNN:** K-Nearest Neighbors Regressor
* **RF:** Random Forest Regressor
* **GPR-RQ:** Gaussian Process (Rational Quadratic Kernel)
* **GPR-SE:** Gaussian Process (Squared Exponential Kernel)
* **MLP:** Multi-Layer Perceptron
* **cGAN:** Conditional GAN
* **NF:** Conditional Normalizing Flow
* **ReFlow:** Rectified Flow (Diffusion-based model)

### 3. Positioning

The final utility is assessed by training a **Weighted KNN (wKNN)** positioning model on the augmented data and testing it on real real data.


### 4. Generating Augmented Data

To generate a specific synthetic dataset:

```
python generate/full_pipeline.py

```

*Output is saved to `augm_data/`.*

### 5. Training Positioning on Augmented Data

To assess if the augmented data improves positioning:

1. Generate data (Step 4).
2. Train wKNN on the new CSV:
```
python positioning/train.py
```

3. Test on test set:
```
python positioning/test.py
```

---

## Citation

If you use this code for your research, please cite:

```
TODO: We need to insert BibTex entry here.
```
