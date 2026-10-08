import sys
import os
import joblib
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from generate.full_pipeline import SPATIAL_MODELS, RADIO_MODELS

REAL_TRAIN_PATH = 'data/train.csv'

def train_positioning(train_data_path, model_name='wknn_baseline', include_real=True):
    """
    Trains a wKNN positioning model using the specified dataset.

    Args:
        train_data_path (str): Path to the CSV file (Real or Synthetic MDT).
        model_name (str): Name to save the model as.
        include_real (bool): If True and train_data_path is a synthetic dataset, the real
            training data is added to it (augmentation = real + synthetic). If False, the
            model is trained on the synthetic data only.
    """
    print(f"Training Positioning Model: {model_name}...")

    # 1. Load Data
    paths = [train_data_path]
    if include_real and os.path.abspath(train_data_path) != os.path.abspath(REAL_TRAIN_PATH):
        paths.insert(0, REAL_TRAIN_PATH)
    print(f"Using Data: {' + '.join(paths)}")

    for path in paths:
        if not os.path.exists(path):
            print(f"Error: Data file {path} not found.")
            return

    df = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)

    # Features: RSRP columns | Targets: Lat, Lon
    rsrp_cols = [c for c in df.columns if 'RSRP' in c]
    X = df[rsrp_cols].values
    y = df[['LATITUDE', 'LONGITUDE']].values

    # 2. Train wKNN (Weighted by inverse distance in signal space)
    wknn = KNeighborsRegressor(n_neighbors=3, weights='distance', metric='euclidean')
    wknn.fit(X, y)

    # 3. Save Model
    os.makedirs('models/positioning', exist_ok=True)
    save_path = f'models/positioning/{model_name}.pkl'
    joblib.dump(wknn, save_path)
    print(f"Positioning model saved to {save_path} ({len(df)} samples)")

if __name__ == "__main__":
    # Usage: python positioning/train.py [all]
    #   all : baseline + every available synthetic dataset, both real+synthetic
    #         (wknn_<spatial>_<radio>) and synthetic-only (wknn_<spatial>_<radio>_synth)
    if sys.argv[1:] == ['all']:
        train_positioning(REAL_TRAIN_PATH, 'wknn_baseline')
        for s in SPATIAL_MODELS:
            for r in RADIO_MODELS:
                path = f'augm_data/synthetic_{s}_{r}.csv'
                if os.path.exists(path):
                    train_positioning(path, f'wknn_{s}_{r}', include_real=True)
                    train_positioning(path, f'wknn_{s}_{r}_synth', include_real=False)
    else:
        #train_positioning('data/train.csv', 'wknn_baseline')
        train_positioning('augm_data/synthetic_kde_knn.csv', 'wknn_kde_knn')
