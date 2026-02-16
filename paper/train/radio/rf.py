import sys
import os
import joblib
from sklearn.ensemble import RandomForestRegressor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.data_loader import load_mdt_data

def train_rf_radio():
    print("Training Radio Random Forest...")
    
    coords, rsrps = load_mdt_data('data/train.csv', mode='radio')
    
    # Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)
    rf.fit(coords.numpy(), rsrps.numpy())
    
    os.makedirs('models/radio', exist_ok=True)
    joblib.dump(rf, 'models/radio/rf.pkl')
    print("Model saved to models/radio/rf.pkl")

if __name__ == "__main__":
    train_rf_radio()