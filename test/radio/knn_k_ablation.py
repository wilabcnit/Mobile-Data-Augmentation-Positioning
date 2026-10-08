import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from train.radio.knn_k_ablation import run_k_ablation

if __name__ == "__main__":
    run_k_ablation()
