#!/usr/bin/env bash
# End-to-end reproduction: mock data -> spatial/radio models -> synthetic data ->
# positioning -> evaluation. Run from the repository root.
set -euo pipefail

echo "== 1. Mock MDT data (70/10/20 train/validation/test) =="
python generate_mock_data.py

echo "== 2. Spatial models =="
for m in kde gmm gan nf; do python train/spatial/$m.py; done

echo "== 3. Radio models =="
for m in knn rf mlp cgan nf rec_flow gpr_se gpr_rq; do python train/radio/$m.py; done

echo "== 4. Model evaluation (test set) =="
for m in kde gmm gan nf; do python test/spatial/$m.py; done
for m in knn rf mlp cgan nf rec_flow gpr_se gpr_rq; do python test/radio/$m.py; done

echo "== 5. Synthetic data (all 32 spatial x radio combinations) =="
python generate/full_pipeline.py all

echo "== 6. Positioning =="
python positioning/train.py all
python positioning/test.py
python positioning/knn_k_ablation.py          # radio KNN K = 1..5: validation + test positioning

echo "== 7. Synthetic data generation time (A = 10, 20, 30) =="
python benchmark_complexity.py
