import numpy as np
from scipy import stats
import pandas as pd

def compare_models_paired(errors1, errors2, label1="Model 1", label2="Model 2", confidence=0.95,
                          n_boot=2000, seed=0, data="mock"):
    """
    Paired comparison of two models on the same test points: bootstrap CI of the median
    error difference (significant if it excludes 0), plus the Wilcoxon p-value.

    In case of this "mock" data of this repository: one run, errors of shape [n_test],
    whereas in case of "real" MDT data like in the paper: several generation runs, errors
    of shape [n_runs, n_test]; the median difference is averaged over the runs.
    """
    if data not in ("mock", "real"):
        raise ValueError('data must be "mock" or "real".')
    errors1 = np.atleast_2d(np.asarray(errors1, float))   # [n_runs, n_test]
    errors2 = np.atleast_2d(np.asarray(errors2, float))
    if data == "mock" and (len(errors1) != 1 or len(errors2) != 1):
        raise ValueError('data="mock" expects the errors of a single run; use data="real" for several runs.')
    errors1, errors2 = np.broadcast_arrays(errors1, errors2)
    n_test = errors1.shape[1]

    # Wilcoxon on the per-point errors (averaged over the runs for real data)
    e1, e2 = errors1.mean(axis=0), errors2.mean(axis=0)
    # Identical errors everywhere: the test is undefined, there is no difference
    p_value = stats.wilcoxon(e1, e2).pvalue if np.any(e1 != e2) else 1.0

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_test, size=(n_boot, n_test))
    boot = np.mean([np.median(r1[idx], axis=1) - np.median(r2[idx], axis=1)
                    for r1, r2 in zip(errors1, errors2)], axis=0)
    alpha = 1 - confidence
    ci_min, ci_max = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    diff = np.mean(np.median(errors1, axis=1) - np.median(errors2, axis=1))

    significant = not (ci_min <= 0 <= ci_max)
    if not significant:
        result = "NO significant difference"
    elif diff < 0:
        result = f"{label1} is better (lower error)"
    else:
        result = f"{label2} is better (lower error)"

    return {
        "Comparison": f"{label1} vs {label2}",
        "Median Difference": round(float(diff), 3),
        f"{confidence:.0%} CI": (round(float(ci_min), 3), round(float(ci_max), 3)),
        "Significant": significant,
        "Wilcoxon p": float(p_value),
        "Result": result
    }
