import numpy as np
from scipy import stats
import pandas as pd

def compare_models_ci(data1, data2, label1="Model 1", label2="Model 2", confidence=0.95):
    """
    Statistically compares two INDEPENDENT distributions (e.g., positioning errors measured
    on different test sets) using Confidence Intervals on the difference of means.

    If both models are evaluated on the SAME test points, use compare_models_paired instead.
    """
    n1, n2 = len(data1), len(data2)
    mean1, mean2 = np.mean(data1), np.mean(data2)
    var1, var2 = np.var(data1, ddof=1), np.var(data2, ddof=1)

    # Standard error of difference
    se_diff = np.sqrt(var1/n1 + var2/n2)

    # Confidence interval calculation
    alpha = 1 - confidence
    z = stats.norm.ppf(1 - alpha / 2)  # Two-tailed test
    diff = mean1 - mean2
    ci_min, ci_max = diff - z * se_diff, diff + z * se_diff

    # Determine statistical significance
    # If 0 is inside the CI, there is no significant difference.
    if ci_min <= 0 <= ci_max:
        result = "NO significant difference"
    elif mean1 < mean2:
        result = f"{label1} is better (lower error)"
    else:
        result = f"{label2} is better (lower error)"

    return {
        "Comparison": f"{label1} vs {label2}",
        "Mean Difference": round(diff, 3),
        f"{confidence:.0%} CI": (round(ci_min, 3), round(ci_max, 3)),
        "Result": result
    }

def compare_models_paired(errors1, errors2, label1="Model 1", label2="Model 2", confidence=0.95,
                          n_boot=2000, seed=0):
    """
    
    Paired comparison of two models evaluated on the same test points: significance is
    decided on the median errors with a paired bootstrap 95% CI of the median difference
    (resampling test points), significant if the CI excludes 0; the Wilcoxon signed-rank
    p-value is also reported. Here each synthetic dataset is generated once, so the CI
    reflects test-set variability only; in the paper (real MDT data), it was computed over
    10 independent generation runs.
    
    """
    errors1, errors2 = np.asarray(errors1, float), np.asarray(errors2, float)
    if errors1.shape != errors2.shape:
        raise ValueError("Paired comparison needs errors on the same test points.")

    # Identical errors everywhere: the test is undefined, there is no difference
    p_value = stats.wilcoxon(errors1, errors2).pvalue if np.any(errors1 != errors2) else 1.0

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(errors1), size=(n_boot, len(errors1)))
    boot = np.median(errors1[idx], axis=1) - np.median(errors2[idx], axis=1)
    alpha = 1 - confidence
    ci_min, ci_max = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    diff = np.median(errors1) - np.median(errors2)

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
