import numpy as np
from scipy import stats
import pandas as pd

def compare_models_ci(data1, data2, label1="Model 1", label2="Model 2", confidence=0.95):
    """
    Statistically compares two distributions (e.g., positioning errors) using
    Confidence Intervals on the difference of means.
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
        "95% CI": (round(ci_min, 3), round(ci_max, 3)),
        "Result": result
    }