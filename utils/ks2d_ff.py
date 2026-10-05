"""
Two-dimensional two-sample Kolmogorov-Smirnov test (Fasano & Franceschini, 1987,
MNRAS 225, 155).

Statistic
---------
For every observed point (X_i, Y_i) of sample 1 *and* of sample 2 (used as origin),
the plane is split into the four quadrants
    (x<X_i, y<Y_i), (x<X_i, y>Y_i), (x>X_i, y<Y_i), (x>X_i, y>Y_i).
In each quadrant the fraction of sample-1 points and of sample-2 points is computed;
D1 (resp. D2) is the largest |F1 - F2| over all quadrants and over all origins taken
from sample 1 (resp. sample 2).   D = (D1 + D2) / 2   and   Z = D * sqrt(n1 n2/(n1+n2)).

p-values
--------
* method="permutation": Monte-Carlo permutation test on the pooled sample using exactly
  the same statistic.  Valid for any n, any dependence structure, no table look-up.
* method="asymptotic":  Press et al. (Numerical Recipes, ks2d2s) approximation based
  on the Kolmogorov distribution corrected for the mean correlation r of the two samples.
* method="ff_table":    the original interpolation of the paper (Appendix A, eq. A1,
  Table A6); only calibrated for SL in [30, 99] %, i.e. p in [0.01, 0.70].
"""
import numpy as np
from scipy.optimize import brentq
from scipy.stats import kstwobign

# ----------------------------------------------------------------------------------
# Quadrant machinery
# ----------------------------------------------------------------------------------
def _quadrant_matrices(xy):
    """Four (N, N) float32 0/1 matrices; entry [i, j] = 1 if point j lies in the
    given (strict) quadrant with point i as origin."""
    x, y = xy[:, 0], xy[:, 1]
    gx, lx = x[None, :] > x[:, None], x[None, :] < x[:, None]
    gy, ly = y[None, :] > y[:, None], y[None, :] < y[:, None]
    return [(a & b).astype(np.float32) for a, b in ((gx, gy), (lx, gy), (lx, ly), (gx, ly))]


def _ff_D(Q, in1, n1, n2):
    """FF statistic D for label matrix in1 of shape (N, B) (True = sample 1)."""
    A = in1.astype(np.float32)
    Bm = 1.0 - A
    Di = np.zeros(in1.shape, dtype=np.float32)
    for q in Q:
        np.maximum(Di, np.abs(q @ A / n1 - q @ Bm / n2), out=Di)
    D1 = np.where(in1, Di, 0.0).max(axis=0)      # origins from sample 1
    D2 = np.where(~in1, Di, 0.0).max(axis=0)     # origins from sample 2
    return 0.5 * (D1 + D2)


# ----------------------------------------------------------------------------------
# Paper's interpolation formula (Appendix A, eq. A1, Table A6)
# ----------------------------------------------------------------------------------
_A6 = {(0,0,0): .7107,    (0,0,1): .9853e-2, (0,0,2): .8561e-2, (0,0,3): .2800e-2,
       (0,1,0): -.5126,   (0,1,1): -.2539,   (0,1,2): .3745e-1, (0,2,0): -.3298,
       (0,2,1): -.1408e-1,(0,3,0): -.7693e-1,(1,0,0): -.6124e-1,(1,0,1): -.1918e-1,
       (1,0,2): .1064e-1, (1,1,0): -.5042e-1,(1,1,1): .3670e-1, (1,2,0): .7750e-3,
       (2,0,0): .7775,    (2,0,1): -.2322,   (2,1,0): -.1273,   (3,0,0): -.9915}


def ff_critical_Z(n, cc, sl_percent):
    """Critical value Z_{n,SL} for sample size n (n = n1 n2/(n1+n2) in the 2-sample case),
    correlation coefficient cc and significance level SL (%)."""
    u = -np.log10((1.268 - cc) / 1.41)
    v = np.log10(100.0 - sl_percent) - 2.0
    s = np.log10(n) + 1.074
    w = s * (4.989 / (4.989 - s)) ** (-0.09418)
    return sum(a * u**i * v**j * w**k for (i, j, k), a in _A6.items())


def ff_pvalue_table(Z, n_eff, cc):
    """p = 1 - SL, with SL obtained by inverting eq. (A1). Returns (p, flag) where flag is
    '=' if Z falls inside the calibrated range, '>' if p > 0.70 (only a bound), '<' if p < 0.01."""
    cc = float(np.clip(abs(cc), 0.0, 0.999))
    f = lambda sl: ff_critical_Z(n_eff, cc, sl) - Z
    lo, hi = 30.0, 99.0
    if f(lo) >= 0:
        return 0.70, ">"
    if f(hi) <= 0:
        return 0.01, "<"
    return (100.0 - brentq(f, lo, hi)) / 100.0, "="


# ----------------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------------
def ks2d2s(sample1, sample2, method="permutation", n_perm=1000, seed=None, chunk=100):
    """Fasano-Franceschini 2-D two-sample KS test.

    Parameters
    ----------
    sample1, sample2 : array-like [n, 2]   e.g. (lat, lon)
    method : {"permutation", "asymptotic", "ff_table"}

    Returns
    -------
    dict(D, Z, p, n1, n2, r, method)
    """
    s1, s2 = np.asarray(sample1, float), np.asarray(sample2, float)
    n1, n2 = len(s1), len(s2)
    pooled = np.vstack([s1, s2])
    N = n1 + n2
    Q = _quadrant_matrices(pooled)

    lab = np.zeros((N, 1), dtype=bool)
    lab[:n1] = True
    D = float(_ff_D(Q, lab, n1, n2)[0])
    n_eff = n1 * n2 / N
    Z = D * np.sqrt(n_eff)
    r = 0.5 * (np.corrcoef(s1.T)[0, 1] + np.corrcoef(s2.T)[0, 1])   # mean correlation

    if method == "permutation":
        rng = np.random.default_rng(seed)
        ge, done = 0, 0
        while done < n_perm:
            b = min(chunk, n_perm - done)
            ranks = rng.random((N, b)).argsort(axis=0).argsort(axis=0)
            Dp = _ff_D(Q, ranks < n1, n1, n2)           # random relabelling, n1 in sample 1
            ge += int(np.sum(Dp >= D - 1e-9))
            done += b
        p = (1 + ge) / (n_perm + 1)                       # add-one: never exactly 0
    elif method == "asymptotic":
        sn = np.sqrt(n_eff)
        p = float(kstwobign.sf(D * sn / (1 + np.sqrt(1 - r**2) * (0.25 - 0.75 / sn))))
    elif method == "ff_table":
        p, _ = ff_pvalue_table(Z, n_eff, r)
    else:
        raise ValueError(method)
    return {"D": D, "Z": Z, "p": p, "n1": n1, "n2": n2, "r": r, "method": method}


if __name__ == "__main__":
    # sanity checks
    # (1) eq. A1 must reproduce Table A1 (CC=0): n=50, SL=90 -> 1.46 ; n=100, SL=50 -> 1.12
    print("A1 check:", round(ff_critical_Z(50, 0, 90), 3), "(1.46)",
          round(ff_critical_Z(100, 0, 50), 3), "(1.12)",
          "| CC=.8 n=50 SL=95:", round(ff_critical_Z(50, .8, 95), 3), "(1.49)")
    rng = np.random.default_rng(0)
    # (2) null: p-values should be ~Uniform(0,1) -> rejection rate ~ 5 %
    rej = {"permutation": 0, "asymptotic": 0, "ff_table": 0}
    R = 150
    for k in range(R):
        a, b = rng.normal(size=(100, 2)), rng.normal(size=(100, 2))
        for m in rej:
            rej[m] += ks2d2s(a, b, m, n_perm=200, seed=k)["p"] < 0.05
    print("type-I error at 5%:", {m: v / R for m, v in rej.items()})
    # (3) alternative: marginals identical, dependence differs (marginal KS blind to it)
    from scipy.stats import ks_2samp
    a = rng.normal(size=(300, 2))
    b = rng.multivariate_normal([0, 0], [[1, .7], [.7, 1]], 300)
    print("marginal KS p:", ks_2samp(a[:, 0], b[:, 0]).pvalue, ks_2samp(a[:, 1], b[:, 1]).pvalue)
    print("2D FF:", ks2d2s(a, b, n_perm=500, seed=1))
