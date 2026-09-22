"""Small dependency-free statistics used by the eval harness."""

from __future__ import annotations

import numpy as np


def kendall_tau(a: np.ndarray, b: np.ndarray) -> float:
    """Kendall tau-b rank correlation between two 1D arrays (ties handled).

    O(n^2), which is fine for the small model counts the eval harness uses. Returns
    a value in [-1, 1], or NaN if undefined (e.g. all tied).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = a.size
    if n < 2 or b.size != n:
        return float("nan")
    concordant = discordant = 0
    ties_a = ties_b = 0
    for i in range(n):
        for j in range(i + 1, n):
            da = a[i] - a[j]
            db = b[i] - b[j]
            if da == 0 and db == 0:
                continue
            if da == 0:
                ties_a += 1
            elif db == 0:
                ties_b += 1
            elif (da > 0) == (db > 0):
                concordant += 1
            else:
                discordant += 1
    n0 = concordant + discordant + ties_a
    n1 = concordant + discordant + ties_b
    denom = np.sqrt(n0 * n1)
    if denom == 0:
        return float("nan")
    return float((concordant - discordant) / denom)


def binom_two_sided(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial p-value: P(a count at least as extreme as k) under
    Binomial(n, p). Dependency-free; used for the sign test. Returns 1.0 for n == 0."""
    from math import comb

    if n <= 0:
        return 1.0
    probs = [comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in range(n + 1)]
    thresh = probs[k] * (1 + 1e-9)
    return float(min(1.0, sum(pr for pr in probs if pr <= thresh)))


def sign_test(values: np.ndarray) -> dict:
    """Two-sided sign test on whether `values` are consistently one sign (ties/NaN
    dropped). Returns counts and an exact binomial p-value."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    pos = int((v > 0).sum())
    neg = int((v < 0).sum())
    n = pos + neg
    k = max(pos, neg)
    return {"n": n, "pos": pos, "neg": neg, "p": binom_two_sided(k, n)}
