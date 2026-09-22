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
