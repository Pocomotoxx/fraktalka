"""Intrinsic, validation-free diagnostics for weights, series, and activations.

Every function here reads an artifact's *own* structure and returns a scalar signal
— no held-out labels, no test set. Whether any of these signals actually predicts
something useful (generalization, drift) is exactly what `fraktalka.eval` measures;
this module only computes them, honestly and deterministically.

A signal is a signal, never a proof. The rank (ordering) is what downstream use
relies on, so we return the raw spectral slope / exponent rather than a "fractal
dimension" with a convention-dependent constant that would only relabel it
monotonically.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Spectral / fractal structure of a 2D field (e.g. a weight matrix)
# ---------------------------------------------------------------------------

def radial_power_spectrum(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (k, P(k)): radially averaged power spectrum of a 2D array.

    Unlike a naive flattened-and-sorted spectrum, this bins the 2D power spectrum
    by radial spatial frequency |k|, which is what a spectral-slope estimate needs.
    """
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    arr = arr - arr.mean()
    fft = np.fft.fft2(arr)
    power = np.abs(np.fft.fftshift(fft)) ** 2

    h, w = power.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    r = r.astype(int)

    nbins = int(r.max()) + 1
    tbin = np.bincount(r.ravel(), power.ravel(), minlength=nbins)
    nr = np.bincount(r.ravel(), minlength=nbins)
    with np.errstate(invalid="ignore", divide="ignore"):
        radial = tbin / np.maximum(nr, 1)
    k = np.arange(nbins)
    # Drop the DC bin (k=0) and empty bins.
    keep = (k > 0) & (nr > 0) & (radial > 0)
    return k[keep], radial[keep]


def spectral_slope(matrix: np.ndarray) -> float:
    """Log-log slope beta of the radially averaged power spectrum P(k) ~ k^(-beta).

    A larger beta means energy concentrated at low frequencies (smoother, more
    self-affine structure); near zero means white-noise-like. Returns beta (>=0 for
    typical decaying spectra). NaN if the spectrum is too small to fit.
    """
    k, p = radial_power_spectrum(matrix)
    if k.size < 4:
        return float("nan")
    slope = np.polyfit(np.log(k), np.log(p), 1)[0]
    return float(-slope)  # report beta = -slope, so a decaying spectrum gives beta > 0


# ---------------------------------------------------------------------------
# Detrended Fluctuation Analysis (DFA) for a 1D series (e.g. a case timeline)
# ---------------------------------------------------------------------------

def dfa_exponent(series: np.ndarray, min_window: int = 4, n_scales: int = 12) -> float:
    """DFA scaling exponent alpha of a 1D series.

    alpha ~ 0.5 for white noise, ~1.0 for 1/f noise, ~1.5 for Brownian motion.
    Robust to trends (each window is linearly detrended), which is why it is
    preferred over a raw FFT slope for non-stationary series.
    """
    x = np.asarray(series, dtype=float).ravel()
    n = x.size
    if n < 2 * min_window:
        return float("nan")
    y = np.cumsum(x - x.mean())

    max_window = n // 4
    if max_window <= min_window:
        max_window = min_window + 1
    scales = np.unique(np.floor(
        np.logspace(np.log10(min_window), np.log10(max_window), n_scales)
    ).astype(int))
    scales = scales[scales >= min_window]

    fluct = []
    used = []
    for s in scales:
        n_seg = n // s
        if n_seg < 1:
            continue
        rms = []
        for i in range(n_seg):
            seg = y[i * s:(i + 1) * s]
            t = np.arange(s)
            coef = np.polyfit(t, seg, 1)
            trend = np.polyval(coef, t)
            rms.append(np.sqrt(np.mean((seg - trend) ** 2)))
        f = np.sqrt(np.mean(np.square(rms)))
        if f > 0:
            fluct.append(f)
            used.append(s)
    if len(used) < 3:
        return float("nan")
    alpha = np.polyfit(np.log(used), np.log(fluct), 1)[0]
    return float(alpha)


# ---------------------------------------------------------------------------
# Cross-layer self-similarity (SS-rate) of activations
# ---------------------------------------------------------------------------

def ss_rate(activations: list[np.ndarray]) -> float:
    """Mean cosine similarity between the Gram matrices of adjacent layers.

    A proxy for how self-similar the internal representation is across depth.
    Expects a list of (batch, features) activation arrays. Returns NaN if < 2 layers.
    """
    if len(activations) < 2:
        return float("nan")
    grams = []
    for act in activations:
        a = np.asarray(act, dtype=float)
        a = a.reshape(a.shape[0], -1)
        norm = np.linalg.norm(a, axis=1, keepdims=True)
        a = a / np.maximum(norm, 1e-12)
        grams.append(a @ a.T)
    sims = []
    for g1, g2 in zip(grams[:-1], grams[1:]):
        v1, v2 = g1.ravel(), g2.ravel()
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        if denom > 0:
            sims.append(float(v1 @ v2 / denom))
    return float(np.mean(sims)) if sims else float("nan")


# ---------------------------------------------------------------------------
# Baselines (what the fractal signal must beat to earn its place)
# ---------------------------------------------------------------------------

def frobenius_norm(matrix: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(matrix, dtype=float)))


def spectral_norm(matrix: np.ndarray) -> float:
    m = np.asarray(matrix, dtype=float)
    if m.ndim == 1:
        m = m[None, :]
    return float(np.linalg.svd(m, compute_uv=False)[0])


def mean_abs(matrix: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(matrix, dtype=float))))
