"""A minimal numpy MLP, only so the eval harness can train *real* models with
genuinely different generalization gaps. Not a training library — just enough to
produce weight matrices whose intrinsic structure we can then probe.
"""

from __future__ import annotations

import numpy as np


def make_classification(n: int, dim: int, n_classes: int, noise: float, rng: np.random.Generator):
    """A simple linearly-structured classification set with optional label noise."""
    centers = rng.normal(size=(n_classes, dim)) * 2.0
    y = rng.integers(0, n_classes, size=n)
    x = centers[y] + rng.normal(size=(n, dim))
    if noise > 0:
        flip = rng.random(n) < noise
        y = y.copy()
        y[flip] = rng.integers(0, n_classes, size=int(flip.sum()))
    return x.astype(float), y.astype(int)


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class TinyMLP:
    def __init__(self, dim: int, hidden: int, n_classes: int, rng: np.random.Generator):
        s1 = np.sqrt(2.0 / dim)
        s2 = np.sqrt(2.0 / hidden)
        self.W1 = rng.normal(scale=s1, size=(dim, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(scale=s2, size=(hidden, n_classes))
        self.b2 = np.zeros(n_classes)

    def forward(self, x):
        h_pre = x @ self.W1 + self.b1
        h = np.maximum(h_pre, 0.0)
        logits = h @ self.W2 + self.b2
        return logits, h

    def activations(self, x):
        logits, h = self.forward(x)
        return [h, _softmax(logits)]

    def accuracy(self, x, y):
        logits, _ = self.forward(x)
        return float((logits.argmax(axis=1) == y).mean())

    def train(self, x, y, epochs: int, lr: float, weight_decay: float, rng: np.random.Generator):
        n, n_classes = x.shape[0], self.W2.shape[1]
        onehot = np.eye(n_classes)[y]
        for _ in range(epochs):
            idx = rng.permutation(n)
            for start in range(0, n, 32):
                b = idx[start:start + 32]
                xb, yb = x[b], onehot[b]
                h_pre = xb @ self.W1 + self.b1
                h = np.maximum(h_pre, 0.0)
                logits = h @ self.W2 + self.b2
                probs = _softmax(logits)
                g_logits = (probs - yb) / len(b)
                gW2 = h.T @ g_logits + weight_decay * self.W2
                gb2 = g_logits.sum(axis=0)
                g_h = g_logits @ self.W2.T
                g_h[h_pre <= 0] = 0.0
                gW1 = xb.T @ g_h + weight_decay * self.W1
                gb1 = g_h.sum(axis=0)
                self.W1 -= lr * gW1
                self.b1 -= lr * gb1
                self.W2 -= lr * gW2
                self.b2 -= lr * gb2
        return self
