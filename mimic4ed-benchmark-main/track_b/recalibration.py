"""M2 Track B -- periodic-recalibration interval (serving-side logic).

Track B code; never imported by Track A. Implements the same rule Track A's
m2_eci_vs_recalibration.py evaluated (M3, W=10,000, R=2,000): the absolute-residual
conformal radius q is the finite-sample ceil-quantile of |y - p| over the most recent W
labelled arrivals, recomputed every R arrivals; the interval is [p - q, p + q] clipped to
[0, 1]. test_model.py checks this class against Track A's function on the same stream.
"""
import json
from collections import deque

import numpy as np

ALPHA = 0.1


def conformal_radius(residuals, alpha=ALPHA):
    n = len(residuals)
    if n == 0:
        return 1.0  # fail-open: widest interval until labels exist
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(np.asarray(residuals, float), level, method="higher"))


class RecalibratingInterval:
    """Rolling window of the most recent W (|y-p|) residuals; q refreshed every R labels."""

    def __init__(self, init_p, init_y, window=10000, refresh_every=2000, alpha=ALPHA):
        self.window, self.refresh_every, self.alpha = window, refresh_every, alpha
        res = np.abs(np.asarray(init_y, float) - np.asarray(init_p, float))[-window:]
        self._buf = deque(res.tolist(), maxlen=window)
        self._since_refresh = 0
        self.q = conformal_radius(res, alpha)

    def interval(self, p):
        return max(0.0, p - self.q), min(1.0, p + self.q)

    def add_label(self, p, y):
        """Call when a patient's outcome label becomes known."""
        self._buf.append(abs(float(y) - float(p)))
        self._since_refresh += 1
        if self._since_refresh >= self.refresh_every:
            self.q = conformal_radius(list(self._buf), self.alpha)
            self._since_refresh = 0

    def state(self):
        return {"window": self.window, "refresh_every": self.refresh_every, "alpha": self.alpha,
                "q": self.q, "residuals": list(self._buf)}

    @classmethod
    def from_state(cls, state):
        obj = cls([], [], state["window"], state["refresh_every"], state["alpha"])
        obj._buf = deque(state["residuals"], maxlen=state["window"])
        obj.q = state["q"]
        return obj

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.state(), f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_state(json.load(f))
