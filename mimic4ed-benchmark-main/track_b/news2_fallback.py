"""M2 Track B -- partial-NEWS2 fallback (rules only, no model).

Same five scored parameters and comparisons as Track A's helpers.add_score_NEWS2: respiratory
rate, temperature, systolic BP, heart rate, SpO2 (Scale 1). Supplemental-oxygen use and level
of consciousness are not available in this data, so this is a PARTIAL NEWS2, exactly as in
Track A. A value that falls in none of the bands (non-integer gaps) or is missing contributes 0
(the Track A np.select default); missing values are counted in `n_missing` so the caller can show
the score as less reliable.

The comparisons are written exactly as in Track A (inclusive/exclusive as there), because
temperatures converted from Fahrenheit sit a hair above round numbers (38.0000000000004).
"""
import numpy as np


def _rr(x):
    if x <= 8: return 3
    if 9 <= x <= 11: return 1
    if 12 <= x <= 20: return 0
    if 21 <= x <= 24: return 2
    if x >= 25: return 3
    return 0


def _temp(x):
    if x <= 35: return 3
    if 35 < x <= 36: return 1
    if 36 < x <= 38: return 0
    if 38 < x <= 39: return 1
    if x > 39: return 2
    return 0


def _sbp(x):
    if x <= 90: return 3
    if 91 <= x <= 100: return 2
    if 101 <= x <= 110: return 1
    if 111 <= x <= 219: return 0
    if x > 219: return 3
    return 0


def _hr(x):
    if x <= 40: return 3
    if 41 <= x <= 50: return 1
    if 51 <= x <= 90: return 0
    if 91 <= x <= 110: return 1
    if 111 <= x <= 130: return 2
    if x > 130: return 3
    return 0


def _spo2(x):
    if x <= 91: return 3
    if 92 <= x <= 93: return 2
    if 94 <= x <= 95: return 1
    if x >= 96: return 0
    return 0


def partial_news2(resprate, temperature, sbp, heartrate, o2sat):
    score, missing = 0, 0
    for x, fn in ((resprate, _rr), (temperature, _temp), (sbp, _sbp), (heartrate, _hr), (o2sat, _spo2)):
        if x is None or np.isnan(float(x)):
            missing += 1
            continue
        score += fn(float(x))
    return {"news2_partial": int(score), "n_missing": missing, "max_possible_partial": 15,
            "note": "partial NEWS2: supplemental O2 and consciousness unavailable", "mode": "fallback_news2",
            "track": "B"}
