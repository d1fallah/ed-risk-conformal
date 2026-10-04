"""M2 Track B -- alignment-safe chronological ordering.

Track A's sort_chronologically() ends with reset_index(drop=True), but prepare_temporal_tensors_v2()
selects the stored sequence rows by df.index.values. Passing a sorted frame therefore pairs the
sequence tensors of the WRONG rows with the labels and static features (measured on the 6 h /
composite Hybrid model: test AUROC 0.9375 -> 0.7324). Track B never uses that combination.

chronological_aligned() sorts the same way but keeps each row's original position as its index,
and assert_aligned() verifies against the stay_id stored with the sequences.
"""
import numpy as np

from m2_hybrid_common import sort_chronologically
from m2_v2_common import load_sequences


def chronological_aligned(df):
    d = df.copy()
    d["_orig_pos"] = d.index.values
    out = sort_chronologically(d)
    out.index = out["_orig_pos"].values
    return out


def assert_aligned(df, split):
    stay = load_sequences(split)[4]
    ok = bool((stay[df.index.values] == df["stay_id"].values).all())
    assert ok, f"{split}: sequence rows are not aligned with the frame's stay_id -- STOP."
    return ok
