"""
Provenance accessor for the revision manuscript. Re-uses the repository's `Evidence` class (src/paper_data.py) unchanged, but points its flat artifact
directory at results/revision_v2/evidence/ (frozen copies with a SHA-256 manifest) and writes sidecars to paper_assets_v2/provenance/.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
import paper_data as pdm  # noqa: E402

EVIDENCE_DIR = os.path.join(ROOT, "results", "revision_v2", "evidence")
ASSET_DIR = os.path.join(ROOT, "paper_assets_v2")
PROVENANCE_DIR = os.path.join(ASSET_DIR, "provenance")
pdm.RESULTS_DIR = EVIDENCE_DIR
pdm.PROVENANCE_DIR = PROVENANCE_DIR
Evidence = pdm.Evidence
resolve_pointer = pdm.resolve_pointer
_load_json = pdm._load_json
_fmt_plain = pdm._fmt


def _fmt(value, fmt):
    """The repository formatter, with a leading hyphen typeset as a minus sign (\\ensuremath{-} works in text and in math mode).
    Installed into paper_data for this process only, so Evidence and the verifier's replay format identically; the original
    manuscript pipeline, which does not import this module, is unchanged."""
    s = _fmt_plain(value, fmt)
    if fmt is not None and isinstance(s, str) and s.startswith("-"):
        s = "\\ensuremath{-}" + s[1:]
    return s


pdm._fmt = _fmt


class EV(Evidence):
    """Evidence with a sample standard deviation reduction ("std") over a wildcard pointer, recorded like the other reductions."""

    def reduce(self, how, artifact, wildcard_pointer, *, fmt=None, note=""):
        if how != "std":
            return super().reduce(how, artifact, wildcard_pointer, fmt=fmt, note=note)
        import math
        values = pdm.resolve_wildcard(pdm._load_json(artifact), wildcard_pointer)
        assert len(values) > 1, "std of fewer than two values"
        mean = sum(values) / len(values)
        raw = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
        formatted = pdm._fmt(raw, fmt)
        self.records.append(pdm._Record("reduce:std", artifact, wildcard_pointer, "std", raw, fmt, formatted, note))
        return formatted if fmt is not None else raw
