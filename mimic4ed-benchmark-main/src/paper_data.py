"""
M2 -- Shared evidence accessor for every number that reaches a paper table, figure, or macro.

Adapted from module 1\\src\\paper_data.py (same Golden Rule: a manuscript number must always trace
to a committed artifact, never be hand-typed). This module is the ONLY place an M2 results artifact
(under mimic4ed-benchmark-main/data/) is read for paper-asset generation. src/generate_latex_tables.py
and src/generate_paper_macros.py go through the `Evidence` class below, so:

  - a number is read from an artifact in exactly one place;
  - every emitted value carries its own provenance (artifact file + pointer), written automatically
    as a sidecar next to the asset that used it;
  - src/verify_paper_assets.py can re-read the RAW artifact, independently replay the pointer or the
    reduction, and assert it still matches the FORMATTED STRING actually printed into the .tex.

Pointer syntax: a '/'-delimited path into the parsed JSON, e.g.
  "/Temporal-Only GRU-D v2/table/M2/coverage"
A path segment of '*' is a wildcard, used only by `reduce()` and `series()` to select every
matching child (e.g. every per-seed entry in a "per_seed" list).

Four kinds of number, four methods, deliberately not fewer (identical contract to M1's):
  get(...)      -- one value at one pointer in one artifact (the common case).
  reduce(...)   -- an aggregate (mean/max/min/len) over a wildcard pointer.
  series(...)   -- an ordered list of values for a figure's plotted series (no formatting).
  external(...) -- a value quoted from a cited external paper. Requires a citation and a locus.
  literal(...)  -- a declared constant, not a measurement. Requires a `reason`.

M2's artifact directory (`data/`) has a much longer revision history than M1's did: many result
files were produced, found defective, and superseded IN PLACE (same directory, different filename,
e.g. "m2_v2_deltas_pre_tuning_symmetry.json" superseded by "m2_v2_deltas.json"), documented
piecemeal across M2_Findings_Dossier.md's 1800+ lines and M2_Hybrid_Deviation_Log.md's 591 lines,
rather than M1's small, cleanly-enumerated list. SUPERSEDED_ARTIFACTS below is deliberately built
INCREMENTALLY, one confirmed entry at a time, as each specific artifact is actually sourced for a
table/figure/macro and checked against the dossier at that point -- never front-loaded from a
guess at file-naming patterns alone. Directories under data/ whose name starts with "archive_" are
excluded structurally (see _load_json): these are pre-fix snapshots explicitly "preserved and never
overwritten in place" (dossier line ~672), so no artifact.json path can ever resolve into one.
"""
from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_HERE, "..", "data")
PROVENANCE_DIR = os.path.join(_HERE, "..", "paper_assets", "provenance")

# Artifacts confirmed superseded by a later, differently-scoped result -- confirmed one at a time
# against M2_Findings_Dossier.md / M2_Hybrid_Deviation_Log.md as each is actually considered as a
# source, never guessed from a filename pattern alone. See this module's docstring.
SUPERSEDED_ARTIFACTS: dict[str, str] = {
    "m2_v2_deltas_pre_mtan_matched.json": "pre-mTAN-matched-tuning snapshot; use m2_v2_deltas.json "
        "(post-tuning-symmetry, post-mTAN-matched -- M2_Findings_Dossier.md Sec 24.2b)",
    "m2_v2_deltas_pre_tuning_symmetry.json": "pre-tuning-symmetry-fix snapshot; use m2_v2_deltas.json",
}


def _artifact_path(artifact: str) -> str:
    """Resolve an artifact filename to a path, enforcing the SUPERSEDED_ARTIFACTS guard and the
    flat-directory-only structural guard shared by both the JSON and CSV loaders."""
    if artifact in SUPERSEDED_ARTIFACTS:
        raise ValueError(
            f"{artifact} is a SUPERSEDED artifact ({SUPERSEDED_ARTIFACTS[artifact]}); "
            f"no paper asset may source it (src/paper_data.py SUPERSEDED_ARTIFACTS)"
        )
    path = os.path.join(RESULTS_DIR, artifact)
    # Structural guard: never resolve into an archive_*/ subdirectory even if a caller's artifact
    # string tried to (defense in depth -- the intended interface is a flat filename only).
    real_results_dir = os.path.realpath(RESULTS_DIR)
    real_path = os.path.realpath(path)
    if os.path.dirname(real_path) != real_results_dir:
        raise ValueError(
            f"{artifact!r} does not resolve to a flat file directly under data/ -- "
            f"artifacts under archive_*/ or other subdirectories are pre-fix snapshots and must "
            f"never be sourced by a paper asset"
        )
    return path


def _load_json(artifact: str) -> Any:
    with open(_artifact_path(artifact), "r", encoding="utf-8") as f:
        return json.load(f)


_CSV_CACHE: dict[str, list[dict[str, str]]] = {}


def _load_csv(artifact: str) -> list[dict[str, str]]:
    """Every M2 result CSV is long-format: one row per (model[, seed[, other key]]), string-typed
    cells (csv.DictReader). Numeric comparison/aggregation happens in the row-filtering helpers
    below, which cast on read -- the raw string is never silently truncated or reformatted here."""
    if artifact not in _CSV_CACHE:
        with open(_artifact_path(artifact), "r", encoding="utf-8", newline="") as f:
            _CSV_CACHE[artifact] = list(csv.DictReader(f))
    return _CSV_CACHE[artifact]


def csv_pointer(filters: dict[str, Any], column: str) -> str:
    """Encode a CSV row-filter + column as a single pointer string recorded in the provenance
    sidecar. JSON-encoded (not Python repr) specifically so verify_paper_assets.py can parse it
    back unambiguously with json.loads -- a dict's repr() is not guaranteed round-trippable the
    same way across Python versions/orderings, and "::" (not "/") separates the filters from the
    column so a value or column name that happened to contain "/" could never be misparsed."""
    return json.dumps(filters, sort_keys=True) + "::" + column


def parse_csv_pointer(pointer: str) -> tuple[dict[str, Any], str]:
    """Inverse of csv_pointer() -- used only by verify_paper_assets.py to independently replay a
    CSV-sourced record."""
    filters_json, column = pointer.rsplit("::", 1)
    return json.loads(filters_json), column


def _filter_rows(rows: list[dict[str, str]], filters: dict[str, Any]) -> list[dict[str, str]]:
    """Rows where every filters[col] == row[col], comparing as strings (CSV cells are always
    strings; the caller passes e.g. seed=42, which str()s to match "42")."""
    out = []
    for row in rows:
        if all(str(row.get(col)) == str(val) for col, val in filters.items()):
            out.append(row)
    return out


def resolve_pointer(data: Any, pointer: str) -> Any:
    """Walk a '/'-delimited pointer into a parsed JSON structure. Integer-looking segments index
    lists; everything else is a dict key. No wildcard support here -- see resolve_wildcard()."""
    node = data
    for seg in pointer.strip("/").split("/"):
        if isinstance(node, list):
            node = node[int(seg)]
        else:
            node = node[seg]
    return node


def resolve_wildcard(data: Any, pointer: str) -> list[Any]:
    """Like resolve_pointer, but exactly one segment may be '*', matching every key (dict) or
    every element (list) at that position. Returns the list of matched leaf values."""
    segs = pointer.strip("/").split("/")
    assert segs.count("*") == 1, f"reduce()/series() pointer needs exactly one '*': {pointer!r}"
    star = segs.index("*")
    head, tail = segs[:star], segs[star + 1:]
    node = data
    for seg in head:
        node = node[int(seg)] if isinstance(node, list) else node[seg]
    children = node if isinstance(node, list) else list(node.values())
    out = []
    for child in children:
        v = child
        for seg in tail:
            v = v[int(seg)] if isinstance(v, list) else v[seg]
        out.append(v)
    return out


def _maybe_float(cell: str) -> Any:
    """CSV cells are always strings; cast to float when the cell parses as a plain number so
    reduce_csv()'s arithmetic works, otherwise leave it as the original string (e.g. a model name
    or "False"/"True")."""
    try:
        return float(cell)
    except (TypeError, ValueError):
        return cell


def _fmt(value: Any, fmt: str | None) -> str:
    if fmt is None:
        return str(value)
    if fmt == "pct":
        return f"{value * 100:.1f}\\%"
    return format(value, fmt)


@dataclass
class _Record:
    kind: str  # "value" | "reduce" | "series" | "external" | "literal"
    artifact: str | None
    pointer: str | None
    reduce_how: str | None
    raw: Any
    fmt: str | None
    formatted: str
    note: str
    citation: str | None = None
    locus: str | None = None
    reason: str | None = None


class Evidence:
    """One instance per emitted asset (one table, one figure, one macro file). Accumulates
    _Record entries as get()/reduce()/series()/external()/literal() are called, then
    write_provenance() dumps them as that asset's sidecar."""

    def __init__(self, asset_id: str):
        self.asset_id = asset_id
        self.records: list[_Record] = []

    # ---- the four accessor kinds -------------------------------------------------------
    def get(self, artifact: str, pointer: str, *, fmt: str | None = None, note: str = "") -> Any:
        raw = resolve_pointer(_load_json(artifact), pointer)
        formatted = _fmt(raw, fmt)
        self.records.append(_Record("value", artifact, pointer, None, raw, fmt, formatted, note))
        return formatted if fmt is not None else raw

    def reduce(self, how: str, artifact: str, wildcard_pointer: str, *, fmt: str | None = None, note: str = "") -> Any:
        values = resolve_wildcard(_load_json(artifact), wildcard_pointer)
        if how == "mean":
            raw = sum(values) / len(values)
        elif how == "max":
            raw = max(values)
        elif how == "min":
            raw = min(values)
        elif how == "len":
            raw = len(values)
        else:
            raise ValueError(f"unknown reduction {how!r}")
        formatted = _fmt(raw, fmt)
        self.records.append(_Record(f"reduce:{how}", artifact, wildcard_pointer, how, raw, fmt, formatted, note))
        return formatted if fmt is not None else raw

    def series(self, artifact: str, wildcard_pointer: str, *, note: str = "") -> list[Any]:
        values = resolve_wildcard(_load_json(artifact), wildcard_pointer)
        self.records.append(_Record("series", artifact, wildcard_pointer, None, values, None, str(values), note))
        return values

    # ---- CSV accessor kinds (most M2 result artifacts are long-format CSV, not JSON) -----
    def get_csv(self, artifact: str, filters: dict[str, Any], column: str, *,
                fmt: str | None = None, note: str = "") -> Any:
        """Exactly one row must match `filters` (e.g. {"model": "Hybrid GRU-D v2", "seed": 42});
        returns that row's `column`, cast to float if it parses as one (CSV cells are strings)."""
        rows = _filter_rows(_load_csv(artifact), filters)
        assert len(rows) == 1, (
            f"get_csv({artifact!r}, {filters!r}) matched {len(rows)} rows, expected exactly 1 "
            f"-- use reduce_csv() to aggregate across multiple rows (e.g. across seeds)"
        )
        raw = _maybe_float(rows[0][column])
        formatted = _fmt(raw, fmt)
        pointer = csv_pointer(filters, column)
        self.records.append(_Record("value", artifact, pointer, None, raw, fmt, formatted, note))
        return formatted if fmt is not None else raw

    def reduce_csv(self, how: str, artifact: str, filters: dict[str, Any], column: str, *,
                    fmt: str | None = None, note: str = "") -> Any:
        """Aggregate `column` (mean/std/max/min/len) across every row matching `filters` -- the
        standard case is aggregating across the 5 confirmatory seeds for one model."""
        rows = _filter_rows(_load_csv(artifact), filters)
        assert rows, f"reduce_csv({artifact!r}, {filters!r}) matched 0 rows"
        values = [_maybe_float(r[column]) for r in rows]
        if how == "mean":
            raw = sum(values) / len(values)
        elif how == "std":
            assert len(values) > 1, f"std of {len(values)} value(s) is undefined"
            mean = sum(values) / len(values)
            raw = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
        elif how == "max":
            raw = max(values)
        elif how == "min":
            raw = min(values)
        elif how == "len":
            raw = len(values)
        else:
            raise ValueError(f"unknown reduction {how!r}")
        formatted = _fmt(raw, fmt)
        pointer = csv_pointer(filters, column)
        self.records.append(_Record(f"reduce:{how}", artifact, pointer, how, raw, fmt, formatted, note))
        return formatted if fmt is not None else raw

    def series_csv(self, artifact: str, filters: dict[str, Any], column: str, *, note: str = "") -> list[Any]:
        """Ordered list of `column` values across every row matching `filters`, for a figure's
        plotted series. Row order follows the CSV's own order (not re-sorted)."""
        rows = _filter_rows(_load_csv(artifact), filters)
        values = [_maybe_float(r[column]) for r in rows]
        pointer = csv_pointer(filters, column)
        self.records.append(_Record("series", artifact, pointer, None, values, None, str(values), note))
        return values

    def external(self, value: Any, *, citation: str, locus: str, fmt: str | None = None, note: str = "") -> str:
        """A number quoted from a cited external paper. NOT a repository artifact -- the Golden
        Rule is satisfied by (citation, locus), which the verifier requires non-empty."""
        assert citation and locus, "external() requires both citation and locus"
        formatted = _fmt(value, fmt) if value != "NR" else "NR"
        self.records.append(_Record("external", None, None, None, value, fmt, formatted, note, citation=citation, locus=locus))
        return formatted

    def literal(self, value: Any, *, reason: str) -> Any:
        """A declared, non-measured constant (a column count, a spacing literal). Requires a
        reason so an unsourced number is a visible, deliberate exemption, not an accident."""
        assert reason, "literal() requires a reason"
        formatted = str(value)
        self.records.append(_Record("literal", None, None, None, value, None, formatted, "", reason=reason))
        return value

    # ---- sidecar --------------------------------------------------------------------------
    def write_provenance(self) -> str:
        os.makedirs(PROVENANCE_DIR, exist_ok=True)
        path = os.path.join(PROVENANCE_DIR, f"{self.asset_id}.json")
        payload = {
            "asset_id": self.asset_id,
            "records": [
                {
                    "kind": r.kind,
                    "artifact": r.artifact,
                    "pointer": r.pointer,
                    "reduce_how": r.reduce_how,
                    "raw": r.raw,
                    "fmt": r.fmt,
                    "formatted": r.formatted,
                    "note": r.note,
                    "citation": r.citation,
                    "locus": r.locus,
                    "reason": r.reason,
                }
                for r in self.records
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return path


def emit_macro_file(macros: list[tuple], out_path: str) -> tuple[str, "Evidence"]:
    r"""Write paper_assets/tables/m2_numbers.tex: one \newcommand per EXPLICITLY NAMED prose/diagram
    quantity, for numbers quoted in the manuscript rather than inside a generated table.

    `macros` is a list of (macro_suffix, kind, artifact, pointer, fmt, note) tuples. `kind` is one
    of:
      "get"                          -- ev.get(artifact, pointer)            (pointer: str)
      "max"/"min"/"mean"/"len"       -- ev.reduce(kind, artifact, pointer)   (pointer: str, wildcard)
      "get_csv"                      -- ev.get_csv(artifact, filters, col)   (pointer: (dict, str))
      "mean_csv"/"std_csv"/"max_csv"/"min_csv"
                                      -- ev.reduce_csv(how, artifact, filters, col) (pointer: (dict, str))
    emitting \newcommand{\MTwoFoo}{value}. macro_suffix must be a fixed, stable name chosen by the
    caller -- never derived from position. M2's result artifacts are mostly CSV (unlike M1's
    JSON-only set), so the CSV kinds exist here for the same reason src/generate_latex_tables.py
    needed get_csv()/reduce_csv(): most headline prose numbers (e.g. the primary model's AUROC, the
    ECI-vs-recalibration coverage/width) live in a long-format CSV, not a JSON pointer path.

    Naming (mirrors M1's found-empirically pitfall): a LaTeX control WORD is a backslash followed
    by a maximal run of LETTERS ONLY -- a macro suffix containing a digit (e.g. "CoverageM1", from
    the conformal-set label "M1") silently breaks \newcommand, parsed as defining \MTwoCoverageM
    with the literal text "1" left over inside the brace group. macro_suffix must therefore spell
    any digit out (e.g. "CoverageSetOne" not "CoverageM1") -- asserted below so this fails loudly
    at generation time, not with an opaque "Missing \begin{document}" error at compile time.
    """
    ev = Evidence("m2_numbers")
    seen: set[str] = set()
    header = [
        "% AUTO-GENERATED by src/paper_data.py:emit_macro_file -- do not hand-edit.",
        "% Each \\newcommand name is an explicit, stable identifier chosen at the call site in",
        "% generate_paper_macros.py -- never derived from position, so it cannot silently rebind.",
        "",
    ]
    lines = list(header)
    for suffix, kind, artifact, pointer, fmt, note in macros:
        assert suffix not in seen, f"duplicate macro suffix {suffix!r}"
        seen.add(suffix)
        assert suffix.isalpha(), (
            f"macro suffix {suffix!r} contains a non-letter (likely a digit) -- LaTeX control "
            f"words cannot contain digits; spell it out in generate_paper_macros.py"
        )
        name = "MTwo" + suffix
        if kind == "get":
            value = ev.get(artifact, pointer, fmt=fmt, note=note)
        elif kind == "get_csv":
            filters, column = pointer
            value = ev.get_csv(artifact, filters, column, fmt=fmt, note=note)
        elif kind.endswith("_csv"):
            filters, column = pointer
            value = ev.reduce_csv(kind[: -len("_csv")], artifact, filters, column, fmt=fmt, note=note)
        else:
            value = ev.reduce(kind, artifact, pointer, fmt=fmt, note=note)
        val_tex = str(value).replace("%", "\\%").replace("_", "\\_")
        lines.append("\\newcommand{\\" + name + "}{" + val_tex + "}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    newline = chr(10)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(newline.join(lines) + newline)
    return out_path, ev
