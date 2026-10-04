"""
M2 -- Independent verification gate for every paper table, figure, and macro. Mirrors module 1's
src/verify_paper_assets.py exactly in design (same CRITICAL DESIGN RULE below), extended to also
independently replay CSV-sourced records, since most of M2's result artifacts are CSV, not JSON.

CRITICAL DESIGN RULE: this script never imports src/paper_data.py's Evidence accessors for
verification. If it did, every check would collapse to "the provider agrees with itself", which
passes forever even if a committed .tex is stale or was hand-edited. Instead it:

  1. Re-reads the RAW data/*.csv and data/*.json files directly (never through Evidence.get/reduce).
  2. Re-reads each emitted asset's provenance sidecar (paper_assets/provenance/<asset>.json), which
     records (artifact, pointer, reduce_how, raw, fmt, formatted) for every number that asset used.
  3. REPLAYS each sidecar record independently: re-resolve the JSON pointer or re-filter the CSV
     rows (using paper_data.py's OWN pointer-parsing/row-filtering helpers, which are pure,
     side-effect-free parsing code, not the Evidence class itself -- see the note on that import
     below), re-run the named reduction, and re-apply the format spec -- then compares both against
     what the sidecar (and, for check 4, the actual committed .tex bytes) say.

Checks performed, in order (mirrors M1's numbering):
  1. Manifest integrity      -- every artifact any sidecar names is in results/m2_evidence_manifest.json
                                 (paths are data/<file>, not results/<file> -- M2's artifacts live
                                 under mimic4ed-benchmark-main/data/, not a separate results/ dir),
                                 and its on-disk SHA-256 matches the manifest's.
  2. Replay                  -- resolving/reducing the raw artifact again gives the same value the
                                 sidecar recorded, for BOTH JSON pointers and CSV row-filters.
  3. Re-format                -- format(raw, fmt) == formatted.
  4. Token scan               -- every numeric literal actually printed in each paper_assets/tables/*.tex
                                 is either one of that asset's own recorded `formatted` values or on
                                 its declared literal() whitelist.
  5. Cross-asset agreement    -- every (artifact, pointer) pair, grouped across ALL sidecars, must
                                 carry the same raw value everywhere it appears.
  6. QC label rules           -- Proxy cohort labeling and the module's negative-result wording
                                 rules (never "proved unnecessary", never an un-scoped "ECI fails"/
                                 "ECI does not work"/"recalibration is better"; "not needed under
                                 these conditions" is the only sanctioned phrasing), checked over
                                 sidecar `note` fields and the actual committed caption text.
  6b. Manuscript wording      -- the same negative-result/"pre-registered" bans, independently
                                 re-checked over every hand-written paper/sections/*.tex and
                                 paper/floats/*.tex file directly (check 6 only ever sees a
                                 table/float file whose name equals a sidecar's asset_id, which
                                 never included hand-written prose).
  7. Document scan            -- every literal with >= 3 decimal digits or scientific notation, in
                                 manuscript-facing documents only, must appear in some sidecar's
                                 formatted-value set.
  8. External citations       -- every external() record carries a non-empty citation and locus,
                                 the citation is a bibkey in paper/refs/m2.bib once that file
                                 exists, and an "NR" raw value always renders as the literal "NR".

Also refuses (fails loudly, does not silently skip) if any sidecar names a SUPERSEDED_ARTIFACTS
entry, or an artifact that does not resolve to a flat file directly under data/ (the same
archive_*/ structural guard paper_data.py itself enforces at generation time -- re-checked here
independently in case a sidecar was ever produced by a version of the generator that didn't).

Exit 0: all checks pass. Exit 1: grouped, explicit failure listing.
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(_HERE, "..")
sys.path.insert(0, _HERE)
# Only pure parsing/guard helpers are imported here, never the Evidence class itself -- see the
# CRITICAL DESIGN RULE above. resolve_pointer/resolve_wildcard/parse_csv_pointer do not read any
# provenance sidecar or cache a "trusted" value; they are re-run from scratch against the raw
# artifact every time this script calls them, exactly as a from-scratch reimplementation would,
# just without duplicating ~30 lines of pointer-walking logic that has no reason to drift.
from paper_data import (  # noqa: E402
    SUPERSEDED_ARTIFACTS,
    resolve_pointer,
    resolve_wildcard,
    parse_csv_pointer,
)

PROVENANCE_DIR = os.path.join(ROOT, "paper_assets", "provenance")
TABLES_DIR = os.path.join(ROOT, "paper_assets", "tables")
FLOATS_DIR = os.path.join(ROOT, "..", "paper", "floats")  # module2/paper/floats (sibling of mimic4ed-benchmark-main)
DATA_DIR = os.path.join(ROOT, "data")
MANIFEST_PATH = os.path.join(ROOT, "results", "m2_evidence_manifest.json")

REPLAY_ATOL = 1e-9

TOKEN_RE = re.compile(r"(?<![\w.])-?\d[\d,]*\.?\d*(?:[eE][+-]?\d+)?(?![\w.])")
DOC_NUMBER_RE = re.compile(r"(?<![\w.])-?\d+\.\d{3,}(?![\w.])|(?<![\w.])-?\d+(?:\.\d+)?[eE][+-]?\d+(?![\w.])")


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_manifest() -> dict:
    if not os.path.exists(MANIFEST_PATH):
        raise SystemExit(f"no evidence manifest at {MANIFEST_PATH} -- run src/freeze_evidence.py first")
    return load_json(MANIFEST_PATH)


def sha256_file(path: str) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def load_sidecars() -> dict[str, dict]:
    out = {}
    for path in sorted(glob.glob(os.path.join(PROVENANCE_DIR, "*.json"))):
        d = load_json(path)
        out[d["asset_id"]] = d
    return out


def _fmt(value, fmt):
    if fmt is None:
        return str(value)
    if fmt == "pct":
        return f"{value * 100:.1f}\\%"
    return format(value, fmt)


def _numeric_like_simple(a: str) -> bool:
    body = a[1:] if a.startswith("-") else a
    return body != "" and body.count(".") <= 1 and body.replace(".", "").replace(",", "").isdigit()


def _close(a, b) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= REPLAY_ATOL
    return a == b


def _artifact_is_allowed(artifact: str) -> tuple[bool, str]:
    """Independent re-check of paper_data.py's own structural guard: SUPERSEDED_ARTIFACTS, and
    every artifact must resolve to a flat file directly under data/ (never archive_*/)."""
    if artifact in SUPERSEDED_ARTIFACTS:
        return False, f"SUPERSEDED artifact ({SUPERSEDED_ARTIFACTS[artifact]})"
    path = os.path.join(DATA_DIR, artifact)
    real_data_dir = os.path.realpath(DATA_DIR)
    real_path = os.path.realpath(path)
    if os.path.dirname(real_path) != real_data_dir:
        return False, "does not resolve to a flat file directly under data/ (archive_*/ or similar)"
    return True, ""


def check_manifest_integrity(sidecars: dict, manifest: dict, failures: list) -> None:
    manifest_files = manifest.get("files", {})
    for asset_id, sc in sidecars.items():
        for rec in sc["records"]:
            artifact = rec.get("artifact")
            if artifact is None:
                continue
            key = f"data/{artifact}"
            if key not in manifest_files:
                failures.append(f"[manifest] {asset_id}: artifact {artifact!r} is not in the evidence manifest "
                               f"(run src/freeze_evidence.py after adding a new source artifact)")
                continue
            on_disk = os.path.join(DATA_DIR, artifact)
            if not os.path.exists(on_disk):
                failures.append(f"[manifest] {asset_id}: artifact {artifact!r} listed in the manifest but missing on disk")
                continue
            actual_hash = sha256_file(on_disk)
            if actual_hash != manifest_files[key]:
                failures.append(f"[manifest] {asset_id}: {artifact!r} hash on disk does not match the frozen manifest "
                               f"-- the evidence freeze is stale (run freeze_evidence.py again)")


def check_superseded_and_structural(sidecars: dict, failures: list) -> None:
    for asset_id, sc in sidecars.items():
        for rec in sc["records"]:
            artifact = rec.get("artifact")
            if artifact is None:
                continue
            allowed, reason = _artifact_is_allowed(artifact)
            if not allowed:
                failures.append(f"[superseded/structural] {asset_id}: sources {artifact!r} -- {reason}")


def _filter_rows(rows: list[dict], filters: dict) -> list[dict]:
    out = []
    for row in rows:
        if all(str(row.get(col)) == str(val) for col, val in filters.items()):
            out.append(row)
    return out


def _maybe_float(cell):
    try:
        return float(cell)
    except (TypeError, ValueError):
        return cell


def check_replay_and_format(sidecars: dict, failures: list) -> None:
    json_cache: dict[str, dict] = {}
    csv_cache: dict[str, list[dict]] = {}
    for asset_id, sc in sidecars.items():
        for i, rec in enumerate(sc["records"]):
            if rec["kind"] in ("external", "literal"):
                continue
            artifact = rec["artifact"]
            is_csv = artifact.endswith(".csv")
            path = os.path.join(DATA_DIR, artifact)
            if not os.path.exists(path):
                failures.append(f"[replay] {asset_id}#{i}: artifact {artifact!r} not found on disk")
                continue

            try:
                if is_csv:
                    if artifact not in csv_cache:
                        with open(path, "r", encoding="utf-8", newline="") as f:
                            csv_cache[artifact] = list(csv.DictReader(f))
                    rows = csv_cache[artifact]
                    filters, column = parse_csv_pointer(rec["pointer"])
                    matched = _filter_rows(rows, filters)
                    values = [_maybe_float(r[column]) for r in matched]
                    if rec["kind"] == "value":
                        assert len(matched) == 1, f"expected 1 row, got {len(matched)}"
                        replayed = values[0]
                    elif rec["kind"].startswith("reduce:"):
                        how = rec["reduce_how"]
                        replayed = _reduce(how, values)
                    elif rec["kind"] == "series":
                        replayed = values
                    else:
                        continue
                else:
                    if artifact not in json_cache:
                        json_cache[artifact] = load_json(path)
                    data = json_cache[artifact]
                    if rec["kind"] == "value":
                        replayed = resolve_pointer(data, rec["pointer"])
                    elif rec["kind"].startswith("reduce:"):
                        values = resolve_wildcard(data, rec["pointer"])
                        replayed = _reduce(rec["reduce_how"], values)
                    elif rec["kind"] == "series":
                        replayed = resolve_wildcard(data, rec["pointer"])
                    else:
                        continue
            except (KeyError, IndexError, ZeroDivisionError, AssertionError) as exc:
                failures.append(f"[replay] {asset_id}#{i}: could not re-resolve {rec['pointer']!r} in {artifact!r}: {exc}")
                continue

            if rec["kind"] == "series":
                if replayed != rec["raw"]:
                    failures.append(f"[replay] {asset_id}#{i}: {artifact}:{rec['pointer']} series re-resolves "
                                   f"differently than the sidecar recorded")
                continue
            if not _close(replayed, rec["raw"]):
                failures.append(f"[replay] {asset_id}#{i}: {artifact}:{rec['pointer']} re-resolves to {replayed!r}, "
                               f"sidecar recorded {rec['raw']!r} -- the source artifact changed since generation")
                continue
            refmt = _fmt(replayed, rec["fmt"])
            if refmt != rec["formatted"]:
                failures.append(f"[format] {asset_id}#{i}: re-formatting {replayed!r} with {rec['fmt']!r} gives "
                               f"{refmt!r}, sidecar says {rec['formatted']!r}")


def _reduce(how: str, values: list):
    if how == "mean":
        return sum(values) / len(values)
    if how == "std":
        mean = sum(values) / len(values)
        return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
    if how == "max":
        return max(values)
    if how == "min":
        return min(values)
    if how == "len":
        return len(values)
    raise ValueError(f"unknown reduction {how!r}")


def check_token_scan(sidecars: dict, failures: list) -> None:
    for asset_id, sc in sidecars.items():
        tex_path = os.path.join(TABLES_DIR, f"{asset_id}.tex")
        if not os.path.exists(tex_path):
            continue
        with open(tex_path, "r", encoding="utf-8") as f:
            tex = f.read()
        allowed = {rec["formatted"] for rec in sc["records"] if rec["kind"] != "series"}
        allowed |= {str(rec["raw"]) for rec in sc["records"] if rec["kind"] == "literal"}
        allowed |= {a.rstrip("%") for a in list(allowed) if a.endswith("%")}
        prose_allowed = [a for a in allowed if not _numeric_like_simple(a) and (len(a) > 6 or a.count(".") >= 2)]

        tokens_found = set(TOKEN_RE.findall(tex))
        suspects = {t for t in tokens_found if ("." in t or len(t.replace(",", "").lstrip("-")) >= 3)}

        def explained(token: str) -> bool:
            # A trailing comma is absorbed into the match by TOKEN_RE's own [\d,]* class when a
            # number sits inside a plain comma-separated prose list ("60, 240, 1000") -- found via
            # M2's "$L\in\{60, 240, 1000\}$" caption tokenizing as "240," (with the list's own
            # separator comma attached). Stripped the same way a sentence-ending period already is.
            stripped = token.rstrip(".").rstrip(",")
            if token in allowed or stripped in allowed:
                return True
            bare = stripped.rstrip("0").rstrip(".")
            if any(bare == a.rstrip("0").rstrip(".") for a in allowed if "." in a):
                return True
            if any(stripped in a for a in prose_allowed):
                return True
            return False

        unexplained = {t for t in suspects if not explained(t)}
        if unexplained:
            failures.append(f"[token-scan] {asset_id}.tex contains numeric literal(s) not traced to any "
                           f"recorded value or literal(): {sorted(unexplained)}")


def check_cross_asset_agreement(sidecars: dict, failures: list) -> None:
    by_key: dict[tuple, list] = {}
    for asset_id, sc in sidecars.items():
        for rec in sc["records"]:
            if rec["kind"] in ("external", "literal") or rec["artifact"] is None:
                continue
            key = (rec["artifact"], rec["pointer"], rec.get("reduce_how"))
            by_key.setdefault(key, []).append((asset_id, rec["raw"]))
    for key, entries in by_key.items():
        values = [v for _, v in entries]
        first = values[0]
        if not all(_close(first, v) for v in values):
            failures.append(f"[cross-asset] {key[0]}:{key[1]} (reduce={key[2]}) disagrees across assets: "
                           f"{[(a, v) for a, v in entries]}")


def _has_caption(tex_path: str) -> bool:
    with open(tex_path, "r", encoding="utf-8") as f:
        return "\\caption{" in f.read()


# Wording the supervisor locked for the ECI-vs-recalibration negative result: only "not needed
# under these conditions" is sanctioned; these unqualified phrasings are banned regardless of
# surrounding context, since the whole point is that the claim must stay scoped to the tested
# setting/metric, never generalized.
BANNED_UNSCOPED_PHRASES = [
    "proved unnecessary",
    "eci fails",
    "eci does not work",
]

# Internal-iteration / imprecise-claim leaks caught by the second supervisor review
# (main(4).pdf -> main(5).pdf, 2026-09-22): "matched-input" implied the Temporal-Only-vs-XGBoost
# delta isolates temporal representation, which it does not (XGBoost-Matched also carries the four
# static features); "tuning-symmetric" overstated the tuning protocol (Methods explicitly says the
# grids are comparable in scale, not identical); "worth its cost" implied a measured
# engineering/deployment cost that was never measured (only latency was). Checked as substrings, not
# whole-word regexes, since none of them are legitimate substrings of any other word used here.
BANNED_STYLE_PHRASES = [
    "matched-input",
    "tuning-symmetric",
    "tuning-symmetry",
    "worth its cost",
]
# Bare "v2" as an internal pipeline-iteration marker leaking into rendered prose/captions -- distinct
# from "v2" inside a LaTeX label/ref/input/includegraphics argument (e.g. \label{tab:m2-v2-headline},
# \includegraphics{...auroc_v2}), which is an internal identifier never rendered in the compiled PDF.
BARE_V2_RE = re.compile(r"(?<![A-Za-z0-9_])v2(?![A-Za-z0-9_])")
_LATEX_REF_ARG_RE = re.compile(r"\\(?:label|ref|input|includegraphics(?:\[[^\]]*\])?|cite[tp]?)\{[^}]*\}")
_MD_CODE_SPAN_RE = re.compile(r"`[^`]*`")


def _strip_nonrendered_latex(text: str) -> str:
    """Drop LaTeX command arguments that never render as visible text (labels, refs, \\input/
    \\includegraphics file paths, cite keys) and Markdown inline-code spans (literal filenames quoted
    in an audit doc, e.g. `m2_v2_deltas_pre_tuning_symmetry.json`), so a leak check on the remaining
    text only sees prose a reader would actually read as a claim, not a quoted artifact filename."""
    return _MD_CODE_SPAN_RE.sub(" ", _LATEX_REF_ARG_RE.sub(" ", text))


def check_qc_labels(sidecars: dict, failures: list) -> None:
    for asset_id, sc in sidecars.items():
        sources = {rec["artifact"] for rec in sc["records"] if rec.get("artifact")}
        notes = " ".join(rec.get("note", "") or "" for rec in sc["records"])
        tex_path = os.path.join(TABLES_DIR, f"{asset_id}.tex")
        float_path = os.path.join(FLOATS_DIR, f"{asset_id}.tex")
        is_rendered_caption = True
        if os.path.exists(tex_path) and _has_caption(tex_path):
            with open(tex_path, "r", encoding="utf-8") as f:
                caption_text = f.read()
        elif os.path.exists(float_path) and _has_caption(float_path):
            with open(float_path, "r", encoding="utf-8") as f:
                caption_text = f.read()
        else:
            caption_text = notes
            is_rendered_caption = False

        if sources and "Proxy" not in caption_text:
            failures.append(f"[qc-label] {asset_id}: sources a results artifact but caption/note lacks "
                           f"'Proxy' cohort labeling")
        lower = caption_text.lower()
        for phrase in BANNED_UNSCOPED_PHRASES:
            if phrase in lower:
                failures.append(f"[qc-label] {asset_id}: caption contains the banned unscoped phrase "
                               f"{phrase!r} -- use the locked 'not needed under these conditions' wording")
        if "pre-registered" in lower:
            failures.append(f"[qc-label] {asset_id}: caption says 'pre-registered' -- these are internal "
                           f"design-freeze documents, not a public registration; use 'pre-specified'")
        # The two checks below only make sense against an actual rendered \caption{} block: the
        # `notes` fallback is internal provenance metadata (never printed in the compiled PDF), and
        # legitimately names the internal "v2" pipeline stage and tuning-protocol shorthand there.
        if is_rendered_caption:
            for phrase in BANNED_STYLE_PHRASES:
                if phrase in lower:
                    failures.append(f"[qc-label] {asset_id}: caption contains the banned style phrase "
                                   f"{phrase!r} -- see BANNED_STYLE_PHRASES for the replacement")
            if BARE_V2_RE.search(_strip_nonrendered_latex(caption_text)):
                failures.append(f"[qc-label] {asset_id}: caption leaks the internal 'v2' pipeline-iteration "
                               f"marker into manuscript-facing text -- use 'primary cohort'/'primary model'/"
                               f"drop the suffix")
        for rec in sc["records"]:
            if rec.get("artifact") in SUPERSEDED_ARTIFACTS:
                failures.append(f"[qc-label] {asset_id}: sources SUPERSEDED artifact {rec['artifact']!r} "
                               f"({SUPERSEDED_ARTIFACTS[rec['artifact']]})")


# Manuscript-facing documents scanned for un-sourced numeric literals. Deliberately narrow, same
# rationale as M1's: the module's closure-phase audit docs (M2_Findings_Dossier.md,
# M2_Hybrid_Deviation_Log.md, the *_PreRegistration.md files) are PRIMARY SOURCE documents in their
# own right, not restatements of a paper-table number -- scanning them would produce the same
# ~100-false-positive "noisy gate nobody reads" failure mode M1 already discovered and rejected.
MANUSCRIPT_FACING_DOC_NAMES = {"M2_FINAL_EVIDENCE_FREEZE.md"}

CITATION_CONTEXT_RE = re.compile(r"\b(doi|arxiv|isbn|issn)\b", re.IGNORECASE)
RETRACTED_CONTEXT_RE = re.compile(r"\b(retracted|superseded|since\s+corrected|was\s+wrong)\b", re.IGNORECASE)


def _manuscript_facing_docs() -> list[str]:
    paths = []
    for name in MANUSCRIPT_FACING_DOC_NAMES:
        p = os.path.join(ROOT, name)
        if os.path.exists(p):
            paths.append(p)
    # The manuscript's own prose, once it exists (module2/paper/sections/*.tex, .../floats/*.tex),
    # mirroring M1's lesson that this must be scanned too, not just the closure-phase docs.
    manuscript_roots = [os.path.join(ROOT, "..", d) for d in ("paper-m2", "paper-m2-v2", "paper_draft")
                        if os.path.isdir(os.path.join(ROOT, "..", d))]  # venue-format manuscript folders
    for manuscript_root in manuscript_roots:
        paths += sorted(glob.glob(os.path.join(manuscript_root, "sections", "*.tex")))
        paths += sorted(glob.glob(os.path.join(manuscript_root, "floats", "*.tex")))
    # supplement.tex is a separate, submitted manuscript document (main.tex's own prose lives in
    # sections/*.tex; supplement.tex carries its own hand-written intro paragraph directly, not
    # split into a sections/ file) -- missed until the second supervisor review, when a leak in its
    # own prose would otherwise have gone unscanned.
    for manuscript_root in manuscript_roots:
        supplement_path = os.path.join(manuscript_root, "supplement.tex")
        if os.path.exists(supplement_path):
            paths.append(supplement_path)
    return sorted(set(paths))


def check_external_citations(sidecars: dict, failures: list) -> None:
    bib_keys: set[str] | None = None
    bib_path = os.path.join(ROOT, "..", "paper-m2", "refs", "m2.bib")
    if os.path.exists(bib_path):
        with open(bib_path, "r", encoding="utf-8") as f:
            bib_text = f.read()
        bib_keys = set(re.findall(r"@\w+\{\s*([^,\s]+)\s*,", bib_text))

    for asset_id, sc in sidecars.items():
        for i, rec in enumerate(sc["records"]):
            if rec["kind"] != "external":
                continue
            if not rec.get("citation") or not rec.get("locus"):
                failures.append(f"[external-citation] {asset_id}#{i}: external() value missing a "
                               f"citation and/or locus -- both are required (paper_data.py docstring)")
            elif bib_keys is not None and rec["citation"] not in bib_keys:
                failures.append(f"[external-citation] {asset_id}#{i}: citation {rec['citation']!r} "
                               f"is not a bibkey in paper/refs/m2.bib")
            if rec["raw"] == "NR" and rec["formatted"] != "NR":
                failures.append(f"[external-citation] {asset_id}#{i}: raw value is 'NR' (not reported) "
                               f"but formatted as {rec['formatted']!r} -- must render as literal 'NR'")


def check_manuscript_wording(failures: list) -> None:
    """Independent of check_qc_labels: that function only scans a TABLE/FLOAT file whose basename
    equals a sidecar's asset_id, so hand-written prose in paper/sections/*.tex (Methods, Results,
    Discussion, ...) was never scanned for the locked negative-result wording or 'pre-registered' --
    a real gap, since those are exactly the files most likely to state the module's headline claim
    in a sentence a reviewer will actually read. Scans every manuscript-facing section/float file
    directly (the same file list check_document_scan already uses for numeric-literal scanning)."""
    for path in _manuscript_facing_docs():
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        lower = text.lower()
        rel = os.path.relpath(path, ROOT)
        for phrase in BANNED_UNSCOPED_PHRASES:
            if phrase in lower:
                failures.append(f"[manuscript-wording] {rel}: contains the banned unscoped phrase "
                               f"{phrase!r} -- use the locked 'not needed under these conditions' wording")
        if "pre-registered" in lower or "preregistered" in lower:
            failures.append(f"[manuscript-wording] {rel}: says 'pre-registered' -- these are internal "
                           f"design-freeze documents, not a public registration; use 'pre-specified'")
        # The style/v2 checks below are scoped to hand-written argumentative prose (sections/,
        # floats/, supplement.tex) only, not to MANUSCRIPT_FACING_DOC_NAMES (M2_FINAL_EVIDENCE_FREEZE.md):
        # that freeze doc legitimately narrates the evidence chain's own history -- superseded
        # artifact filenames like `m2_v2_deltas_pre_tuning_symmetry.json` and the internal fix they
        # name -- which is accurate self-documentation, not a claim a reviewer would read as
        # asserting anything about the paper's methodology.
        if os.path.basename(path) not in MANUSCRIPT_FACING_DOC_NAMES:
            for phrase in BANNED_STYLE_PHRASES:
                if phrase in lower:
                    failures.append(f"[manuscript-wording] {rel}: contains the banned style phrase "
                                   f"{phrase!r} -- see BANNED_STYLE_PHRASES for the replacement")
            if BARE_V2_RE.search(_strip_nonrendered_latex(text)):
                failures.append(f"[manuscript-wording] {rel}: leaks the internal 'v2' pipeline-iteration "
                               f"marker into manuscript-facing text -- use 'primary cohort'/'primary model'/"
                               f"drop the suffix")


def check_document_scan(sidecars: dict, failures: list) -> None:
    all_formatted: set[str] = set()
    for sc in sidecars.values():
        all_formatted |= {rec["formatted"] for rec in sc["records"] if rec["kind"] != "series"}
    all_formatted_bare = {v.replace("\\%", "").replace("$", "").strip() for v in all_formatted}

    for path in _manuscript_facing_docs():
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        lines = text.split("\n")
        for m in DOC_NUMBER_RE.finditer(text):
            token = m.group(0)
            line_no = text.count("\n", 0, m.start())
            if CITATION_CONTEXT_RE.search(lines[line_no]):
                continue
            window = "\n".join(lines[max(0, line_no - 1):line_no + 4])
            if RETRACTED_CONTEXT_RE.search(window):
                continue
            if token in all_formatted or token in all_formatted_bare:
                continue
            if any(abs(float(token.replace(",", "")) - float(a.replace(",", ""))) < 1e-6
                   for a in all_formatted_bare if _numeric_like_simple(a)):
                continue
            failures.append(f"[doc-scan] {os.path.relpath(path, ROOT)}: literal {token!r} does not match any "
                           f"recorded paper-asset value -- verify it traces to a committed artifact")


def main() -> int:
    manifest = load_manifest()
    sidecars = load_sidecars()
    if not sidecars:
        print("NOTE: no provenance sidecars found under paper_assets/provenance/ -- "
              "run generate_latex_tables.py first. Nothing to verify.")
        return 0

    failures: list[str] = []
    check_manifest_integrity(sidecars, manifest, failures)
    check_superseded_and_structural(sidecars, failures)
    check_replay_and_format(sidecars, failures)
    check_token_scan(sidecars, failures)
    check_cross_asset_agreement(sidecars, failures)
    check_qc_labels(sidecars, failures)
    check_manuscript_wording(failures)
    check_external_citations(sidecars, failures)
    check_document_scan(sidecars, failures)

    if failures:
        print(f"--- FAIL: {len(failures)} paper-asset verification failure(s) ---")
        for f in failures:
            print(f"  {f}")
        return 1

    n_records = sum(len(sc["records"]) for sc in sidecars.values())
    print(f"paper-asset verification: PASS ({len(sidecars)} assets, {n_records} recorded values, "
          f"all replayed/reformatted/cross-checked against {len(manifest.get('files', {}))} manifested artifacts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
