"""
Independent re-verification of the revision manuscript assets.

1. Every file in the evidence folder matches the SHA-256 of the manifest, and each manifest source file still hashes the same (so the
   manuscript numbers were generated from the current results).
2. Every provenance record (paper_assets_v2/provenance/*.json) is replayed: the raw value is re-read from the frozen artifact by its pointer
   (or re-derived by its reduction) and its formatted string recomputed; both must equal the record.
3. Every macro value in rv_numbers.tex equals the formatted string of its record, and every formatted string recorded for a table appears in
   that table's .tex file (so no cell was typed or edited after generation).
4. Every \\RV macro used in the manuscript sections is defined, and no digit string of four or more digits or decimal number appears in the prose
   outside macros, tables, citations, and a short allow-list (dates, years, the pre-specified constants).
Exit status 1 on any failure.
"""
import glob
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_data_v2 as PD  # noqa: E402

PAPER = os.path.abspath(os.path.join(PD.ROOT, "..", "paper-m2-rev"))
fails = []


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check(c, msg):
    if not c:
        fails.append(msg)


def main():
    man = json.load(open(os.path.join(PD.EVIDENCE_DIR, "MANIFEST.json")))
    for name, rec in man["files"].items():
        p = os.path.join(PD.EVIDENCE_DIR, name)
        check(os.path.exists(p) and sha(p) == rec["sha256"], f"evidence file changed or missing: {name}")
        src = rec["source"]
        sp = os.path.join(PD.ROOT, src) if not src.startswith("results/") else os.path.join(PD.ROOT, src)
        if os.path.exists(sp):
            check(sha(sp) == rec["sha256"], f"source file changed after freeze: {src}")
    n_rec = 0
    texts = {}
    for sidecar in sorted(glob.glob(os.path.join(PD.PROVENANCE_DIR, "*.json"))):
        d = json.load(open(sidecar))
        asset = d["asset_id"]
        tex_path = os.path.join(PAPER, "rv_numbers.tex") if asset == "rv_numbers" else os.path.join(PAPER, "floats", asset + ".tex")
        is_fig = asset.startswith("fig_")
        tex = open(tex_path, encoding="utf-8").read() if os.path.exists(tex_path) else ""
        texts[asset] = tex
        for r in d["records"]:
            n_rec += 1
            if r["kind"] == "value":
                raw = PD.resolve_pointer(PD._load_json(r["artifact"]), r["pointer"])
                check(raw == r["raw"], f"{asset}: raw value differs for {r['artifact']}{r['pointer']}")
                check(PD._fmt(raw, r["fmt"]) == r["formatted"], f"{asset}: formatted differs for {r['pointer']}")
                f = r["formatted"]
                if asset == "rv_numbers":
                    check(("{" + f + "}") in tex, f"rv_numbers.tex lacks value {f} ({r['pointer']})")
                elif not is_fig and "not displayed" not in (r.get("note") or "") and r["formatted"] not in tex:
                    check(False, f"{asset}.tex lacks formatted value {r['formatted']} ({r['pointer']})")
            elif r["kind"] == "series":
                vals = PD.pdm.resolve_wildcard(PD._load_json(r["artifact"]), r["pointer"])
                check(vals == r["raw"], f"{asset}: series differs for {r['pointer']}")
            elif r["kind"] == "external":
                check(bool(r["citation"]) and bool(r["locus"]), f"{asset}: external value without citation and locus")
                check(("{" + r["formatted"] + "}") in tex, f"rv_numbers.tex lacks external value {r['formatted']}")
            elif r["kind"].startswith("reduce:"):
                ev = PD.EV("verify")
                raw = ev.reduce(r["kind"].split(":")[1], r["artifact"], r["pointer"], fmt=r["fmt"])
                rec = ev.records[-1]
                check(abs(rec.raw - r["raw"]) < 1e-12, f"{asset}: reduction differs for {r['pointer']}")
                check(rec.formatted == r["formatted"], f"{asset}: reduction format differs for {r['pointer']}")
                f = r["formatted"]
                if asset == "rv_numbers":
                    check(("{" + f + "}") in tex, f"rv_numbers.tex lacks value {f}")
                elif not is_fig and "not displayed" not in (r.get("note") or "") and r["formatted"] not in tex:
                    check(False, f"{asset}.tex lacks formatted value {r['formatted']} ({r['pointer']})")
            else:
                check(False, f"{asset}: unsupported record kind {r['kind']}")
    # macros used vs defined
    defined = set(re.findall(r"\\newcommand\{\\(RV[A-Za-z]+)\}", open(os.path.join(PAPER, "rv_numbers.tex"), encoding="utf-8").read()))
    used = set()
    # both versions: the submitted paper and the extended version (paper-m2-rev/extended), which share floats/ and rv_numbers.tex
    section_files = glob.glob(os.path.join(PAPER, "sections", "*.tex")) + glob.glob(os.path.join(PAPER, "extended", "sections", "*.tex"))
    for f in section_files + glob.glob(os.path.join(PAPER, "floats", "*.tex")) + [os.path.join(PAPER, "main.tex"), os.path.join(PAPER, "extended", "main.tex")]:
        used |= set(re.findall(r"\\(RV[A-Za-z]+)", open(f, encoding="utf-8").read()))
    for m in sorted(used - defined):
        check(False, f"macro used but not defined: \\{m}")
    # hand-typed numbers in prose
    # 20 = minimum events per evaluated stratum (p2_16); 0.40 = upper end of the decision-curve grid (p2_13); both fixed in code before running.
    allow = re.compile(r"^(0\.(05|10|20|01|89|90|40)|0\.9|0\.80|0\.95|0\.99|2[0-9]{3}|10|20|24|12|18|42|50|65|90|95|100|128|3050|2\.13|3\.4|1\.9|1\.5|60|240|1,?000|2,?000|10,?000|40,?000)$")
    for f in section_files:
        if os.path.basename(f) == "supp_tripod.tex":
            continue  # reporting checklist: its digits are TRIPOD+AI item numbers, and it holds no result
        for i, line in enumerate(open(f, encoding="utf-8").read().split("\n"), 1):
            if line.lstrip().startswith("%") or "includegraphics" in line:
                continue
            s = line.replace("{,}", ",")
            s = re.sub(r"\\RV[A-Za-z]+", "", s)
            s = re.sub(r"\\(cite|ref|label|input|eqref|mc)\{[^}]*\}", "", s)  # \mc: supplement citation by key
            s = re.sub(r"\$[^$]*\$", "", s)
            s = re.sub(r"\\[A-Za-z]+\{[^}]*\}", lambda m: m.group(0) if "\\" not in m.group(0)[1:] else "", s)
            for tok in re.findall(r"(?<![A-Za-z\\{])\d[\d,\.]*\d|\b\d\b", s):
                if not allow.match(tok) and not re.match(r"^\d$", tok):
                    check(False, f"{os.path.basename(f)}:{i}: possible hand-typed number {tok!r}")
    print(f"records replayed: {n_rec}; macros defined {len(defined)}, used {len(used)}; section files scanned {len(section_files)}")
    for m in fails[:80]:
        print("FAIL:", m)
    print("RESULT:", "FAIL" if fails else "PASS", f"({len(fails)} problems)")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
