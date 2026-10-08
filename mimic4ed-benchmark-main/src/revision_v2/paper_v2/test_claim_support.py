"""Regression tests for check 5 of verify_assets_v2.py (claim support of the submitted Abstract, Discussion and Conclusion).

Each test builds a small synthetic paper folder (no manuscript text, no data) and runs submitted_support_check on it. The three
failures that the check found in the real manuscript are kept as permanent cases:
  1. a Discussion sentence about landmark times whose numbers appear only in the extended version;
  2. a Limitations number (a share of ineligible stays) that no submitted section or table shows;
  3. a sentence that points only to the extended version, with no number.
A clean fixture, an unsigned alias of a signed macro, and a number shown only in a table must pass.

Run from this folder:  python -m unittest test_claim_support -v
"""
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify_assets_v2 as V  # noqa: E402

# (macro suffix, value, artifact pointer, format) in the order of rv_numbers.tex
MACROS = [
    ("AucOne", "0.905", "/a/auc"),
    ("GainSigned", "+0.027", "/a/gain"),
    ("GainU", "0.027", "/a/gain"),  # same source value as GainSigned, printed without the sign
    ("TableOnly", "0.512", "/d/t"),
    ("FixOne", "0.821", "/b/one"),
    ("FixSix", "0.860", "/b/six"),
    ("IneligPct", "6.8\\%", "/c/inelig"),
    ("Eligible", "78,492", "/c/elig"),
    ("Stays", "84,241", "/c/stays"),
]

ABSTRACT = "\\begin{abstract}\nThe AUROC was \\RVAucOne{}.\n\\end{abstract}\n"
RESULTS = (
    "\\section{Results}\n\\label{sec:results}\n"
    "The AUROC was \\RVAucOne{}, a gain of \\RVGainSigned{}; \\RVEligible{} of \\RVStays{} test stays were eligible (Table~\\ref{tab:x}).\n"
    "\\input{tab_x}\n"
)
DISCUSSION = "\\section{Discussion}\nThe AUROC was \\RVAucOne{}, and the two hours add a further \\RVGainU{} (Section~\\ref{sec:results}, Table~\\ref{tab:x}).\n"
CONCLUSION = "\\section{Conclusion}\nThe AUROC was \\RVAucOne{}.\n"
TABLE = "\\begin{table}\n\\caption{Values.}\n\\label{tab:x}\n\\begin{tabular}{l}\n0.512 \\\\\n\\end{tabular}\n\\end{table}\n"


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)


class ClaimSupport(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="claim_support_")
        self.paper = os.path.join(self.root, "paper")
        self.prov = os.path.join(self.root, "prov")
        write(os.path.join(self.paper, "main.tex"), "\\input{sections/00_abstract}\n\\input{sections/04_results}\n\\input{sections/05_discussion}\n\\input{sections/06_conclusion}\n")
        write(os.path.join(self.paper, "supplement.tex"), "\\section{Supplement}\n")
        write(os.path.join(self.paper, "rv_numbers.tex"), "".join("\\newcommand{\\RV%s}{%s}\n" % (n, v) for n, v, _ in MACROS))
        write(os.path.join(self.prov, "rv_numbers.json"), json.dumps({"asset_id": "rv_numbers", "records": [
            {"artifact": "synthetic.json", "pointer": ptr, "reduce_how": None, "formatted": v} for _n, v, ptr in MACROS]}))
        write(os.path.join(self.paper, "floats", "tab_x.tex"), TABLE)
        self.sections = {"00_abstract": ABSTRACT, "04_results": RESULTS, "05_discussion": DISCUSSION, "06_conclusion": CONCLUSION}
        self.write_sections()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def write_sections(self):
        for name, text in self.sections.items():
            write(os.path.join(self.paper, "sections", name + ".tex"), text)

    def run_check(self):
        self.write_sections()
        found = []
        V.submitted_support_check(paper=self.paper, prov_dir=self.prov, report=lambda ok, msg: None if ok else found.append(msg))
        return found

    def test_clean_fixture_passes(self):
        # a macro used in Results, an unsigned alias of a signed macro, a number shown only in a table, and resolving references
        self.sections["05_discussion"] += "A table value of \\RVTableOnly{} is shown in Table~\\ref{tab:x}.\n"
        self.assertEqual(self.run_check(), [])

    def test_landmark_time_sentence_is_flagged(self):
        # case 1: numbers that only the extended version contains, with a pointer to it
        self.sections["05_discussion"] += "On one fixed set of stays the AUROC rose from \\RVFixOne{} at one hour to \\RVFixSix{} at six hours (extended version).\n"
        found = self.run_check()
        self.assertTrue(any("RVFixOne" in m for m in found), found)
        self.assertTrue(any("RVFixSix" in m for m in found), found)
        self.assertTrue(any("pointer to the extended version" in m for m in found), found)

    def test_ineligible_share_is_flagged(self):
        # case 2: a number that no submitted section or table shows, although the numbers it derives from are shown
        self.sections["05_discussion"] += "The cohort excludes stays that ended early (\\RVIneligPct{} of test-period stays).\n"
        found = self.run_check()
        self.assertEqual(len(found), 1, found)
        self.assertIn("RVIneligPct", found[0])

    def test_sentence_pointing_only_to_the_extended_version_is_flagged(self):
        # case 3: no number at all, the claim rests on the extended version
        self.sections["05_discussion"] += "Discrimination by calendar year was similar in every year (extended version).\n"
        found = self.run_check()
        self.assertEqual(len(found), 1, found)
        self.assertIn("pointer to the extended version", found[0])

    def test_unresolved_reference_is_flagged(self):
        self.sections["06_conclusion"] += "See Table~\\ref{tab:missing}.\n"
        found = self.run_check()
        self.assertEqual(len(found), 1, found)
        self.assertIn("tab:missing", found[0])


if __name__ == "__main__":
    unittest.main()
