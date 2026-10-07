"""
Phase 0, task 0.7 (follow-up) -- re-run the repository's own ECI equivalence tests (T8-T13 in m2_label_lag.run_tests) against the
vendored reference implementation. Read-only: no result file is written, the ECI algorithm is not modified. Prints PASS lines only.
"""
import os, sys
SCRIPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "Benchmark_scripts"))
sys.path.insert(0, SCRIPT_DIR)
import m2_label_lag as ll  # noqa: E402
from m2_mondrian_eci_pilot import PRIMARY_MODEL  # noqa: E402

preds, y, year = ll.get_all_predictions()
m0 = (PRIMARY_MODEL, ll.FINAL_SEEDS[0]); pr = preds[m0]
print("model/seed used for the tests:", m0)
ll.run_tests(pr["cs"], y["cs"], pr["fc"], y["fc"], pr["te"], y["te"])
print("ALL EXISTING ECI EQUIVALENCE TESTS PASSED")
