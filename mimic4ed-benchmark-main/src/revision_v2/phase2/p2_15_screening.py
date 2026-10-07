"""
Screening of the recorded literature search (p2_14_literature_search.json). Title-level screening by the author; decisions are the lists below (substrings of titles).
Two levels per query: `related` = same clinical or methodological area as the claim the query tests; `meets` = reports the specific analysis the manuscript says it did not find.
Outputs counts only; the manuscript's claims about what was and was not found rest on these counts.
"""
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
D = os.path.join(ROOT, "results", "revision_v2", "phase2")
RELATED = {
    "Q1_workflow_shortcut_ed": ["Diagnostic Codes in AI Prediction Models", "Machine learning-based model for triage-stage prediction of emergency department disposition",
                                "Explainable Machine Learning Model for Early Hospital Admission Prediction", "DeepTriage-CN"],
    "Q2_conformal_ed_or_deterioration": ["uncertainty-aware machine learning model for predicting patient disposition", "Conformal selective prediction with cost aware deferral",
                                         "Understanding uncertainty in large language model predictions of early death", "Conformal reliability assessment of PPG-based ICU arrhythmia",
                                         "SENTINEL: A cascade-aware temporal framework", "Distribution-Free Uncertainty Quantification in Mechanical Ventilation"],
    "Q3_conformal_class_conditional_clinical": ["Reliable Uncertainty Under Class Imbalance", "Development and Validation of an Interpretable Conformal Predictor to Predict Sepsis Mortality",
                                                "Improving Mortality Prediction in Adults with Kidney Failure", "Reliable classification of imbalanced lung cancer data",
                                                "Conformal reliability assessment of PPG-based ICU arrhythmia", "Uncertainty-aware sepsis survival prediction using conformal XGBoost",
                                                "Understanding uncertainty in large language model predictions of early death"],
    "Q4_online_conformal_clinical": ["Conformal selective prediction with cost aware deferral", "Reliable Uncertainty Under Class Imbalance", "Conformal Prediction in Clinical Artificial Intelligence",
                                     "Estimating Test Performance for AI Medical Devices under Distribution Shift", "Making Conformal Predictors Robust in Healthcare Settings",
                                     "Safe and reliable transport of prediction models"],
    "Q5_conformal_delayed_feedback": [],
    "Q6_monitoring_pattern_leakage_clinical_prediction": ["Informative presence and observation in routine health data", "Informative missingness in electronic health record systems",
                                                         "A framework for understanding label leakage", "Preserving Informative Presence", "Incorporating informatively collected laboratory data",
                                                         "MADE: leveraging informative missingness", "Diagnostic Codes in AI Prediction Models", "Imputation and missing indicators for handling missing data",
                                                         "Beyond the AUROC: data leakage"],
    "Q7_ed_prediction_time_landmark": ["Early prediction of in-hospital deterioration after emergency department", "Deep-Learning Approaches to Identify Critically Ill Patients at Emergency Department Triage",
                                       "Early prediction of intensive care unit admission in emergency department patients", "Early Prediction of Patient Disposition for Emergency Department",
                                       "Emergency department early mortality model", "In-Hospital Mortality Prediction Model for Critically Ill Older Adult",
                                       "Early prediction of clinical deterioration in older adults with suspected infection"],
}
MEETS = {k: [] for k in RELATED}  # none of the screened records reports the specific analysis (decided by reading titles and, where present, abstracts)


def main():
    d = json.load(open(os.path.join(D, "p2_14_literature_search.json"), encoding="utf-8"))
    out = {"date": d["date"], "source": d["source"], "queries": {}}
    for k, v in d["queries"].items():
        titles = {(r["title"] or "").lower() for r in v["records"]}
        rel = sum(1 for sub in RELATED[k] if any(sub.lower() in t for t in titles))
        out["queries"][k] = {"total_hits": v["total_hits"], "screened": len(v["records"]), "related": rel, "meets": len(MEETS[k])}
    json.dump(out, open(os.path.join(D, "p2_15_screening.json"), "w", encoding="utf-8"), indent=1)
    print({k: (v["total_hits"], v["screened"], v["related"], v["meets"]) for k, v in out["queries"].items()}, out["date"])


if __name__ == "__main__":
    main()
