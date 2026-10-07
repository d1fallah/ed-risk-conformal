"""
Recorded literature search for the related-work claims of the revision manuscript (public bibliographic metadata only).
Source: OpenAlex works search (title and abstract index), restricted to publication years 2015 onward, English, sorted by relevance;
for each query the total hit count and the first 60 records (title, first author, year, venue, DOI) are stored in results/revision_v2/phase2/p2_14_literature_search.json
together with the date of the search. The records are screened by hand afterwards; the screening decision is written to the manuscript's supplement, not here.
"""
import datetime
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.path.join(ROOT, "results", "revision_v2", "phase2", "p2_14_literature_search.json")
QUERIES = {
    "Q1_workflow_shortcut_ed": '("emergency department" OR "emergency departments") AND (leakage OR shortcut OR "measurement frequency" OR "measurement counts" OR "measurement patterns" OR "informative missingness" OR "informative presence") AND (prediction OR "risk model")',
    "Q2_conformal_ed_or_deterioration": '("conformal prediction" OR "conformal inference") AND ("emergency department" OR "clinical deterioration" OR "patient deterioration" OR "intensive care" OR "triage")',
    "Q3_conformal_class_conditional_clinical": '("conformal prediction" OR "conformal inference") AND ("class-conditional" OR "class conditional" OR "Mondrian" OR "minority class" OR "imbalanced") AND (clinical OR patients OR mortality OR "electronic health")',
    "Q4_online_conformal_clinical": '("adaptive conformal" OR "online conformal" OR "conformal prediction") AND ("distribution shift" OR "temporal shift" OR "non-stationary" OR "nonstationary" OR "covariate shift") AND (clinical OR patients OR hospital OR "electronic health" OR "MIMIC")',
    "Q6_monitoring_pattern_leakage_clinical_prediction": '("informative missingness" OR "informative presence" OR "measurement frequency" OR "measurement patterns" OR "monitoring frequency" OR "monitoring intensity" OR "label leakage" OR "temporal leakage" OR "data leakage") AND ("clinical prediction" OR "risk prediction" OR "early warning" OR "deterioration" OR "mortality prediction")',
    "Q7_ed_prediction_time_landmark": '("emergency department" OR "emergency departments") AND ("prediction time" OR "time of prediction" OR "landmark" OR "first hour" OR "early prediction" OR "available at triage") AND (deterioration OR "critical care" OR "ICU admission" OR "in-hospital mortality" OR "critical illness") AND ("machine learning" OR "deep learning" OR "prediction model")',
    "Q5_conformal_delayed_feedback": '("conformal prediction" OR "conformal inference") AND ("delayed feedback" OR "delayed labels" OR "label delay" OR "delayed label" OR "feedback delay")',
}


def fetch(q):
    params = {"filter": "title_and_abstract.search:" + q + ",from_publication_date:2015-01-01,language:en", "per-page": "60", "select": "id,doi,title,publication_year,primary_location,authorships,cited_by_count"}
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params)
    for attempt in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "m2-literature-search"}), timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise
            time.sleep(10 * (attempt + 1))
    raise RuntimeError("rate limited")


def main():
    res = {"date": datetime.date.today().isoformat(), "source": "OpenAlex works search", "filter": "publication year >= 2015, English", "queries": {}}
    for k, q in QUERIES.items():
        d = fetch(q)
        recs = []
        for w in d["results"]:
            loc = (w.get("primary_location") or {}).get("source") or {}
            au = (w.get("authorships") or [{}])[0].get("author", {}).get("display_name") if w.get("authorships") else None
            recs.append({"title": w.get("title"), "first_author": au, "year": w.get("publication_year"), "venue": loc.get("display_name"), "doi": w.get("doi")})
        res["queries"][k] = {"query": q, "total_hits": d["meta"]["count"], "records": recs}
        print(k, d["meta"]["count"], flush=True)
        time.sleep(5)
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
