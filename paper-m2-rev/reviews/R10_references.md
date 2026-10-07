# Round 10: References (2026-10-07)

Same rubric and weights as round 5. Scorer: the drafting assistant.

## What changed (53 cited entries; the 18 uncited entries in the .bib were not touched)
- One DOI style: IEEEtran does not print the `doi` field, so 18 entries with a DOI had printed none while others printed it through a note. Every entry with a
  DOI now prints "doi: ..." the same way; URLs and DOIs are in the body font.
- Metadata added, each value checked at its source:
  - DOIs: Angelopoulos et al. and Ding et al. (NeurIPS 2023, 10.52202/075280-...), El Halabi and Brandt and Keoliya et al. (arXiv DataCite DOIs), Millar et al.
    (medRxiv); all resolve at doi.org.
  - Proceedings URLs where the venue issues no DOI, each checked to load and, where the page is static, to contain the title: PMLR (Horn, Zaffran, Nestor,
    Cordier), NeurIPS proceedings (Gibbs, Tibshirani), OpenReview conference records (Shukla, Wu; confirmed through the OpenReview API as the ICLR records, not
    the DBLP mirrors), the RCP NEWS2 page, and the 2003 Mondrian working paper (first page read: Vovk, Lindsay, Nouretdinov, Gammerman, Working Paper 4).
  - Page ranges from PMLR and NeurIPS metadata; author lists compared with PMLR, arXiv and OpenReview (all matched; OpenAlex had a wrong second author for
    Wu et al., the .bib was right).
- Preprints: Solaiman et al. is now cited in its published version (IEEE ICHI 2026, pp. 1116-1125, doi 10.1109/ICHI69079.2026.00137); two published papers lost
  their "Preprint: arXiv" suffix. Three preprints remain (El Halabi and Brandt, CAREBench, Millar et al.): none has a published version in Crossref, arXiv or
  Semantic Scholar.
- Style: IEEE journal abbreviations (Nat. Mach. Intell., Scand. J. Statist., Crit. Care, PLoS ONE, Chest) and proceedings abbreviations; titles in sentence case.
- Defect found: IEEEtran sentence-cases every title, so unbraced capitals are lower-cased. "non-ICU" had printed as "non-icu" in all earlier builds, and the
  NEWS2 report title lost its capitals. Fixed by bracing; the built reference list was read to confirm.

## References: 9.6 (was 8.4)
| Criterion | W | Credit | Reason |
|---|---|---|---|
| Every entry verified | 3 | 1 | every added or changed value checked at its source |
| Complete metadata | 2 | 1 | every cited entry has a DOI or a verified URL; every article has volume and pages or article number |
| Recent and relevant | 2 | 1 | |
| Consistent IEEE style | 1.5 | 1 | one DOI format, abbreviated venues, sentence-case titles |
| Limited reliance on preprints | 1.5 | 0.75 | three of 53 remain; the Millar preprint supports a comparison in Related Work and the Discussion (labelled as a preprint in the text) |
