# Author review checklist for the ICASSP 2027 draft

The active four-page PDF is a compact English rewrite of the project's existing
Chinese manuscript, plans, code descriptions, and frozen result summaries. It
is ready for technical author review, not for direct submission.

Before calling it submission-ready, the authors should:

- confirm that the problem statement and three contributions express their own
  intended idea, especially the distinction from EcoServe;
- verify every configuration and number against `paper/data/manifest.json` once
  that manifest is complete;
- confirm the actual T6 selector settings, overrides, attention backend, graph
  mode, concurrency and KV-cache configuration;
- decide whether the fixed 2,048-token vLLM-Ascend CP baseline is sufficiently
  fair without chunk-budget sensitivity;
- inspect the MaaS cohort denominator and unmatched telemetry treatment;
- replace `authors.tex` placeholders with the exact author order and
  affiliations, and collect an ORCID for every author;
- check the title, 100--150 word abstract, and up to five index terms against
  the online submission fields;
- edit and approve every paragraph in their own voice and take responsibility
  for all claims, citations, figures, and code under the ICASSP 2027 policy.
