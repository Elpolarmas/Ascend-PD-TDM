# Frozen paper data

This directory will contain small, generated, paper-facing aggregates and the
authoritative `manifest.json`. It must not duplicate large raw experiment
directories from `results/`.

Every manifest entry should record:

- claim/figure/table identifier;
- raw input path;
- config and paper-facing paradigm name;
- model, hardware, seed, arrival mode, warm-up, and measurement window;
- raw SLO or post-hoc SLO rule;
- generating script and command;
- source commit;
- caveats such as timeout filtering or single-seed status.

`figure_metrics.json` is the current machine-readable output of the paper
aggregation workflow. It records T6 headline cells, the F5 matrix values and
caveats, and the MaaS values recomputed from raw requests. The MaaS-specific
reaggregation is reproducible with:

```bash
python3 experiments/posthoc_maas_paper.py
```

It uses the active compressed arrival interval `[30,4560)` seconds and counts
unmatched telemetry as a failed SLO attainment.
