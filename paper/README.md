# PD-TDM paper workspace

Current milestone (D-020, 2026-09-07): restart today and freeze the complete English
R0 by **2026-09-09 22:00 Asia/Shanghai**. See [TASKS.md](TASKS.md) and
[the readiness audit](READINESS_AUDIT_2026-09-07.md). The official ICASSP 2027
layout is integrated. The active manuscript is an English four-page review
draft, including references. Author names and affiliations remain placeholders.

This directory is the authoritative writing workspace for the PD-TDM ICASSP
2027 submission. `TASKS.md` is the execution queue and `WRITING_PLAN.md`
defines the four-page paper structure.

## Build

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The PDF is written to `paper/build/main.pdf`.

If GNU Make is available, `make`, `make watch`, and `make clean` provide
shortcuts for build, continuous preview, and cleanup.

## Layout

- `main.tex`: document assembly and section order
- `preamble.tex`: packages and formatting
- `macros.tex`: title, system names, and repeated terminology
- `sections/`: compact English manuscript; implementation is merged into design,
  related work into background, and limitations into evaluation
- `figures/`: figures referenced by the active manuscript
- `references.bib`: verified bibliography entries
- `MATERIALS.md`: authoritative material inventory and paper-use policy
- `CLAIMS.md`: claim--evidence ledger and prohibited overclaims
- `TASKS.md`: dependency-ordered execution plan and completion criteria
- `WRITING_PLAN.md`: section-level outline, evidence mapping, and page budget

The active entry uses `article` + the unmodified official `spconf.sty` and
`IEEEbib.bst` downloaded from the ICASSP 2027 Paper Kit. Source URLs and SHA-256
hashes are in `template/official/SOURCES.json`; the untouched official example
is `template/official/Template.tex`. Its first comment still says ICASSP-2026,
but it is the file currently linked by the 2027 Paper Kit and its title says 2027.

`authors.tex` contains explicit draft placeholders; replace them with real names
and affiliations. Superseded plans, the former Chinese-template entry files,
unused figure candidates, and the deferred VLM report are preserved under
`../memory/archive/paper_cleanup_2026-09-07/`.

The build resolves citations and references, embeds all fonts, uses US Letter,
and is four pages. It is a technical review draft rather than a submission-ready
manuscript: authors must review every statement and figure, finish the numerical
manifest/fairness audit, and replace author placeholders. GNU Make is absent in
the current environment, so use the direct `latexmk` command above.

## Draft language

The archived Chinese draft remains source material. The active paper is an
English four-page rewrite around the ICASSP storyboard.

## Conventions

1. Every numerical claim must map to a frozen result path and analysis script
   in `TASKS.md`.
2. Cite only m31-fix results from `results/m31fix_validate/`.
3. Distinguish raw measurements from post-hoc SLO reclassification.
4. Mark unresolved prose with `\paperTODO{...}`.
5. Keep historical research notes in `memory/`; keep the final paper narrative
   here.
6. Treat phase ratio as a scheduling tendency rather than a strict wall-clock
   share; do not claim the current controller finds the online optimum.

## Submission dates (verified 2026-09-07)

No separate abstract deadline is listed for regular papers in the official
[dates](https://2027.ieeeicassp.org/important-dates/),
[Paper Kit](https://cmsworkshops.com/ICASSP2027/papers/paper_kit.php), or submission
instructions. Abstract text is entered with the paper submission. The kit suggests
100–150 words; the initial submission form caps it at 200 words.

The official [submission page](https://cmsworkshops.com/ICASSP2027/papers.php)
explicitly gives **2026-09-17 20:00 Beijing time**, corresponding to September 16
24:00 at the International Date Line. Keep the internal September 15 target.
