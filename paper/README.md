# `paper/` — CoRL 2026 submission draft

## Files

| file              | purpose                                                                      |
|-------------------|------------------------------------------------------------------------------|
| `main.tex`        | **canonical source** — full LaTeX paper, bibliography, TikZ overview figure   |
| `references.bib`  | bibliography (real published citations + 1 placeholder for concurrent work)   |
| `Makefile`        | `make` -> compile via `pdflatex + bibtex + pdflatex × 2`                       |
| `main.md`         | markdown mirror of `main.tex` for quick reading without LaTeX                  |

## Status (Day 1, May 2, 2026)

- **Methods & system sections:** complete and locked.
- **Experimental protocol:** pre-registered before data collection.
- **System validation results (§7.1):** complete — all 15 Day-1 sanity
  checks pass; gradient-norm table populated.
- **Headline empirical results (§7.2):** **TBD on Day 14.** The result
  table structure is locked; the cells fill once the 100-episode evals
  run.
- **Figures:** the architecture overview is rendered in TikZ and
  compiles inline. Sensor-importance heatmap and qualitative ToF figures
  are Day 12–13 deliverables.

## Why pre-registration?

The paper makes one falsifiable headline claim — PLA > VLM-only ACT on
the near-contact task with paired p < 0.05. Pre-registering the tasks,
sample size, statistical procedure, and acceptance criterion *before*
the experiment runs prevents post-hoc rationalisation if the data
disagree with the prediction. We commit to reporting the headline
number whether or not it is significant, and to declaring deviations
from this protocol explicitly.

See `docs/STATISTICAL_PROTOCOL.md` for the full protocol; this paper's
§6 is its prose form.

## Build

```bash
# Full compile with bibliography:
make

# Single-pass compile (no bib refresh):
make quick

# Watch / rebuild (requires latexmk):
make watch

# Word count:
make wordcount

# Clean:
make clean
```

If you don't have a LaTeX toolchain, `main.md` is the readable mirror.
The two should agree on content; if they drift, `main.tex` is canonical
and `main.md` should be regenerated.

## Length budget (CoRL)

CoRL caps the main paper at 8 pages excluding references and
appendices. The current draft is well under this — the appendices
(timeline, variants, hyperparameters) are deliberately on the back
matter so the front matter stays readable. Once the headline numbers
land we expect to add ~1 page of results and discussion to bring the
front matter to ~7.5 pages.

## Anonymisation

The submission is double-blind. The `\author{}` block is anonymised in
`main.tex`. References to the lab's own prior work (e.g. GenTact,
Roncone et al. 2016 on peripersonal space) are cited in third person.
The repository URL in the reproducibility statement will be a fresh
anonymised release; the working repository here is *not* what the
reviewer sees.

## What's missing on purpose

- **Real numbers in Table 2.** Day-14 deliverable. The placeholders
  are explicit.
- **Sensor-importance heatmap figure.** Depends on the post-hoc sweep
  on a trained PLA checkpoint; Day 11 deliverable.
- **Qualitative ToF heatmap sequence (far / mid / near / pre-grasp).**
  Depends on a real near-contact rollout to extract; Day 12 deliverable.
- **Concurrent work citation.** `tactilevla2025` is a placeholder;
  replace with the appropriate citation by Day 22 once the related-work
  search is finalised.
