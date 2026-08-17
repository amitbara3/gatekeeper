# Gatekeeper

**An experimentation & causal inference workbench, validated against a real A/B test.**

Most teams can run an A/B test. Far fewer can analyse one in a way that survives
scrutiny. Gatekeeper is a typed, tested Python library plus a notebook curriculum
that implements the trustworthy-experimentation toolkit — and then checks its own
work against data whose answer is known.

## The thesis

The Kaggle *Cookie Cats* dataset is a **randomised** experiment (~90,189 players,
progression gate at level 30 vs level 40), so it yields a trustworthy estimate of
the true effect. That estimate becomes **ground truth**.

We then deliberately break randomisation in ways we control — confounded selection,
simulated non-compliance, a withheld confounder — and score each causal-inference
estimator on a question with a known answer:

> Does this method recover the experimental ground truth, and how badly does it fail
> when its assumptions are violated?

That turns "learning causal inference" from reading about identification assumptions
into **measuring what happens when they break**.

## Planning documents

| Document | Contents |
|---|---|
| [PRD.md](PRD.md) | Problem, users, features, success metrics, and an honest account of what this dataset cannot support |
| [Architecture.md](Architecture.md) | Stack and rejected alternatives, the `estimand → estimator → estimate` core, folder structure, testing strategy |
| [Rules.md](Rules.md) | Statistical, honesty, library, and code rules — including the post-treatment-covariate ban |
| [Phases.md](Phases.md) | Ten phases with deliverables and exit criteria |
| [Design.md](Design.md) | Validated colour palette, typography, chart specifications |

## Status

**Phase 0 — scaffolding.** Planning documents complete; no implementation yet.

See [Phases.md](Phases.md) for the plan. Phases 3 (first end-to-end readout) and 6
(the estimator benchmark) are the milestones.

## Planned stack

Python 3.11+ · numpy · pandas · scipy · statsmodels · matplotlib · plotly ·
Streamlit · pytest + hypothesis · ruff · mypy (strict)

Deliberately **not** used: DuckDB, polars, FastAPI/React, PyMC, and `econml`/`dowhy`
in the core path — reasoning in [Architecture.md §1.1](Architecture.md).

## Getting the data

The dataset is **not committed** to this repository. Download `cookie_cats.csv` from
the Kaggle *Cookie Cats* A/B test dataset and place it at `data/raw/cookie_cats.csv`.

## Design principles worth stating up front

- Uncertainty is never optional — no point estimate ships without an interval.
- Plot the *difference*, not the two levels.
- Practical significance is the headline; statistical significance is a footnote.
- Synthetic data is always visibly labelled.
- A failed sanity check is a wall, not a warning.

## Reading path

- Kohavi, Tang & Xu — *Trustworthy Online Controlled Experiments*
- [*Causal Inference for the Brave and True*](https://matheusfacure.github.io/python-causality-handbook/) (free, Python-native)
- Udacity *A/B Testing* (originally Google)
