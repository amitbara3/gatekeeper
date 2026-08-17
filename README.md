# Gatekeeper

**An experimentation & causal inference workbench, validated against a known answer.**

Most teams can run an A/B test. Far fewer can analyse one in a way that survives scrutiny.
Gatekeeper is a typed, tested Python library that implements the trustworthy-experimentation
toolkit — and then checks its own work by measuring itself against effects it already knows.

**710 tests · 94% coverage · `mypy --strict` clean**

---

## The headline result

The project's thesis is that a causal estimator should be judged against a **known**
answer, not against another estimator. So the benchmark builds data with an exact true
effect, breaks randomisation in controlled ways, and scores each method on whether it
recovers the truth.

True ATE = 1.0, 40 replications × 4,000 units, estimators given only the covariate `x`:

| regime | estimator | bias | RMSE | coverage |
|---|---|---:|---:|---:|
| randomised | naive difference | −0.017 | 0.085 | 97.5% |
| randomised | IPW | −0.020 | 0.065 | 100% |
| randomised | outcome regression | −0.020 | 0.065 | 97.5% |
| randomised | AIPW | −0.020 | 0.065 | 97.5% |
| **selection** | **naive difference** | **+1.854** | **1.857** | **0%** |
| selection | IPW | −0.004 | 0.131 | 100% |
| selection | outcome regression | −0.028 | 0.115 | 95.0% |
| selection | AIPW | −0.029 | 0.125 | 90.0% |
| **unobserved** | naive difference | +1.850 | 1.854 | 0% |
| **unobserved** | IPW | +1.829 | 1.831 | 0% |
| **unobserved** | outcome regression | +1.829 | 1.831 | 0% |
| **unobserved** | **AIPW** | **+1.829** | **1.831** | **0%** |

**Read the last block carefully.** Under unobserved confounding, the doubly robust
estimator is no better than doing nothing. Supply the hidden confounder and the same three
methods land at −0.0003, +0.0001, and −0.0014. The failure was **missing data, not
inadequate methods**.

That is the whole point: double robustness protects against *misspecification*, not
*omission*. It is routinely oversold as though it were the latter.

## The peeking problem, measured

True effect zero, α = 0.05, 3,000 simulated experiments:

| looks | naive (stop at first p < 0.05) | always-valid (mSPRT) |
|---|---:|---:|
| 1 | 5.2% | 0.3% |
| 5 | 13.9% (2.8×) | 0.9% |
| 10 | 18.7% (3.7×) | 1.2% |
| 50 | 31.1% (6.2×) | 1.7% |

Reading once at the end gives 5.23%, confirming the harness is unbiased rather than the
inflation being a simulation artefact. The always-valid rate sits *well below* α, so it
buys its guarantee with real power — that is the honest trade for unrestricted monitoring,
not a free lunch.

## CUPED variance reduction

| ρ | ρ² (theory) | achieved | CI width vs. plain | effective n |
|---|---:|---:|---:|---:|
| 0.3 | 0.090 | 0.090 | 0.95× | 1.10× |
| 0.5 | 0.250 | 0.251 | 0.87× | 1.33× |
| 0.7 | 0.490 | 0.492 | 0.71× | 1.97× |
| 0.9 | 0.810 | 0.812 | 0.43× | 5.31× |

---

## What makes this different from a tutorial

**Rules are enforced in code, not documented in prose.** The statistical rules in
[Rules.md](Rules.md) are mechanically enforced:

- `EffectEstimate` cannot be constructed without declaring its `assumptions` and whether
  the data was real or synthetic.
- `estimate_cuped` **raises** if handed a post-treatment covariate — and the guard fires
  before any arithmetic, so the error is about the analysis plan.
- The spec refuses an underpowered plan (`mde > practical_threshold`), which could not
  answer its own question.
- Multiplicity correction refuses a metric family that doesn't exactly match the
  pre-registered one, because dividing by *m* is meaningless if *m* grows as you look.
- A fixed-horizon spec raises on a second look.
- The palette refuses to cycle past three series rather than silently reusing a colour.

**Calibration, not just tests.** Every estimator is checked by simulation: error rates
match α at α ∈ {0.01, 0.05, 0.10}, and 95% intervals cover at 93–97% — on normal, skewed,
and binary data. That layer caught things the others couldn't, and it also caught cases
where *the test* was wrong rather than the code (see the retrospective).

**Honest about what the data cannot support.** Cookie Cats has no pre-experiment covariate
and no timestamps, so CUPED, DiD, and real sequential accrual are impossible on it. Those
are demonstrated on synthetic data with known ground truth and labelled `SYNTHETIC`
everywhere. A test asserts the dataset has zero usable covariates, so if that ever changes
the conclusion gets revisited rather than silently inherited.

---

## Quick start

```bash
python -m venv .venv --system-site-packages
.venv/Scripts/python -m pip install -e ".[dev,viz,ml,app]"   # or bin/python on POSIX
.venv/Scripts/python -m pytest -m "not slow"                 # fast suite
.venv/Scripts/python -m streamlit run app/streamlit_app.py   # the readout app
```

The whole suite runs on synthetic data, so **no download is required** to build or test.

### Using the real dataset

Download `cookie_cats.csv` from the Kaggle *Cookie Cats* A/B test dataset to
`data/raw/cookie_cats.csv`. Nothing under `data/` is ever committed.

```python
from gatekeeper import load_cookie_cats, load_spec, run_sanity_checks
from gatekeeper.report import build_readout

spec = load_spec("specs/cookie_cats_gate.yaml")
data = load_cookie_cats()
sanity = run_sanity_checks(
    data, expected_shares=spec.expected_shares, srm_threshold=spec.srm_threshold
)
sanity.raise_if_failed()  # the gate: blocks loudly, overrides are recorded
```

## Layout

| Module | Contents |
|---|---|
| `types.py` | `EffectEstimate`, `Estimand`, `SanityReport` — the uniform contract every estimator returns |
| `spec.py` | Pre-registration, **enforced** rather than advisory |
| `data/` | Strict schema validation, Parquet cache, synthetic generators with exact ground truth |
| `design/` | Power, sample size, MDE (by inverting one power function), SRM |
| `checks/` | The sanity gate: SRM, uniqueness, cross-arm leakage, outlier leverage |
| `frequentist/` | Two-proportion z, Welch, bootstrap (percentile + BCa), delta-method ratios, BH/Bonferroni |
| `sequential/` | Always-valid p-values (mSPRT) and the peeking simulation |
| `variance/` | CUPED, with the post-treatment guard |
| `bayesian/` | Beta-Binomial conjugate posterior, `P(B>A)` by quadrature, expected-loss decisions |
| `causal/` | Confounding simulator, propensity/IPW, outcome regression, AIPW, E-values |
| `hte/` | S/T/X-learners, Qini and uplift curves |
| `benchmark/` | The harness and scoring behind the table above |
| `report/`, `viz/` | Decision logic, self-contained HTML/Markdown, the validated palette |

## Documents

| | |
|---|---|
| [PRD.md](PRD.md) | Problem, users, features, and an explicit account of what this dataset cannot support |
| [Architecture.md](Architecture.md) | Stack with rejected alternatives, the `estimand → estimator → estimate` core, testing strategy |
| [Rules.md](Rules.md) | The statistical and honesty rules the code enforces |
| [Phases.md](Phases.md) | Ten phases with exit criteria and what actually happened |
| [Design.md](Design.md) | The validated palette, typography, chart specifications |
| [RETROSPECTIVE.md](RETROSPECTIVE.md) | What the benchmark showed, and every bug worth learning from |

## Stack

Python 3.11+ · numpy · pandas · scipy · statsmodels · scikit-learn (nuisance models only) ·
Streamlit · pytest + hypothesis · ruff · mypy strict

Deliberately **not** used, with reasoning in [Architecture.md §1.1](Architecture.md):
DuckDB, polars, FastAPI/React, PyMC, and `econml`/`dowhy` in the core path.

## Still open

- **PRD O4** — whether the real split clears the SRM threshold. The check is built and
  cross-validated against scipy to 1e-9, but has never touched real data.
- **PRD O5** — the spec carries one practical threshold in the primary metric's units,
  which is dimensionally meaningless for a guardrail measured in different units. Guardrails
  are judged on adjusted statistical significance instead; per-metric thresholds are the
  proper fix.
- **The notebooks** — the curriculum was deferred in favour of getting the library right.
  Every result above is reproducible from the test suite.
- **Alpha-spending** (O'Brien–Fleming, Pocock) — deferred in favour of the mSPRT, which is
  closed-form and directly verifiable by simulation.
