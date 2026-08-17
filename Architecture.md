# Architecture — Gatekeeper

**Last updated** 2026-08-17 · Companion to [PRD.md](PRD.md) and [Rules.md](Rules.md)

---

## 1. Stack, and why

The deciding constraint: both primary references (*Causal Inference for the Brave
and True*, and the Kohavi CUPED/sequential material) are Python-native, the dataset
is a ~4 MB CSV, and the deliverable is statistical correctness rather than a
product surface. So this is a **Python library with a thin app on top**, not a web
application.

| Layer | Choice | Why this, not the alternative |
|---|---|---|
| Language | **Python 3.11+** | The causal-inference ecosystem lives here. 3.11+ for `Self`, better error locations, and `tomllib`. |
| Env & deps | **uv** + `pyproject.toml` | Fast, lockfile-based, single tool for venv + install. Falls back to plain `pip install -e .` for anyone without uv. |
| Numerics | **numpy**, **pandas** | 90k rows fits in memory with room to spare. Polars would be faster and irrelevant at this size. |
| Statistics | **scipy.stats**, **statsmodels** | Trusted reference implementations to compose *and* to cross-validate our own against (Rules.md R5.3). |
| Static charts | **matplotlib** | Notebook figures and report export. Deterministic, no JS runtime. |
| Interactive charts | **plotly** | Streamlit hover/tooltip layer without hand-writing JS. |
| App | **Streamlit** | The readout is a handful of controls over a computed result. Reaching for FastAPI + React here would add two deploy targets and a serialisation boundary to display ~6 numbers. |
| Storage | **Parquet** via **pyarrow** | Cached clean dataset. Columnar, typed, gitignored. |
| Testing | **pytest**, **hypothesis** | Property-based tests are the natural fit for estimator invariants (§6). |
| Quality | **ruff** (lint + format), **mypy** (strict) | One tool for lint+format. Strict typing because the result objects are the contract. |
| Notebooks | **jupyter** + **nbstripout** | Curriculum. `nbstripout` keeps output out of git diffs. |

### 1.1 Deliberately rejected

Recording these so they aren't relitigated:

- **DuckDB** — genuinely nice, and pointless for 90k rows in pandas. Revisit only
  if the simulation harness starts generating tens of millions of rows.
- **FastAPI + React** — see the app row above. The cost is two runtimes and an API
  contract; the benefit over Streamlit is nil for this deliverable.
- **PyMC / full MCMC** — the Bayesian A/B case (F4.4) is Beta-Binomial, which is
  **conjugate and closed-form**. MCMC would be slower, harder to test, and would
  obscure the mathematics that is the point of the exercise. Available as an
  optional extra if a later model has no closed form.
- **econml / dowhy** — excellent libraries, but importing a causal-forest
  implementation would skip the learning goal in Phase 6. They enter as an optional
  extra to *cross-validate* our learners, not to replace them.
- **A database / ORM** — there is no mutable application state. Inputs are files;
  outputs are files.

### 1.2 Optional extras

```toml
[project.optional-dependencies]
advanced = ["econml", "dowhy"]   # cross-validation of CATE learners
bayes    = ["pymc", "arviz"]     # only if a non-conjugate model appears
```

---

## 2. Core design: estimand → estimator → estimate

Every analysis in this codebase, frequentist or causal, follows the same three-step
shape. This is the single most important structural decision, because it makes the
§5 benchmark and the §6 calibration tests uniform across otherwise unrelated
methods.

```
                    ┌──────────────────────────────────────┐
                    │  ExperimentSpec  (pre-registered)    │
                    │  primary metric, guardrails, α,      │
                    │  power, MDE, practical threshold,    │
                    │  stopping rule, declared subgroups   │
                    └──────────────────┬───────────────────┘
                                       │  read, never inferred
                                       ▼
   raw CSV ──► ingest ──► ExperimentData ──► SanityReport ──► [GATE]
                          (typed frame)      SRM, dupes,        │
                                             integrity          │ blocks on failure
                                                                ▼
                                                    ┌───────────────────────┐
                                    Estimand  ─────►│      Estimator        │
                                    "ATE of gate_40 │  ztest / welch /      │
                                     on retention_7"│  bootstrap / ipw /    │
                                                    │  aipw / iv / cuped    │
                                                    └───────────┬───────────┘
                                                                ▼
                                                        EffectEstimate
                                                   point, ci, se, p, method,
                                                   assumptions, data_source,
                                                   n_per_arm, diagnostics
                                                                │
                                       ┌────────────────────────┼─────────────┐
                                       ▼                        ▼             ▼
                                    Readout               Benchmark        Report
                                  (Streamlit)          (vs ground truth)   (HTML/MD)
```

**Why the uniform `EffectEstimate` return type matters:**

1. The benchmark harness (F5.8) can score *any* estimator without knowing what it is.
2. Calibration tests can be written once and parametrised over every estimator.
3. The report layer never contains method-specific branching.
4. `assumptions` and `data_source` are **required fields**, so an estimate
   physically cannot be produced without declaring what it assumes and whether the
   data was real or synthetic (PRD §6, Rules R2.4).

### 2.1 The gate

`SanityReport` sits between data and analysis, not beside it. `analyze()` takes a
`SanityReport` and raises `SanityCheckFailure` if it is failing. Overriding
requires an explicit `override_reason: str` that is recorded on the resulting
`EffectEstimate`. Making the safe path the default path is the whole design intent.

---

## 3. Folder structure

```
Project2/
├── PRD.md
├── Architecture.md
├── Rules.md
├── Phases.md
├── Design.md
├── README.md
├── pyproject.toml
├── uv.lock
├── .gitignore
├── .pre-commit-config.yaml
│
├── data/                          # gitignored (except README)
│   ├── README.md                  # how to obtain cookie_cats.csv
│   ├── raw/                       # cookie_cats.csv — never committed
│   └── processed/                 # cached Parquet
│
├── specs/                         # pre-registration files, COMMITTED
│   └── cookie_cats_gate.yaml      # the spec that locks the analysis plan
│
├── src/gatekeeper/
│   ├── __init__.py
│   ├── types.py                   # EffectEstimate, Estimand, SanityReport, enums
│   ├── spec.py                    # ExperimentSpec load/validate (pydantic)
│   │
│   ├── data/
│   │   ├── ingest.py              # CSV → typed ExperimentData, Parquet cache
│   │   ├── schema.py              # column contract + dtype validation
│   │   └── synthetic.py           # generators with exactly known ground truth
│   │
│   ├── design/
│   │   ├── power.py               # sample size, power, MDE, duration
│   │   └── srm.py                 # sample ratio mismatch
│   │
│   ├── checks/
│   │   ├── integrity.py           # dupes, cross-arm leakage, nulls
│   │   ├── outliers.py            # tail profile, leverage (reports, never trims)
│   │   └── balance.py             # standardised mean differences
│   │
│   ├── frequentist/
│   │   ├── proportions.py         # two-proportion z, absolute/relative lift
│   │   ├── means.py               # Welch t
│   │   ├── bootstrap.py           # percentile + BCa
│   │   ├── ratio.py               # delta method
│   │   └── multiplicity.py        # BH, Bonferroni
│   │
│   ├── sequential/
│   │   ├── alpha_spending.py      # O'Brien–Fleming, Pocock
│   │   └── always_valid.py        # mSPRT / always-valid p-values
│   │
│   ├── variance/
│   │   └── cuped.py               # CUPED (+ guard: no post-treatment covariates)
│   │
│   ├── bayesian/
│   │   └── beta_binomial.py       # conjugate posterior, P(B>A), expected loss
│   │
│   ├── causal/
│   │   ├── confounding.py         # F5.1 — the confounding simulator
│   │   ├── propensity.py          # estimation, IPW, stabilised weights, matching
│   │   ├── outcome.py             # regression adjustment
│   │   ├── aipw.py                # doubly robust
│   │   ├── iv.py                  # 2SLS / Wald IV, LATE
│   │   ├── did.py                 # synthetic panel only
│   │   └── sensitivity.py         # E-value, Rosenbaum bounds
│   │
│   ├── hte/
│   │   ├── learners.py            # S-, T-, X-learner
│   │   └── uplift.py              # Qini, uplift curves
│   │
│   ├── benchmark/
│   │   ├── harness.py             # run estimators × regimes → scores
│   │   └── scoring.py             # bias, variance, RMSE, CI coverage
│   │
│   ├── viz/
│   │   ├── theme.py               # Design.md palette as code — single source
│   │   ├── static.py              # matplotlib figures
│   │   └── interactive.py         # plotly figures
│   │
│   └── report/
│       ├── readout.py             # EffectEstimate → decision
│       └── render.py              # → HTML / Markdown
│
├── app/
│   └── streamlit_app.py
│
├── notebooks/                     # one per phase; import from src, no logic
│   ├── 01_eda_and_sanity.ipynb
│   ├── 02_frequentist_core.ipynb
│   ├── 03_cookie_cats_readout.ipynb
│   ├── 04_variance_and_sequential.ipynb
│   ├── 05_bayesian.ipynb
│   ├── 06_causal_benchmark.ipynb
│   └── 07_heterogeneous_effects.ipynb
│
└── tests/
    ├── conftest.py
    ├── fixtures/                  # known-answer cases w/ hand-computed values
    ├── unit/                      # per-module
    ├── calibration/               # null uniformity + CI coverage (slow marker)
    └── reference/                 # cross-checks vs scipy/statsmodels
```

**Structural rule:** `src/gatekeeper/` never imports from `notebooks/` or `app/`.
Dependencies point inward:

```
app → report → frequentist|causal|sequential|variance|bayesian|hte
                    ↓
              checks → spec → types
                    ↓
                  data → types
```

`spec` sits inside the analysis layers but outside `types`, and `checks` is allowed
to depend on it: `check_outlier_leverage` takes the spec's `OutlierRule` so it can
say *which* rule was pre-declared rather than just that one existed. That coupling
buys a materially better error message, and `spec` itself imports only `types`, so
there is no cycle.

---

## 4. Key types

Sketches, not final signatures — the shape is the commitment.

```python
# types.py
from dataclasses import dataclass
from enum import StrEnum


class DataSource(StrEnum):
    REAL = "real"
    SYNTHETIC = "synthetic"
    SEMI_SYNTHETIC = "semi_synthetic"  # real data, injected confounding


class Decision(StrEnum):
    SHIP = "ship"
    HOLD = "hold"
    INCONCLUSIVE = "inconclusive"
    BLOCKED = "blocked"  # sanity checks failed


@dataclass(frozen=True, slots=True)
class Estimand:
    """What we are trying to estimate, stated before estimating it."""

    outcome: str
    treatment: str
    target: str  # "ATE" | "ATT" | "LATE" | "CATE"
    population: str
    scale: str  # "absolute" | "relative"


@dataclass(frozen=True, slots=True)
class EffectEstimate:
    estimand: Estimand
    point: float
    ci: tuple[float, float]
    ci_level: float
    se: float | None
    p_value: float | None
    method: str
    assumptions: tuple[str, ...]  # REQUIRED — no silent estimates
    data_source: DataSource  # REQUIRED — real vs synthetic
    n_per_arm: dict[str, int]
    diagnostics: dict[str, float]
    override_reason: str | None = None

    def is_practically_significant(self, threshold: float) -> bool: ...
```

Frozen and slotted: an estimate is a record of a computation, not a mutable buffer.

Every estimator is a plain function with the same silhouette:

```python
def estimate_<method>(
    data: ExperimentData,
    estimand: Estimand,
    *,
    alpha: float = 0.05,
    **method_specific,
) -> EffectEstimate: ...
```

Functions, not classes: these are stateless mathematical transforms, and a uniform
callable signature is what lets the benchmark harness iterate over them.

---

## 5. The benchmark harness (F5.8)

The mechanism behind PRD §1.1, and the reason `causal/confounding.py` exists.

```
Cookie Cats RCT
      │
      ├─► estimate_two_proportion(...)  ──►  ground truth τ̂*  (+ its own CI)
      │
      └─► confounding.make_regime(seed, strength, kind)
                │
                ├─ "selection"        P(include) depends on a covariate + arm
                ├─ "noncompliance"    one-sided; assignment ≠ treatment received
                └─ "unobserved"       confounder exists but is withheld from the estimator
                          │
                          ▼
              for each regime × estimator × seed:
                  τ̂ = estimator(sample)
                  record bias = τ̂ − τ̂*, |CI| , covered = τ̂* ∈ CI
                          │
                          ▼
              scoring.py → bias · variance · RMSE · coverage
```

The expected and instructive results: **naive difference-in-means is badly biased
under "selection"**; IPW and AIPW largely recover τ̂\* when the confounder is
observed; **every adjustment method fails under "unobserved"**, which is precisely
what `sensitivity.py` is for; and IV recovers LATE — not ATE — under
non-compliance, so the two must not be compared as if they were the same quantity.

Writing down these expectations *before* running the benchmark is itself the
pre-registration discipline applied to our own project.

---

## 6. Testing strategy

Statistical code fails silently. Four layers, in increasing cost:

1. **Known-answer fixtures** (`tests/fixtures/`) — hand-computed or
   textbook-sourced cases. A 2×2 table with a z-statistic worked out by hand.
2. **Reference cross-checks** (`tests/reference/`) — where scipy or statsmodels
   implements the same thing, agree to `< 1e-6`. Applies to the z-test, Welch,
   BH, and 2SLS.
3. **Property tests** (hypothesis) — invariants that must hold for all inputs:
   swapping arms flips the sign of the effect but not its magnitude; a relative
   lift of 0 implies a p-value of 1 for identical arms; CI width shrinks
   monotonically as n grows; bootstrap CIs are order-invariant under permutation.
4. **Calibration tests** (`tests/calibration/`, `@pytest.mark.slow`) — the layer
   that catches the errors the other three miss. Simulate under a true null ≥1,000
   times: p-values must be uniform (KS test), and 95% CIs must cover at 93–97%.
   Run under a fixed seed in CI; nightly with fresh seeds.

Every RNG-touching function takes an explicit `rng: np.random.Generator` or
`seed: int`. No module-level `np.random` calls anywhere (Rules R4.2).

**Where ground truth comes from.** `data/synthetic.py` sets marginal outcome
probabilities *directly per arm*, so the true absolute effect is exactly
`p_treatment − p_control` rather than a logit-scale coefficient that would need
marginalising. Tests can therefore assert recovery of an exact number. The whole
suite runs on synthetic data, so the Kaggle CSV is never needed in CI and is never
committed.

---

## 7. Data flow & reproducibility

```
data/raw/cookie_cats.csv          (manual download, gitignored)
        │  ingest.load_cookie_cats()
        │    • validate schema + dtypes against schema.py
        │    • fail on unexpected variant labels
        │    • no row drops, no imputation
        ▼
data/processed/cookie_cats.parquet (cached, gitignored)
        │
        ▼
ExperimentData ── + specs/cookie_cats_gate.yaml ──► analysis
```

Reproducibility contract:

- `specs/*.yaml` is committed; raw and processed data are not.
- One command reproduces the full readout from a fresh clone plus the CSV.
- Every seed is explicit and recorded in the resulting `EffectEstimate`.
- Notebooks are stripped of output on commit and executed in CI, so a notebook that
  no longer runs is a build failure rather than rot.

---

## 8. CI

GitHub Actions, on push and PR:

| Job | Does |
|---|---|
| `lint` | `ruff check` + `ruff format --check` |
| `types` | `mypy --strict src/` |
| `test` | `pytest -m "not slow"` + coverage gate (≥ 85%) |
| `notebooks` | Execute every notebook headlessly; fail on error |
| `calibration` | `pytest -m slow` — nightly schedule, not per-push (it is minutes, not seconds) |
