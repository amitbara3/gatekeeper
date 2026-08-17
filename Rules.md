# Rules — Gatekeeper

**Last updated** 2026-08-17

Constraints for anyone working in this repo, human or AI. Ordered by how much
damage a violation does.

**§1 and §2 are the important ones.** Nobody ships a broken statistic because they
forgot to run `ruff`; they ship it because they let the data pick the hypothesis.
A rule marked **[HARD]** must never be violated. **[SOFT]** rules may be broken
with a written justification in the PR description or a code comment.

---

## §1 Statistical rules

### R1.1 Declare the estimand before estimating it — [HARD]

State outcome, treatment, target quantity (ATE / ATT / LATE / CATE), population,
and scale (absolute / relative) *before* running the estimator. This is enforced in
code: every estimator takes an `Estimand`.

Rationale: "the effect of the gate on retention" is not one number. The ATE and the
LATE genuinely differ under non-compliance, and comparing them as if they were the
same quantity is the most common causal-inference error.

### R1.2 The pre-registration spec is the source of truth — [HARD]

The primary metric, guardrails, α, power, MDE, practical-significance threshold,
stopping rule, and any subgroups live in `specs/*.yaml`, committed **before** the
first analysis run. The analysis reads the spec; it never infers intent from the
data.

Changing a spec after seeing results is permitted — science is iterative — but it
requires a **new spec file** and the resulting analysis is labelled *exploratory*,
never *confirmatory*. Never edit a spec in place to match a result.

### R1.3 Sanity checks gate the readout — [HARD]

SRM and assignment-integrity checks run before any effect is reported. A failing
`SanityReport` raises `SanityCheckFailure`. Overriding requires
`override_reason: str`, which is recorded on the `EffectEstimate` and rendered in
the report.

An SRM failure means the *instrumentation* is suspect. A beautiful p-value from
broken assignment is not a weak result; it is not a result.

### R1.4 Never report a p-value alone — [HARD]

Every reported result carries: point estimate, confidence interval, and the
practical-significance threshold it is being judged against. `p < 0.05` is not a
finding, and "not significant" is never reported as "no effect" — report the CI and
say what effect sizes the test could and could not rule out.

### R1.5 No peeking without a sequential correction — [HARD]

Repeated looks at accumulating data require alpha-spending or always-valid
p-values. A fixed-horizon test may be read **once**, at the pre-registered n.

### R1.6 No post-hoc outlier removal — [HARD]

Outlier handling rules are declared in the spec before analysis. `checks/outliers.py`
**reports and never trims** — trimming is an explicit, spec-driven caller decision.

Cookie Cats has at least one player with an extreme `sum_gamerounds`. Discovering
it, removing it, and re-running is p-hacking, however reasonable it feels. Declare
the rule (e.g. winsorise at p99.9), then apply it to both arms identically, and
report results with and without it.

### R1.7 Never condition on a post-treatment variable — [HARD]

This resolves PRD open question **O1** and it is the rule most likely to be
violated by accident on this dataset.

`sum_gamerounds` is measured **after** the gate is encountered, so it is a
*mediator*, not a covariate. Therefore:

- ❌ It must **never** be a CUPED covariate. CUPED requires a **pre**-experiment
  covariate; using a post-treatment one biases the estimate while *appearing* to
  reduce variance. This is the single most tempting mistake available in this
  repo. `variance/cuped.py` must raise on any covariate not marked pre-period.
- ❌ It must never be a control in a regression whose estimand is the total effect
  of the gate on retention (that induces post-treatment/collider bias).
- ✅ It may be an **outcome** in its own right.
- ✅ It may be used for descriptive segmentation clearly labelled as descriptive.
- ✅ It may be a mediator in an explicitly specified mediation analysis, where the
  estimand says so.

### R1.8 Multiplicity is corrected across the declared metric set — [HARD]

Testing k metrics or k subgroups requires a correction (BH by default). The metric
set comes from the spec, so it cannot expand silently as the analysis proceeds.

### R1.9 Subgroups: report the interaction, not the subgroup p-value — [HARD]

Heterogeneity claims require a test of the treatment × subgroup **interaction**.
"Significant in this subgroup, not in that one" is not evidence of heterogeneity —
it is usually a power artifact.

### R1.10 Causal language requires stated identification assumptions — [HARD]

Any word implying causation ("increases", "causes", "drives", "lifts") is licensed
either by randomisation or by explicitly stated identification assumptions
(conditional ignorability, positivity, exclusion restriction, parallel trends…).
`EffectEstimate.assumptions` is a required field for exactly this reason.

Observational estimates additionally require sensitivity analysis (§F5.7): state
how strong unmeasured confounding would need to be to overturn the conclusion.

### R1.11 Label real vs synthetic — [HARD]

`EffectEstimate.data_source` is required. Any figure, table, or report line derived
from synthetic or semi-synthetic data is visibly labelled as such (Design.md
mandates the badge). Methods that Cookie Cats cannot support (CUPED, DiD, RDD,
real sequential accrual — PRD §6) are **never** demonstrated on the real data as
though they were valid.

### R1.12 Prefer Welch; don't assume equal variance — [SOFT]

Default to Welch's t-test. Equal-variance pooling requires justification.
For heavily skewed metrics like `sum_gamerounds`, prefer bootstrap or a
rank-based test and say which was pre-registered.

### R1.13 Match the unit of analysis to the unit of randomisation — [HARD]

Cookie Cats randomises by `userid` and metrics are per-user, so this is
satisfied — but any future per-round or per-session metric needs delta-method or
cluster-robust variance. Independence violations shrink standard errors and
manufacture significance.

---

## §2 Honesty rules

### R2.1 Negative and null results are shipped — [HARD]

If the analysis says the gate change did nothing, that is the finding. This project
has no incentive to find an effect, and pretending otherwise would defeat its
purpose.

### R2.2 Write the expected result before running the benchmark — [HARD]

Architecture.md §5 states what each estimator is expected to do before the
benchmark runs. Compare outcomes against the written prediction and record
surprises. Retrofitting an explanation to whatever came out is the failure mode.

### R2.3 Document what was tried and abandoned — [SOFT]

Rejected approaches go in Architecture.md §1.1 or a phase note, with the reason.

### R2.4 No estimate without assumptions — [HARD]

Enforced by the type system: `assumptions` and `data_source` are non-defaulted
required fields on `EffectEstimate`.

---

## §3 Library rules

### Use

`numpy` · `pandas` · `scipy` · `statsmodels` · `matplotlib` · `plotly` ·
`streamlit` · `pydantic` (spec validation) · `pyarrow` · `pytest` · `hypothesis` ·
`ruff` · `mypy`

### Do not add without discussion — [HARD]

- **No new stats/ML dependency** to solve something already covered by
  scipy/statsmodels.
- **No sklearn for the core estimators.** It is permitted for nuisance models
  (propensity, outcome regression) in `causal/` and for CATE learners in `hte/`.
  It is not the implementation of a t-test.
- **No `econml` / `dowhy` in the core path.** Optional extras, used to
  *cross-validate* our implementations (Architecture §1.2). Importing a
  causal-forest to replace Phase 6 skips the learning goal.
- **No PyMC unless a model has no closed form.** Beta-Binomial is conjugate.
- **No polars, no DuckDB, no ORM.** See Architecture §1.1.
- **No plotting library beyond matplotlib + plotly.**
- **No LLM/API dependency.** There is nothing in this project an LLM should be in
  the runtime path of.

### Forbidden outright — [HARD]

- `scipy.stats.ttest_ind(..., equal_var=True)` as a default (R1.12).
- Any module-level `np.random.seed()` or bare `np.random.*` call (R4.2).
- `pandas` chained assignment / `inplace=True`.
- `except:` or `except Exception:` without re-raise (§5).
- Committing anything under `data/raw/` or `data/processed/`.

---

## §4 Code rules

### R4.1 No analysis logic in notebooks — [HARD]

Notebooks import from `src/gatekeeper/` and contain narrative, calls, and charts.
A function defined in a notebook cell cannot be tested, reused, or reviewed. If
you write one there, move it into the package before committing.

### R4.2 Randomness is explicit — [HARD]

Every stochastic function takes `rng: np.random.Generator` or `seed: int`. The seed
is recorded in the output. Two runs with the same seed produce identical numbers.

### R4.3 Estimators are pure functions — [HARD]

No mutation of inputs, no I/O, no global state, no logging side effects inside a
computation. Given the same data and parameters, the same result.

### R4.4 Typed and strict — [HARD]

`mypy --strict` passes on `src/`. Public functions carry full annotations. Result
objects are frozen dataclasses.

### R4.5 Every estimator ships with tests — [HARD]

New estimator ⇒ (a) a known-answer fixture, (b) a calibration test, and (c) a
reference cross-check where an equivalent exists. Not "later"; in the same PR.

### R4.6 Match the surrounding code — [SOFT]

Read a neighbouring module first. Consistent naming (`estimate_*` for estimators,
`check_*` for sanity checks, `plot_*` for figures), consistent docstring style
(numpy format, with an **Assumptions** section on every estimator).

### R4.7 Calibration failures are bugs until proven otherwise — [HARD]

Every estimator carries a calibration test: under a true null, p-values must be
uniform, and 95% intervals must cover at 93–97% (Architecture §6).

A failure blocks the phase. **The exception, and it is narrow:** some estimators
have *documented, expected* finite-sample under-coverage — AIPW and IV are the
known cases, where nuisance-model error and weak instruments respectively degrade
coverage at realistic n. That is a property of the method, not a defect in our code.

To invoke the exception, all four must hold:

1. A **citation or derivation** for the expected behaviour — not a hunch.
2. The test asserts the **degraded** bound explicitly (e.g. `coverage > 0.88`) with
   the reason in the test's docstring. It never simply widens 93–97% and moves on.
3. Coverage **improves toward nominal as n grows**, demonstrated by a test at two
   sample sizes. This is the check that separates a real finite-sample effect from a
   bug, and it is the one that actually does the work.
4. The shortfall is recorded in the estimator's `EffectEstimate.assumptions`, so a
   consumer of the number sees it.

Absent all four, a calibration failure is a bug. "AIPW is known to undercover" is
not a licence to skip the investigation — it is a hypothesis that item 3 tests.

### R4.8 The palette lives in one place — [HARD]

`viz/theme.py` is the only place colour hexes appear. No inline hex in a plotting
call. Design.md is the spec; `theme.py` is its implementation.

---

## §5 Error handling

- **Fail loudly on invalid statistical input.** Negative variance, `n < 2`, α
  outside (0,1), an empty arm, a proportion outside [0,1] → raise. Never return
  `NaN` and continue.
- **Custom exceptions**, defined in `types.py`:
  `SanityCheckFailure` · `SpecViolation` · `AssumptionViolation` ·
  `InsufficientData` · `PostTreatmentCovariateError` (R1.7).
- **No silent coercion.** Schema validation at ingest fails on unexpected variant
  labels, wrong dtypes, or missing columns. Do not `fillna` to make the code run.
- **Never swallow exceptions.** Catch a specific type, add context, re-raise.
- **Warnings for degraded results, exceptions for invalid ones.** Poor propensity
  overlap → `UserWarning` plus a `diagnostics` entry. Zero overlap → raise.
- **Error messages state the fix.** Not `"invalid input"` but
  `"alpha must be in (0, 1), got 1.5"`.

---

## §6 Git & repo rules

- `main` is protected in spirit: work on `phase-N-description` branches, merge via PR.
- Conventional commits: `feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `chore:`.
- One phase per branch; a phase merges only when its exit criteria in Phases.md are met.
- Notebook outputs stripped before commit (`nbstripout` via pre-commit).
- Never commit data, `.env`, credentials, or Kaggle tokens.
- CI must be green to merge.

---

## §7 Rules for AI assistants

### Do

- **Read `specs/*.yaml` before analysing anything.** The spec is the intent.
- **Cite the rule** you are honouring when it shapes a design decision (e.g. "not
  using `sum_gamerounds` as the CUPED covariate per R1.7").
- **Say when the data cannot support the method.** PRD §6 lists these. The correct
  response is "this needs synthetic data, here is why", never a plausible-looking
  number on the real data.
- **Write the calibration test with the estimator**, in the same change (R4.5).
- **Report failing tests and their output verbatim.** Do not summarise a failure as
  a success.
- **State assumptions in the docstring** under an `Assumptions` heading.
- **Ask when the estimand is ambiguous.** "Effect on retention" underspecifies
  which retention metric, which target quantity, and on what scale.

### Do not

- **Do not invent statistical results.** Never write a number into a doc, comment,
  or commit message that was not produced by running code. If a value is
  illustrative, mark it `EXAMPLE`.
- **Do not choose the metric or threshold that makes the result significant.**
  Those come from the spec (R1.2).
- **Do not silently widen scope.** Building Phase 5 while Phase 2 is open is not
  helpful. Phases exist because the subject is bottomless.
- **Do not add dependencies** to avoid implementing something that is the learning
  goal (§3).
- **Do not "fix" a failing calibration test by loosening its tolerance.** A
  coverage of 87% on a nominal 95% CI is a bug until proven otherwise. Investigate
  the estimator, not the threshold. See R4.7 for the one legitimate exception and
  the bar it has to clear.
- **Do not soften an SRM or assumption failure** into a warning to let a pipeline
  finish.
- **Do not fabricate an interpretation.** If the result is confusing, say it is
  confusing and propose diagnostics.
- **Do not reformat, restructure, or "tidy" unrelated files** as part of a task.
- **Do not commit or push unless asked.**
