# PRD — Gatekeeper

**An experimentation & causal inference workbench, validated against a real A/B test.**

| | |
|---|---|
| **Status** | Draft — Phase 0 |
| **Owner** | @amitbara3 |
| **Last updated** | 2026-08-17 |
| **Primary dataset** | Kaggle *Cookie Cats* mobile game A/B test (~90,189 players) |

---

## 1. Why this project exists

Most teams can *run* an A/B test. Far fewer can analyse one in a way that survives
scrutiny. The failure modes are well documented and depressingly common:

- **Peeking.** Checking the dashboard daily and stopping at the first `p < 0.05`,
  which inflates the false-positive rate far above the nominal 5%.
- **Ignoring sample ratio mismatch (SRM).** A 50/50 split that arrives as 49.2/50.8
  usually means the *instrumentation* is broken, which invalidates the result
  regardless of how pretty the p-value is.
- **No power analysis.** Running a test that could never have detected the effect
  size the team cares about, then reading "not significant" as "no effect".
- **Naked p-values.** Reporting significance with no effect size, no confidence
  interval, and no practical-significance threshold.
- **Metric fishing.** Testing eight metrics and six subgroups, reporting the one
  that moved, and never correcting for multiplicity.
- **Causal overreach.** When randomisation is unavailable or broken, reaching for
  a group-mean difference and describing it in causal language.

This project builds the tooling and the judgement to avoid all six. It is a
**learning project with a real deliverable**: a typed, tested Python library plus a
notebook curriculum, where every method is validated on data whose answer we can
check.

### 1.1 The thesis — the RCT is the answer key

This is the idea that distinguishes Gatekeeper from a tutorial rehash.

Cookie Cats is a **randomised** experiment, so it yields a trustworthy estimate of
the true effect of moving the progression gate from level 30 to level 40. That
estimate becomes **ground truth**.

We then deliberately handicap ourselves: from the same data we construct
observational samples where randomisation is *broken in a way we control* —
confounded selection, simulated non-compliance, an induced time dimension. Each
causal-inference estimator is then scored on a question with a known answer:

> **Does this method recover the experimental ground truth, and how badly does it
> fail when its assumptions are violated?**

That turns "learning causal inference" from reading about identification
assumptions into measuring what happens when they break. It is also the
project's most defensible artifact: a benchmark table of estimator performance
against a known target.

---

## 2. Target users

| Priority | User | Needs |
|---|---|---|
| **P0** | **The builder** (me) — analyst/DS building experimentation depth | A structured path from z-tests to CATE estimation, with every claim checkable against ground truth. Code that is reusable afterwards, not throwaway notebook cells. |
| **P1** | **A working analyst** | A correct, typed library for day-to-day readouts: power/MDE before the test, SRM + sanity checks during, effect size + CI + decision after. Sensible defaults that make the safe thing the easy thing. |
| **P2** | **A PM / stakeholder** | A one-page readout that states the decision, the effect size with uncertainty, and the caveats — without needing to read the notebook. |
| **P3** | **A reviewer / hiring manager** | Evidence of statistical judgement: honest treatment of what the data *cannot* support, calibration tests, documented assumptions. |

---

## 3. Goals & non-goals

### Goals

- **G1** Implement the trustworthy-experiment toolkit end-to-end: design → sanity
  checks → analysis → decision → readout.
- **G2** Implement the core causal-inference toolkit for when randomisation is
  absent or broken, and **benchmark each estimator against the RCT ground truth**.
- **G3** Every estimator is **calibration-tested**: under a simulated null,
  p-values are uniform and 95% CIs cover at ~95%. A statistical library that is
  merely "tested" for return types is untested.
- **G4** Be explicit and honest about **what Cookie Cats cannot support** (see §6).
  Methods that need data this dataset lacks are demonstrated on synthetic data and
  labelled as such — never faked on the real data.
- **G5** Produce a shareable readout: one Streamlit app + one exportable report.

### Non-goals

- **N1** Not an experiment *assignment/serving* platform. No bucketing service, no
  feature flags, no SDK. Analysis only.
- **N2** Not a general BI tool. One dataset shape, deeply understood.
- **N3** Not a novel-methods research project. Textbook methods, correctly
  implemented and honestly evaluated.
- **N4** No user accounts, multi-tenancy, or hosted deployment in v1.
- **N5** Not a from-scratch reimplementation of scipy/statsmodels. Wrap and compose
  trusted primitives; implement from scratch only where it is the learning goal
  (CUPED, sequential tests, IPW, the learners) — and even then, cross-check against
  a reference implementation.

---

## 4. Features

Priority: **P0** = v1 must ship · **P1** = v1 should ship · **P2** = nice to have.

### 4.1 Experiment design (pre-test) — P0

| ID | Feature | Detail |
|---|---|---|
| F1.1 | Power & sample size | Required n per arm given baseline rate, MDE, α, power. Proportions and means. |
| F1.2 | MDE calculator | Inverse of F1.1 — given available traffic and duration, what is the smallest detectable effect? |
| F1.3 | Duration estimator | n → days, given traffic per day. |
| F1.4 | Pre-registration stub | Emit a machine-readable experiment spec (primary metric, guardrails, α, power, MDE, planned analysis, stopping rule) that the analysis step *reads* — so the analysis cannot silently deviate from the plan. |

**F1.4 is the keystone feature.** Pre-registration is what makes the other guards
enforceable rather than advisory: the analysis function refuses to label a metric
"primary" unless the spec said so.

### 4.2 Sanity checks (the gate before any result is read) — P0

| ID | Feature | Detail |
|---|---|---|
| F2.1 | SRM check | Chi-square goodness-of-fit against the intended split. Fails loudly. Default threshold `p < 0.0005` (Kohavi's recommendation — a strict threshold because this test is run on every experiment and a false alarm is expensive). |
| F2.2 | Assignment integrity | Duplicate unit IDs, units in both arms, null assignments, unexpected variant labels. |
| F2.3 | Outlier profile | Report the metric distribution's tail before any decision; flag single units with outsized leverage. Never auto-trim (see Rules.md). |
| F2.4 | Pre-period balance | Where pre-period covariates exist, standardised mean differences across arms. |

The analysis API is designed so that **F2.1–F2.2 run automatically and a failure
blocks the readout** unless explicitly overridden with a recorded reason.

### 4.3 Frequentist analysis — P0

| ID | Feature | Detail |
|---|---|---|
| F3.1 | Two-proportion test | Binary metrics (`retention_1`, `retention_7`). Absolute + relative lift, CI, p-value. |
| F3.2 | Means test | Welch's t-test (unequal variance is the default assumption, not pooled). |
| F3.3 | Non-parametric / robust | Bootstrap CIs (percentile + BCa) and Mann–Whitney, for the heavily skewed `sum_gamerounds`. |
| F3.4 | Ratio metrics | Delta-method variance for ratios where the unit of analysis ≠ unit of randomisation. |
| F3.5 | Multiplicity control | Benjamini–Hochberg (default) and Bonferroni across the declared metric set. |

### 4.4 Trustworthy-experiment extensions — P1

| ID | Feature | Detail |
|---|---|---|
| F4.1 | Sequential testing | Alpha-spending (O'Brien–Fleming, Pocock) and always-valid p-values. Directly addresses the peeking problem. |
| F4.2 | CUPED | Variance reduction via a pre-experiment covariate. **Synthetic data only** — Cookie Cats has no pre-period (§6). |
| F4.3 | Variance-reduction benchmark | Measured variance reduction vs. the theoretical `1 − ρ²`. |
| F4.4 | Bayesian A/B | Beta-Binomial conjugate posterior for binary metrics: P(treatment > control), expected loss, credible intervals. Closed-form first; MCMC only if a model needs it. |

### 4.5 Causal inference — P1

Operates on the **deliberately confounded** samples described in §1.1.

| ID | Feature | Detail |
|---|---|---|
| F5.1 | Confounding simulator | Construct observational samples from the RCT with controlled, known confounding. The harness that makes the whole benchmark possible. |
| F5.2 | Propensity scores | Estimation, IPW + stabilised weights, matching, overlap diagnostics, trimming. |
| F5.3 | Balance diagnostics | Standardised mean differences pre/post weighting, love plots, positivity checks. |
| F5.4 | Outcome regression & AIPW | Regression adjustment and doubly-robust estimation. |
| F5.5 | IV / LATE | With simulated one-sided non-compliance, assignment is a valid instrument; recover the complier effect. |
| F5.6 | DiD | Synthetic panel only — Cookie Cats has no time dimension (§6). |
| F5.7 | Sensitivity analysis | E-values and Rosenbaum-style bounds: how strong would unmeasured confounding need to be to overturn the conclusion? |
| F5.8 | **Estimator benchmark** | The headline deliverable: each estimator's bias, variance, and CI coverage against the RCT ground truth, per confounding regime. |

### 4.6 Heterogeneous effects — P2

| ID | Feature | Detail |
|---|---|---|
| F6.1 | CATE learners | S-, T-, and X-learner. |
| F6.2 | Uplift evaluation | Qini and uplift curves, decile lift. |
| F6.3 | Honest subgroup analysis | Subgroups must be pre-declared in the spec; multiplicity-corrected; interaction test reported rather than per-subgroup significance. |

### 4.7 Readout & app — P1

| ID | Feature | Detail |
|---|---|---|
| F7.1 | Readout object | One typed result carrying estimate, CI, p-value, method, assumptions, and sanity-check status. |
| F7.2 | Streamlit app | Load an experiment, see sanity checks, metric results with CIs, and the decision. |
| F7.3 | Report export | Static HTML/Markdown one-pager from the readout object. |
| F7.4 | Decision framing | Ship / hold / inconclusive against the *pre-registered* practical-significance threshold — not against `p < 0.05`. |

### 4.8 Notebook curriculum — P0

One notebook per phase, each pairing an implementation with the matching reading
(§8). Notebooks **import from the library** and contain narrative + charts only —
no analysis logic defined in a notebook cell (Rules.md).

---

## 5. Success metrics

This is a learning project, so the metrics are about capability and correctness,
not usage.

| Metric | Target |
|---|---|
| Calibration: p-value uniformity under the null | KS test p > 0.05 for every estimator |
| Calibration: 95% CI coverage | 93–97% over ≥ 1,000 simulations |
| Cross-validation against reference implementations | Agreement to < 1e-6 where a scipy/statsmodels equivalent exists |
| Estimator benchmark completeness | All F5 estimators scored on ≥ 3 confounding regimes |
| Cookie Cats readout | Reproducible end-to-end from raw CSV with one command |
| Test coverage on `src/gatekeeper/` | ≥ 85%, with every estimator having a known-answer fixture |
| Curriculum | One notebook per phase, all executing top-to-bottom in CI |

---

## 6. Dataset reality check — what Cookie Cats can and cannot support

Being explicit here prevents the most likely form of self-deception in this
project. **Schema:** `userid`, `version` (`gate_30` / `gate_40`), `sum_gamerounds`,
`retention_1`, `retention_7`.

**Supports directly:**
- Two-proportion tests on `retention_1` / `retention_7`
- Means/robust tests on `sum_gamerounds` (heavily right-skewed, with at least one
  extreme outlier that must be handled by a pre-declared rule, not discovered and
  then removed)
- SRM and assignment-integrity checks — the split is *not* exactly 50/50, which
  makes this a live exercise rather than a toy one
- Ground truth for the §1.1 benchmark

**Cannot support without synthetic augmentation — must be labelled as such:**

| Method | Why not | How we handle it |
|---|---|---|
| **CUPED** | No pre-experiment covariate exists. | Demonstrate on synthetic data with known pre/post correlation. **`sum_gamerounds` is post-treatment and must never be used as the covariate** — that is a rules-level violation, not a judgement call. |
| **DiD** | No time dimension; one observation per user. | Synthetic panel only. |
| **RDD** | The level-30 threshold is crossed endogenously by player skill/engagement. | Illustrative synthetic example, with the endogeneity stated. |
| **Sequential testing** | No timestamps, so no real accrual order. | Simulate accrual by permuting arrival order; state that this is a simulation of the peeking problem, not an observed one. |

**Open question (O1):** `sum_gamerounds` is measured *after* treatment, so it is a
mediator, not a covariate. Any use of it as a control variable changes the estimand
from a total effect to something conditional. This must be settled in Phase 1 and
recorded in Rules.md.

---

## 7. Constraints & risks

| Risk | Mitigation |
|---|---|
| **Scope sprawl** — causal inference is bottomless | Phases.md defines exit criteria per phase. A method not in the phase plan does not get built. |
| **Silent statistical error** — plausible code, wrong numbers | Calibration tests (G3) + cross-validation against reference implementations. This is the top technical risk. |
| **Learning without retention** | Each notebook ends with a written interpretation in plain language, plus what would change the conclusion. |
| **Fake rigour** — dressing up synthetic demos as real findings | §6 table, plus a mandatory `data_source: real \| synthetic` field on every readout. |
| Kaggle dataset access requires an account | Document the manual download path; cache a Parquet copy locally; never commit the raw data. |

---

## 8. Reading path

Each is mapped to a specific phase in Phases.md.

| Source | Use |
|---|---|
| **Kohavi, Tang & Xu — _Trustworthy Online Controlled Experiments_** | The spine. Ch. 1–4 for fundamentals; the CUPED and SRM material for Phase 4. |
| **_Causal Inference for the Brave and True_** (free, Python-native) | The causal half — propensity scores, DiD, IV, sensitivity. |
| **Udacity "A/B Testing"** (originally Google) | Optional guided pass over Phases 1–3; useful for framing the design step. |
| Kaggle *Cookie Cats* dataset | The data. |

---

## 9. Open questions

- **O1** — Is `sum_gamerounds` ever admissible as a control? (See §6. Blocking for Phase 1.)
- **O2** — Which is the primary metric, `retention_1` or `retention_7`? Must be
  fixed *before* the first analysis and recorded in the pre-registration spec.
  Picking after seeing results is the exact failure this project exists to avoid.
- **O3** — What is the practical-significance threshold? A statistically
  significant 0.1pp retention change may be commercially irrelevant. Needs a stated
  number, even if arbitrary, before Phase 3.
- **O4** — Does the observed split imbalance pass the pre-registered SRM threshold?
  Deliberately left to be *measured* in Phase 1, not assumed here.
