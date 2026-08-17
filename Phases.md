# Phases — Gatekeeper

**Last updated** 2026-08-17 · Start date **2026-08-17**

Causal inference is bottomless, so this plan exists to bound it. **A phase is done
when its exit criteria are met — not when it feels done, and not when the next
topic looks more interesting.** One branch per phase (`phase-N-slug`), merged by PR
with CI green.

Durations are effort estimates at a few evenings per week, not commitments. The
sequence is the real content; slide the dates freely.

| Phase | Focus | Est. | Target start |
|---|---|---|---|
| 0 | Scaffold & tooling | 1 day | 2026-08-17 |
| 1 | Data, EDA & sanity checks | 1 wk | 2026-08-18 |
| 2 | Frequentist core | 1.5 wk | 2026-08-25 |
| 3 | **The Cookie Cats readout** | 0.5 wk | 2026-09-04 |
| 4 | Variance reduction & sequential | 1.5 wk | 2026-09-08 |
| 5 | Bayesian A/B | 1 wk | 2026-09-18 |
| 6 | **Causal inference + benchmark** | 3 wk | 2026-09-25 |
| 7 | Heterogeneous effects | 1.5 wk | 2026-10-16 |
| 8 | App & report | 1 wk | 2026-10-26 |
| 9 | Hardening & write-up | 1 wk | 2026-11-02 |

**Phases 3 and 6 are the milestones.** Phase 3 is the first end-to-end result;
Phase 6 is the project's thesis (PRD §1.1). Everything else is scaffolding for
those two.

---

## Phase 0 — Scaffold & tooling

**Goal:** a repo where the quality gates exist before there is any code to gate.

**Build**
- `pyproject.toml` (uv), pinned lockfile, `src/gatekeeper/` package skeleton
- `ruff` + `mypy --strict` + `pytest` configured and passing on an empty package
- `.pre-commit-config.yaml` including `nbstripout`
- `.gitignore` — `data/raw/`, `data/processed/`, `.env`, `.venv/`, `__pycache__/`
- GitHub Actions: `lint`, `types`, `test` (Architecture §8)
- `README.md` with setup + how to obtain the dataset
- `data/README.md` explaining the manual Kaggle download

**Exit criteria**
- [ ] Fresh clone → `uv sync` → `pytest` passes
- [ ] CI green on a trivial PR
- [ ] `mypy --strict src/` clean
- [ ] Pre-commit hooks fire on a test commit

**Note:** doing this first is deliberate. Retrofitting `mypy --strict` onto a
codebase of statistical helpers is far more painful than starting with it.

---

## Phase 1 — Data, EDA & sanity checks

**Reading:** Kohavi ch. 1–3 (what a controlled experiment is, the metrics
taxonomy, common pitfalls) · Kohavi's SRM material

**Goal:** know the data cold, and build the gate that stands between data and
results. **No effect estimates in this phase** — resisting that is the point.

**Build**
- `data/schema.py` — column contract, dtype validation, allowed variant labels
- `data/ingest.py` — CSV → `ExperimentData`, Parquet cache, fail-loud validation
- `types.py` — `Estimand`, `EffectEstimate`, `SanityReport`, `DataSource`, exceptions
- `spec.py` + `specs/cookie_cats_gate.yaml` — the pre-registration (R1.2)
- `design/srm.py` — chi-square SRM, default threshold `p < 0.0005`
- `checks/integrity.py` — duplicate ids, cross-arm units, nulls
- `checks/outliers.py` — tail profile and leverage; **reports, never trims** (R1.6)
- `notebooks/01_eda_and_sanity.ipynb`

**Deliverables**
- Distribution of `sum_gamerounds` (log scale — it is severely right-skewed),
  `retention_1` / `retention_7` rates per arm
- A written SRM verdict against the pre-registered threshold
- The outlier rule, **declared in the spec before Phase 2**

**Exit criteria**
- [ ] Raw CSV loads with full schema validation; unexpected labels raise
- [ ] SRM check implemented, tested against a hand-computed 2×2 fixture
- [ ] `specs/cookie_cats_gate.yaml` committed and locked: primary metric,
      guardrails, α, power, MDE, practical threshold, outlier rule, subgroups
- [ ] PRD **O2** (primary metric) and **O3** (practical threshold) resolved in writing
- [ ] PRD **O4** answered with a measured number, not an assumption
- [ ] Zero effect estimates produced

**Trap to expect:** the arms are *not* an exact 50/50 split. Whether that
constitutes an SRM depends entirely on the threshold — which is why the threshold
goes in the spec **before** the number is looked at. Whichever way it lands, write
the reasoning down; this is the phase's real lesson.

---

## Phase 2 — Frequentist core

**Reading:** Kohavi ch. 4 (statistical fundamentals) · Udacity A/B Testing (design
and sizing sections)

**Goal:** correct, calibrated implementations of the everyday tests.

**Build**
- `design/power.py` — sample size, power, MDE, duration (proportions + means)
- `frequentist/proportions.py` — two-proportion z, absolute + relative lift, CI
- `frequentist/means.py` — Welch's t (R1.12)
- `frequentist/bootstrap.py` — percentile + BCa
- `frequentist/ratio.py` — delta method
- `frequentist/multiplicity.py` — BH + Bonferroni
- `tests/calibration/` — the null-uniformity and CI-coverage harness
- `notebooks/02_frequentist_core.ipynb`

**Exit criteria**
- [ ] Every estimator returns an `EffectEstimate` with populated `assumptions`
- [ ] Known-answer fixture per estimator (hand-computed)
- [ ] Reference cross-check vs scipy/statsmodels to `< 1e-6` where one exists
- [ ] **Calibration passes:** p-values uniform under the null (KS p > 0.05) and 95%
      CI coverage in 93–97% over ≥ 1,000 sims, for every estimator
- [ ] Power curve reproduces a textbook sample-size example
- [ ] Property tests: arm-swap sign symmetry, CI width monotonic in n

**Do not** proceed to Phase 3 with a failing calibration test. A miscalibrated
z-test invalidates every downstream phase, and loosening the tolerance to get green
is explicitly forbidden (Rules §7).

---

## Phase 3 — The Cookie Cats readout 🎯

**Goal:** the first real end-to-end answer. Spec → sanity gate → analysis →
decision, on real data, using only what Phases 1–2 built.

**Build**
- `report/readout.py` — `EffectEstimate` → `Decision` against the spec's practical
  threshold (R1.4, F7.4)
- `notebooks/03_cookie_cats_readout.ipynb`
- First figures via `viz/theme.py` (Design.md palette)

**Deliverables**
- Effect of `gate_40` vs `gate_30` on the **pre-registered primary metric**, with
  CI, p-value, and a ship/hold/inconclusive decision
- Guardrail metrics, BH-corrected
- A plain-language interpretation: what the CI rules out, what it does not, and
  what would change the conclusion

**Exit criteria**
- [ ] One command reproduces the readout from raw CSV
- [ ] Decision stated against the **practical** threshold, not `p < 0.05`
- [ ] Sanity gate demonstrably blocks on a deliberately corrupted input
- [ ] Interpretation written before consulting any public analysis of this dataset

**Rule for this phase:** Cookie Cats is a popular Kaggle dataset with many public
notebooks. **Do not read them until this phase's interpretation is written.**
Comparing afterwards is valuable; anchoring beforehand wastes the exercise. Also
record the estimate carefully — it becomes the **ground truth τ̂\*** for Phase 6.

---

## Phase 4 — Variance reduction & sequential testing

**Reading:** Kohavi's CUPED chapter · Kohavi on sequential testing / peeking

**Goal:** the two techniques that most change practice — smaller tests, and honest
early stopping.

**Build**
- `variance/cuped.py` — CUPED, with a **hard guard rejecting post-treatment
  covariates** (R1.7, `PostTreatmentCovariateError`)
- `sequential/alpha_spending.py` — O'Brien–Fleming, Pocock
- `sequential/always_valid.py` — mSPRT / always-valid p-values
- Simulation harness: peeking under a true null, with and without correction
- `notebooks/04_variance_and_sequential.ipynb`

**Exit criteria**
- [ ] CUPED achieves measured variance reduction matching theoretical `1 − ρ²` on
      synthetic data with known ρ
- [ ] CUPED **raises** when handed `sum_gamerounds` as the covariate — with a test
      asserting the raise
- [ ] Peeking simulation reproduces the inflated false-positive rate for
      uncorrected repeated looks, and shows corrected methods holding at α
- [ ] Everything CUPED/sequential labelled `data_source=SYNTHETIC` (R1.11)

**Honesty requirement:** Cookie Cats has no pre-period covariate and no timestamps
(PRD §6). CUPED and sequential accrual are therefore **synthetic-only** here.
Applying CUPED to the real data with a post-treatment covariate would produce a
number that looks like variance reduction and is actually bias. Naming that trap is
a deliverable of this phase.

---

## Phase 5 — Bayesian A/B

**Reading:** Bayesian A/B chapters in *Causal Inference for the Brave and True*

**Goal:** the same question in a different framework, and a clear account of how
the two differ.

**Build**
- `bayesian/beta_binomial.py` — conjugate posterior, `P(treatment > control)`,
  credible intervals, expected loss
- Prior sensitivity analysis
- `notebooks/05_bayesian.ipynb`

**Exit criteria**
- [ ] Closed-form posterior verified against a large-sample simulation
- [ ] Bayesian and frequentist conclusions on Cookie Cats compared side by side,
      with the interpretive difference stated precisely (a credible interval is not
      a confidence interval, and `P(B > A)` is not `1 − p`)
- [ ] Prior sensitivity shown: where the prior does and does not matter
- [ ] Decision rule via **expected loss**, not just `P(B > A) > 0.95`

Closed-form only — no MCMC (Architecture §1.1).

---

## Phase 6 — Causal inference & the estimator benchmark 🎯

**Reading:** *Causal Inference for the Brave and True* — propensity scores, IPW,
doubly robust, IV, DiD, sensitivity analysis

**Goal:** the project's thesis. Break randomisation deliberately, then measure which
estimators recover the known answer and by how much they miss.

Largest phase; three sub-steps.

### 6a — The confounding simulator
- `causal/confounding.py` — `make_regime(seed, strength, kind)` for
  `"selection"`, `"noncompliance"`, `"unobserved"` (Architecture §5)
- `benchmark/harness.py` + `benchmark/scoring.py` — bias, variance, RMSE, CI coverage
- **Write down expected results before running anything** (R2.2)

### 6b — The estimators
- `causal/propensity.py` — estimation, IPW, stabilised weights, matching, trimming
- `checks/balance.py` — SMDs, love plots, positivity/overlap diagnostics
- `causal/outcome.py` — regression adjustment
- `causal/aipw.py` — doubly robust
- `causal/iv.py` — Wald / 2SLS, LATE under simulated non-compliance
- `causal/did.py` — synthetic panel only
- `causal/sensitivity.py` — E-values, Rosenbaum-style bounds

### 6c — The benchmark
- `notebooks/06_causal_benchmark.ipynb` — the headline artifact

**Exit criteria**
- [ ] Every estimator scored on ≥ 3 confounding regimes × ≥ 100 seeds
- [ ] Ground-truth τ̂\* from Phase 3 used as the target
- [ ] Naive difference-in-means shown to be biased under `"selection"`
- [ ] IPW/AIPW shown to recover τ̂\* when the confounder is **observed**
- [ ] **All** adjustment methods shown to fail under `"unobserved"`, with
      sensitivity analysis quantifying how much confounding it took
- [ ] IV recovers **LATE**, and the write-up states explicitly why LATE ≠ ATE here
      rather than treating the gap as estimator error
- [ ] Balance diagnostics reported for every propensity-based estimate
- [ ] Written comparison of predicted vs actual outcomes, surprises included (R2.2)

**Watch for:** poor overlap silently producing enormous IPW weights. Diagnose it,
report it in `diagnostics`, and warn — never quietly trim to make the estimate look
better.

---

## Phase 7 — Heterogeneous effects

**Goal:** who is affected, without falling into subgroup fishing.

**Build**
- `hte/learners.py` — S-, T-, X-learner
- `hte/uplift.py` — Qini curves, uplift curves, decile lift
- Optional: cross-validate against `econml` (extras only, Rules §3)
- `notebooks/07_heterogeneous_effects.ipynb`

**Exit criteria**
- [ ] Learners recover a **known** synthetic CATE function
- [ ] Qini/uplift curves implemented and validated against a hand-computed case
- [ ] Subgroups pre-declared in the spec; **interaction test reported** (R1.9)
- [ ] A written statement of how much heterogeneity this dataset can actually
      detect given its covariates — which is very little, since it has almost none.
      Saying so is the correct outcome.

Cookie Cats has essentially no pre-treatment covariates, so honest CATE estimation
on the real data is close to impossible. The value here is the machinery plus the
judgement to recognise that limit.

---

## Phase 8 — App & report

**Goal:** make the results legible to someone who will not read the notebooks.

**Build**
- `viz/theme.py` finalised — the palette from Design.md, single source (R4.7)
- `viz/static.py`, `viz/interactive.py`
- `app/streamlit_app.py` — spec + data in; sanity checks, metric results, decision out
- `report/render.py` — one-page HTML/Markdown export

**Exit criteria**
- [ ] App runs from a fresh clone with one command
- [ ] Sanity-check failures surface as a **blocking** state in the UI, not a footnote
- [ ] Synthetic/semi-synthetic results carry the visible badge (Design.md)
- [ ] Every chart follows Design.md: CI always shown, no dual axes, legend + direct
      labels for variant identity
- [ ] Light and dark modes both verified by eye
- [ ] Report export reproduces the Phase 3 readout

---

## Phase 9 — Hardening & write-up

**Goal:** leave it in a state that is defensible six months from now.

**Build**
- Coverage to ≥ 85% on `src/gatekeeper/`
- Nightly calibration job in CI (Architecture §8)
- Docstrings with `Assumptions` sections throughout (R4.6)
- `README.md` rewritten as the project's front door: thesis, benchmark table, how
  to reproduce
- A written retrospective: what the benchmark showed, what surprised me, what I
  would do differently

**Exit criteria**
- [ ] All CI jobs green, including notebook execution
- [ ] `pytest -m slow` (calibration) passes end to end
- [ ] Coverage gate met
- [ ] Every PRD open question (O1–O4) closed in writing
- [ ] README carries the estimator benchmark table as the headline result
- [ ] Retrospective written

---

## Rules for the plan itself

1. **Do not start a phase before the previous one's exit criteria are checked off.**
2. **Do not build ahead.** A method from Phase 6 does not get a head start during
   Phase 2. This plan exists because the subject has no natural stopping point.
3. **A failing calibration test blocks the phase.** Always.
4. **Update this file when reality diverges.** A plan quietly abandoned is worse
   than a plan revised in writing.
5. **Phases 4, 5, and 7 are individually skippable** if time runs short. Phases
   0–3 and 6 are the project; the rest are enrichment.
