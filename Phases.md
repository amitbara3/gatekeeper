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

**Status at close of Phase 9: all ten phases delivered at the library level.**
710 tests, 94% coverage, `ruff` + `ruff format` + `mypy --strict` clean, CI green.

What did **not** land, and why:

- **The notebook curriculum.** Deferred behind the dataset, then behind the palette, then
  behind the library. A correct library with no notebook beats the reverse, but this was a
  stated goal that did not arrive. Every published result is reproducible from the test
  suite instead.
- **PRD O4** — whether the real split clears the SRM threshold. Needs the Kaggle CSV.
- **Alpha-spending** (O'Brien–Fleming, Pocock) — traded for the mSPRT, which is closed-form
  and directly verifiable by simulation. See Phase 4.
- **IV, DiD, Rosenbaum bounds** — cut. Cookie Cats has neither covariates nor a time
  dimension, so these would be synthetic-only demonstrations with low yield relative to
  cost. E-values carry most of the sensitivity intuition.

Full account in [RETROSPECTIVE.md](RETROSPECTIVE.md).

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
- [x] Fresh clone → install → `pytest` passes *(via `pip install -e ".[dev,viz,ml]"`;
      `uv` was not available on the dev machine, so the documented pip fallback is
      the working path — Architecture §1)*
- [x] `ruff check` + `ruff format --check` clean
- [x] `mypy --strict src/` clean
- [ ] CI green on a trivial PR — workflow written and YAML-validated locally, not yet
      exercised on GitHub
- [ ] Pre-commit hooks fire on a test commit — config written, `pre-commit install`
      not yet run

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
- [x] CSV loads with full schema validation; unexpected variant labels raise
- [x] SRM check implemented, tested against hand-computed fixtures *and* cross-checked
      against `scipy.stats.chisquare` to 1e-9
- [x] `specs/cookie_cats_gate.yaml` committed and locked: primary metric,
      guardrails, α, power, MDE, practical threshold, outlier rule, subgroups
- [x] PRD **O2** (primary metric → `retention_7`) and **O3** (practical threshold →
      1pp) resolved in writing, with reasoning, in the spec file
- [x] PRD **O1** (post-treatment covariates) closed — enforced in code by
      `ExperimentData.assert_pre_treatment`
- [x] Zero effect estimates produced
- [ ] **PRD O4** (does the real split clear the SRM threshold?) — blocked on the
      Kaggle download. The check is built and tested; it has not been run on the real
      data. Deliberately left unanswered rather than assumed.
- [ ] `notebooks/01_eda_and_sanity.ipynb` — deferred: the EDA notebook needs both the
      real data and the `viz/theme.py` palette (Phase 8). The library layer it would
      call is complete.

**Status: substantially complete.** 169 tests, 96% coverage, all three quality gates
green. The two open items both depend on the dataset download.

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
- [x] Every estimator returns an `EffectEstimate` with populated `assumptions`
- [x] Known-answer fixture per estimator, hand-computed (Welch's df = 8 case;
      pooled vs unpooled SE; BH step-up values; the zero-variance ratio)
- [x] Reference cross-check vs scipy/statsmodels — agreement to ~1e-16, far inside
      the 1e-6 bar, for the z-test, Welch (statistic, df, p, CI), BH, Bonferroni,
      and `TTestIndPower`
- [x] **Calibration passes** for every estimator: error rate matches α at α ∈
      {0.01, 0.05, 0.10}, and 95% CI coverage inside 93–97% over ≥ 1,000 sims
- [x] Power validated two ways: analytic vs `statsmodels.TTestIndPower` (means) and
      analytic vs empirical rejection rate over 4,000 simulations (proportions).
      *Stronger than the originally planned "textbook example", so that is what was
      done instead.*
- [x] Property tests: arm-swap sign symmetry, CI width monotonic in n, point
      estimate inside its own interval, shift invariance
- [ ] `notebooks/02_frequentist_core.ipynb` — deferred with Phase 1's notebook,
      pending the dataset and the Phase 8 palette

**Two corrections made during this phase** (both recorded because R2.3 asks for it):

1. **The calibration instrument was wrong, not the estimators.** A KS test against a
   continuous uniform rejected the two-proportion p-values — but that p-value is
   *discrete*: at a 0.5 base rate with n=800, 2,000 draws give only ~973 distinct
   values, one of them carrying 2.4% of the mass. KS is invalid there. Replaced with a
   level test (`P(p ≤ α) ≈ α`), which is valid for discrete and continuous alike, plus
   a test asserting the discreteness premise so the reasoning is verified rather than
   assumed. Welch's failure was separately shown to be pure seed noise (3 of 36 KS runs
   below 0.05 across 12 seeds, pooled `P(p ≤ 0.05) = 0.0488`); its test now checks the
   median KS p across seeds. **Neither change loosened a tolerance** — R4.7's
   distinction between a bug and a bad measurement is exactly this.

2. **`scipy.stats.nct` returns NaN at large df**, which broke the sample-size solver
   and, separately, makes `statsmodels.TTestIndPower` return NaN at d=0.8, n=400.
   `power_means` now falls back to the normal approximation above df=1,000 and on any
   non-finite result, so it is strictly more robust than the reference there. The
   reference test asserts our finiteness rather than reproducing the NaN.

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
- [x] `report/readout.py` built: decision stated against the **practical** threshold,
      not `p < 0.05`. Distinguishes an *informative null* (interval entirely inside
      ±threshold, so a meaningful effect is ruled out) from an *underpowered* one
      (interval neither inside nor outside), which is the R1.4 distinction that
      "not significant" usually erases.
- [x] Sanity gate demonstrably blocks: a failing report yields `Decision.BLOCKED`, no
      metric numbers appear in the render at all, and an override is recorded on the
      readout
- [ ] **One command reproduces the readout from raw CSV** — blocked on the Kaggle
      download
- [ ] **Interpretation written before consulting any public analysis** — blocked on
      the same

**Status: library layer complete, deliverable blocked on the dataset.** The decision
logic is built and tested against synthetic estimates; what remains is running it on the
real numbers.

**New limitation found while building this — PRD open question O5.** The spec carries a
single `practical_threshold` in the primary metric's units (retention proportion).
Applying that same `0.01` to `sum_gamerounds`, measured in rounds, would be
dimensionally meaningless. So guardrails are judged on BH-adjusted *statistical*
significance and a moved guardrail downgrades a ship to hold pending explanation, while
only the primary metric is judged on *practical* significance. Inventing per-guardrail
thresholds would have been worse — it would put numbers nobody chose into a decision
rule.

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
- [x] CUPED's measured variance reduction matches theory on synthetic data with known
      ρ. **Note the direction:** CUPED *multiplies* variance by `1 − ρ²`, so the
      fraction *removed* is `ρ²`. This line originally said the reduction equals
      `1 − ρ²`, and that phrasing is exactly what produced an inverted
      `theoretical_reduction` in the code — caught by a test at ρ=0.3 (achieved 0.092
      vs claimed 0.910). Measured across ρ ∈ {0, 0.3, 0.5, 0.7, 0.9}:

      | ρ | ρ² | achieved | CI width vs plain | effective n |
      |---|---|---|---|---|
      | 0.0 | 0.000 | 0.000 | 1.00× | 1.00× |
      | 0.3 | 0.090 | 0.090 | 0.95× | 1.10× |
      | 0.5 | 0.250 | 0.251 | 0.87× | 1.33× |
      | 0.7 | 0.490 | 0.492 | 0.71× | 1.97× |
      | 0.9 | 0.810 | 0.812 | 0.43× | 5.31× |

- [x] CUPED **raises** `PostTreatmentCovariateError` when handed `sum_gamerounds`,
      with tests asserting the raise, the message, and that the guard fires *before*
      any arithmetic
- [x] Peeking simulation reproduces the inflated false-positive rate and shows the
      always-valid p-value holding at α (true effect 0, α=0.05, 3,000 sims):

      | looks | naive | always-valid |
      |---|---|---|
      | 1 | 0.0523 (1.0×) | 0.0027 |
      | 2 | 0.0853 (1.7×) | 0.0053 |
      | 5 | 0.1390 (2.8×) | 0.0090 |
      | 10 | 0.1870 (3.7×) | 0.0120 |
      | 20 | 0.2380 (4.8×) | 0.0140 |
      | 50 | 0.3110 (6.2×) | 0.0173 |

      Reading once at the end gives 0.0523, confirming the harness itself is unbiased
      rather than the inflation being a simulation artefact.
- [x] Everything CUPED/sequential is labelled `data_source=SYNTHETIC` (R1.11)
- [ ] `notebooks/04_variance_and_sequential.ipynb` — deferred with the other notebooks

**Scope change: alpha-spending deferred, always-valid p-values built instead.**
O'Brien–Fleming and Pocock boundaries need the joint distribution of the sequential
statistics via recursive numerical integration — tractable, but a substantial piece of
numerical work whose correctness is hard to verify independently. The mSPRT is
closed-form, needs no pre-committed number of looks, and its guarantee
(`P(∃n: pₙ ≤ α) ≤ α`) is directly checkable by simulation, which is how it is validated
here. This matches the earlier recommendation to cut O'Brien–Fleming as a difficulty
spike. Alpha-spending remains available as a later addition.

**Honest note on conservatism:** the always-valid rate sits well *below* α (0.3–1.7%
where α=5%), so it is buying its guarantee with real power. That is the correct
trade-off for something monitored continuously, and it is stated rather than presented
as a free lunch.

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
- [x] Closed-form posterior verified. `P(B > A)` is computed by **quadrature**, not
      sampling, so the headline number is deterministic — cross-checked against 10 million
      Monte Carlo draws at four count configurations
- [x] Interpretive differences enforced in code, not just prose: `p_value` is deliberately
      `None` (there is no p-value in this framework, and filling it with `1 − P(B>A)` would
      invite the exact misreading), and a test documents that `P(B>A)` tracks the *one*-sided
      p only approximately while being emphatically not `1 − p`
- [x] Prior sensitivity shown **both ways** — swamped at n=45,000, dominant at n=10, since
      "the prior doesn't matter much" is a per-dataset claim rather than a general one
- [x] Decision rule via **expected loss**. A test shows why: at a million per arm,
      `P(B>A) = 0.99` for a 0.13pp effect, which a probability threshold would ship and
      expected loss correctly declines
- [ ] Side-by-side comparison on the **real** Cookie Cats data — blocked on the download.
      Agreement with the frequentist implementation is asserted on synthetic data instead
      (point estimates to 1e-4, interval widths to 5%)
- [ ] `notebooks/05_bayesian.ipynb` — deferred with the rest of the curriculum

Closed-form only — no MCMC (Architecture §1.1). One numerical subtlety worth recording: at
n=45,000 the posterior sd is ~0.0019, so the density occupies under 1% of [0,1] and adaptive
quadrature over the naive unit interval steps straight over the mass. The integration range
is derived from the posteriors' own moments, with the upper tail added analytically.

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
- [x] Every estimator scored on 4 scenarios (3 regimes + a randomised control) × 40 seeds
- [x] Naive difference-in-means biased under `"selection"`: **+1.854, 14.6 SE, 0% coverage**
- [x] IPW/AIPW recover τ when the confounder is observed: **−0.004 / −0.029**
- [x] **All** adjustment methods fail under `"unobserved"` (+1.829, identical to naive),
      and — the contrast that makes it a finding — the same methods land at −0.0003,
      +0.0001, −0.0014 once `u` is supplied
- [x] Overlap diagnostics on every propensity estimate: propensity range, max weight, and
      Kish effective sample size, with a warning below 0.05/0.95 and a **raise** on outright
      positivity failure. Trimming is off by default and, when used, announces that the
      estimand changed to the overlap region
- [x] Predicted vs actual compared in writing, surprises included (R2.2) —
      see [RETROSPECTIVE.md](RETROSPECTIVE.md) §1
- [ ] **Ground truth from Phase 3** — *deliberately changed.* The plan scored against
      τ̂\* from the real RCT, but τ̂\* is itself an estimate whose interval (±0.5pp) is the
      same order as the biases being measured. Scoring against it would conflate estimator
      bias with sampling error in the yardstick. The benchmark uses a synthetic DGP with an
      **exact** τ instead; the finite-population framing for Cookie Cats is documented and
      blocked on the download.
- [ ] **IV / LATE** — cut. With a homogeneous effect LATE equals ATE, so the demonstration
      would be vacuous; introducing heterogeneity purely to make IV interesting would be
      contrived. `true_late` is carried on the scenario so adding it later is mechanical.
- [ ] `causal/did.py`, Rosenbaum bounds — cut for the reasons in the phase table above
- [ ] `notebooks/06_causal_benchmark.ipynb` — deferred

**The overlap warning fired as predicted.** IPW's effective sample size drops sharply at
higher confounding strength, which a test asserts. Nothing trims automatically.

---

## Phase 7 — Heterogeneous effects

**Goal:** who is affected, without falling into subgroup fishing.

**Build**
- `hte/learners.py` — S-, T-, X-learner
- `hte/uplift.py` — Qini curves, uplift curves, decile lift
- Optional: cross-validate against `econml` (extras only, Rules §3)
- `notebooks/07_heterogeneous_effects.ipynb`

**Exit criteria**
- [x] Learners recover a known synthetic CATE function `τ(x) = 1.0 + 1.5x`. The T- and
      X-learners recover the *shape* at correlation > 0.7, not merely the average
- [x] The S-learner's shrinkage toward zero is **demonstrated, not asserted**: a test shows
      it finds less heterogeneity than T and X on identical data. A flexible learner handed
      one treatment column among many under-uses it, because dropping a weak feature costs
      little prediction loss
- [x] Qini/uplift curves validated three ways: a perfect ranking beats random, a random
      ranking does not, and a **reversed** ranking scores negative — a real finding, since a
      backwards model targets exactly the people the treatment harms
- [x] The dataset limit stated **and enforced**: a test enumerates Cookie Cats' usable
      pre-treatment covariates and requires the list to be **empty**, so if the schema ever
      gains one, this conclusion is revisited rather than silently inherited
- [ ] **Interaction test for pre-declared subgroups (R1.9)** — not built. The spec declares
      zero subgroups, correctly, because there are no covariates to form them from. Building
      the machinery with nothing to point it at would be speculative; R1.9 remains enforced
      at the spec level via `assert_subgroup_declared`.
- [ ] `econml` cross-validation, `notebooks/07_*.ipynb` — deferred

Cookie Cats has no pre-treatment covariates, so honest CATE estimation on the real data is
impossible. The machinery works; the dataset cannot feed it, and that is Phase 7's finding.

---

## Phase 8 — App & report

**Goal:** make the results legible to someone who will not read the notebooks.

**Build**
- `viz/theme.py` finalised — the palette from Design.md, single source (R4.8)
- `viz/static.py`, `viz/interactive.py`
- `app/streamlit_app.py` — spec + data in; sanity checks, metric results, decision out
- `report/render.py` — one-page HTML/Markdown export

**Exit criteria**
- [x] App runs with one command; **verified booting headless and serving HTTP 200** with no
      runtime errors
- [x] Sanity failures are a **blocking** state: the app withholds every metric number, and
      the override requires a typed reason that is stamped onto the readout
- [x] Synthetic results carry the badge in the app, the HTML, and the Markdown — a page
      lifted into a slide deck carries its own provenance
- [x] Design.md rules enforced in code rather than trusted: `series_color` **raises** past
      three series instead of cycling, `requires_relief` exposes the non-dismissable
      light-mode contrast WARN, and property tests assert the diverging midpoint is neutral
      and the sequential ramp monotone
- [x] Light and dark both emitted, under `prefers-color-scheme` **and** a `data-theme` scope,
      with an explicit `body` background so nothing borrows the host page's colours
- [x] HTML export is self-contained — no scripts, no external CSS or fonts — and escapes
      untrusted text (a test feeds it a `<script>` tag)
- [x] The app cannot override the spec: a test greps the source to confirm it reads
      `spec.practical_threshold`, `spec.alpha`, and `spec.srm_threshold` rather than
      hardcoding any of them. A UI able to change those would make pre-registration
      decorative (R1.2)
- [ ] `viz/static.py` / `viz/interactive.py` (matplotlib/plotly figures) — not built. The
      readout's signature chart is a difference-with-interval, which the HTML renders
      directly; chart modules were the lower-value half of this phase.
- [ ] Report export reproducing the **Phase 3** readout — blocked on the dataset, like
      Phase 3 itself

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
- [x] All CI jobs green — `lint`, `types`, `test`, `notebooks` (which passes trivially with
      no notebooks present, by design)
- [x] `pytest -m slow` passes end to end: the calibration suite, the peeking guarantee, and
      the benchmark predictions
- [x] Coverage gate met — **94%**, against an 85% floor
- [x] README carries the benchmark table as the headline result
- [x] [RETROSPECTIVE.md](RETROSPECTIVE.md) written: what the benchmark showed, all six real
      bugs with the pattern connecting them, and what I would do differently
- [x] **O1** (post-treatment covariates) closed and enforced in code;
      **O2** (primary metric) and **O3** (practical threshold) closed in the spec with
      reasoning; **O5** raised and documented during Phase 3
- [ ] **O4** (does the real split clear the SRM threshold) — the one open question. The
      check is built and cross-validated against scipy to 1e-9 but has never touched real
      data. Deliberately left unanswered rather than assumed.

**Nightly calibration** is configured in CI on a `schedule` trigger, skipped on pull
requests so the fast suite stays fast (Architecture §8).

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
