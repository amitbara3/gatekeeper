# Retrospective

Phases 0–9, written at the close of Phase 9. Three parts: what the benchmark showed, the
bugs worth learning from, and what I would do differently.

---

## 1. What the benchmark showed

The predictions in Architecture §5 were written **before** the benchmark ran, and all three
held. The interesting part is which one was *worth* writing down.

**Prediction 1 (randomised data is easy) was never in doubt** and earned its place anyway,
as the control condition. When `outcome_regression` later turned out to have a broken
standard error, the randomised cell is what proved the *point estimate* was fine and
localised the fault to the variance. A benchmark without a known-good baseline cannot
distinguish "the regime broke it" from "it never worked".

**Prediction 2 (adjustment fixes observed confounding) held cleanly.** Naive bias +1.85
versus −0.004 for IPW is not a subtle difference. Worth noting how *large* the naive bias
is: 14.6 standard errors, which is why its coverage is 0.0% rather than merely poor. A
badly confounded estimate isn't a bit wrong — it's confidently wrong, with an interval that
excludes the truth every single time.

**Prediction 3 is the one that mattered.** I expected every adjustment method to fail under
unobserved confounding, and they did. What I did not anticipate was how *identical* the
failures would be: naive +1.850, IPW +1.829, outcome regression +1.829, AIPW +1.829. All
the sophistication bought exactly nothing.

The contrast test is what turns that from discouraging into informative. Supply `u` and
the same three estimators land at −0.0003, +0.0001, −0.0014. So the failure was diagnostic
of *missing data*, not of the methods. I would not have felt the force of "doubly robust
means robust to misspecification, not to omission" from reading it. Seeing AIPW post the
same bias as a difference in means made it stick.

**A surprise worth recording:** IPW over-covers, hitting 100% in two cells. I had expected
the propensity-as-fixed sandwich to be *anti*-conservative. It is conservative instead, at
least with stabilised weights in this regime. I recorded it in the estimator's own
assumptions rather than tuning it away — intervals that are too wide are a defensible
choice, but the reader should know which way they err.

---

## 2. Bugs worth learning from

Six real defects. The pattern across them is what I find most useful.

### The plausible-but-wrong standard error

`estimate_outcome_regression` computed `(mu1 - mu0).std() / sqrt(n)`. That looks like a
standard error and is not one: it measures how much the predicted contrast varies *across
units*, not how uncertain the *average* is. With a homogeneous effect the per-unit contrast
is nearly constant, so it collapsed toward zero — coverage 2.5%, |bias|/se of 1402, while
the bias itself was a healthy −0.02.

**Why it survived my own review:** every individual piece was defensible. It was an
average, divided by a root-n. Only the benchmark's coverage column caught it, because
coverage is the one diagnostic that tests the *uncertainty* rather than the estimate.

This is the single strongest argument for the four-layer testing strategy. Fixtures,
property tests, and reference cross-checks would all have passed.

### The inverted variance reduction, caused by my own documentation

`theoretical_reduction` was set to `1 - ρ²` when it should be `ρ²`. CUPED *multiplies*
variance by `1 - ρ²`, so the fraction *removed* is `ρ²`.

The root cause was Phases.md, which said "variance reduction matching theoretical `1 − ρ²`".
That phrasing is ambiguous in English and I implemented the wrong reading of my own spec.
I fixed the document as well as the code, because leaving the ambiguity would have
regenerated the bug.

### The gate that crashed instead of blocking

`check_srm` read its arm universe from the *observed* data. When an arm received zero
units — the most extreme SRM possible — it silently became a one-arm test and raised
`InsufficientData`. Since SRM runs first, the whole gate blew up. The worst possible
experiment failure produced the worst possible diagnostic, contradicting a rule I had
written hours earlier.

**The lesson:** "read the universe from the data" is exactly the class of shortcut that
works on healthy inputs and fails on the pathological ones a guard exists to catch. The
universe now comes from what was *intended*.

### The cache that outlived its schema

Parquet cache freshness compared mtimes against the CSV only. Because the cache holds
*validated* data, a hit skips revalidation — so narrowing the schema returned rows checked
against the old contract, including columns the new schema didn't declare. A schema
fingerprint is now part of the filename, so editing a schema misses the cache by
construction.

### Three times the *test* was wrong, not the code

This is the pattern I most want to remember.

1. **A KS test on discrete p-values.** The two-proportion p-value has atoms — at a 0.5 base
   rate with n=800, 2,000 draws give ~973 distinct values with one carrying 2.4% of the
   mass. KS assumes continuity and rejects systematically. The estimator was fine;
   `P(p ≤ 0.05)` came out at 0.053.
2. **An assertion about scipy's version.** I asserted `statsmodels` returns NaN at a
   specific input. True locally, false in CI. The test encoded a property of one scipy
   build rather than of my code.
3. **A hard-coded tolerance.** I allowed 0.15 for how far CUPED moves a point estimate; an
   unlucky seed produced 2.4 SD. The shift is *exactly* θ × covariate imbalance, whose SD
   is computable, so the bound is now derived from the data.

In all three the temptation was to loosen a threshold. R4.7 exists precisely because that
temptation is strongest when the code is actually correct — the failure feels spurious, so
widening feels harmless. The discipline that worked was asking **"is this assertion about
my code or about my environment?"** before touching anything.

### And twice I made the *same* mistake

I got Benjamini–Hochberg's tie behaviour wrong in Phase 2, wrote a test documenting the
correct behaviour, and then got it wrong again in Phase 3. Tied p-values inherit the
*least* conservative adjustment, because the adjusted value at rank *i* is the minimum over
*j ≥ i*. Writing something down is evidently not the same as learning it.

---

## 3. What I would do differently

**The original scope was about 2× too large**, and I said so before building. Cutting
O'Brien–Fleming and Rosenbaum bounds in favour of the mSPRT and E-values was right: both
survivors are closed-form, verifiable by simulation, and carry most of the intuition. I
should have made that cut in the plan rather than discovering it.

**I would build the calibration layer earlier.** It found the outcome-regression bug, the
peeking inflation, and the CUPED inversion. It is also the only layer that tests what an
estimator actually *claims*.

**The notebooks never happened.** They were deferred behind the dataset and the palette,
then behind the library. I do not regret the priority — a correct library with a broken
notebook is better than the reverse — but the curriculum was a stated goal and it did not
land. Every result in the README is reproducible from the test suite, which is a partial
substitute at best.

**The dataset dependency should have been faced on day one.** Three phases carry blocked
exit criteria for want of one CSV. Everything that *could* be built without it was, and
the synthetic generators turned out better than the real data would have been for
benchmarking — a known-by-construction τ beats an estimated one. But PRD O4 is still open,
and "did the real split pass SRM" was supposed to be Phase 1's lesson.

**What I would keep without changes:** the uniform `EffectEstimate` return type. It let the
benchmark harness score arbitrary estimators without knowing what they are, let calibration
tests be written once and parametrised, and made `assumptions` a required field so no
estimate can exist without declaring what it assumes. Every downstream convenience traces
back to that one decision in Phase 1.
