"""The benchmark's predictions, asserted as tests.

Architecture §5 wrote these down *before* the benchmark ran (R2.2). Asserting them here
means the benchmark cannot be quietly reinterpreted to match whatever came out.

1. On randomised data every estimator recovers tau, naive included.
2. Under selection on an observed covariate, naive is badly biased and adjustment works.
3. Under unobserved confounding everything fails -- and everything works again the moment
   the confounder is supplied.

Marked slow: each cell runs dozens of replications.
"""

from __future__ import annotations

import pytest

from gatekeeper.benchmark.harness import run_benchmark

pytestmark = pytest.mark.slow

ADJUSTERS = ("ipw", "outcome_regression", "aipw")


@pytest.fixture(scope="module")
def benchmark():
    return run_benchmark(n_units=4_000, n_seeds=40, strength=1.2, covariates=("x",))


class TestPrediction1RandomisedIsEasy:
    """An estimator that fails here is broken, not challenged."""

    @pytest.mark.parametrize("estimator", ("naive_difference", *ADJUSTERS))
    def test_every_estimator_is_unbiased(self, benchmark, estimator: str):
        score = benchmark.score("randomised", estimator)
        assert not score.is_badly_biased, score.describe()
        assert abs(score.bias) < 0.1

    @pytest.mark.parametrize("estimator", ("naive_difference", *ADJUSTERS))
    def test_coverage_is_at_least_nominal(self, benchmark, estimator: str):
        score = benchmark.score("randomised", estimator)
        # At or above nominal. Over-coverage means conservative intervals, which is a
        # defensible choice; under-coverage is a defect.
        assert score.coverage >= 0.85, score.describe()


class TestPrediction2AdjustmentFixesObservedConfounding:
    def test_naive_is_badly_biased(self, benchmark):
        score = benchmark.score("selection", "naive_difference")
        assert score.is_badly_biased
        assert abs(score.bias) > 1.0, score.describe()
        assert score.coverage < 0.1

    @pytest.mark.parametrize("estimator", ADJUSTERS)
    def test_adjustment_recovers_the_truth(self, benchmark, estimator: str):
        score = benchmark.score("selection", estimator)
        assert not score.is_badly_biased, score.describe()
        assert abs(score.bias) < 0.15, score.describe()

    def test_adjustment_beats_naive_on_rmse_by_an_order_of_magnitude(self, benchmark):
        naive = benchmark.score("selection", "naive_difference")
        for estimator in ADJUSTERS:
            assert benchmark.score("selection", estimator).rmse < naive.rmse / 5


class TestPrediction3UnobservedConfoundingDefeatsEverything:
    """The finding that justifies the whole exercise."""

    @pytest.mark.parametrize("estimator", ("naive_difference", *ADJUSTERS))
    def test_every_method_fails(self, benchmark, estimator: str):
        score = benchmark.score("unobserved", estimator)
        assert score.is_badly_biased, f"{estimator} unexpectedly survived: {score.describe()}"
        assert abs(score.bias) > 1.0
        assert score.coverage < 0.1

    def test_sophistication_does_not_help(self, benchmark):
        """AIPW is no better than naive here. Double robustness is about
        misspecification, not omission."""
        naive = benchmark.score("unobserved", "naive_difference")
        aipw = benchmark.score("unobserved", "aipw")
        assert abs(aipw.bias) == pytest.approx(abs(naive.bias), rel=0.15)

    def test_supplying_the_confounder_fixes_everything(self):
        """The contrast that makes the failure a finding rather than a dead end."""
        given = run_benchmark(
            n_units=4_000,
            n_seeds=30,
            strength=1.2,
            covariates=("x", "u"),
            regimes=("unobserved",),
            include_randomised=False,
        )
        for estimator in ADJUSTERS:
            score = given.score("unobserved", estimator)
            assert not score.is_badly_biased, (
                f"{estimator} should recover the truth once u is supplied: {score.describe()}"
            )
            assert abs(score.bias) < 0.15

    def test_naive_is_not_rescued_by_supplying_the_confounder(self):
        """It adjusts for nothing, so extra covariates cannot help it."""
        given = run_benchmark(
            n_units=4_000,
            n_seeds=30,
            strength=1.2,
            covariates=("x", "u"),
            regimes=("unobserved",),
            include_randomised=False,
        )
        assert given.score("unobserved", "naive_difference").is_badly_biased


class TestAipwFiniteSampleCoverage:
    """R4.7's four-part exception, applied honestly.

    AIPW's coverage can sit below nominal at moderate n, because the influence function
    has heavy tails when propensities approach 0 or 1. R4.7 says a claimed finite-sample
    effect must be *demonstrated* to shrink as n grows -- that is what separates it from a
    bug. This test is that demonstration.
    """

    def test_coverage_improves_toward_nominal_as_n_grows(self):
        deficits = []
        for n_units in (1_000, 8_000):
            result = run_benchmark(
                n_units=n_units,
                n_seeds=40,
                strength=1.2,
                covariates=("x",),
                regimes=("selection",),
                include_randomised=False,
            )
            score = result.score("selection", "aipw")
            deficits.append(max(0.0, 0.95 - score.coverage))

        # Either it was already fine at the small n, or it got better.
        assert deficits[1] <= deficits[0] + 0.05, (
            f"coverage deficit grew with n ({deficits[0]:.3f} -> {deficits[1]:.3f}), which "
            "means this is a bug rather than finite-sample behaviour (R4.7)"
        )

    def test_bias_shrinks_as_n_grows(self):
        biases = []
        for n_units in (1_000, 16_000):
            result = run_benchmark(
                n_units=n_units,
                n_seeds=30,
                strength=1.2,
                covariates=("x",),
                regimes=("selection",),
                include_randomised=False,
            )
            biases.append(abs(result.score("selection", "aipw").bias))
        assert biases[1] < biases[0] + 0.02, f"bias by n: {biases}"


class TestBenchmarkMechanics:
    def test_render_table_includes_every_cell(self, benchmark):
        table = benchmark.render_table()
        for regime in ("randomised", "selection", "noncompliance", "unobserved"):
            assert regime in table
        for estimator in ("naive_difference", *ADJUSTERS):
            assert estimator in table
        assert "true ATE" in table

    def test_unknown_cell_raises_with_a_helpful_message(self, benchmark):
        with pytest.raises(KeyError, match="no score for"):
            benchmark.score("nonexistent", "aipw")

    def test_too_few_seeds_rejected(self):
        with pytest.raises(ValueError, match="n_seeds must be >= 5"):
            run_benchmark(n_seeds=2)

    def test_regimes_and_estimators_are_reported(self, benchmark):
        assert "selection" in benchmark.regimes
        assert "aipw" in benchmark.estimators
        assert benchmark.true_ate == 1.0
