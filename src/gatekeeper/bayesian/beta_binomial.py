"""Bayesian A/B testing for binary metrics: the Beta-Binomial conjugate model.

The same question as the frequentist z-test, asked in a different framework -- and the
answers mean genuinely different things. Two distinctions the docstrings return to
because they are the ones people get wrong:

**A credible interval is not a confidence interval.** A 95% *credible* interval is a
statement about the parameter given this data: there is a 95% posterior probability the
rate difference lies inside it. A 95% *confidence* interval is a statement about the
procedure: intervals built this way cover the true value 95% of the time across
hypothetical repetitions. The first is what people usually *mean* when they read a
frequentist interval, which is exactly why the two get conflated.

**P(treatment > control) is not 1 - p.** The p-value is
``P(data at least this extreme | no effect)``. ``P(B > A)`` is
``P(treatment is better | this data and this prior)``. They often land at similar
numbers, which is a coincidence of the arithmetic rather than a correspondence of
meaning; a p-value of 0.04 does not license "96% chance treatment wins".

**Why no MCMC.** The Beta prior is conjugate to the Binomial likelihood, so the
posterior is another Beta -- available in closed form (Architecture §1.1). MCMC would be
slower, harder to test, and would hide the one piece of mathematics that is the point of
the exercise. ``P(B > A)`` is computed by quadrature so the headline number is
deterministic, not sampled; only quantities with no closed form (the posterior of the
*difference*, and expected loss) use sampling, with an explicit seed (R4.2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import integrate, stats

from gatekeeper.data.schema import ExperimentData
from gatekeeper.types import DataSource, EffectEstimate, Estimand, InsufficientData, Scale

__all__ = [
    "JEFFREYS_PRIOR",
    "UNIFORM_PRIOR",
    "BayesianComparison",
    "BetaPosterior",
    "BetaPrior",
    "compare_beta_binomial",
    "estimate_beta_binomial",
    "prior_sensitivity",
    "prob_b_beats_a",
]


@dataclass(frozen=True, slots=True)
class BetaPrior:
    """A Beta prior on a rate."""

    alpha: float
    beta: float
    name: str = "custom"

    def __post_init__(self) -> None:
        if self.alpha <= 0 or self.beta <= 0:
            raise ValueError(
                f"Beta prior parameters must be positive, got alpha={self.alpha}, beta={self.beta}"
            )

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def prior_sample_size(self) -> float:
        """``alpha + beta`` -- how many observations the prior is "worth".

        The most useful way to judge whether a prior is doing too much work: compare it
        against the real sample size. A Beta(1,1) prior carries the weight of two
        observations, which is nothing beside 45,000.
        """
        return self.alpha + self.beta

    def update(self, successes: int, trials: int) -> BetaPosterior:
        """Conjugate update: ``Beta(a + s, b + n - s)``. The whole model, one line."""
        if trials < 0 or successes < 0 or successes > trials:
            raise ValueError(
                f"need 0 <= successes <= trials, got successes={successes}, trials={trials}"
            )
        return BetaPosterior(
            alpha=self.alpha + successes,
            beta=self.beta + (trials - successes),
            prior=self,
            successes=successes,
            trials=trials,
        )


UNIFORM_PRIOR = BetaPrior(1.0, 1.0, name="uniform")
"""Beta(1,1) -- flat on [0,1]. Worth two observations."""

JEFFREYS_PRIOR = BetaPrior(0.5, 0.5, name="jeffreys")
"""Beta(0.5,0.5) -- the reference prior for a binomial rate.

Invariant under reparameterisation, and slightly less informative than uniform in the
tails. The conventional default when you want the data to speak.
"""


@dataclass(frozen=True, slots=True)
class BetaPosterior:
    """A Beta posterior over a rate."""

    alpha: float
    beta: float
    prior: BetaPrior
    successes: int
    trials: int

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def sd(self) -> float:
        a, b = self.alpha, self.beta
        return math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))

    @property
    def mode(self) -> float:
        """Posterior mode. Defined only when both parameters exceed 1."""
        if self.alpha <= 1 or self.beta <= 1:
            return self.mean
        return (self.alpha - 1) / (self.alpha + self.beta - 2)

    def credible_interval(self, level: float = 0.95) -> tuple[float, float]:
        """Equal-tailed credible interval.

        **Not** a confidence interval: this says there is ``level`` posterior
        probability that the rate lies inside, given the data and the prior.
        """
        if not 0.0 < level < 1.0:
            raise ValueError(f"level must be in (0, 1), got {level}")
        tail = (1.0 - level) / 2.0
        return (
            float(stats.beta.ppf(tail, self.alpha, self.beta)),
            float(stats.beta.ppf(1.0 - tail, self.alpha, self.beta)),
        )

    def sample(self, size: int, rng: np.random.Generator) -> np.ndarray:
        return rng.beta(self.alpha, self.beta, size)


def prob_b_beats_a(posterior_a: BetaPosterior, posterior_b: BetaPosterior) -> float:
    """``P(rate_b > rate_a)`` by quadrature.

    Computed as ``integral of f_B(x) * F_A(x) dx``, which is exact for smooth integrands
    and -- unlike sampling -- deterministic, so the headline number does not wobble
    between runs.

    Returns
    -------
    float
        Posterior probability that B's rate exceeds A's.

    Notes
    -----
    At realistic sample sizes both posteriors are extremely peaked: at n=45,000 and a
    19% rate the posterior sd is about 0.0019, so the density is concentrated in a
    window narrower than 1% of [0, 1]. Integrating naively over the whole unit interval
    invites an adaptive quadrature routine to step straight over the mass and return
    nonsense. So the integration range is set from the posteriors' own means and
    standard deviations, and the mass beyond the upper limit is added analytically::

        for x > hi, F_A(x) ~= 1, so that tail contributes P(B > hi) exactly

    A test compares this against Monte Carlo with ten million draws.
    """
    lo_center = min(posterior_a.mean, posterior_b.mean)
    hi_center = max(posterior_a.mean, posterior_b.mean)
    spread = 12.0 * max(posterior_a.sd, posterior_b.sd)
    lo = max(0.0, lo_center - spread)
    hi = min(1.0, hi_center + spread)

    if lo >= hi:
        # Degenerate; fall back to the full interval.
        lo, hi = 0.0, 1.0

    def integrand(x: float) -> float:
        return float(
            stats.beta.pdf(x, posterior_b.alpha, posterior_b.beta)
            * stats.beta.cdf(x, posterior_a.alpha, posterior_a.beta)
        )

    value, _ = integrate.quad(integrand, lo, hi, limit=200)
    # Everything above `hi` has F_A ~= 1 and contributes B's own upper tail.
    tail = float(stats.beta.sf(hi, posterior_b.alpha, posterior_b.beta))
    return float(min(1.0, max(0.0, value + tail)))


@dataclass(frozen=True, slots=True)
class BayesianComparison:
    """Posterior comparison of two rates."""

    control: BetaPosterior
    treatment: BetaPosterior
    prob_treatment_better: float
    lift_absolute: float
    lift_credible_interval: tuple[float, float]
    expected_loss_ship: float
    """Expected regret from shipping treatment: ``E[max(rate_c - rate_t, 0)]``."""
    expected_loss_keep: float
    """Expected regret from keeping control: ``E[max(rate_t - rate_c, 0)]``."""
    credible_level: float
    n_samples: int
    seed: int

    @property
    def prob_control_better(self) -> float:
        return 1.0 - self.prob_treatment_better

    def decision_at(self, loss_threshold: float) -> Literal["ship", "keep", "undecided"]:
        """Decide by **expected loss**, not by ``P(B > A) > 0.95``.

        The expected-loss rule asks "how much do I stand to lose if I am wrong?", which
        is the question a decision-maker actually has. ``P(B > A)`` alone ignores
        magnitude entirely: a 99% probability of a 0.01pp gain is a strong probability
        about something not worth shipping for.

        Ship when the expected regret from shipping is below the threshold; keep when
        the regret from keeping is below it; otherwise gather more data.
        """
        if loss_threshold <= 0:
            raise ValueError(f"loss_threshold must be positive, got {loss_threshold}")
        ship_ok = self.expected_loss_ship < loss_threshold
        keep_ok = self.expected_loss_keep < loss_threshold
        if ship_ok and not keep_ok:
            return "ship"
        if keep_ok and not ship_ok:
            return "keep"
        if ship_ok and keep_ok:
            # Both options are cheap to get wrong: the arms are near-equivalent, so
            # prefer the status quo rather than churning for nothing.
            return "keep"
        return "undecided"

    def describe(self) -> str:
        lo, hi = self.lift_credible_interval
        pct = round(self.credible_level * 100)
        return (
            f"P(treatment > control) = {self.prob_treatment_better:.4f}\n"
            f"  absolute lift {self.lift_absolute:+.5f} "
            f"[{lo:+.5f}, {hi:+.5f}] ({pct}% credible interval)\n"
            f"  control  posterior mean {self.control.mean:.5f} (sd {self.control.sd:.5f})\n"
            f"  treatment posterior mean {self.treatment.mean:.5f} "
            f"(sd {self.treatment.sd:.5f})\n"
            f"  expected loss: ship {self.expected_loss_ship:.6f}, "
            f"keep {self.expected_loss_keep:.6f}\n"
            f"  prior: {self.control.prior.name} "
            f"(worth {self.control.prior.prior_sample_size:g} observations)"
        )


def compare_beta_binomial(
    successes_control: int,
    trials_control: int,
    successes_treatment: int,
    trials_treatment: int,
    *,
    prior: BetaPrior = JEFFREYS_PRIOR,
    credible_level: float = 0.95,
    n_samples: int = 200_000,
    seed: int = 0,
) -> BayesianComparison:
    """Compare two rates under a Beta-Binomial model.

    Parameters
    ----------
    successes_control, trials_control
        Control arm counts.
    successes_treatment, trials_treatment
        Treatment arm counts.
    prior
        Beta prior, applied to **both** arms. Defaults to Jeffreys.
    credible_level
        Level for the credible interval on the difference.
    n_samples
        Posterior draws used for the *difference* and the expected losses. The headline
        ``P(B > A)`` does not use sampling -- see :func:`prob_b_beats_a`.
    seed
        RNG seed; recorded on the result (R4.2).

    Returns
    -------
    BayesianComparison

    Notes
    -----
    The posterior of the **difference** of two Betas has no standard closed form, so it
    is sampled. That is an honest split: exact where exactness is available,
    Monte Carlo where it is not, and the result records which is which.
    """
    if trials_control < 1 or trials_treatment < 1:
        raise InsufficientData(
            f"both arms need at least one trial, got {trials_control} and {trials_treatment}"
        )
    if n_samples < 1_000:
        raise ValueError(
            f"n_samples must be >= 1000 for a usable credible interval, got {n_samples}"
        )
    if not 0.0 < credible_level < 1.0:
        raise ValueError(f"credible_level must be in (0, 1), got {credible_level}")

    control = prior.update(successes_control, trials_control)
    treatment = prior.update(successes_treatment, trials_treatment)

    prob_better = prob_b_beats_a(control, treatment)

    rng = np.random.default_rng(seed)
    draws_c = control.sample(n_samples, rng)
    draws_t = treatment.sample(n_samples, rng)
    diff = draws_t - draws_c

    tail = (1.0 - credible_level) / 2.0
    interval = (
        float(np.quantile(diff, tail)),
        float(np.quantile(diff, 1.0 - tail)),
    )

    return BayesianComparison(
        control=control,
        treatment=treatment,
        prob_treatment_better=prob_better,
        lift_absolute=treatment.mean - control.mean,
        lift_credible_interval=interval,
        expected_loss_ship=float(np.mean(np.maximum(draws_c - draws_t, 0.0))),
        expected_loss_keep=float(np.mean(np.maximum(draws_t - draws_c, 0.0))),
        credible_level=credible_level,
        n_samples=n_samples,
        seed=seed,
    )


def prior_sensitivity(
    successes_control: int,
    trials_control: int,
    successes_treatment: int,
    trials_treatment: int,
    *,
    priors: tuple[BetaPrior, ...] = (),
    seed: int = 0,
) -> dict[str, BayesianComparison]:
    """Re-run the comparison under several priors, to show where the prior matters.

    At large n the prior is swamped and the answer barely moves; at small n it dominates.
    Demonstrating that rather than asserting it is the point -- "the prior doesn't matter
    much" is a claim that has to be checked per dataset, not assumed.

    Parameters
    ----------
    priors
        Priors to compare. Defaults to Jeffreys, uniform, a sceptical
        Beta(50,50) centred on 0.5, and a strong Beta(2,98) centred on 0.02.
    """
    if not priors:
        priors = (
            JEFFREYS_PRIOR,
            UNIFORM_PRIOR,
            BetaPrior(50.0, 50.0, name="sceptical_50_50"),
            BetaPrior(2.0, 98.0, name="strong_low_rate"),
        )
    return {
        p.name: compare_beta_binomial(
            successes_control,
            trials_control,
            successes_treatment,
            trials_treatment,
            prior=p,
            seed=seed,
        )
        for p in priors
    }


def estimate_beta_binomial(
    data: ExperimentData,
    estimand: Estimand,
    *,
    prior: BetaPrior = JEFFREYS_PRIOR,
    credible_level: float = 0.95,
    n_samples: int = 200_000,
    seed: int = 0,
    treatment_arm: str | None = None,
) -> EffectEstimate:
    """Bayesian estimate, packaged as an :class:`EffectEstimate` for uniformity.

    The interval carried in ``ci`` is a **credible** interval, not a confidence
    interval, and ``p_value`` is deliberately ``None`` -- there is no p-value in this
    framework, and filling the field with ``1 - P(B > A)`` would invite exactly the
    misreading this module warns about. ``P(B > A)`` lives in ``diagnostics``.
    """
    if estimand.scale is Scale.RELATIVE:
        raise NotImplementedError(
            "relative-scale Bayesian comparison is not implemented; report the absolute "
            "difference, or sample the ratio directly"
        )

    control = data.control
    treatment = treatment_arm if treatment_arm is not None else data.treatment

    values_c = data.outcome(estimand.outcome, control)
    values_t = data.outcome(estimand.outcome, treatment)
    for label, values in (("control", values_c), ("treatment", values_t)):
        if not set(values.tolist()) <= {0.0, 1.0}:
            raise ValueError(
                f"{estimand.outcome!r} is not binary in the {label} arm; the "
                "Beta-Binomial model applies to binary metrics only"
            )

    comparison = compare_beta_binomial(
        int(values_c.sum()),
        int(values_c.size),
        int(values_t.sum()),
        int(values_t.size),
        prior=prior,
        credible_level=credible_level,
        n_samples=n_samples,
        seed=seed,
    )

    assumptions = [
        f"units ({data.schema.unit_col}) are independent Bernoulli draws",
        f"Beta({prior.alpha:g}, {prior.beta:g}) prior ({prior.name}) on BOTH arms' rates, "
        f"worth {prior.prior_sample_size:g} observations against "
        f"{values_c.size + values_t.size:,} real ones",
        "the interval is a CREDIBLE interval: given this data and prior, there is "
        f"{credible_level:.0%} posterior probability the effect lies inside it. It is "
        "NOT a confidence interval and must not be described as one",
        "P(treatment > control) in diagnostics is a posterior probability, NOT 1 minus "
        "a p-value; no p-value is reported because the framework has none",
        f"posterior of the DIFFERENCE sampled with {n_samples:,} draws (seed {seed}); "
        "P(B > A) itself computed exactly by quadrature",
    ]
    if data.data_source is DataSource.REAL:
        assumptions.append("assignment was randomised, licensing a causal reading")
    else:
        assumptions.append(f"data is {data.data_source}, not a real experiment (R1.11)")

    return EffectEstimate(
        estimand=estimand,
        point=comparison.lift_absolute,
        ci=comparison.lift_credible_interval,
        ci_level=credible_level,
        se=None,
        p_value=None,
        method="beta_binomial_posterior",
        assumptions=tuple(assumptions),
        data_source=data.data_source,
        n_per_arm={control: int(values_c.size), treatment: int(values_t.size)},
        diagnostics={
            "prob_treatment_better": comparison.prob_treatment_better,
            "posterior_mean_control": comparison.control.mean,
            "posterior_mean_treatment": comparison.treatment.mean,
            "posterior_sd_control": comparison.control.sd,
            "posterior_sd_treatment": comparison.treatment.sd,
            "expected_loss_ship": comparison.expected_loss_ship,
            "expected_loss_keep": comparison.expected_loss_keep,
            "prior_alpha": prior.alpha,
            "prior_beta": prior.beta,
        },
        seed=seed,
    )
