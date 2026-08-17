"""Break randomisation on purpose, in ways we control.

This is the machinery behind the project's thesis (PRD §1.1). A randomised experiment
gives a trustworthy effect estimate; we then construct observational samples in which
randomisation is broken in a *known* way, and score each causal estimator on a question
whose answer we already have.

**Where ground truth comes from, and a correction to the original plan.** PRD §1.1
proposed scoring estimators against ``tau_hat*``, the Cookie Cats RCT estimate. That is
subtly wrong: ``tau_hat*`` is itself an *estimate*, and at n=45,000 per arm with a 19%
base rate its 95% interval is roughly ±0.5pp -- the same order as the biases being
measured. Scoring against it would conflate estimator bias with sampling error in the
yardstick.

So the benchmark uses a **fully synthetic DGP whose true tau is exact by construction**.
Two consequences worth being explicit about:

- Bias is measured against a parameter, not an estimate, so a measured bias of 0.01 means
  bias and nothing else.
- Applying this to Cookie Cats needs the *finite-population* framing instead: treat the
  90,189 rows as the population, so the population ATE is exactly the observed difference
  in means and carries no sampling error. That path is available but blocked on the
  dataset download.

**The three regimes**, and what each is designed to break:

``selection``
    Units enter the sample with probability depending on both the covariate and the arm.
    Treatment and control populations therefore differ on ``x``, which is *observed*. An
    estimator that adjusts for ``x`` should recover tau; a naive difference in means
    should not.

``noncompliance``
    One-sided: some assigned-to-treatment units do not take it. Assignment remains random,
    so it is a valid instrument, but the effect of *receiving* treatment among compliers
    (LATE) is a different quantity from the ATE. This regime exists to make that
    difference measurable rather than definitional.

``unobserved``
    Selection depends on ``u``, which the estimator never sees. **Every** adjustment
    method should fail here, and the point of including it is to show that no amount of
    statistical sophistication substitutes for having measured the confounder. This is
    what sensitivity analysis is for.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from gatekeeper.data.schema import ColumnSpec, DatasetSchema, ExperimentData
from gatekeeper.types import DataSource, InsufficientData

__all__ = [
    "CONFOUNDED_SCHEMA",
    "CausalScenario",
    "ConfoundingRegime",
    "covariate_imbalance",
    "make_confounded",
    "make_heterogeneous",
    "make_randomised",
]

ConfoundingRegime = Literal["selection", "noncompliance", "unobserved"]

CONFOUNDED_SCHEMA = DatasetSchema(
    name="causal_benchmark",
    unit_col="unit_id",
    variant_col="assigned",
    control="control",
    columns=(
        ColumnSpec("unit_id", "int", unique=True, description="randomisation unit"),
        ColumnSpec(
            "assigned",
            "str",
            allowed_values=frozenset({"control", "treatment"}),
            description="ASSIGNED arm; random by construction, so a valid instrument",
        ),
        ColumnSpec(
            "x",
            "float",
            post_treatment=False,
            description="observed pre-treatment covariate; drives both selection and outcome",
        ),
        ColumnSpec(
            "u",
            "float",
            post_treatment=False,
            description=(
                "pre-treatment confounder that is observable in the frame but withheld "
                "from estimators in the 'unobserved' regime"
            ),
        ),
        ColumnSpec(
            "received",
            "float",
            post_treatment=True,
            description="1.0 if treatment was actually received; differs from assigned under noncompliance",
        ),
        ColumnSpec("y", "float", post_treatment=True, description="the outcome"),
    ),
)
"""Schema for the causal benchmark.

``u`` is present as a column so the generator can build a genuine confounder, but the
``unobserved`` regime's whole point is that estimators are not given it. Withholding is
enforced by the benchmark harness passing an explicit covariate list, not by hiding the
column -- so a test can verify that adjusting for ``u`` *would* have worked, which is the
finding that makes the regime informative rather than merely discouraging.
"""


@dataclass(frozen=True, slots=True)
class CausalScenario:
    """A dataset plus the exact parameter that generated it."""

    data: ExperimentData
    true_ate: float
    """Exact by construction -- a parameter, not an estimate."""
    true_late: float | None
    """Complier effect, where non-compliance was simulated. Equals ``true_ate`` when
    every unit complies, and differs from it when compliance depends on ``x``."""
    regime: ConfoundingRegime | None
    """``None`` for the unconfounded randomised sample."""
    strength: float
    seed: int
    n_retained: int
    n_generated: int

    @property
    def retention_rate(self) -> float:
        return self.n_retained / self.n_generated

    def describe(self) -> str:
        regime = self.regime or "randomised"
        late = "n/a" if self.true_late is None else f"{self.true_late:+.4f}"
        return (
            f"{regime:14} strength={self.strength:.2f}  true ATE={self.true_ate:+.4f}  "
            f"LATE={late}  n={self.n_retained:,}/{self.n_generated:,} "
            f"({self.retention_rate:.1%} retained)"
        )


def _generate(
    n: int,
    *,
    true_ate: float,
    seed: int,
    x_on_y: float = 2.0,
    u_on_y: float = 2.0,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Build a randomised population with an exactly known ATE.

    The outcome is additive and linear::

        y = 1.0 + x_on_y * x + u_on_y * u + true_ate * received + noise

    so the average treatment effect is exactly ``true_ate`` regardless of the covariate
    distribution -- there is no heterogeneity to average over and nothing to marginalise.
    That is deliberate: it makes the benchmark target a parameter we can write down.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    u = rng.standard_normal(n)
    assigned_treatment = rng.random(n) < 0.5
    noise = rng.standard_normal(n)

    frame = pd.DataFrame(
        {
            "unit_id": np.arange(1, n + 1, dtype=np.int64),
            "assigned": np.where(assigned_treatment, "treatment", "control"),
            "x": x,
            "u": u,
            "received": assigned_treatment.astype(float),
            "y": 1.0 + x_on_y * x + u_on_y * u + true_ate * assigned_treatment + noise,
        }
    )
    return frame, x, u


def make_randomised(n: int = 20_000, *, true_ate: float = 1.0, seed: int = 0) -> CausalScenario:
    """A clean randomised experiment: the control condition for the benchmark.

    Every estimator should recover ``true_ate`` here, including the naive difference in
    means. An estimator that fails on *this* is broken, and running it first is what
    separates "the regime broke it" from "it never worked".
    """
    if n < 100:
        raise InsufficientData(f"need >= 100 units, got {n}")
    frame, _, _ = _generate(n, true_ate=true_ate, seed=seed)
    return CausalScenario(
        data=ExperimentData.from_frame(
            frame, schema=CONFOUNDED_SCHEMA, data_source=DataSource.SYNTHETIC
        ),
        true_ate=true_ate,
        true_late=true_ate,
        regime=None,
        strength=0.0,
        seed=seed,
        n_retained=len(frame),
        n_generated=n,
    )


def make_confounded(
    regime: ConfoundingRegime,
    n: int = 20_000,
    *,
    true_ate: float = 1.0,
    strength: float = 1.5,
    seed: int = 0,
) -> CausalScenario:
    """Construct a sample in which randomisation is broken in a controlled way.

    Parameters
    ----------
    regime
        Which mechanism to apply -- see the module docstring.
    n
        Units generated *before* selection. The retained count is reported on the result,
        since selection regimes discard units.
    true_ate
        The exact treatment effect built into the outcome.
    strength
        How hard the confounding bites. ``0`` reduces every regime to the randomised
        case, which a test uses as a continuity check.
    seed
        RNG seed (R4.2).

    Returns
    -------
    CausalScenario
        Tagged ``SEMI_SYNTHETIC``: real *mechanism*, deliberately injected confounding
        (R1.11).
    """
    if n < 100:
        raise InsufficientData(f"need >= 100 units, got {n}")
    if strength < 0:
        raise ValueError(f"strength must be non-negative, got {strength}")

    frame, x, u = _generate(n, true_ate=true_ate, seed=seed)
    rng = np.random.default_rng(seed + 10_000)
    is_treated = frame["assigned"].to_numpy() == "treatment"
    true_late: float | None = true_ate

    if regime in ("selection", "unobserved"):
        # Keep units with probability depending on the arm AND a covariate, so the two
        # arms end up with different covariate distributions. `selection` uses the
        # observed x; `unobserved` uses u, which estimators are not given.
        driver = x if regime == "selection" else u
        logit = strength * driver * np.where(is_treated, 1.0, -1.0)
        keep_prob = 1.0 / (1.0 + np.exp(-logit))
        keep = rng.random(n) < keep_prob
        if keep.sum() < 100:
            raise InsufficientData(
                f"selection at strength={strength} retained only {int(keep.sum())} units; "
                "lower the strength or raise n"
            )
        frame = frame.loc[keep].reset_index(drop=True)

    elif regime == "noncompliance":
        # One-sided: some assigned-to-treatment units do not take it, with compliance
        # depending on x. Assignment stays random, so it remains a valid instrument.
        comply_logit = strength * x
        comply_prob = 1.0 / (1.0 + np.exp(-comply_logit))
        complies = rng.random(n) < comply_prob
        received = is_treated & complies
        frame["received"] = received.astype(float)
        # Rebuild the outcome from what was actually RECEIVED, not what was assigned.
        frame["y"] = (
            1.0
            + 2.0 * frame["x"].to_numpy()
            + 2.0 * frame["u"].to_numpy()
            + true_ate * received
            + rng.standard_normal(n)
        )
        # With a homogeneous effect the complier effect equals the ATE. Kept explicit
        # so that introducing heterogeneity later forces this to be revisited rather
        # than silently inherited.
        true_late = true_ate
        if received.sum() < 50:
            raise InsufficientData(
                f"compliance at strength={strength} left only {int(received.sum())} treated units"
            )
    else:
        raise ValueError(f"unknown regime {regime!r}")

    return CausalScenario(
        data=ExperimentData.from_frame(
            frame, schema=CONFOUNDED_SCHEMA, data_source=DataSource.SEMI_SYNTHETIC
        ),
        true_ate=true_ate,
        true_late=true_late,
        regime=regime,
        strength=strength,
        seed=seed,
        n_retained=len(frame),
        n_generated=n,
    )


def make_heterogeneous(
    n: int = 20_000,
    *,
    base_effect: float = 1.0,
    effect_slope: float = 1.5,
    seed: int = 0,
    treatment_share: float = 0.5,
) -> tuple[ExperimentData, np.ndarray]:
    """A randomised experiment with a **known** conditional treatment effect.

    The effect varies linearly in the observed covariate::

        tau(x) = base_effect + effect_slope * x

    so the true per-unit effect is known exactly and the CATE learners can be scored
    against it rather than against each other. Assignment stays random, so there is no
    confounding to untangle -- the only question is whether a learner recovers the
    *shape* of ``tau(x)``.

    Returns
    -------
    tuple[ExperimentData, numpy.ndarray]
        The data, and the true ``tau(x_i)`` per unit in frame order.
    """
    if n < 100:
        raise InsufficientData(f"need >= 100 units, got {n}")
    if not 0.0 < treatment_share < 1.0:
        raise ValueError(f"treatment_share must be in (0, 1), got {treatment_share}")

    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    u = rng.standard_normal(n)
    treated = rng.random(n) < treatment_share
    tau = base_effect + effect_slope * x

    frame = pd.DataFrame(
        {
            "unit_id": np.arange(1, n + 1, dtype=np.int64),
            "assigned": np.where(treated, "treatment", "control"),
            "x": x,
            "u": u,
            "received": treated.astype(float),
            "y": 1.0 + 2.0 * x + 0.5 * u + tau * treated + rng.standard_normal(n),
        }
    )
    data = ExperimentData.from_frame(
        frame, schema=CONFOUNDED_SCHEMA, data_source=DataSource.SYNTHETIC
    )
    return data, tau


def covariate_imbalance(scenario: CausalScenario, covariate: str = "x") -> float:
    """Standardised mean difference in ``covariate`` between arms.

    A quick check that a regime actually did something: under randomisation this is near
    zero, and selection regimes should push it well away from zero. Reported in units of
    pooled standard deviations, where |SMD| > 0.1 is the conventional threshold for
    meaningful imbalance.
    """
    data = scenario.data
    treated = data.outcome(covariate, "treatment")
    control = data.outcome(covariate, "control")
    pooled_sd = math.sqrt((treated.var(ddof=1) + control.var(ddof=1)) / 2.0)
    if pooled_sd == 0.0:
        return 0.0
    return float((treated.mean() - control.mean()) / pooled_sd)
