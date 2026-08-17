"""Synthetic experiment generation with exactly known ground truth.

Two jobs:

1. **Testing.** Every estimator needs data whose true effect is known by
   construction, so a known-answer or calibration test has something to assert
   against.
2. **Methods this dataset cannot support.** Cookie Cats has no pre-period covariate
   and no timestamps, so CUPED, DiD, and real sequential accrual are demonstrated
   here instead of faked on the real data (PRD §6). Anything built on this module
   carries ``DataSource.SYNTHETIC`` and renders with the badge (R1.11, Design §7).

Marginal outcome probabilities are set *directly* per arm, so the true absolute
effect is exactly ``p_treatment - p_control`` -- not a logit-scale coefficient that
would need marginalising. Tests can therefore assert recovery of an exact number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from gatekeeper.data.schema import COOKIE_CATS, ColumnSpec, DatasetSchema, ExperimentData
from gatekeeper.types import DataSource

__all__ = [
    "PRE_PERIOD_SCHEMA",
    "SyntheticExperiment",
    "make_cookie_cats_like",
    "make_null_experiment",
    "make_pre_period_experiment",
]

PRE_PERIOD_SCHEMA = DatasetSchema(
    name="synthetic_pre_period",
    unit_col="userid",
    variant_col="version",
    control="control",
    columns=(
        ColumnSpec("userid", "int", unique=True, description="randomisation unit"),
        ColumnSpec(
            "version",
            "str",
            allowed_values=frozenset({"control", "treatment"}),
            description="control / treatment",
        ),
        ColumnSpec(
            "pre_rounds",
            "float",
            post_treatment=False,
            description=(
                "rounds played BEFORE the experiment started; a genuine pre-experiment "
                "covariate and therefore valid for CUPED"
            ),
        ),
        ColumnSpec(
            "rounds",
            "float",
            post_treatment=True,
            description="rounds played during the experiment; the outcome",
        ),
    ),
)
"""A schema with a real pre-experiment covariate.

Cookie Cats has none, which is why CUPED cannot be applied to it (PRD §6). This schema
exists so CUPED can be demonstrated and calibrated on data where a valid covariate
genuinely exists, rather than by pressing a post-treatment column into service and
calling the resulting bias a variance reduction (R1.7).
"""


@dataclass(frozen=True, slots=True)
class SyntheticExperiment:
    """A synthetic dataset bundled with the true effects used to generate it.

    Attributes
    ----------
    data
        The generated experiment, tagged ``DataSource.SYNTHETIC``.
    true_effects
        True *absolute* effect per metric, exact by construction.
    seed
        The seed used, for reproduction (R4.2).
    """

    data: ExperimentData
    true_effects: dict[str, float]
    seed: int

    def true_effect(self, metric: str) -> float:
        if metric not in self.true_effects:
            raise KeyError(
                f"no true effect recorded for {metric!r}; have {sorted(self.true_effects)}"
            )
        return self.true_effects[metric]


def make_cookie_cats_like(
    n: int = 90_000,
    *,
    seed: int = 0,
    retention_1_control: float = 0.448,
    retention_1_effect: float = -0.006,
    retention_7_control: float = 0.190,
    retention_7_effect: float = -0.008,
    gamerounds_effect: float = 0.0,
    treatment_share: float = 0.5,
    schema: DatasetSchema = COOKIE_CATS,
) -> SyntheticExperiment:
    """Generate a dataset matching the Cookie Cats schema with known effects.

    Defaults are in the same ballpark as the real dataset so that plots and
    diagnostics look realistic, but they are **inputs, not findings** -- nothing here
    is evidence about the real experiment (Rules §7: no invented statistics).

    Parameters
    ----------
    n
        Total number of units across both arms.
    seed
        RNG seed. Explicit and recorded (R4.2).
    retention_1_control, retention_7_control
        Control-arm marginal rates.
    retention_1_effect, retention_7_effect
        True absolute effects (treatment rate minus control rate), in proportion
        units. ``-0.008`` means treatment is 0.8pp lower.
    gamerounds_effect
        Additive shift in the log-mean of ``sum_gamerounds`` for the treatment arm.
    treatment_share
        Fraction assigned to treatment. Set away from 0.5 to generate a sample
        ratio mismatch for testing the SRM check.
    schema
        Dataset contract to conform to.

    Returns
    -------
    SyntheticExperiment
        Data plus the exact true effects.

    Notes
    -----
    ``sum_gamerounds`` is drawn from a lognormal whose location depends on
    ``retention_7``, which induces the realistic positive correlation between
    engagement and retention -- and preserves the fact that ``sum_gamerounds`` is
    downstream of the outcome, hence post-treatment (R1.7).
    """
    if not 0.0 < treatment_share < 1.0:
        raise ValueError(f"treatment_share must be in (0, 1), got {treatment_share}")
    if n < 2:
        raise ValueError(f"n must be at least 2, got {n}")

    rates = {
        "retention_1": (retention_1_control, retention_1_control + retention_1_effect),
        "retention_7": (retention_7_control, retention_7_control + retention_7_effect),
    }
    for metric, (p_c, p_t) in rates.items():
        for label, p in (("control", p_c), ("treatment", p_t)):
            if not 0.0 < p < 1.0:
                raise ValueError(
                    f"{metric} {label} rate must be in (0, 1), got {p:.4f}; "
                    "check the base rate and effect size"
                )

    rng = np.random.default_rng(seed)
    control, treatment = schema.control, schema.variants[1]

    is_treated = rng.random(n) < treatment_share
    version = np.where(is_treated, treatment, control)

    out: dict[str, np.ndarray] = {
        schema.unit_col: np.arange(1, n + 1, dtype=np.int64),
        schema.variant_col: version,
    }

    for metric, (p_c, p_t) in rates.items():
        probs = np.where(is_treated, p_t, p_c)
        out[metric] = rng.random(n) < probs

    # Engagement is higher among retained players; treatment shifts the log-mean.
    log_mu = np.where(out["retention_7"], 3.6, 1.9) + np.where(is_treated, gamerounds_effect, 0.0)
    out["sum_gamerounds"] = np.floor(rng.lognormal(mean=log_mu, sigma=1.4)).astype(np.int64)

    data = ExperimentData.from_frame(
        pd.DataFrame(out), schema=schema, data_source=DataSource.SYNTHETIC
    )
    return SyntheticExperiment(
        data=data,
        true_effects={
            "retention_1": retention_1_effect,
            "retention_7": retention_7_effect,
        },
        seed=seed,
    )


def make_pre_period_experiment(
    n: int = 20_000,
    *,
    seed: int = 0,
    rho: float = 0.7,
    effect: float = 2.0,
    pre_mean: float = 20.0,
    pre_sd: float = 8.0,
    outcome_sd: float = 8.0,
    treatment_share: float = 0.5,
) -> SyntheticExperiment:
    """Generate an experiment with a pre-experiment covariate of known correlation.

    The data-generating process is built so ``rho`` really is the population
    correlation between the covariate and the outcome::

        pre     ~ Normal(pre_mean, pre_sd)
        outcome = mu + outcome_sd * [ rho * z_pre + sqrt(1 - rho^2) * noise ] + effect*T

    where ``z_pre`` is the standardised covariate. Because the covariate's coefficient
    is ``rho`` on the standardised scale, the theoretical CUPED variance reduction is
    exactly ``1 - rho^2`` -- which is what makes this generator useful as a test
    oracle rather than merely as plausible-looking data.

    The covariate is drawn **independently of treatment**, as a genuine pre-period
    quantity must be. That independence is what keeps CUPED unbiased.

    Parameters
    ----------
    n
        Total units across both arms.
    seed
        RNG seed (R4.2).
    rho
        Population correlation between covariate and outcome. Must be in (-1, 1).
    effect
        True additive treatment effect on the outcome.
    pre_mean, pre_sd
        Covariate distribution.
    outcome_sd
        Outcome standard deviation before the treatment shift.
    treatment_share
        Fraction assigned to treatment.

    Returns
    -------
    SyntheticExperiment
        ``true_effects["rounds"]`` is ``effect``, exact by construction.
    """
    if not -1.0 < rho < 1.0:
        raise ValueError(f"rho must be in (-1, 1), got {rho}")
    if pre_sd <= 0 or outcome_sd <= 0:
        raise ValueError("pre_sd and outcome_sd must be positive")
    if not 0.0 < treatment_share < 1.0:
        raise ValueError(f"treatment_share must be in (0, 1), got {treatment_share}")
    if n < 2:
        raise ValueError(f"n must be at least 2, got {n}")

    rng = np.random.default_rng(seed)
    is_treated = rng.random(n) < treatment_share

    z_pre = rng.standard_normal(n)
    pre = pre_mean + pre_sd * z_pre
    noise = rng.standard_normal(n)
    outcome = (
        pre_mean
        + outcome_sd * (rho * z_pre + math.sqrt(1.0 - rho**2) * noise)
        + np.where(is_treated, effect, 0.0)
    )

    frame = pd.DataFrame(
        {
            "userid": np.arange(1, n + 1, dtype=np.int64),
            "version": np.where(is_treated, "treatment", "control"),
            "pre_rounds": pre,
            "rounds": outcome,
        }
    )
    data = ExperimentData.from_frame(
        frame, schema=PRE_PERIOD_SCHEMA, data_source=DataSource.SYNTHETIC
    )
    return SyntheticExperiment(data=data, true_effects={"rounds": effect}, seed=seed)


def make_null_experiment(
    n: int = 20_000, *, seed: int = 0, base_rate: float = 0.2
) -> SyntheticExperiment:
    """Generate an experiment with a true effect of exactly zero.

    The workhorse of the calibration suite (Architecture §6): under a true null,
    p-values must be uniform and 95% intervals must cover zero about 95% of the
    time. An estimator that fails this is broken regardless of how reasonable its
    code looks.
    """
    return make_cookie_cats_like(
        n=n,
        seed=seed,
        retention_1_control=base_rate,
        retention_1_effect=0.0,
        retention_7_control=base_rate,
        retention_7_effect=0.0,
        gamerounds_effect=0.0,
    )
