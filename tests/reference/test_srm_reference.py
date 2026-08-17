"""Cross-check our SRM implementation against scipy's chi-square goodness of fit.

Architecture §6 layer 2: where a trusted reference implementation exists, agree with
it to < 1e-6. This is what catches an algebra slip that a hand-written fixture and a
property test would both sail past.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats

from gatekeeper.design.srm import srm_test

TOL = 1e-9


def _scipy_reference(counts: dict[str, int], shares: dict[str, float] | None = None):
    arms = sorted(counts)
    obs = np.array([counts[a] for a in arms], dtype=float)
    total = obs.sum()
    if shares is None:
        exp = np.full(len(arms), total / len(arms))
    else:
        exp = np.array([shares[a] for a in arms], dtype=float) * total
    res = stats.chisquare(f_obs=obs, f_exp=exp)
    return float(res.statistic), float(res.pvalue)


class TestAgreementWithScipy:
    @pytest.mark.parametrize(
        "counts",
        [
            {"a": 500, "b": 500},
            {"a": 600, "b": 400},
            {"a": 44_700, "b": 45_489},
            {"a": 1, "b": 999_999},
            {"a": 120, "b": 90, "c": 90},
            {"a": 10, "b": 20, "c": 30, "d": 40},
        ],
    )
    def test_equal_shares(self, counts: dict[str, int]):
        ours = srm_test(counts)
        theirs = _scipy_reference(counts)
        assert ours[0] == pytest.approx(theirs[0], rel=TOL, abs=TOL)
        assert ours[1] == pytest.approx(theirs[1], rel=TOL, abs=TOL)

    @pytest.mark.parametrize(
        ("counts", "shares"),
        [
            ({"a": 700, "b": 300}, {"a": 0.7, "b": 0.3}),
            ({"a": 700, "b": 300}, {"a": 0.5, "b": 0.5}),
            ({"a": 100, "b": 200, "c": 700}, {"a": 0.1, "b": 0.2, "c": 0.7}),
            ({"a": 333, "b": 333, "c": 334}, {"a": 0.2, "b": 0.3, "c": 0.5}),
        ],
    )
    def test_unequal_shares(self, counts: dict[str, int], shares: dict[str, float]):
        ours = srm_test(counts, shares)
        theirs = _scipy_reference(counts, shares)
        assert ours[0] == pytest.approx(theirs[0], rel=TOL, abs=TOL)
        assert ours[1] == pytest.approx(theirs[1], rel=TOL, abs=TOL)

    @settings(max_examples=300, deadline=None)
    @given(
        a=st.integers(min_value=1, max_value=10**6),
        b=st.integers(min_value=1, max_value=10**6),
    )
    def test_agreement_holds_across_the_input_space(self, a: int, b: int):
        counts = {"a": a, "b": b}
        ours = srm_test(counts)
        theirs = _scipy_reference(counts)
        assert ours[0] == pytest.approx(theirs[0], rel=1e-9, abs=1e-9)
        assert ours[1] == pytest.approx(theirs[1], rel=1e-9, abs=1e-12)
