"""Gatekeeper readout app.

Run with::

    streamlit run app/streamlit_app.py

Design.md §4.1's layout, top to bottom: provenance badge, then the **sanity gate**, then
the decision, then the metrics. The gate comes before the numbers deliberately -- a reader
who sees an effect size first has already formed an opinion by the time they reach the
caveat.

Every control here is a *display* choice. Nothing in this app can change the primary
metric, the practical threshold, or the multiplicity method: those come from the committed
spec, because letting a UI adjust them would make pre-registration decorative (R1.2).
The one exception is the sanity override, which requires a typed reason and is stamped
onto the readout.

No real dataset is required. When ``data/raw/cookie_cats.csv`` is absent the app runs on
synthetic data and says so, loudly.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from gatekeeper.checks.integrity import run_sanity_checks
from gatekeeper.checks.outliers import check_outlier_leverage, profile_metric
from gatekeeper.data.ingest import load_cookie_cats, project_root
from gatekeeper.data.synthetic import make_cookie_cats_like
from gatekeeper.frequentist.bootstrap import estimate_bootstrap
from gatekeeper.frequentist.proportions import estimate_two_proportion
from gatekeeper.report.readout import build_readout
from gatekeeper.report.render import render_markdown
from gatekeeper.spec import load_spec
from gatekeeper.types import DataSource, Decision, Estimand
from gatekeeper.viz.theme import decision_style

st.set_page_config(page_title="Gatekeeper readout", layout="centered")

SPEC_PATH = project_root() / "specs" / "cookie_cats_gate.yaml"


@st.cache_data(show_spinner=False)
def _load(use_real: bool, n_synthetic: int, seed: int):
    """Load the experiment, falling back to synthetic data when the CSV is absent."""
    if use_real:
        return load_cookie_cats(), None
    exp = make_cookie_cats_like(n=n_synthetic, seed=seed)
    return exp.data, exp


def main() -> None:
    spec = load_spec(SPEC_PATH)
    csv_path = project_root() / "data" / "raw" / "cookie_cats.csv"
    real_available = csv_path.exists()

    with st.sidebar:
        st.header("Data")
        if real_available:
            use_real = st.toggle("Use the real Cookie Cats dataset", value=True)
        else:
            use_real = False
            st.info(
                "`data/raw/cookie_cats.csv` not found, so this is running on **synthetic "
                "data**. See `data/README.md` to obtain the real file."
            )
        n_synthetic = st.select_slider(
            "Synthetic sample size", options=[5_000, 20_000, 90_000, 200_000], value=90_000
        )
        seed = st.number_input("Seed", min_value=0, max_value=9_999, value=42, step=1)

        st.header("Display")
        show_diagnostics = st.toggle("Show diagnostics", value=False)

        st.caption(
            "These controls affect **display and data source only**. The primary metric, "
            "practical threshold, and multiplicity method come from the committed spec — "
            "a UI that could change them would make pre-registration decorative (R1.2)."
        )

    data, _synthetic = _load(use_real, int(n_synthetic), int(seed))

    # ---- header: provenance first -------------------------------------------
    st.title("Gatekeeper readout")
    if data.data_source is DataSource.REAL:
        st.caption(f"**real data** · spec `{spec.name}` · registered {spec.registered_on}")
    else:
        st.warning(
            f"**SYNTHETIC DATA** — not a real experiment. spec `{spec.name}`, "
            f"registered {spec.registered_on}. Nothing here is evidence about the real "
            "Cookie Cats experiment (R1.11).",
            icon="⚠",
        )

    # ---- the gate, ABOVE the results ----------------------------------------
    extra = [
        check_outlier_leverage(
            data, "sum_gamerounds", declared_rule=spec.outlier_rule_for("sum_gamerounds")
        )
    ]
    sanity = run_sanity_checks(
        data,
        expected_shares=spec.expected_shares,
        srm_threshold=spec.srm_threshold,
        min_per_arm=spec.min_per_arm,
        extra=extra,
    )

    st.subheader(sanity.summary())
    for check in sanity.checks:
        (st.success if check.passed else st.error)(
            f"**{check.name}** — {check.detail}", icon="✔" if check.passed else "✕"
        )

    override_reason = None
    if not sanity.passed:
        st.divider()
        st.error(
            "Metric results are withheld. A failed sanity check is a wall, not a "
            "footnote: assignment or logging is suspect, so an effect size shown here "
            "would be read as a finding (R1.3).",
            icon="✕",
        )
        with st.expander("Show anyway — requires a recorded reason"):
            typed = st.text_input("Why is it safe to proceed? This is stamped on the readout.")
            if typed.strip():
                override_reason = typed.strip()

    # ---- estimates over the pre-registered family ---------------------------
    estimates = {}
    for metric in spec.all_metrics:
        estimand = Estimand(outcome=metric, treatment=data.schema.variant_col)
        if metric == "sum_gamerounds":
            # Severely skewed: the bootstrap, not a t-test (R1.12).
            estimates[metric] = estimate_bootstrap(
                data, estimand, n_resamples=2_000, seed=int(seed)
            )
        else:
            estimates[metric] = estimate_two_proportion(data, estimand, alpha=spec.alpha)

    readout = build_readout(spec, sanity, estimates, override_reason=override_reason)

    st.divider()
    color, icon = decision_style(readout.decision)
    st.markdown(
        f"### <span style='color:{color}'>{icon} Decision: {readout.decision.value.upper()}</span>",
        unsafe_allow_html=True,
    )
    st.write(readout.rationale)

    if readout.decision is not Decision.BLOCKED:
        st.divider()
        primary = readout.primary
        lo, hi = primary.estimate.ci
        left, right = st.columns([2, 3])
        with left:
            st.metric(
                label=f"{primary.metric} (primary)",
                value=f"{primary.estimate.point:+.4f}",
            )
        with right:
            st.write(
                f"**[{lo:+.4f}, {hi:+.4f}]** "
                f"({round(primary.estimate.ci_level * 100)}% interval)  \n"
                f"practical threshold ±{spec.practical_threshold:g} "
                "*(primary metric only — PRD O5)*"
            )

        st.subheader("Metrics")
        st.dataframe(
            [
                {
                    "metric": m.metric,
                    "role": "primary" if m.is_primary else "guardrail",
                    "effect": round(m.estimate.point, 6),
                    "interval": f"[{m.estimate.ci[0]:+.4f}, {m.estimate.ci[1]:+.4f}]",
                    "adj. p": "—" if m.adjusted_p is None else round(m.adjusted_p, 5),
                    "moved": "⚠" if (not m.is_primary and m.statistically_significant) else "",
                    "method": m.estimate.method,
                }
                for m in readout.metrics
            ],
            hide_index=True,
            use_container_width=True,
        )

        with st.expander("Assumptions behind every estimate"):
            for m in readout.metrics:
                st.markdown(f"**{m.metric}** (`{m.estimate.method}`)")
                for assumption in m.estimate.assumptions:
                    st.markdown(f"- {assumption}")

    if show_diagnostics:
        st.divider()
        st.subheader("Diagnostics")
        for variant in data.variants:
            st.text(profile_metric(data, "sum_gamerounds", variant).describe())

    st.divider()
    with st.expander("Export this readout as Markdown"):
        st.code(render_markdown(readout), language="markdown")


def _repo_has_spec() -> bool:
    """Guard so an import in a test environment does not explode on a missing spec."""
    return Path(SPEC_PATH).exists()


if __name__ == "__main__":
    if not _repo_has_spec():
        raise SystemExit(f"no spec at {SPEC_PATH}; run from the repository root")
    main()
