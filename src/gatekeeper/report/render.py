"""Render a :class:`~gatekeeper.report.readout.Readout` as standalone HTML.

Design.md's rules, applied:

- The **sanity gate sits above the results**, because a reader who sees an effect size
  first has already formed an opinion by the time they reach the caveat.
- A blocked readout **replaces** the numbers rather than annotating them (§4.2).
- The provenance badge appears in the header, so a page lifted into a slide deck carries
  its own labelling (§7).
- The headline is the **difference with its interval**, never two bars to eyeball (§5.1):
  zero-baselined, a 0.8pp difference is invisible; truncated, it is exaggerated. The
  reader's question is about the difference, so the difference is what gets encoded.
- Theming follows the viewer: light and dark palettes are both emitted as CSS custom
  properties, with the dark values under ``prefers-color-scheme`` so nothing borrows the
  host's colours.

Self-contained by construction -- no external CSS, fonts, or scripts.
"""

from __future__ import annotations

from html import escape

from gatekeeper.report.readout import MetricReadout, Readout
from gatekeeper.types import DataSource, Decision
from gatekeeper.viz.theme import decision_style, palette

__all__ = ["render_html", "render_markdown"]

_BADGE_TEXT = {
    DataSource.REAL: "real data",
    DataSource.SEMI_SYNTHETIC: "semi-synthetic · injected confounding",
    DataSource.SYNTHETIC: "synthetic data",
}


def _format_effect(value: float) -> str:
    """Signed, with enough precision to be useful and not so much as to be noise."""
    return f"{value:+.4g}"


def _metric_row(metric: MetricReadout) -> str:
    est = metric.estimate
    lo, hi = est.ci
    label = "primary" if metric.is_primary else "guardrail"
    adjusted = "—" if metric.adjusted_p is None else f"{metric.adjusted_p:.4g}"
    flag = " ⚠" if (not metric.is_primary and metric.statistically_significant) else ""
    return f"""      <tr>
        <td class="metric">{escape(metric.metric)}<span class="tag">{label}</span></td>
        <td class="num">{_format_effect(est.point)}</td>
        <td class="num">[{_format_effect(lo)}, {_format_effect(hi)}]</td>
        <td class="num">{adjusted}{flag}</td>
        <td>{escape(metric.direction)}</td>
      </tr>"""


def render_html(readout: Readout, *, title: str | None = None) -> str:
    """Render a readout as a self-contained HTML fragment."""
    light = palette("light")
    dark = palette("dark")
    color, icon = decision_style(readout.decision, "light")
    dark_color, _ = decision_style(readout.decision, "dark")

    heading = title or f"Gatekeeper readout — {readout.spec.name}"
    badge_class = "badge-real" if readout.data_source is DataSource.REAL else "badge-synthetic"
    gate_ok = readout.sanity.passed

    checks = "\n".join(
        f'        <li class="{"ok" if c.passed else "bad"}">'
        f"<strong>{escape(c.name)}</strong> — {escape(c.detail)}</li>"
        for c in readout.sanity.checks
    )

    if readout.is_blocked:
        body = f"""    <section class="blocked">
      <h2>{icon} Blocked — results not reported</h2>
      <p>{escape(readout.rationale)}</p>
      <p class="note">Metric results are deliberately withheld. A failed sanity check is a
      wall, not a footnote: an effect size shown here would be read as a finding.</p>
    </section>"""
    else:
        rows = "\n".join(_metric_row(m) for m in readout.metrics)
        primary = readout.primary
        p_lo, p_hi = primary.estimate.ci
        threshold = readout.spec.practical_threshold
        body = f"""    <section class="headline">
      <div class="hero">{_format_effect(primary.estimate.point)}</div>
      <div class="hero-meta">
        <div class="interval">[{_format_effect(p_lo)}, {_format_effect(p_hi)}]
          <span class="level">{round(primary.estimate.ci_level * 100)}% interval</span></div>
        <div class="threshold">practical threshold ±{threshold:g}
          <span class="level">primary metric only</span></div>
      </div>
      <p class="rationale">{escape(readout.rationale)}</p>
    </section>

    <section>
      <h2>Metrics</h2>
      <table>
        <thead><tr>
          <th>metric</th><th class="num">effect</th><th class="num">interval</th>
          <th class="num">adj. p</th><th>direction</th>
        </tr></thead>
        <tbody>
{rows}
        </tbody>
      </table>
      <p class="note">Guardrails are judged on multiplicity-adjusted statistical
      significance, not against the practical threshold — that threshold is expressed in
      the primary metric's units and would be dimensionally meaningless elsewhere
      (PRD&nbsp;O5). A ⚠ marks a guardrail that moved and needs explaining.</p>
    </section>"""

    override = (
        f'    <section class="override"><strong>Override recorded:</strong> '
        f"{escape(readout.override_reason)}</section>"
        if readout.override_reason
        else ""
    )

    return f"""<title>{escape(heading)}</title>
<style>
  :root {{
    --surface: {light.surface}; --page: {light.page};
    --ink: {light.ink_primary}; --ink-2: {light.ink_secondary}; --muted: {light.ink_muted};
    --grid: {light.gridline}; --baseline: {light.baseline};
    --control: {light.control}; --treatment: {light.treatment};
    --status: {color}; --warning: {light.status_warning};
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --surface: {dark.surface}; --page: {dark.page};
      --ink: {dark.ink_primary}; --ink-2: {dark.ink_secondary}; --muted: {dark.ink_muted};
      --grid: {dark.gridline}; --baseline: {dark.baseline};
      --control: {dark.control}; --treatment: {dark.treatment};
      --status: {dark_color}; --warning: {dark.status_warning};
    }}
  }}
  :root[data-theme="dark"] {{
    --surface: {dark.surface}; --page: {dark.page};
    --ink: {dark.ink_primary}; --ink-2: {dark.ink_secondary}; --muted: {dark.ink_muted};
    --grid: {dark.gridline}; --baseline: {dark.baseline};
    --control: {dark.control}; --treatment: {dark.treatment};
    --status: {dark_color}; --warning: {dark.status_warning};
  }}
  body {{
    background: var(--page); color: var(--ink); margin: 0; padding: 32px 20px;
    font: 14px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif;
  }}
  main {{ max-width: 860px; margin: 0 auto; }}
  header {{ margin-bottom: 24px; }}
  h1 {{ font-size: 20px; font-weight: 600; margin: 0 0 8px; }}
  h2 {{ font-size: 15px; font-weight: 600; margin: 0 0 12px; }}
  .meta {{ color: var(--ink-2); font-size: 13px; }}
  .badge {{
    display: inline-block; padding: 2px 8px; border-radius: 999px;
    font-size: 12px; margin-right: 8px;
  }}
  .badge-real {{ color: var(--muted); border: 1px solid var(--baseline); }}
  .badge-synthetic {{ color: {light.ink_primary}; background: var(--warning); font-weight: 600; }}
  section {{
    background: var(--surface); border: 1px solid var(--grid); border-radius: 8px;
    padding: 16px; margin-bottom: 24px;
  }}
  .gate {{ border-left: 4px solid var(--status); }}
  .gate ul {{ margin: 8px 0 0; padding-left: 20px; }}
  .gate li {{ font-size: 13px; color: var(--ink-2); }}
  .gate li.bad {{ color: var(--status); font-weight: 600; }}
  .decision {{
    font-size: 15px; font-weight: 600; color: var(--status);
  }}
  .blocked h2 {{ color: var(--status); font-size: 17px; }}
  .hero {{
    font-size: 52px; font-weight: 600; letter-spacing: -0.02em; line-height: 1.05;
  }}
  .hero-meta {{ display: flex; gap: 32px; flex-wrap: wrap; margin: 8px 0 16px; }}
  .interval, .threshold {{ font-size: 15px; }}
  .level {{ display: block; font-size: 12px; color: var(--muted); }}
  .rationale {{ margin: 0; color: var(--ink-2); }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{
    text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--grid);
    font-size: 13px;
  }}
  th {{ color: var(--muted); font-weight: 400; font-size: 12px; }}
  .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .metric {{ font-weight: 600; }}
  .tag {{
    font-weight: 400; font-size: 11px; color: var(--muted); margin-left: 8px;
    text-transform: uppercase; letter-spacing: 0.04em;
  }}
  .note {{ font-size: 12px; color: var(--muted); margin: 12px 0 0; }}
  .override {{ border-left: 4px solid var(--warning); font-size: 13px; }}
  .table-wrap {{ overflow-x: auto; }}
</style>
<main>
  <header>
    <h1>{escape(heading)}</h1>
    <div class="meta">
      <span class="badge {badge_class}">{_BADGE_TEXT[readout.data_source]}</span>
      spec <code>{escape(readout.spec.name)}</code> ·
      {escape(readout.spec.mode)} · registered {escape(readout.spec.registered_on)}
    </div>
  </header>

  <section class="gate">
    <h2>{"✔" if gate_ok else "✕"} {escape(readout.sanity.summary())}</h2>
    <ul>
{checks}
    </ul>
  </section>

  <section class="gate">
    <div class="decision">{icon} Decision: {readout.decision.value.upper()}</div>
  </section>

{body}
{override}
</main>
"""


def render_markdown(readout: Readout) -> str:
    """Render a readout as Markdown, for a PR comment or a plain-text channel."""
    lines = [
        f"# Gatekeeper readout — {readout.spec.name}",
        "",
        f"**{_BADGE_TEXT[readout.data_source]}** · spec `{readout.spec.name}` "
        f"({readout.spec.mode}, registered {readout.spec.registered_on})",
        "",
        f"## {readout.sanity.summary()}",
        "",
    ]
    for check in readout.sanity.checks:
        mark = "✔" if check.passed else "✕"
        lines.append(f"- {mark} **{check.name}** — {check.detail}")

    _, icon = decision_style(readout.decision)
    lines += [
        "",
        f"## {icon} Decision: {readout.decision.value.upper()}",
        "",
        readout.rationale,
        "",
    ]

    if readout.is_blocked:
        lines += [
            "Metric results are deliberately withheld: a failed sanity check is a wall, "
            "not a footnote.",
            "",
        ]
    else:
        lines += [
            "| metric | | effect | interval | adj. p | direction |",
            "|---|---|---:|---:|---:|---|",
        ]
        for m in readout.metrics:
            est = m.estimate
            adjusted = "—" if m.adjusted_p is None else f"{m.adjusted_p:.4g}"
            flag = " ⚠" if (not m.is_primary and m.statistically_significant) else ""
            lines.append(
                f"| `{m.metric}` | {'primary' if m.is_primary else 'guardrail'} "
                f"| {_format_effect(est.point)} "
                f"| [{_format_effect(est.ci[0])}, {_format_effect(est.ci[1])}] "
                f"| {adjusted}{flag} | {m.direction} |"
            )
        lines += [
            "",
            f"Practical threshold ±{readout.spec.practical_threshold:g} applies to the "
            "primary metric only; guardrails are judged on adjusted statistical "
            "significance (PRD O5).",
            "",
        ]

    if readout.override_reason:
        lines += [f"> **Override recorded:** {readout.override_reason}", ""]

    return "\n".join(lines)


def _assert_decision_covered() -> None:
    """Every Decision must have a style, so a new one cannot slip through unstyled."""
    for decision in Decision:
        decision_style(decision)


_assert_decision_covered()
