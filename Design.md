# Design — Gatekeeper

**Last updated** 2026-08-17

This project's UI is almost entirely **charts and numbers**, so "design" here means
*encoding decisions* far more than decoration. A dashboard that makes a
non-significant result look decisive is a design failure, not a stylistic one.

`src/gatekeeper/viz/theme.py` is the single implementation of everything below. No
hex value appears anywhere else in the codebase (Rules R4.8).

---

## 1. Design principles

1. **Uncertainty is never optional.** Every effect estimate renders with its
   confidence or credible interval. A point estimate without an interval is not
   shippable output.
2. **Plot the difference, not the two levels.** The reader's question is "did it
   move, and by how much" — that is one number with an interval, not two bars to
   eyeball. See §5.1; this is the most important rule on the page.
3. **Practical significance is the headline; statistical significance is a
   footnote.** The decision reads against the pre-registered threshold (R1.4).
4. **Synthetic data is always visibly labelled.** Non-negotiable (R1.11, §7).
5. **A blocked result is a wall, not a warning.** SRM failure replaces the numbers;
   it does not sit beside them in small text.
6. **Recessive chrome, prominent data.** Hairline grids, muted axes, no chartjunk.

---

## 2. Colour

Palette values come from the validated reference instance. **Both modes were run
through the six-check validator before this file was written**, including the harder
all-pairs test because propensity-overlap and CUPED pre/post plots are scatters:

```
node scripts/validate_palette.js "#2a78d6,#eb6834,#1baf7a" --mode light --pairs all
node scripts/validate_palette.js "#3987e5,#d95926,#199e70" --mode dark --surface "#1a1a19" --pairs all
```

Results — **all checks pass in both modes**:

| Check | Light | Dark |
|---|---|---|
| Lightness band | PASS | PASS |
| Chroma floor | PASS | PASS |
| CVD separation (all-pairs, deutan) | PASS — worst ΔE 9.2 | PASS — worst ΔE 9.4 |
| Normal-vision floor | PASS — worst ΔE 24.0 | PASS — worst ΔE 20.9 |
| Contrast vs surface | **WARN** — aqua 2.74:1 | PASS |

### 2.1 Variant identity (categorical)

Fixed slot order, assigned by variant, **never cycled and never re-assigned by
rank** — the control keeps its colour when a filter changes the variant set.

| Slot | Role | Light | Dark |
|---|---|---|---|
| 1 | **Control** (`gate_30`) | `#2a78d6` blue | `#3987e5` |
| 2 | **Treatment** (`gate_40`) | `#eb6834` orange | `#d95926` |
| 3 | Third variant (A/B/n) | `#1baf7a` aqua | `#199e70` |

Beyond three variants: fold into "Other" or facet into small multiples. Do not
extend the palette — three slots are what clears the all-pairs floors.

**Relief rule (from the WARN above):** light-mode aqua sits at 2.74:1 against the
light surface, below the 3:1 mark threshold. Wherever slot 3 is used on the light
surface it **must** ship visible direct labels or the table view. This is not
dismissable.

**Never** colour a variant by whether its result was significant. Identity and
outcome are different encoding jobs; conflating them means the chart repaints when
the statistics change.

### 2.2 Magnitude (sequential) — one hue, light→dark

Blue ramp, for heatmaps and continuous magnitude (propensity-score density,
correlation matrices, power surfaces):

| step | 100 | 250 | 400 | 550 | 700 |
|---|---|---|---|---|---|
| hex | `#cde2fb` | `#86b6ef` | `#3987e5` | `#1c5cab` | `#0d366b` |

Full 100→700 for continuous encoding. For **ordinal** marks (funnel stages, weight
deciles) start no lighter than step 250 on light, no darker than step 600
(`#184f95`) on dark, so the end nearest the surface still clears 2:1.

### 2.3 Polarity (diverging) — lift direction, balance plots

**Blue ↔ red** with a **neutral gray** midpoint — never a hue at zero, never a
rainbow. Equal step count per arm.

| Role | Light | Dark |
|---|---|---|
| Negative pole | `#d03b3b`-family red | same family |
| Neutral midpoint (zero) | `#f0efec` | `#383835` |
| Positive pole | `#2a78d6` blue | `#3987e5` |

Used for: relative lift by segment, standardised mean differences in love plots
(where zero is the target), correlation matrices.

### 2.4 Decision state (status — reserved, never themed)

The four `Decision` values map onto the reserved status scale. These colours are
**never** reused for a series.

| Decision | Status role | Hex | Icon |
|---|---|---|---|
| `SHIP` | good | `#0ca30c` | ✔ |
| `HOLD` | serious | `#ec835a` | ▲ |
| `INCONCLUSIVE` | warning | `#fab219` | ● |
| `BLOCKED` (sanity failed) | critical | `#d03b3b` | ✕ |

**Always icon + label, never colour alone.** Two reasons, and the second is
specific to this palette:

1. On the light surface, `warning` (1.79:1) and `serious` (2.57:1) are sub-3:1 by
   design — the icon+label pairing is the mitigation.
2. **Treatment-orange (`#eb6834`) sits only ΔE ≈ 5.8 from status-serious
   (`#ec835a`) in light mode.** A bare orange `HOLD` chip beside an orange
   treatment series is genuinely ambiguous. So the chip always carries its icon and
   the word "Hold", and never sits inside the plot area where the series colour
   lives.

### 2.5 Chrome & ink

| Role | Light | Dark |
|---|---|---|
| Chart surface | `#fcfcfb` | `#1a1a19` |
| Page plane | `#f9f9f7` | `#0d0d0d` |
| Primary ink | `#0b0b0b` | `#ffffff` |
| Secondary ink | `#52514e` | `#c3c2b7` |
| Muted (axis, tick labels) | `#898781` | `#898781` |
| Gridline (hairline) | `#e1e0d9` | `#2c2c2a` |
| Baseline / axis | `#c3c2b7` | `#383835` |
| Positive delta text | `#006300` | `#0ca30c` |
| Hairline ring | `rgba(11,11,11,.10)` | `rgba(255,255,255,.10)` |

**Text always wears text tokens, never a series colour.** A numeric label next to a
coloured mark stays in primary/secondary ink; the mark carries the identity.

### 2.6 Dark mode

Dark is a **selected** set of steps from the same hues, validated against the dark
surface — not an inverted or auto-darkened light palette. `theme.py` exposes
`palette(mode: Literal["light","dark"])`; Streamlit and matplotlib both read from it.

### 2.7 Texture (the accessibility channel)

One hand-drawn `Lines` fill at **45° and its 135° mirror only**, inked tone-on-tone.
On value scales the rotation is *ordered* with magnitude. Triggered by the
accessibility setting, print, or `forced-colors` — never decorative, never default.

---

## 3. Typography

System sans throughout — no display or serif face anywhere:

```
system-ui, -apple-system, "Segoe UI", sans-serif
```

| Role | Size | Weight | Token |
|---|---|---|---|
| Hero figure (the effect size) | 48–56px | 600 | primary ink |
| Section heading | 20px | 600 | primary ink |
| Chart title | 15px | 600 | primary ink |
| Chart subtitle / estimand | 13px | 400 | secondary ink |
| Body | 14px | 400 | primary ink |
| Axis ticks, small labels | 12px | 400 | muted |
| Table numerals | 13px | 400 | primary, `tabular-nums` |

**Figures:** proportional by default, including the hero number. Reserve
`font-variant-numeric: tabular-nums` for columns that must align vertically — table
rows and axis ticks.

**Number formatting**
- Rates as percentages, 2 dp: `44.82%`
- Absolute lift in percentage points, always signed: `−0.59pp`
- Relative lift signed with a percent: `−1.3%`
- CIs as `[−1.35pp, +0.17pp]` — brackets, signed, same unit as the estimate
- p-values: 3 dp, and `p < 0.001` below that. **Never** bare `p < 0.05`
- Always write **pp** for percentage points vs **%** for relative change. Confusing
  the two is the most common misreading of an A/B readout

---

## 4. Layout

### 4.1 Streamlit readout, top to bottom

```
┌──────────────────────────────────────────────────────────────┐
│ Gatekeeper · Cookie Cats — gate_30 vs gate_40                │
│ [ real data ]   spec: cookie_cats_gate.yaml   seed: 42       │
├──────────────────────────────────────────────────────────────┤
│  ✔ SANITY CHECKS PASSED    SRM p=0.0086 · no dupes · 0 leak  │  ← gate, full width
├──────────────────────────────────────────────────────────────┤
│                                                              │
│   retention_7   ── PRIMARY ──                                │
│                                                              │
│         −0.82pp            ▲ HOLD                            │
│    ┌────────────────────┐                                    │
│    │  [−1.35, −0.29] pp │   p = 0.002                        │
│    └────────────────────┘   threshold: ±0.50pp               │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│   Guardrails (BH-corrected)         [forest plot, 1 axis]    │
│   retention_1   ├──●──┤                                      │
│   gamerounds  ├────●────┤                                    │
│                     0                                        │
├──────────────────────────────────────────────────────────────┤
│   Diagnostics ▸  (collapsed: distributions, overlap, tails)  │
└──────────────────────────────────────────────────────────────┘
```

*(Numbers above are `EXAMPLE` placeholders for layout only — per Rules §7, no
statistic is written into a doc before code produces it.)*

**Order is deliberate:** the sanity gate is above the results, because a reader who
sees an effect size first has already formed an opinion by the time they reach the
caveat.

- Filters (metric, segment, date) sit in **one row above** the charts, never between them
- One decision per screen; guardrails are secondary and visually subordinate
- Diagnostics collapsed by default, one click away — present but not competing

### 4.2 Blocked state

When sanity checks fail, the results region is **replaced**, not annotated:

```
┌──────────────────────────────────────────────────────────────┐
│  ✕ BLOCKED — SAMPLE RATIO MISMATCH                           │
│                                                              │
│  Observed 48.3 / 51.7 vs intended 50 / 50 · χ² p = 3.1e-07   │
│  Threshold p < 0.0005.  Assignment is suspect; metric         │
│  results are not shown.                                      │
│                                                              │
│  [ Show anyway — requires a recorded reason ]                 │
└──────────────────────────────────────────────────────────────┘
```

Overriding demands a typed reason, which is then stamped onto the readout and the
export (R1.3).

### 4.3 Spacing

8px base scale (8 / 16 / 24 / 32 / 48). 24px between chart blocks, 16px inside a
card, 48px between major sections. Charts sit on `--surface-1` cards with a hairline
ring and 8px radius.

---

## 5. Chart specifications

### 5.1 The effect-size plot — the project's signature chart

**Form:** hero number + horizontal interval; a **forest plot** for multiple metrics.

**Not** two side-by-side bars of the control and treatment rates. That form fails
either way you scale it: zero-baselined, a 0.8pp difference is invisible; truncated,
it is wildly exaggerated. The reader's actual question is about the *difference and
its uncertainty*, so encode the difference directly.

- Zero reference line: 1px, baseline token, drawn **behind** the marks
- Practical-significance threshold: 1px dashed, muted, labelled `±0.50pp`
- Interval: 2px line, dot ≥ 8px, in the variant's slot colour
- Direct value labels on the point and both interval ends — always, since this is
  the headline
- A mark overlapping another gets a **2px surface-coloured ring** for separation

### 5.2 Distribution plots (`sum_gamerounds`)

- **Log x-axis**, labelled as log — the metric is severely right-skewed
- Overlaid density per arm at ~70% opacity, or offset small multiples
- Mean *and* median marked, since they diverge sharply here
- The extreme-outlier tail stays visible; annotate it rather than clipping it (R1.6)

### 5.3 Sequential / peeking plots

- x = sample size or look number; y = cumulative estimate
- Alpha-spending boundaries as a shaded band from the sequential ramp
- The naive fixed-horizon α = 0.05 line dashed for contrast
- Annotate the point where a naive reading would have stopped — that annotation *is*
  the lesson

### 5.4 Balance / love plots

- Standardised mean differences, before vs after weighting
- Diverging scale centred on zero; ±0.1 threshold lines
- One row per covariate, sorted by pre-weighting imbalance
- Before and after as slots 1 and 2, both direct-labelled

### 5.5 Propensity overlap

- Density by treatment arm; trimmed region shaded, never silently dropped
- **`--pairs all` palette rules apply** (scatter/overlap forms) — hence the
  three-slot cap validated in §2

### 5.6 Benchmark results (Phase 6 headline)

- Grouped horizontal bars: bias per estimator × confounding regime
- Zero line = ground truth τ̂\*
- **2px surface gap between adjacent bars and between stacked segments**
- Coverage shown as a separate small-multiple, **not** a second y-axis

---

## 6. Universal chart rules

- **One axis. Never a dual-axis chart.** Two measures of different scale become two
  charts, small multiples, or an indexed common base. This is the single most common
  chart mistake and it is banned outright.
- **Legend present whenever there are ≥ 2 series**; ≤ 4 series are also
  direct-labelled, so identity is never carried by colour alone. A single series
  needs no legend box — the title names it.
- **Selective labels only.** Endpoints, extremes, and the headline value. Never a
  number on every point.
- **Marks:** 2px lines; markers ≥ 8px; bars thin with 4px rounded data-ends anchored
  to the baseline; 2px surface gap between adjacent fills.
- **Chrome recedes:** hairline gridlines on one axis only, muted ticks, no borders
  around the plot area beyond the card ring.
- **Hover by default** on every interactive chart — crosshair + tooltip on
  line/area, per-mark tooltip on bar/dot/cell. Hit targets larger than the mark. The
  only exception is a bare stat tile with no plot.
- **A table view exists for every chart**, which is also what discharges the §2.1
  contrast relief obligation.
- **Never colour by rank.** Sorting a chart must not repaint its series.
- **Render it and look at it.** The validator checks colour, not layout — open every
  figure and check for label collisions and overflow before calling it done.

---

## 7. The synthetic-data badge

Required wherever results appear (R1.11). Rendered from
`EffectEstimate.data_source`, so it cannot be forgotten:

| `data_source` | Badge | Style |
|---|---|---|
| `REAL` | `real data` | muted ink, hairline ring, no fill |
| `SEMI_SYNTHETIC` | `semi-synthetic · injected confounding` | warning `#fab219` + ● icon |
| `SYNTHETIC` | `synthetic data` | warning `#fab219` + ● icon |

Badges sit in the header **and** on every exported figure — a chart lifted out of
the app into a slide deck must carry its own provenance.

---

## 8. Accessibility checklist

Before any chart ships:

- [ ] Validator run for **both** modes; all six checks pass (§2)
- [ ] `--pairs all` used for scatter / overlap / small-multiple forms
- [ ] Identity never colour-alone — legend plus direct labels
- [ ] Light-mode slot 3 (aqua) carries visible labels or the table view
- [ ] Status colours always paired with icon + label
- [ ] Table view reachable for every chart
- [ ] Dark mode verified by eye, not assumed
- [ ] Texture channel available for CVD / print / `forced-colors`
- [ ] No dual axes anywhere
