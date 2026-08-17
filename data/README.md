# Data

**Nothing in `raw/` or `processed/` is committed** (Rules §6). This file explains how
to populate them.

## Cookie Cats

The primary dataset: a mobile-game A/B test in which the first progression gate was
placed at level 30 (control) or level 40 (treatment).

1. Download `cookie_cats.csv` from the Kaggle *Cookie Cats* / *Mobile Games A/B
   Testing* dataset.
2. Place it at `data/raw/cookie_cats.csv`.
3. Verify it loads:

   ```bash
   python -c "from gatekeeper.data import load_cookie_cats; print(load_cookie_cats().summary())"
   ```

Ingest validates the schema strictly and **fails loudly** rather than coercing
(Rules §5), so a wrong file or an unexpected variant label raises immediately.

### Expected schema

| Column | Type | Notes |
|---|---|---|
| `userid` | int | Randomisation unit. Must be unique. |
| `version` | str | `gate_30` (control) or `gate_40` (treatment). No other value permitted. |
| `sum_gamerounds` | int | Rounds played in the first 14 days. **Post-treatment** — see R1.7. |
| `retention_1` | bool | Returned 1 day after install. |
| `retention_7` | bool | Returned 7 days after install. |

`sum_gamerounds` is measured *after* the player encounters the gate. It is a
mediator, not a covariate, and must never be used as a control or as a CUPED
covariate (Rules R1.7). `variance/cuped.py` raises `PostTreatmentCovariateError`
if you try.

## Layout

```
data/
├── README.md          # this file (committed)
├── raw/               # cookie_cats.csv — gitignored
└── processed/         # Parquet cache, written by ingest — gitignored
```

## Working without the dataset

The library does not require the real CSV. `gatekeeper.data.synthetic` generates
schema-identical frames with a known ground-truth effect, which is what the test
suite runs on. You can build and test every phase before obtaining the download —
you just cannot produce the real Phase 3 readout.
