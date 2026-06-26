# 2026-06-26 Official Sales Evidence Tightening

## What Changed

- Replaced the demo sales evidence file with official-domain public rank evidence only.
- Added an `official_only` evidence path for repository queries, CLI comparison, ML calibration, API `/api/sales-gap`, and the Streamlit workbench.
- Kept non-official public claims available only through explicit opt-in (`--include-non-official`, `official_only=false`, or the Streamlit sidebar toggle).

## Current Evidence Boundary

- Public official sources found so far provide skin sales rankings, not exact unit sales.
- Sources are treated as `official_public_rank`, with `measurement_kind=rank_proxy`.
- Exact B2B-grade sales validation still requires client-authorized data: unit sales, revenue, conversion rate, refund/chargeback if relevant, campaign spend, time window, and channel split.

## Current Local Result

- Official public rank evidence: 53 skins / 76 evidence items.
- Calibration scope: `official_only`.
- Training calibration: 2/53 samples with `abs(gap) > 10`, p=0.089799.
- Leave-one-out validation: 32/53 samples with `abs(gap) > 10`, so the model is not yet generalizable.

## Next Step

The next B2B milestone should be an authorized sales-data connector or import format. Public official ranks can calibrate directionality, but they cannot prove exact sales lift.
