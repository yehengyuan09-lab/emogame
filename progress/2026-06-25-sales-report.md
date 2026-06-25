# 2026-06-25 Sales Report Progress

## Done

- Added `business/sales_advisor.py` for evidence-first sales action reports.
- Added `scripts/generate_sales_report.py` CLI.
- Added tests for:
  - insufficient evidence -> collect more evidence;
  - validated purchase intent -> scale marketing;
  - weak value perception -> fix price or bundle.
- Verified the CLI on local `107-08`:
  - no market evidence produces `collect_more_evidence`;
  - no final sales decision is made from official metadata alone.

## Notes

- This layer does not predict sales volume directly.
- It converts market evidence into operational decisions and evidence gaps.
- Real sales/ownership data still needs to be ingested before calibration.
