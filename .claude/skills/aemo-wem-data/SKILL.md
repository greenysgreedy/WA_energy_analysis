---
name: aemo-wem-data
description: Use when downloading, loading or cleaning AEMO WA Wholesale Electricity Market (WEM) public data files — prices, FCESS, facility SCADA, capacity credits.
---

# AEMO WEM public data

Source: https://data.wa.aemo.com.au/public/public-data/datafiles/<dataset>/ — each dataset folder is an HTML index of CSV links. Use the public files, not developer APIs.

## Downloading

1. Parse the index page for `.csv` links and resolve them against the folder URL.
2. Save to `data/raw/<dataset>/<filename>` unchanged. Skip files that already exist, except the current period's file, which AEMO keeps updating — re-download that one.
3. Wait about 6.7 seconds between requests. Use `raise_for_status()` and a 30-second timeout.
4. Don't overwrite processed outputs from the download step.

## File conventions

- Timestamps are AWST (UTC+8, no daylight saving) and mark the start of the interval. Parse with `format="%d/%m/%Y %H:%M:%S"` (day first). Never let pandas infer the format.
- The trading day runs 08:00 to 08:00. Yearly and monthly files are grouped by trading day, so a "2023" file ends at 01/01/2024 07:55. Derive `trading_date = (ts - 8h).date()`.
- Rows come newest first, so sort ascending after loading.
- `Extracted At` is filled on the first row only. Drop it.
- `Failure Reason` on price files: keep it, and count non-null rows in the data checks.

## Known datasets

| Dataset | Grain | Key columns |
|---|---|---|
| market-clearing-prices-csv | 5 min | Energy, Contingency Raise/Lower, Regulation Raise/Lower, RoCoF clearing prices ($/MWh) |
| reference-trading-price-csv | 30 min | Reference Trading Price |
| facility-scada-csv | 5 min, monthly files | Facility Code, Average MWh (confirm the unit before revenue use) |
| fcess | standing | Max accredited FCESS quantities per facility (not enablement) |
| capacity-credits-since-market-start | capacity year | Credits per facility by capacity year (wide format; melt it) |

If a needed quantity (FCESS enablement, offers, loss factors, facility MW/MWh) isn't in these, find the dataset on the index and add it to the table in the project CLAUDE.md.

## Cleaning output

Write tidy parquet to `data/processed/` with a UTC-naive AWST `interval_start` column, `trading_date`, and one row per interval (per facility where relevant).

## Checks — every cleaned file

- Expected count per trading day: 288 (5 min) or 48 (30 min). List any shortfalls, don't fill them silently.
- No duplicate (interval, facility) keys.
- No null prices outside rows that have a Failure Reason.
- Price range sanity: report the min and max, and the number of intervals at the price cap or floor.
- First and last interval match the expected period.

Print a one-table check summary per dataset and fail loudly on duplicates.
