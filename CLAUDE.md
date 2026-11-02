# WA_energy_analysis

How much could a grid battery in WA's SWIS have earned since the WEM reform (1 October 2023), and how much of that could it realistically capture? Claude builds the pipeline end to end (code, tests, notebooks, write-up). Keep explanations brief and commit after each stage.

## Scope

- Post-reform data only: trading days from 1 October 2023 onwards.
- Source: AEMO public data files at https://data.wa.aemo.com.au/public/public-data/datafiles. No developer APIs.
- Design strategies on 2024. Test them unchanged on 2025, then on 2026 to date. Never tune on the test years.
- Battery cases: 100 MW at 1, 2 and 4 hours unless stated otherwise.
- Australian English in all prose. Write-ups in a plain academic register.

## Layout

- `src/` modules: download, clean, model, strategies, backcast, economics
- `tests/` pytest. Toy-day dispatch tests must pass before any real-data run.
- `notebooks/` one per stage, numbered, each runs top to bottom from `data/`
- `data/raw/<dataset>/` untouched AEMO files (git-ignored)
- `data/processed/` cleaned parquet (git-ignored)
- `docs/assumptions.md` every rule-derived or judgement-call assumption
- `docs/results.md` the write-up

## AEMO data conventions (checked against the files in data/raw)

- Timestamps are Perth time (AWST, UTC+8, no daylight saving) and mark the start of the interval.
- Dates are day-first: `01/02/2024 07:55:00` is 1 February.
- Trading day runs 08:00 to 08:00. Yearly and monthly files are grouped by trading day: `MarketClearingPrices-2023.csv` ends at `01/01/2024 07:55`, and `FacilityScada-2023-10.csv` contains `01/11/2023 07:55`.
- Rows are sorted newest first, so sort by time after loading.
- `Extracted At` is filled on the first row only. Drop it.
- `market-clearing-prices-csv` (5-minute): energy plus five FCESS clearing prices (contingency raise/lower, regulation raise/lower, RoCoF), and a `Failure Reason` column.
- `reference-trading-price-csv` is a 30-minute price. Use it only if docs/assumptions.md shows settlement uses it.
- `facility-scada-csv`: `Dispatch Interval, Facility Code, Average MWh`. Confirm the unit (MW average or MWh per 5 minutes) before using it for revenue.
- `fcess/fcess.csv` holds accreditation standing data (max accredited quantities per facility), not enablement. Enablement quantities need a different dataset.
- Download politely: about 6.7 seconds between requests, and skip files that already exist.

## Market rules: never from memory

For any market-rule detail (settlement price and interval, loss factors, FCESS payment, ESR obligation intervals, capacity credit rules, Reserve Capacity Price):

1. Find the governing clause in the current WEM Rules or WEM Procedure.
2. Record it in `docs/assumptions.md` as: assumption | clause reference | short quote | date checked | effect on the model.
3. If it can't be confirmed, mark it UNCONFIRMED, state the value used, and run a sensitivity case on it.

Training-data knowledge of the WEM may be out of date, so it never counts as a source.

## Checks that must always hold

- Perfect-foresight revenue is at least as high as every strategy's revenue on the same data.
- Revenue does not rise when efficiency falls or the throughput cost rises.
- Cleaned series have the expected interval counts per trading day (288 at 5 minutes, 48 at 30 minutes), with no duplicates.
- Modelled FCESS totals reconcile with AEMO/ERA published figures within the tolerance in docs/assumptions.md, or the gap is explained.
