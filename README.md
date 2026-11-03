# WA_energy_analysis

How much could a grid battery in Western Australia's South West Interconnected System
have earned since the WEM reform of 1 October 2023, and how much of that could it
realistically have captured?

**Answers and evidence: [`docs/results.md`](docs/results.md).**
**Every assumption, with its clause reference: [`docs/assumptions.md`](docs/assumptions.md).**

## In one paragraph

A 100 MW battery could have earned a great deal in the first full year after the reform
and almost nothing by the third. Perfect foresight over the whole period is worth $45m to
$76m depending on duration; a strategy without foresight captures 31% to 50% of that. By
2026 the spread has closed far enough that naive strategies lose money, while the real
batteries commissioned that year did too. What makes a SWIS battery viable is not trading
revenue but the Reserve Capacity Mechanism: a 4-hour battery earns $164,000 to $212,000
per MW-year from capacity credits alone, and no merchant case reaches a positive NPV
except at four hours on 2024 revenue.

## Layout

| Path | What is in it |
|---|---|
| `src/download_data.py` | fetch AEMO public data files into `data/raw/` |
| `src/download_docs.py` | fetch the Rules, Procedures and cost reports every assumption cites |
| `src/ruletext.py` | render those documents to searchable text so clauses can be quoted |
| `src/clean.py` | parse raw CSVs to tidy parquet with schema checks |
| `src/data.py` | the one place the settlement-price convention is applied |
| `src/loss_factors.py` | per-facility loss factors parsed from the annual Loss Factor Reports |
| `src/model.py` | the co-optimised energy and FCESS dispatch LP, solved with HiGHS |
| `src/cases.py` | the headline battery cases, defined once |
| `src/fcess_market.py` | how much FCESS enablement a new entrant could actually win |
| `src/strategies.py` | perfect foresight and rolling-horizon strategies |
| `src/backcast.py` | what the real SWIS batteries actually earned |
| `src/reconcile.py` | modelled FCESS totals against AEMO's published figures |
| `src/price_impact.py` | spread evolution and the price-taker question |
| `src/economics.py` | NPV, IRR, degradation and sensitivity analysis |
| `notebooks/` | one per stage, each runs top to bottom from `data/` |
| `tests/` | pytest; the toy-day dispatch cases gate every real-data run |
| `scripts/` | runners and data-integrity checks |
| `outputs/` | generated summary tables the write-up quotes |

## Running it

Uses the existing virtual environment at the repository root.

```
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe scripts\check_env.py            # verify, and solve a small LP
.venv\Scripts\python.exe -m pytest tests\ -q             # toy-day tests must pass first
.venv\Scripts\python.exe -m src.download_data            # about 2.7 GB, roughly 30 minutes
.venv\Scripts\python.exe -m src.download_docs
.venv\Scripts\python.exe scripts\rerun_pipeline.py       # clean, model, notebooks
```

`scripts/rerun_pipeline.py --from strategies` restarts part way through. Tests that read
real data are marked `slow` and need `--run-slow`.

## Three things worth knowing before reading the numbers

**Energy settles on the thirty-minute price, not the five-minute one.** ESM Rules 9.9.4
settles energy on the Reference Trading Price, the time-weighted average of the six
five-minute clearing prices in the Trading Interval, while clause 9.10.6 pays FCESS per
five-minute Dispatch Interval. Roughly a quarter of the apparent daily spread is an
artefact of looking at the wrong series. The model dispatches at five minutes and settles
energy at thirty.

**How much FCESS a new entrant can win is worth an order of magnitude.** Capped only by
accreditation, a 100 MW battery would hold a quarter of the entire SWIS contingency raise
requirement continuously, which no battery has ever done. The central case instead gives
it the enablement per accredited MW that the operating fleet actually achieved. All three
candidate ceilings are reported side by side in notebook 03.

**The contingency raise and lower columns are transposed at source.** In
`market-clearing-prices-csv` and `market-schedule-csv` the columns headed "Contingency
Raise" and "Contingency Lower" carry each other's data; the requirement dataset is
correct. Because the transposition is consistent, price × quantity products and revenue
totals are unaffected, but service attribution and headroom direction are not.
`scripts/check_service_mapping.py` asserts the correction is still in place.

## Data provenance

All market data is from AEMO's public data file portal at
`https://data.wa.aemo.com.au/public/public-data/datafiles`. No developer APIs are used,
which is why there is no offer-curve analysis: post-reform Real-Time Market Submissions
are participant-only. Rules, Procedures, Loss Factor Reports, Quarterly Energy Dynamics
and CSIRO GenCost are fetched by `src/download_docs.py` into `data/reference/docs/`.

Nothing under `data/` is tracked except `data/reference/loss_factors.csv`, which is a
cited input rather than an output: rebuilding it needs the annual Loss Factor Report
PDFs, which are third-party documents and are left to be re-downloaded. Everything else
is reproducible — raw files from `src/download_data.py`, parquet from `src/clean.py`,
reference documents from `src/download_docs.py`.
