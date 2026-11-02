# Assumptions

Every rule-derived or judgement-call assumption in this project, with its source.

Market-rule detail is never taken from training data. Each rule row below cites a clause
in a primary document under `data/reference/docs/`, fetched by `src/download_docs.py` and
rendered to searchable text by `src/ruletext.py`. Quotes are from those files and can be
re-checked with:

```
.venv\Scripts\python.exe -m src.ruletext --doc esm_rules_2026-10-01 --find "<phrase>"
```

**Naming.** The Wholesale Electricity Market Rules were renamed the **Electricity System
and Market Rules (ESM Rules)** on 6 February 2025. Clause numbers cited below are from the
1 October 2026 consolidation unless stated otherwise, cross-checked against the
1 October 2023 version that was in force at the start of the study period.

**Period.** Trading days from 1 October 2023 (the New WEM Commencement Day) to the most
recent complete trading day in the data, inclusive. The WEM trading day runs 08:00 to
08:00 AWST.

---

## 1. Settlement of energy

| # | Assumption | Clause | Quote | Checked | Effect on the model |
|---|---|---|---|---|---|
| E1 | **Energy is settled on the 30-minute Reference Trading Price, not the 5-minute Energy Clearing Price.** | ESM Rules 9.9.4 | "EnergyTradingAmount(p,t) = ReferenceTradingPrice(t) × NetTradingQuantity(p,t)" where "ReferenceTradingPrice(t) is the Final Reference Trading Price for Trading Interval t" | 2 Nov 2026 | Decisive. Revenue is computed against the 30-minute price, so arbitrage *within* a half hour earns nothing. Dispatch is still optimised at 5 minutes because the physical and FCESS constraints bind at 5 minutes. |
| E2 | The Reference Trading Price is the time-weighted average of the 5-minute energy Market Clearing Prices in the Trading Interval. | ESM Rules 7.11A.1(b) | "a Reference Trading Price is determined by AEMO as the time-weighted average of the Market Clearing Prices for energy for each Dispatch Interval in a Trading Interval" | 2 Nov 2026 | `reference-trading-price-csv` is used directly for settlement. The derivation from the 5-minute series is reproduced as a validation check, not as the source. |
| E3 | The same price applies to both sales and purchases, so charging and discharging face one price. | ESM Rules 7.11A.1(g) | "subject to section 9.9, AEMO must apply the Reference Trading Price to both sales and purchases of energy in the relevant Trading Interval" | 2 Nov 2026 | No bid/offer spread in the energy settlement. The only round-trip loss is physical efficiency plus the throughput cost. |
| E4 | For a standalone merchant battery with no bilateral contracts, the Net Contract Position is zero, so the Net Trading Quantity is the Metered Schedule. | ESM Rules 9.9.5 | "NetTradingQuantity(p,t) = Σ MeteredSchedule(f,t) − NetContractPosition(p,t)" | 2 Nov 2026 | Modelling choice for the hypothetical battery. Real batteries may hold bilateral contracts, which is one reason the backcast can differ from the model. |
| E5 | **Energy Uplift Payments are excluded from modelled revenue.** | ESM Rules 9.9.6–9.9.11 | "EnergyUpliftPayment(f,DI) = IsMisPriced(f,DI) × EnergyUpliftPrice(f,DI) × EnergyUpliftQuantity(f,DI)" | 2 Nov 2026 | Judgement call. Uplift compensates a facility dispatched against its own offer, which requires offer data that is not public (see D3). Excluding it makes the modelled revenue a *lower* bound on this component. |
| E6 | Sent-out energy is positive and consumed energy is negative. | ESM Rules 9.5.5 | "a quantity of energy generated and sent out into the relevant Network has a positive value and a quantity of energy consumed has a negative value" | 2 Nov 2026 | Sign convention for the backcast: charging intervals produce negative revenue at the prevailing price. |

## 2. Loss factors

| # | Assumption | Clause | Quote | Checked | Effect on the model |
|---|---|---|---|---|---|
| L1 | **The loss factor adjusts the settled quantity, not the price.** | ESM Rules 9.5.2 | "the net quantity of energy generated and sent out into the relevant Network or consumed by the Facility during that Trading Interval, **Loss Factor adjusted to the Reference Node**" | 2 Nov 2026 | Revenue = price × metered MWh × loss factor. Because the same factor scales both charging and discharging, a loss factor other than 1.0 scales net arbitrage revenue roughly linearly, and does *not* change the optimal dispatch pattern. |
| L2 | The Loss Factor is the product of a Transmission Loss Factor and a Distribution Loss Factor; transmission-connected points have a Distribution Loss Factor of 1. | ESM Rules 2.27.3, 2.27.4(c) | "each Loss Factor must be expressed as the product of a Transmission Loss Factor and a Distribution Loss Factor"; "Measurement Points on the transmission system are assigned to a Distribution Loss Factor Class with a Distribution Loss Factor equal to one" | 2 Nov 2026 | All nine SWIS batteries are transmission-connected Scheduled Facilities, so their Loss Factor is their Transmission Loss Factor. |
| L3 | Transmission Loss Factors are marginal, relative to the Reference Node, averaged over a year, and recalculated annually by 1 June for the following financial year. | ESM Rules 2.27.5(a), 2.27.6 | "Transmission Loss Factors must notionally represent the marginal transmission system losses for a Measurement Point relative to the Reference Node, averaged over all Trading Intervals in a year"; "Each year by 1 June each Network Operator must … recalculate the Loss Factors" | 2 Nov 2026 | Loss factors are a step function of financial year, applied per facility per trading day. |
| L4 | **JUDGEMENT CALL — loss factor values are hand-curated.** Per-facility loss factors are not published in the AEMO public data file portal, and the AEMO loss-factors web page rejects automated requests. | — | — | 2 Nov 2026 | Values are entered in `data/reference/loss_factors.csv` from the annual WEM Loss Factor Reports, one row per facility per financial year, with the report and page cited in the file. Any facility-year that cannot be sourced uses 1.0 and is flagged in the file. **Sensitivity case LF-1** varies the modelled battery's loss factor over 0.95–1.05. |

## 3. FCESS

| # | Assumption | Clause | Quote | Checked | Effect on the model |
|---|---|---|---|---|---|
| F1 | There are five Frequency Co-optimised Essential System Services: Contingency Reserve Raise, Contingency Reserve Lower, Regulation Raise, Regulation Lower and the RoCoF Control Service. | ESM Rules 9.10.3(a)–(d) | "CR_Payable … CL_Payable … RCS_Payable … Regulation_Payable" | 2 Nov 2026 | Matches the five FCESS price columns in `market-clearing-prices-csv` and the five schedule columns in `market-schedule-csv`. System Restart Service and NCESS are separate, are not co-optimised, and are excluded. |
| F2 | **FCESS is paid per 5-minute Dispatch Interval on the enablement quantity at that interval's Market Clearing Price.** | ESM Rules 9.10.6 | "CR_Payable(f,DI) = CR_MCP(DI) × 5/60 × CR_EnablementQuantity(f,DI) × CR_PerformanceFactor(f,DI) + CR_AvailabilityPayment(f,DI) − CR_SESSMRefund(f,DI)", where "5/60 represents the period of a Dispatch Interval in hours" | 2 Nov 2026 | FCESS revenue is a 5-minute calculation while energy is a 30-minute one. The model co-optimises at 5 minutes and settles energy at 30. |
| F3 | No loss factor applies to FCESS payments. | ESM Rules 9.10.6 | The formula contains no loss factor term, unlike 9.5.2 for energy. | 2 Nov 2026 | FCESS revenue is not scaled by the loss factor. |
| F4 | **ASSUMED — the Performance Factor is 1.0.** | ESM Rules 9.10.6(c) and following | — | 2 Nov 2026 | Performance factors are derived from facility response measurement that is not published per facility per interval. Setting it to 1.0 makes modelled FCESS revenue an **upper** bound on this component. **Sensitivity case F-1** applies 0.90 and 0.95. |
| F5 | **ASSUMED — Availability Payments and SESSM Refunds are zero.** | ESM Rules 9.10.6 | — | 2 Nov 2026 | Both are zero outside a Supplementary Essential System Service Mechanism event. `fcess-uplift-payments` is downloaded and checked to confirm the magnitude of uplift over the period; if material it is reported separately rather than folded into the modelled revenue. |
| F6 | FCESS Uplift Payments are excluded from modelled revenue. | ESM Rules 9.10.3A–9.10.3C | "FCESSUpliftPayment(f,DI) = Max(0, RTMDispatchCost(f,DI) − RTMBaseCompensation(f,DI))" | 2 Nov 2026 | Same reason as E5: the calculation needs offer data. Exclusion makes modelled revenue a lower bound on this component. |
| F7 | **JUDGEMENT CALL — enablement is taken from the market schedule dataset.** | — | — | 2 Nov 2026 | Clause 9.10.6(c) settles on the Essential System Service Enablement Quantity published under 7.13.1E(c). The public portal exposes the *schedule* quantities in `market-schedule-csv`, which are the dispatch run's enablement quantities. These are used as the enablement series. They may differ from the final settled quantities after outage adjustments under 9.10.6(c)(ii). The gap is quantified in stage 5 by reconciling against published FCESS totals. |
| F10 | **The hypothetical battery is accredited for 50 % of its rating in each of the four raise and lower services, and nothing for RoCoF.** | Derived from the FCESS accreditation register for the real SWIS batteries: accredited quantity divided by registered MW has a median of 0.48 for contingency raise and lower and 0.47 for regulation raise and lower, with a maximum of exactly 0.50 across all six accredited facilities (COLLIE_BESS2 300/600, COLLIE_ESR1 200/400, COLLIE_ESR4 and ESR5 240.6/500, KWINANA_ESR2 199.7/450, KWINANA_ESR1 50/200). **No SWIS battery holds any RoCoF accreditation.** | 2 Nov 2026 | Caps enablement in the dispatch LP at 50 MW per service for a 100 MW battery, and excludes RoCoF entirely. Using the fleet's own observed ceiling rather than the power rating avoids overstating FCESS revenue, which the headroom constraints alone would allow up to 100 MW. RoCoF prices are in any case near zero over the period (mean $0.01/MWh in 2023, $0.00 thereafter), so excluding it changes almost nothing. |
| F11 | **Enablement is capped at the entrant's pro-rata share of the published market requirement, not at its accreditation.** | `market-requirements-csv` gives the whole-SWIS requirement per five-minute interval: contingency raise averages 187 MW in 2024 and regulation raise 87 MW. The FCESS accreditation register totals 2,000 MW of contingency raise and 1,569 MW of regulation raise across 29 facilities. | 2 Nov 2026 | **Material.** Without this cap the optimiser enables to the accreditation in every interval, which for a 100 MW battery is a quarter of the entire SWIS contingency raise requirement held continuously, and produces FCESS revenue an order of magnitude above anything any real battery earned. The share used is `own / (fleet + own)`, about 2.4 % for a battery accredited at 50 MW. The resulting enablement is consistent with what real batteries actually achieved: KWINANA_ESR1 averaged 7.6 MW of contingency raise and 23.2 MW of regulation raise over 2024. **Caveats:** the register is a late-2026 snapshot, so the share is conservative for the earlier years when the fleet was smaller; and pro-rata allocation is a simplification of a merit-order clearing. Both are varied in **sensitivity case F-2** (share × 0.5 and × 2). The uncapped case is also reported, labelled as an upper bound. |
| F9 | **A blank FCESS schedule quantity means no enablement, so it is read as zero.** | `market-schedule-csv` leaves the five FCESS columns blank rather than zero for facilities not enabled for a service: 20.6 M of 28.4 M rows are blank for regulation raise. | 2 Nov 2026 | `src.data.load_schedule` fills them with zero. Treating a blank as missing instead would drop most of the dataset; treating it as anything other than zero would invent enablement that was never scheduled. |
| F8 | Where there is an FCESS Accreditation Shortfall, the clearing price is set to the difference between the Energy Offer Price Ceiling and Floor. | ESM Rules 7.11A.1(i) | "where there is a FCESS Accreditation Shortfall in a Frequency Co-optimised Essential System Service, AEMO must set the Market Clearing Price for that service to the difference between the Energy Offer Price Ceiling and the Energy Offer Price Floor" | 2 Nov 2026 | Explains extreme FCESS price spikes. Those intervals are reported separately so a handful of them cannot be mistaken for a durable revenue stream. |

## 4. Capacity Credits and the ESR obligation

| # | Assumption | Clause | Quote | Checked | Effect on the model |
|---|---|---|---|---|---|
| C1 | A storage facility's Peak Certified Reserve Capacity is the capacity it can sustain over the Peak ESR Obligation Duration, derated to 41 °C. | ESM Rules 4.11.3 | "AEMO's reasonable expectation of the Linearly Derating Capacity that each Electric Storage Resource can sustain over the Peak Electric Storage Resource Obligation Duration after netting off capacity required to serve Loads" | 2 Nov 2026 | A 1-hour 100 MW battery cannot be accredited for 100 MW over a 4-hour obligation. Accredited MW ≈ `p_max × min(1, duration_h / esr_duration_h)`, which is why the 1-hour case earns far less capacity revenue per MW than the 4-hour case. |
| C2 | **The Electric Storage Resource Obligation Duration is 8 contiguous Trading Intervals (4 hours) throughout the study period.** | WEM Rules (1 Oct 2023) glossary, "Electric Storage Resource Obligation Duration"; ESM Rules (1 Oct 2026) 1.63.1 | 2023: "The **eight contiguous** Electric Storage Resource Obligation Intervals which apply each Trading Day and commence at the time published by AEMO in accordance with clause 4.11.3A." 2026: "the ESR Duration Requirement for the 2024 Reserve Capacity Cycle is 8 Trading Intervals." | 2 Nov 2026 | Sets the derating denominator for the study period. Later cycles lengthen it under the Reserve Capacity Reform, which is noted but falls outside the period. |
| C3 | An ESROI is a Trading Interval in which a non-zero Reserve Capacity Obligation Quantity applies to the storage facility. | WEM Rules (1 Oct 2023) glossary, "Electric Storage Resource Obligation Interval" | "A Trading Interval, that AEMO has determined in accordance with the WEM Procedure referred to in clause 4.11.3A, in which a **non-zero Reserve Capacity Obligation Quantity is applied** to an Electric Storage Resource." | 2 Nov 2026 | The obligation is an availability obligation over a published window, which is exactly what the `electric-storage-resource-obligation-intervals` dataset publishes. Validates the interpretation in D2. |
| C4 | A facility holding Capacity Credits is expected to be available for dispatch at its Certified Reserve Capacity across the whole obligation duration. | ESM Rules (1 Oct 2026) 4.11.4; WEM Rules (1 Oct 2023) 4.11.3, 4.11.3A | 2026: "expects it to be available to be dispatched for the MW quantity of its Peak Certified Reserve Capacity during all Default Peak Electric Storage Resource Obligation Intervals on each Business Day" | 2 Nov 2026 | Implemented as a hard constraint in the dispatch LP: across the published ESROI window the battery must retain enough state of charge to discharge at its accredited MW for the whole window, and may not be net charging within it. |
| C5 | **JUDGEMENT CALL — capacity revenue uses published Reserve Capacity Prices, not the rule formula.** | ESM Rules 4.29.1 | "The Reserve Capacity Prices for a Reserve Capacity Cycle to apply during the period specified in clause 4.1.29 are to be calculated as follows…" | 2 Nov 2026 | The formula depends on the Benchmark Reserve Capacity Price, excess capacity and several factors, and the regime changed within the period. The realised prices from the `reserve-capacity-prices` dataset are used instead. This is an observed value, not a model output. |
| C6 | Capacity Credits for a cycle apply from the trading day starting 1 October of Year 3 to the trading day ending 1 October of Year 4. | ESM Rules 4.1.29, 4.1.30 | "apply from the start of the Trading Day commencing on 1 October of Year 3 of the Reserve Capacity Cycle to the end of the Trading Day ending on 1 October of Year 4" | 2 Nov 2026 | Capacity years run 1 October to 1 October, matching the `capacity-credits-since-market-start` column labels. |
| C7 | **NOT MODELLED — Capacity Cost Refunds for failing the obligation.** | ESM Rules 4.26 | — | 2 Nov 2026 | The modelled battery satisfies C4 as a hard constraint, so no refund arises. Real batteries may trade off a refund against energy revenue, which is one candidate explanation for a backcast gap. Listed in stage 5's gap diagnosis. |

## 5. Offer price limits

| # | Assumption | Clause | Quote | Checked | Effect on the model |
|---|---|---|---|---|---|
| P1 | An Energy Offer Price Ceiling and Floor bound energy offers, are reviewed by the ERA, and changed within the study period. | ESM Rules 2.26.2, 2.26.2D–2.26.2G | "allow for the Real-Time Market for energy to clear without the Reference Trading Price being equal to the Energy Offer Price Floor in most circumstances" | 2 Nov 2026 | Used only to interpret price extremes in the data, not as a model input. Observed minima and maxima are reported directly from the cleaned price series, so the model needs no assumed cap. |

## 6. Data conventions and judgement calls

| # | Assumption | Source | Checked | Effect on the model |
|---|---|---|---|---|
| D1 | Timestamps are AWST (UTC+8, no daylight saving) and mark the start of the interval; dates are day-first. The trading day runs 08:00 to 08:00, so `trading_date = (interval_start − 8h).date()`. | AEMO file conventions, confirmed against the raw files: `MarketClearingPrices-2024.csv` ends at `01/01/2025 07:55`. | 2 Nov 2026 | Parsed with an explicit `%d/%m/%Y %H:%M:%S` format in `src/clean.py`. Inferred parsing would silently read 1 February as 2 January. |
| D2 | The published `ESROI Trading Interval End` is the **start** of the final obligation Trading Interval, so a window of 17:30–21:00 is 8 Trading Intervals. | `electric-storage-resource-obligation-intervals-2026.csv` gives 17:30–21:00 windows; the ESR Duration Requirement is 8 Trading Intervals (C2), and 17:30 to 21:00 inclusive of both endpoints is exactly 8 half hours. | 2 Nov 2026 | Determines the length of the ESROI constraint window. `src/clean.py` asserts the implied length and reports the distribution of window lengths so a change in regime is visible. |
| D3 | **JUDGEMENT CALL — there is no public post-reform offer data.** `effective-balancing-submission` stops at September 2023; post-reform Real-Time Market Submissions are available only to participants through WEMS and its API, which this project does not use. | AEMO public data file portal index; AEMO Real-Time Market Submissions participant documentation. | 2 Nov 2026 | The price-impact analysis cannot use observed offer curves. Instead an implied supply relationship is estimated empirically from `market-schedule-csv` quantities, SCADA and clearing prices (stage 6). The result is an **estimated elasticity, not observed offers**, and is labelled as such everywhere it appears. |
| D4 | **JUDGEMENT CALL — battery energy capacity (MWh) is not published.** `post-facilities` gives registered MW only. | `post-facilities/facilities.csv` columns. | 2 Nov 2026 | Durations for the nine real batteries are entered in `data/reference/storage_durations.csv` with a cited public source per facility, and the accredited capacity implied by C1 and C2 is used as a cross-check. Where the two disagree the discrepancy is reported rather than reconciled. |
| D5 | **CONFIRMED — SCADA `Average MWh` is MWh per five-minute interval, not average MW.** So `MW = average_mwh × 12`. | `scripts/diagnose_scada_unit.py` over the first six months of cleaned SCADA. For all 20 facilities above 100 MW the observed peak divided by registered MW / 12 clusters at 1.00 (thermal plant: NEWGEN_NEERABUP_GT1 1.005, COLLIE_G1 1.017, BW2_BLUEWATERS_G1 1.016, ALINTA_PNJ_U2 1.003), while the peak divided by registered MW is about 0.084, that is 1/12. | 2 Nov 2026 | Energy quantities are summed directly from `average_mwh`; power is derived as `× 12`. The verdict is unanimous across every large facility, so there is no residual ambiguity. Two facilities sit well below 1.0 for reasons unrelated to the unit: WARRADARGE_WF1 at 0.63 is a wind farm that never reaches nameplate, and KWINANA_ESR1 at 0.50 peaks at 100 MW against a registered 200 MW, which is followed up in stage 5. |
| D6 | Rows are sorted newest first in the raw files and `Extracted At` is populated on the first row only. | Confirmed against the raw files. | 2 Nov 2026 | `src/clean.py` sorts ascending and drops `Extracted At`. |
| D7 | Missing intervals are reported, never filled. Duplicate keys are fatal. | — | 2 Nov 2026 | Filling would invent price or quantity data; a duplicate key would double-count revenue. `src/clean.py` raises on duplicates and lists short trading days. |

## 7. Model and environment judgement calls

| # | Assumption | Reason | Checked | Effect on the model |
|---|---|---|---|---|
| M1 | **HiGHS is driven directly through `highspy` rather than through cvxpy.** | `import cvxpy` segfaults in this environment: its `qdldl` dependency (already at its latest release, 0.1.9.post1) raises an access violation against numpy 2.4.6 on Windows. Downgrading numpy would break pandas 3.1.0. | 2 Nov 2026 | The solver is still HiGHS, as required. Driving it directly also lets the LP be built once and re-solved with warm starts, which the rolling horizon needs; a Pyomo or cvxpy layer would rebuild the model on every one of several hundred thousand solves. Correctness is defended by the hand-calculated toy-day tests rather than by a second modelling layer. |
| M2 | Charge and discharge efficiencies are separate and equal to `sqrt(round_trip)` by default. | Modelling convention; no rule governs it. | — | `BatteryParams.eta_ch`, `eta_dis`. Varied in the tornado chart. |
| M3 | State of charge starts and ends each trading day at 50 % of usable energy. | Prevents the optimiser from earning revenue by draining a battery it never refills, which would overstate the first and last day of every run. | — | Hard constraint in the LP. The alternative — a free terminal state of charge — is run as a sensitivity to show how much of the perfect-foresight bound depends on it. |
| M4 | A per-MWh throughput cost represents degradation inside the optimisation, and is charged on half of total throughput so one full cycle is charged once. | Standard practice; ties the dispatch decision to the degradation cost. | — | `BatteryParams.c_thru`. The economics stage then recomputes degradation from the *modelled* throughput rather than assuming a cycle count. |
| M5 | Simultaneous charging and discharging is left unconstrained in the LP unless it appears in the solution. | A binary would make the model a MILP and multiply solve time across hundreds of thousands of rolling-horizon solves. | — | The toy-day tests assert it does not occur at positive prices. If it appears at negative prices the binary is switched on and the revenue effect is reported. |

---

## Registered sensitivity cases

| Case | Varies | Range | Reason |
|---|---|---|---|
| LF-1 | Modelled battery loss factor | 0.95 – 1.05 | L4: values are hand-curated, not machine-readable. |
| F-1 | FCESS Performance Factor | 0.90, 0.95, 1.00 | F4: not published per facility per interval. |
| M3-1 | Terminal state of charge | fixed at 50 % vs free | M3: tests how much of the bound depends on the boundary condition. |
| D4-1 | Real battery durations | ±25 % | D4: MWh is not published. |
| PI-1 | Price impact of a 100 MW entrant | zero vs estimated elasticity | D3: no offer data, so the elasticity is estimated. |

## Open items

* **D5** — the SCADA unit verdict, to be filled in from `diagnose_scada_unit` once `facility_scada` is cleaned.
* **Clause cross-check** — clause numbers are from the 1 October 2026 consolidation. The
  1 October 2023 and 1 January 2025 versions are downloaded; the cross-check that the
  cited clauses carry the same numbering and wording throughout the study period is
  recorded in section 8 below once complete.
* **WEM Procedure: Electric Storage Resource Obligation Intervals** — the procedure URL in
  `src/download_docs.py` returned HTTP 403/404 on 2 November 2026. The ESROI
  interpretation is therefore sourced from the Rules definition (C3) and validated against
  the published data (D2) rather than from the procedure.
* **FCESS reconciliation tolerance** — set in stage 5 against AEMO quarterly WEM reports
  and ERA reports, with source and page recorded there.

## 8. Clause cross-check across versions

Checked 2 November 2026 by locating each cited clause in both
`wem_rules_2023-10-01.txt` (in force at the start of the study period) and
`esm_rules_2026-10-01.txt` (current).

| Cited clause | 1 Oct 2023 | 1 Oct 2026 | Verdict |
|---|---|---|---|
| 9.9.4 energy trading amount | present, identical formula | present, identical formula | **stable** — the settlement price for energy is the Reference Trading Price across the whole period |
| 9.9.5 net trading quantity | present, identical | present, identical | stable |
| 9.5.2 Metered Schedule, loss factor adjusted | present, identical wording | present, identical wording | stable |
| 2.27.3 Loss Factor = TLF × DLF | present, identical | present, identical | stable |
| 7.11A.1(g) one price for sales and purchases | present, identical | present, identical | stable |
| 9.10.6 FCESS payment formula | present, identical formula | present, identical formula | stable |
| 4.11.3 ESR Certified Reserve Capacity | "Certified Reserve Capacity … over the **Electric Storage Resource Obligation Duration**" | "**Peak** Certified Reserve Capacity … over the **Peak** Electric Storage Resource Obligation Duration" | **renamed, same mechanism.** The Reserve Capacity Reform split capacity into Peak and Flexible after the study period began. The derating logic in C1 is unchanged. |
| ESR Obligation Duration | glossary: "eight contiguous" Trading Intervals | 1.63.1: 8 Trading Intervals for the 2024 cycle, lengthening in later cycles | 4 hours throughout the study period |

Conclusion: every clause the revenue model depends on is stable across the study period.
The only drift is terminology around the Reserve Capacity Mechanism, which affects naming
rather than calculation. Capacity revenue is in any case taken from published Reserve
Capacity Prices (C5) rather than from the rule formula.
