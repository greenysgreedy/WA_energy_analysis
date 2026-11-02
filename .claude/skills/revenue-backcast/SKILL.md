---
name: revenue-backcast
description: Use when rebuilding what real SWIS batteries actually earned from AEMO SCADA and prices, and comparing that with modelled revenue or published figures.
---

# Revenue backcast of real batteries

The purpose is to test the model against what actually happened. If the model's realistic strategy says a battery could earn far more than real batteries did, either the model is missing a constraint or the real operators are doing something else — find out which.

## Steps

1. **Identify the batteries.** Pull storage facilities from facility standing data. Record the facility code, registered MW and MWh, commissioning date, and owner. Only count a facility from its first full month of normal operation, and exclude commissioning tests.
2. **Confirm the SCADA unit.** Check whether `Average MWh` is an average MW or MWh per interval. Test it: peak values should be close to registered MW (if MW) or MW/12 (if 5-minute MWh). Write the answer in docs/assumptions.md.
3. **Energy revenue.** Per interval: `net_export_MWh * settlement_price * loss_factor`. Use the settlement price and interval confirmed in assumptions.md. Charging is negative revenue.
4. **FCESS revenue.** Only if per-facility enablement data is available: `enablement * service_price * dt`. If it isn't, say so and estimate an upper bound from accredited quantities — label it as a bound.
5. **Capacity revenue.** Capacity credits for the capacity year × the Reserve Capacity Price for that year (cite the source).
6. **Normalise.** Report $/MW-year and $/MWh-year of capacity, by month and by year, so batteries of different sizes compare.
7. **Compare with the model.** Run the dispatch model with each real battery's MW, MWh and an assumed efficiency over the same period. Report the actual/perfect-foresight ratio next to the model's own capture rate.
8. **Reconcile with published figures.** Compare summed FCESS revenue with AEMO quarterly WEM reports and ERA reports. Record the tolerance used and the source and page for each published figure.

## Explaining gaps

For any gap over the tolerance, check in this order: wrong unit or sign, wrong price series, missing loss factor, partial operating months, ESR obligation intervals, FCESS enablement reducing energy dispatch, outages, and network constraints. Write the explanation into docs/results.md. Don't tune the model to close the gap.

## Output

One table per battery: period, MWh charged and discharged, cycles, energy $, FCESS $ (actual or bound), capacity $, total $/MW-year, model perfect-foresight $, ratio. Plus a short note on each material gap.
