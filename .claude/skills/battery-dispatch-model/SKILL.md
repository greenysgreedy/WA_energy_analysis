---
name: battery-dispatch-model
description: Use when building, changing or running the battery dispatch optimisation (energy + FCESS co-optimisation, perfect foresight or rolling horizon) for the WEM battery project.
---

# Battery dispatch model

## Conventions

- Units: MW for power, MWh for energy, $/MWh for prices, interval length `dt` in hours (5 min = 1/12).
- Separate variables `p_ch >= 0` and `p_dis >= 0`. Net export = `p_dis - p_ch`.
- Efficiency is split: `soc[t+1] = soc[t] + eta_ch*p_ch*dt - p_dis*dt/eta_dis`, with `eta_ch = eta_dis = sqrt(round_trip)` unless stated.
- SoC limits are `[soc_min, soc_max]` as fractions of usable MWh. Start and end SoC are equal (default 50%) so horizons don't drain the battery for free.
- Every parameter lives in one dataclass. No magic numbers in the model code.

## Objective

Maximise `sum_t [ price_e[t]*(p_dis-p_ch)*dt*loss_factor + sum_s price_s[t]*enable_s[t]*dt - c_thru*(p_dis+p_ch)*dt/2 ]`

`c_thru` ($/MWh throughput) carries degradation into dispatch. Confirm in docs/assumptions.md how each FCESS service is actually paid before treating `price_s*enable_s*dt` as revenue.

## FCESS co-optimisation

- Raise services: `p_dis - p_ch + raise_enable <= p_max`. Lower services: `p_ch - p_dis + lower_enable <= p_max`. Contingency and regulation stack within these unless the rules say otherwise.
- Energy headroom: SoC must cover delivering the enabled raise for the required duration (and the absorbing capacity for lower). Put the duration in assumptions.md.
- Cap each enablement at the facility's accredited maximum.
- ESR obligation intervals: if capacity credits are held, require available discharge capacity and SoC during those intervals per docs/assumptions.md.

## Solver

Pyomo or cvxpy with HiGHS. LP by default. Add a binary to stop simultaneous charge and discharge only if the LP output shows it happening at negative prices — and report whether it changed revenue.

## Tests before real data (pytest)

1. Two-price day: low then high, spread above the efficiency loss — expect one full cycle and a hand-calculated revenue.
2. Spread below the efficiency loss — expect zero cycling.
3. Flat price — expect zero revenue.
4. Negative prices — expect charging, with revenue matching the hand calculation.
5. A high FCESS price with zero energy spread — expect enablement up to the headroom limit.
6. SoC never leaves its bounds; start SoC equals end SoC.

## Strategies

- Perfect foresight: solve per trading day (or a longer window with terminal SoC fixed) on actual prices. This is the upper bound.
- Rolling horizon: at each decision point solve on forecast prices, commit only the first interval(s), settle on actual prices, roll forward. Forecasts: persistence, same time yesterday, same weekday last week.
- Report capture rate = strategy revenue / perfect-foresight revenue, by month and by year.
- Design on 2024. Run 2025 and 2026 unchanged.

## Invariants to assert on every run

Perfect foresight >= every strategy. Revenue is non-increasing in `c_thru` and non-decreasing in efficiency. Energy balance closes to solver tolerance.
