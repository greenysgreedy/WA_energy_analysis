"""Reconcile modelled FCESS payments to storage against AEMO's published market costs.

The backcast rebuilds FCESS revenue from published enablement schedules and published
clearing prices. AEMO separately publishes total FCESS **enablement** costs for the
whole market each quarter in its Quarterly Energy Dynamics reports, along with the share
of contingency and regulation volume that batteries took. Those two things should agree:
the modelled storage total should be close to the published market total multiplied by
the battery volume share.

Published figures are quoted below with their source and page so they can be re-checked.
Only enablement costs are used, never the totals that include Uplift, because modelled
revenue excludes Uplift by construction (``docs/assumptions.md`` E5, F6).
"""

from __future__ import annotations

import pandas as pd

from src.paths import OUTPUTS

#: Published FCESS enablement costs for the whole SWIS, and the battery share of
#: contingency and regulation volume. All from AEMO Quarterly Energy Dynamics.
#:
#: Q4 2025 and Q4 2024: QED Q4 2025, section 4.5.2 (p. 73) — "FCESS enablement costs
#: decreased to $7.2 million in Q4 2025, a reduction of $34.5 million (-83%) compared to
#: Q4 2024", which puts Q4 2024 at $41.7 million. Battery volume share from section 4.4
#: (p. 71) — "batteries increase their market share of the contingency and regulation
#: markets, reaching 84% of total volume ... an increase of 48 pp on Q4 2024", which puts
#: Q4 2024 at 36%.
PUBLISHED = pd.DataFrame(
    [
        {
            "quarter": "2024Q4",
            "market_enablement_cost_m": 41.7,
            "battery_volume_share": 0.36,
            "source": "AEMO QED Q4 2025, s4.5.2 p.73 and s4.4 p.71 (derived from the stated change)",
        },
        {
            "quarter": "2025Q4",
            "market_enablement_cost_m": 7.2,
            "battery_volume_share": 0.84,
            "source": "AEMO QED Q4 2025, s4.5.2 p.73 and s4.4 p.71",
        },
    ]
)

#: Tolerance on the reconciliation. The battery share published by AEMO is a share of
#: *volume*, not of cost, and the two differ whenever batteries are enabled in the
#: cheaper services, so a wide band is appropriate. A gap outside this band is reported
#: and explained in docs/results.md rather than closed by adjusting the model.
TOLERANCE = 0.40

#: Tolerance on the whole-market check. This one compares like with like, so it is tight.
MARKET_TOLERANCE = 0.15


def modelled_by_quarter(path=None) -> pd.DataFrame:
    """Modelled FCESS payments to storage by calendar quarter, in $ million."""
    path = path or OUTPUTS / "backcast_daily.parquet"
    daily = pd.read_parquet(path)
    daily["trading_date"] = pd.to_datetime(daily["trading_date"])
    daily["quarter"] = daily["trading_date"].dt.to_period("Q").astype(str)
    grouped = daily.groupby("quarter")[["fcess_revenue", "energy_revenue"]].sum() / 1e6
    return grouped.rename(
        columns={"fcess_revenue": "modelled_storage_fcess_m", "energy_revenue": "modelled_energy_m"}
    )


def modelled_market_cost_by_quarter() -> pd.Series:
    """Whole-market FCESS enablement cost per quarter, in $ million.

    Built from every facility's published enablement times the clearing price, so it
    depends on nothing about storage and can be compared directly with AEMO's published
    total. This is the primary check.
    """
    from src import fcess_market

    cost = fcess_market.load_market_cost().copy()
    cost.index = pd.to_datetime(cost.index)
    quarterly = cost["total_cost"].resample("QS").sum() / 1e6
    quarterly.index = quarterly.index.to_period("Q").astype(str)
    return quarterly


def reconcile(path=None) -> pd.DataFrame:
    """Two checks against AEMO's published figures, strongest first.

    **Market total.** Modelled whole-market enablement cost against AEMO's published
    quarterly figure. This depends on nothing about storage, so it tests whether the
    enablement and price series are being read correctly at all.

    **Storage share.** Modelled payments to storage as a share of the modelled market
    total, against AEMO's published battery share. AEMO publishes a share of *volume*,
    not of cost, and the two diverge whenever storage is concentrated in the cheaper
    services, so this check is weaker and is interpreted with the service mix in hand.
    """
    modelled = modelled_by_quarter(path)
    table = PUBLISHED.set_index("quarter").join(modelled, how="left")

    market = modelled_market_cost_by_quarter()
    table["modelled_market_cost_m"] = table.index.map(market)
    table["market_ratio"] = table["modelled_market_cost_m"] / table["market_enablement_cost_m"]
    table["market_within_tolerance"] = (table["market_ratio"] - 1.0).abs() <= MARKET_TOLERANCE

    table["modelled_storage_cost_share"] = (
        table["modelled_storage_fcess_m"] / table["modelled_market_cost_m"]
    )
    table["implied_storage_cost_m"] = (
        table["market_enablement_cost_m"] * table["battery_volume_share"]
    )
    table["difference_m"] = table["modelled_storage_fcess_m"] - table["implied_storage_cost_m"]
    table["ratio"] = table["modelled_storage_fcess_m"] / table["implied_storage_cost_m"]
    table["within_tolerance"] = (table["ratio"] - 1.0).abs() <= TOLERANCE
    return table


def service_mix(quarter: str) -> pd.DataFrame:
    """Storage share of enablement in each FCESS service for one quarter.

    The published battery share is a share of *volume across all services combined*.
    Using it to imply a share of *cost* only works if storage is spread across the
    services in the same proportion as the cost is. This table shows whether it is, and
    is the first place to look when the reconciliation ratio is away from one.
    """
    from src import data, fcess_market
    from src.model import SERVICES

    fleet = fcess_market.load().copy()
    fleet["quarter"] = fleet["trading_date"].dt.to_period("Q").astype(str)
    fleet = fleet.loc[fleet["quarter"] == quarter]

    requirements = data.load_market_requirements().copy()
    requirements["quarter"] = requirements["trading_date"].dt.to_period("Q").astype(str)
    requirements = requirements.loc[requirements["quarter"] == quarter]

    prices = data.load_prices().copy()
    prices["quarter"] = prices["trading_date"].dt.to_period("Q").astype(str)
    prices = prices.loc[prices["quarter"] == quarter]

    rows = []
    for service in SERVICES:
        enabled = fleet[f"{service}_enabled_mw"].mean()
        required = requirements[f"{service}_requirement"].mean()
        price = prices[f"{service}_price"].mean()
        rows.append(
            {
                "service": service,
                "storage enabled MW": enabled,
                "market requirement MW": required,
                "storage share of requirement": enabled / required if required else float("nan"),
                "mean clearing price": price,
                "market cost weight": required * price,
            }
        )

    frame = pd.DataFrame(rows).set_index("service")
    total_weight = frame["market cost weight"].sum()
    frame["share of market cost"] = frame["market cost weight"] / total_weight
    return frame


def report(path=None) -> str:
    """A readable reconciliation summary."""
    table = reconcile(path)
    lines = ["FCESS reconciliation against AEMO Quarterly Energy Dynamics", ""]
    for quarter, row in table.iterrows():
        lines.append(f"{quarter}")
        lines.append("  check 1, whole market (depends on nothing about storage)")
        lines.append(f"    published enablement cost             ${row['market_enablement_cost_m']:6.1f}m")
        lines.append(f"    modelled enablement cost              ${row['modelled_market_cost_m']:6.1f}m")
        lines.append(f"    ratio                                 {row['market_ratio']:6.2f}")
        lines.append(
            f"    within the {MARKET_TOLERANCE:.0%} tolerance                "
            f"{'yes' if row['market_within_tolerance'] else 'NO - explain in docs/results.md'}"
        )
        lines.append("  check 2, storage share (volume share used as a cost proxy)")
        lines.append(f"    published battery share of volume     {row['battery_volume_share']:6.0%}")
        lines.append(f"    modelled storage share of cost        {row['modelled_storage_cost_share']:6.0%}")
        lines.append(f"    modelled payment to storage           ${row['modelled_storage_fcess_m']:6.1f}m")
        lines.append(f"    implied by the volume share           ${row['implied_storage_cost_m']:6.1f}m")
        lines.append(f"    ratio                                 {row['ratio']:6.2f}")
        lines.append(
            f"    within the {TOLERANCE:.0%} tolerance                "
            f"{'yes' if row['within_tolerance'] else 'NO - see the service mix below'}"
        )
        lines.append(f"  source: {row['source']}")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    pd.set_option("display.width", 220)
    print(report())
    print("\nmodelled storage FCESS by quarter, $ million")
    print(modelled_by_quarter().round(2).to_string())

    for quarter in PUBLISHED["quarter"]:
        print(f"\nservice mix, {quarter}")
        print(service_mix(quarter).round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
