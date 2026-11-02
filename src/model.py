"""Co-optimised energy and FCESS battery dispatch, solved as an LP with HiGHS.

The model dispatches at five-minute resolution because the physical and FCESS
constraints bind at five minutes, but settles energy on the thirty-minute Reference
Trading Price, because that is what the Rules settle on (``docs/assumptions.md`` E1,
ESM Rules 9.9.4). FCESS is paid per five-minute Dispatch Interval on the enablement
quantity at that interval's clearing price (F2, ESM Rules 9.10.6).

Sign and unit conventions
-------------------------
* power in MW, energy in MWh, prices in $/MWh, ``dt`` in hours (five minutes = 1/12)
* ``p_ch`` and ``p_dis`` are both non-negative; net export is ``p_dis - p_ch``
* state of charge is in MWh of stored energy, bounded by fractions of usable energy
* charge and discharge efficiencies are separate:
  ``soc[t+1] = soc[t] + eta_ch * p_ch[t] * dt - p_dis[t] * dt / eta_dis``

Objective (maximised)
---------------------
``sum_t  rtp[t] * (p_dis[t] - p_ch[t]) * dt * loss_factor``
``+ sum_t sum_s  fcess_price[s,t] * enable[s,t] * dt * performance_factor``
``- sum_t  c_thru * (p_ch[t] + p_dis[t]) * dt / 2``

where ``rtp[t]`` is the Reference Trading Price of the Trading Interval containing
dispatch interval ``t``. Because that price is constant across the six dispatch
intervals in a Trading Interval, this is identical to settling the net Trading Interval
quantity at the Reference Trading Price, and needs no extra variables.

Why HiGHS directly rather than Pyomo or cvxpy: see ``docs/assumptions.md`` M1.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from src.paths import DT_HOURS

#: The five Frequency Co-optimised Essential System Services, in a fixed order used for
#: every array in this module. ESM Rules 9.10.3 (docs/assumptions.md F1).
SERVICES: tuple[str, ...] = (
    "contingency_raise",
    "contingency_lower",
    "regulation_raise",
    "regulation_lower",
    "rocof",
)

#: Which services consume raise headroom (the battery must be able to inject more) and
#: which consume lower headroom (it must be able to withdraw more). RoCoF is treated as
#: a raise-direction service; recorded as an assumption in docs/assumptions.md.
RAISE_SERVICES: tuple[str, ...] = ("contingency_raise", "regulation_raise", "rocof")
LOWER_SERVICES: tuple[str, ...] = ("contingency_lower", "regulation_lower")

RAISE_IDX = tuple(SERVICES.index(s) for s in RAISE_SERVICES)
LOWER_IDX = tuple(SERVICES.index(s) for s in LOWER_SERVICES)

N_SERVICES = len(SERVICES)


class InfeasibleDispatch(RuntimeError):
    """The dispatch LP had no feasible solution.

    In a rolling-horizon run this is reachable without a modelling error: a battery can
    carry forward a state of charge too low to meet an obligation window that opens
    inside the horizon. The strategy catches it, records the breach and relaxes the
    obligation for that solve rather than silently failing.
    """


@dataclass(frozen=True)
class BatteryParams:
    """Every parameter of the battery and its market treatment. No magic numbers."""

    p_max_mw: float = 100.0
    duration_h: float = 2.0

    #: Round-trip efficiency, split evenly between charging and discharging unless
    #: eta_ch and eta_dis are given explicitly.
    round_trip_efficiency: float = 0.88
    eta_ch: float | None = None
    eta_dis: float | None = None

    #: State of charge limits as fractions of usable energy.
    soc_min_frac: float = 0.0
    soc_max_frac: float = 1.0
    soc_initial_frac: float = 0.5

    #: Require the final state of charge to equal the initial one, so the optimiser
    #: cannot earn revenue by draining a battery it never refills.
    cyclic_soc: bool = True

    #: Degradation charged inside the optimisation, in $ per MWh of throughput. Charged
    #: on half of total throughput so one full cycle is charged once.
    c_thru: float = 0.0

    #: Loss factor applied to settled energy quantities (ESM Rules 9.5.2).
    loss_factor: float = 1.0

    #: FCESS performance factor (ESM Rules 9.10.6). 1.0 unless a sensitivity varies it.
    performance_factor: float = 1.0

    #: Maximum accredited enablement per service in MW. ``None`` means the power rating.
    accredited_mw: dict[str, float] = field(default_factory=dict)

    #: Hours of energy that enabled raise and lower quantities must be deliverable for.
    fcess_raise_duration_h: float = 0.25
    fcess_lower_duration_h: float = 0.25

    #: Services the battery is allowed to offer at all.
    enabled_services: tuple[str, ...] = SERVICES

    #: Capacity Credits. When held, the ESROI availability constraint applies and the
    #: accredited MW is derated by duration (ESM Rules 4.11.3, docs/assumptions.md C1).
    has_capacity_credits: bool = False
    esr_duration_requirement_h: float = 4.0

    def __post_init__(self) -> None:
        if self.p_max_mw <= 0 or self.duration_h <= 0:
            raise ValueError("p_max_mw and duration_h must be positive")
        if not 0 < self.round_trip_efficiency <= 1:
            raise ValueError("round_trip_efficiency must be in (0, 1]")
        if not 0 <= self.soc_min_frac <= self.soc_initial_frac <= self.soc_max_frac <= 1:
            raise ValueError("state of charge fractions must satisfy min <= initial <= max <= 1")
        unknown = set(self.accredited_mw) - set(SERVICES)
        if unknown:
            raise ValueError(f"unknown service(s) in accredited_mw: {sorted(unknown)}")

    # -- derived quantities ------------------------------------------------------------

    @property
    def energy_mwh(self) -> float:
        """Usable energy capacity."""
        return self.p_max_mw * self.duration_h

    @property
    def efficiency_ch(self) -> float:
        return self.eta_ch if self.eta_ch is not None else float(np.sqrt(self.round_trip_efficiency))

    @property
    def efficiency_dis(self) -> float:
        return self.eta_dis if self.eta_dis is not None else float(np.sqrt(self.round_trip_efficiency))

    @property
    def soc_min_mwh(self) -> float:
        return self.soc_min_frac * self.energy_mwh

    @property
    def soc_max_mwh(self) -> float:
        return self.soc_max_frac * self.energy_mwh

    @property
    def soc_initial_mwh(self) -> float:
        return self.soc_initial_frac * self.energy_mwh

    @property
    def accredited_capacity_mw(self) -> float:
        """Capacity Credits a battery of this duration can be accredited for.

        ESM Rules 4.11.3 assigns the capacity the resource can **sustain over** the ESR
        Obligation Duration (docs/assumptions.md C1, C2). That is a sent-out capacity,
        so the deliverable energy is the usable state-of-charge range net of discharge
        losses, and the accredited capacity is that energy spread over the obligation
        duration, capped by the power rating.

        Deriving it this way rather than as a flat ``duration / requirement`` ratio
        matters: it guarantees the obligation constraint is feasible, because the energy
        the battery must hold at the start of the window is then at most its usable
        energy. A flat ratio demands ``energy / eta_dis``, which no battery can store.
        """
        if not self.has_capacity_credits:
            return 0.0
        deliverable_mwh = (self.soc_max_mwh - self.soc_min_mwh) * self.efficiency_dis
        sustainable_mw = deliverable_mwh / self.esr_duration_requirement_h
        return min(self.p_max_mw, sustainable_mw)

    def accredited_for(self, service: str) -> float:
        """Maximum enablement for ``service`` in MW."""
        if service not in self.enabled_services:
            return 0.0
        return float(self.accredited_mw.get(service, self.p_max_mw))


@dataclass
class DispatchResult:
    """Solved dispatch and its revenue decomposition."""

    status: str
    objective: float
    p_ch: np.ndarray
    p_dis: np.ndarray
    enablement: dict[str, np.ndarray]
    soc: np.ndarray
    dt: float
    params: BatteryParams

    energy_revenue: float = 0.0
    fcess_revenue: dict[str, float] = field(default_factory=dict)
    throughput_cost: float = 0.0

    @property
    def net_export_mwh(self) -> np.ndarray:
        return (self.p_dis - self.p_ch) * self.dt

    @property
    def discharge_mwh(self) -> float:
        return float(self.p_dis.sum() * self.dt)

    @property
    def charge_mwh(self) -> float:
        return float(self.p_ch.sum() * self.dt)

    @property
    def throughput_mwh(self) -> float:
        return self.discharge_mwh + self.charge_mwh

    @property
    def equivalent_full_cycles(self) -> float:
        """Throughput expressed in full cycles of usable energy."""
        return self.throughput_mwh / (2.0 * self.params.energy_mwh)

    @property
    def total_fcess_revenue(self) -> float:
        return float(sum(self.fcess_revenue.values()))

    @property
    def total_revenue(self) -> float:
        return self.energy_revenue + self.total_fcess_revenue - self.throughput_cost

    def energy_balance_error(self) -> float:
        """Largest absolute violation of the state-of-charge recursion, in MWh."""
        p = self.params
        expected = self.soc[:-1] + p.efficiency_ch * self.p_ch * self.dt - self.p_dis * self.dt / p.efficiency_dis
        return float(np.abs(self.soc[1:] - expected).max())

    def summary(self) -> dict:
        row = {
            "status": self.status,
            "total_revenue": self.total_revenue,
            "energy_revenue": self.energy_revenue,
            "fcess_revenue": self.total_fcess_revenue,
            "throughput_cost": self.throughput_cost,
            "discharge_mwh": self.discharge_mwh,
            "charge_mwh": self.charge_mwh,
            "cycles": self.equivalent_full_cycles,
        }
        row.update({f"fcess_{k}": v for k, v in self.fcess_revenue.items()})
        return row


# --------------------------------------------------------------------------------------
# LP construction
# --------------------------------------------------------------------------------------


class DispatchLP:
    """A built LP for one dispatch window, re-solvable with new prices.

    Building the constraint matrix is the expensive part and depends only on the window
    length and the battery, not on prices. The rolling-horizon strategy therefore builds
    one of these per window length and calls :meth:`solve` repeatedly with new price
    vectors and a new opening state of charge, which lets HiGHS warm start.
    """

    def __init__(
        self,
        n_intervals: int,
        params: BatteryParams,
        *,
        dt: float = DT_HOURS,
        esroi_mask: np.ndarray | None = None,
        fcess_cap: np.ndarray | None = None,
    ) -> None:
        """Build the LP.

        Parameters
        ----------
        fcess_cap
            Optional ``(5, T)`` per-interval ceiling on enablement, in :data:`SERVICES`
            order, applied on top of the accredited maxima. This is how the market
            requirement enters the model: a battery cannot be paid for holding more
            reserve than the market procured, and a single entrant wins only a share of
            what was procured. Without it the optimiser enables to its accreditation in
            every interval, which for a 100 MW battery would be a quarter of the whole
            SWIS contingency raise requirement held continuously.
        """
        import highspy

        if n_intervals < 1:
            raise ValueError("n_intervals must be at least 1")

        self.T = int(n_intervals)
        self.params = params
        self.dt = float(dt)
        self.esroi_mask = (
            np.zeros(self.T, dtype=bool) if esroi_mask is None else np.asarray(esroi_mask, dtype=bool)
        )
        if self.esroi_mask.shape != (self.T,):
            raise ValueError(f"esroi_mask must have shape ({self.T},)")

        if fcess_cap is not None:
            fcess_cap = np.asarray(fcess_cap, dtype=float)
            if fcess_cap.shape != (N_SERVICES, self.T):
                raise ValueError(f"fcess_cap must have shape ({N_SERVICES}, {self.T})")
            if (fcess_cap < -1e-9).any():
                raise ValueError("fcess_cap must be non-negative")
        self.fcess_cap = fcess_cap

        #: Obligation state-of-charge floor, kept separately from the live column bounds
        #: because those also carry the pinned opening and closing values.
        self._soc_floor = np.full(self.T + 1, params.soc_min_mwh)

        self._highspy = highspy
        self.h = highspy.Highs()
        self.h.setOptionValue("output_flag", False)
        self.h.setOptionValue("solver", "simplex")
        self.h.changeObjectiveSense(highspy.ObjSense.kMaximize)

        self._build()

    # -- variable index helpers --------------------------------------------------------

    @property
    def n_vars(self) -> int:
        return (2 + N_SERVICES) * self.T + (self.T + 1)

    def i_ch(self, t: int) -> int:
        return t

    def i_dis(self, t: int) -> int:
        return self.T + t

    def i_enable(self, service_index: int, t: int) -> int:
        return (2 + service_index) * self.T + t

    def i_soc(self, t: int) -> int:
        return (2 + N_SERVICES) * self.T + t

    # -- construction ------------------------------------------------------------------

    def _column_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        p = self.params
        lower = np.zeros(self.n_vars)
        upper = np.zeros(self.n_vars)

        upper[self.i_ch(0) : self.i_ch(0) + self.T] = p.p_max_mw
        upper[self.i_dis(0) : self.i_dis(0) + self.T] = p.p_max_mw

        for s, service in enumerate(SERVICES):
            start = self.i_enable(s, 0)
            ceiling = np.full(self.T, p.accredited_for(service))
            if self.fcess_cap is not None:
                ceiling = np.minimum(ceiling, self.fcess_cap[s])
            upper[start : start + self.T] = ceiling

        soc_start = self.i_soc(0)
        lower[soc_start : soc_start + self.T + 1] = p.soc_min_mwh
        upper[soc_start : soc_start + self.T + 1] = p.soc_max_mwh

        # Opening state of charge is fixed; the closing one too when cyclic.
        lower[self.i_soc(0)] = upper[self.i_soc(0)] = p.soc_initial_mwh
        if p.cyclic_soc:
            lower[self.i_soc(self.T)] = upper[self.i_soc(self.T)] = p.soc_initial_mwh

        # The ESR obligation is imposed entirely through column bounds rather than extra
        # rows, so a rolling-horizon run can move the obligation window between solves
        # with two bound updates instead of rebuilding the model.
        self._apply_obligation(lower, upper)

        return lower, upper

    def _obligation_soc_floor(self) -> np.ndarray:
        """Lower bound on stored energy at each interval, given the obligation window.

        A battery holding Capacity Credits must be able to discharge at its credited
        capacity across the whole window (docs/assumptions.md C4), so when the window
        opens it must already hold that much energy. The requirement is capped at the
        usable range, so a published window longer than the obligation duration cannot
        make the LP infeasible; in that case the strongest obligation the battery can
        physically meet is to be full when the window opens.
        """
        p = self.params
        floor = np.full(self.T + 1, p.soc_min_mwh)
        if not (p.has_capacity_credits and self.esroi_mask.any()):
            self._soc_floor = floor
            return floor

        usable = p.soc_max_mwh - p.soc_min_mwh
        for start, length in _mask_runs(self.esroi_mask):
            required = min(
                p.accredited_capacity_mw * length * self.dt / p.efficiency_dis, usable
            )
            floor[start] = p.soc_min_mwh + required
        self._soc_floor = floor
        return floor

    def _apply_obligation(self, lower: np.ndarray, upper: np.ndarray) -> None:
        """Write the obligation into the charge upper bounds and state-of-charge floors."""
        p = self.params
        if not p.has_capacity_credits:
            return

        # No charging inside the window: the battery must be available to inject.
        charge = upper[self.i_ch(0) : self.i_ch(0) + self.T]
        charge[self.esroi_mask] = 0.0

        soc_start = self.i_soc(0)
        lower[soc_start : soc_start + self.T + 1] = self._obligation_soc_floor()

        # The pinned opening and closing states must still lie inside the floor.
        opening = max(p.soc_initial_mwh, lower[self.i_soc(0)])
        lower[self.i_soc(0)] = upper[self.i_soc(0)] = opening
        if p.cyclic_soc:
            closing = max(p.soc_initial_mwh, lower[self.i_soc(self.T)])
            lower[self.i_soc(self.T)] = upper[self.i_soc(self.T)] = closing

    def _rows(self) -> tuple[list[float], list[float], list[list[tuple[int, float]]]]:
        """Constraint rows as ``(lower, upper, [(column, coefficient), ...])``."""
        p = self.params
        inf = self._highspy.kHighsInf
        lower: list[float] = []
        upper: list[float] = []
        rows: list[list[tuple[int, float]]] = []

        def add(lo: float, hi: float, terms: list[tuple[int, float]]) -> None:
            lower.append(lo)
            upper.append(hi)
            rows.append(terms)

        eta_ch, eta_dis = p.efficiency_ch, p.efficiency_dis

        for t in range(self.T):
            # State of charge recursion, as an equality.
            add(
                0.0,
                0.0,
                [
                    (self.i_soc(t + 1), 1.0),
                    (self.i_soc(t), -1.0),
                    (self.i_ch(t), -eta_ch * self.dt),
                    (self.i_dis(t), self.dt / eta_dis),
                ],
            )

            # Raise headroom: net injection plus enabled raise cannot exceed the rating.
            raise_terms = [(self.i_ch(t), -1.0), (self.i_dis(t), 1.0)]
            raise_terms += [(self.i_enable(s, t), 1.0) for s in RAISE_IDX]
            add(-inf, p.p_max_mw, raise_terms)

            # Lower headroom: net withdrawal plus enabled lower cannot exceed the rating.
            lower_terms = [(self.i_ch(t), 1.0), (self.i_dis(t), -1.0)]
            lower_terms += [(self.i_enable(s, t), 1.0) for s in LOWER_IDX]
            add(-inf, p.p_max_mw, lower_terms)

            # Energy headroom for enabled raise: the stored energy above the floor must
            # cover delivering the enabled raise quantity for its required duration.
            if p.fcess_raise_duration_h > 0:
                coeff = p.fcess_raise_duration_h / eta_dis
                terms = [(self.i_enable(s, t), coeff) for s in RAISE_IDX]
                terms.append((self.i_soc(t), -1.0))
                add(-inf, -p.soc_min_mwh, terms)

            # Energy headroom for enabled lower: the space below the ceiling must cover
            # absorbing the enabled lower quantity for its required duration.
            if p.fcess_lower_duration_h > 0:
                coeff = p.fcess_lower_duration_h * eta_ch
                terms = [(self.i_enable(s, t), coeff) for s in LOWER_IDX]
                terms.append((self.i_soc(t), 1.0))
                add(-inf, p.soc_max_mwh, terms)

        return lower, upper, rows

    def _build(self) -> None:
        col_lower, col_upper = self._column_bounds()
        self.h.addVars(self.n_vars, col_lower, col_upper)

        row_lower, row_upper, rows = self._rows()
        starts: list[int] = []
        indices: list[int] = []
        values: list[float] = []
        for terms in rows:
            starts.append(len(indices))
            for column, coefficient in terms:
                indices.append(column)
                values.append(coefficient)

        self.h.addRows(
            len(rows),
            np.array(row_lower, dtype=np.double),
            np.array(row_upper, dtype=np.double),
            len(values),
            np.array(starts, dtype=np.int32),
            np.array(indices, dtype=np.int32),
            np.array(values, dtype=np.double),
        )
        self._all_columns = np.arange(self.n_vars, dtype=np.int32)
        self._charge_columns = np.arange(self.i_ch(0), self.i_ch(0) + self.T, dtype=np.int32)
        self._soc_columns = np.arange(self.i_soc(0), self.i_soc(0) + self.T + 1, dtype=np.int32)
        self._enable_columns = np.arange(
            self.i_enable(0, 0), self.i_enable(N_SERVICES - 1, 0) + self.T, dtype=np.int32
        )
        self._col_lower = col_lower
        self._col_upper = col_upper

    # -- moving the window between solves ----------------------------------------------

    def set_fcess_cap(self, fcess_cap: np.ndarray | None) -> None:
        """Replace the per-interval enablement ceiling in place.

        One bulk bound update rather than a rebuild, so a rolling-horizon run can step
        the window forward cheaply.
        """
        p = self.params
        ceilings = []
        for s, service in enumerate(SERVICES):
            ceiling = np.full(self.T, p.accredited_for(service))
            if fcess_cap is not None:
                ceiling = np.minimum(ceiling, np.asarray(fcess_cap, dtype=float)[s])
            ceilings.append(ceiling)

        upper = np.concatenate(ceilings)
        lower = np.zeros_like(upper)
        self.fcess_cap = fcess_cap
        self._col_upper[self._enable_columns] = upper
        self.h.changeColsBounds(len(self._enable_columns), self._enable_columns, lower, upper)

    def set_esroi_mask(self, esroi_mask: np.ndarray | None) -> None:
        """Replace the obligation window in place."""
        p = self.params
        self.esroi_mask = (
            np.zeros(self.T, dtype=bool) if esroi_mask is None else np.asarray(esroi_mask, dtype=bool)
        )
        if self.esroi_mask.shape != (self.T,):
            raise ValueError(f"esroi_mask must have shape ({self.T},)")

        charge_upper = np.full(self.T, p.p_max_mw)
        soc_lower = np.full(self.T + 1, p.soc_min_mwh)
        soc_upper = np.full(self.T + 1, p.soc_max_mwh)
        if p.has_capacity_credits:
            charge_upper[self.esroi_mask] = 0.0
            soc_lower = self._obligation_soc_floor()

        self._col_upper[self._charge_columns] = charge_upper
        self.h.changeColsBounds(
            len(self._charge_columns), self._charge_columns, np.zeros(self.T), charge_upper
        )
        self._col_lower[self._soc_columns] = soc_lower
        self._col_upper[self._soc_columns] = soc_upper
        self.h.changeColsBounds(
            len(self._soc_columns), self._soc_columns, soc_lower, soc_upper
        )

    def soc_floor_at(self, index: int) -> float:
        """Obligation state-of-charge floor at an interval.

        Read from the obligation floor rather than from the live column bounds, which
        also carry the pinned opening and closing values.
        """
        return float(self._soc_floor[index])

    # -- solving -----------------------------------------------------------------------

    def _costs(self, energy_price: np.ndarray, fcess_prices: np.ndarray) -> np.ndarray:
        p = self.params
        costs = np.zeros(self.n_vars)

        energy_coeff = energy_price * self.dt * p.loss_factor
        costs[self.i_dis(0) : self.i_dis(0) + self.T] = energy_coeff - p.c_thru * self.dt / 2.0
        costs[self.i_ch(0) : self.i_ch(0) + self.T] = -energy_coeff - p.c_thru * self.dt / 2.0

        for s in range(N_SERVICES):
            start = self.i_enable(s, 0)
            costs[start : start + self.T] = fcess_prices[s] * self.dt * p.performance_factor

        return costs

    def solve(
        self,
        energy_price: np.ndarray,
        fcess_prices: np.ndarray | None = None,
        *,
        soc_initial_mwh: float | None = None,
        cyclic: bool | None = None,
    ) -> DispatchResult:
        """Solve for the given prices.

        Parameters
        ----------
        energy_price
            Settlement price per dispatch interval, length ``T``. For the Rules-correct
            case this is the Reference Trading Price of the containing Trading Interval,
            repeated across its six dispatch intervals.
        fcess_prices
            Shape ``(5, T)`` in :data:`SERVICES` order. Zeros if omitted.
        soc_initial_mwh
            Overrides the opening state of charge, for rolling-horizon re-solves.
        cyclic
            Overrides whether the closing state of charge is pinned to the opening one.
        """
        energy_price = np.asarray(energy_price, dtype=np.double)
        if energy_price.shape != (self.T,):
            raise ValueError(f"energy_price must have shape ({self.T},), got {energy_price.shape}")

        if fcess_prices is None:
            fcess_prices = np.zeros((N_SERVICES, self.T))
        fcess_prices = np.asarray(fcess_prices, dtype=np.double)
        if fcess_prices.shape != (N_SERVICES, self.T):
            raise ValueError(f"fcess_prices must have shape ({N_SERVICES}, {self.T})")

        self.h.changeColsCost(self.n_vars, self._all_columns, self._costs(energy_price, fcess_prices))

        # The opening state of charge is pinned to its actual value, overriding any
        # obligation floor at that index: the battery is where it is.
        opening = self.params.soc_initial_mwh if soc_initial_mwh is None else soc_initial_mwh
        if soc_initial_mwh is None:
            opening = max(opening, self.soc_floor_at(0))
        self.h.changeColBounds(self.i_soc(0), opening, opening)

        pin_end = self.params.cyclic_soc if cyclic is None else cyclic
        if pin_end:
            closing = max(opening, self.soc_floor_at(self.T))
            self.h.changeColBounds(self.i_soc(self.T), closing, closing)
        else:
            self.h.changeColBounds(
                self.i_soc(self.T), self.soc_floor_at(self.T), self.params.soc_max_mwh
            )

        self.h.run()
        status = self.h.modelStatusToString(self.h.getModelStatus())
        if self.h.getModelStatus() != self._highspy.HighsModelStatus.kOptimal:
            raise InfeasibleDispatch(f"dispatch LP did not solve to optimality: {status}")

        x = np.array(self.h.getSolution().col_value, dtype=np.double)
        return self._result(x, status, energy_price, fcess_prices)

    def _result(
        self, x: np.ndarray, status: str, energy_price: np.ndarray, fcess_prices: np.ndarray
    ) -> DispatchResult:
        p = self.params
        p_ch = x[self.i_ch(0) : self.i_ch(0) + self.T]
        p_dis = x[self.i_dis(0) : self.i_dis(0) + self.T]
        soc = x[self.i_soc(0) : self.i_soc(0) + self.T + 1]

        enablement = {}
        fcess_revenue = {}
        for s, service in enumerate(SERVICES):
            start = self.i_enable(s, 0)
            quantity = x[start : start + self.T]
            enablement[service] = quantity
            fcess_revenue[service] = float(
                (fcess_prices[s] * quantity).sum() * self.dt * p.performance_factor
            )

        energy_revenue = float((energy_price * (p_dis - p_ch)).sum() * self.dt * p.loss_factor)
        throughput_cost = float((p_ch + p_dis).sum() * self.dt / 2.0 * p.c_thru)

        return DispatchResult(
            status=status,
            objective=float(self.h.getObjectiveValue()),
            p_ch=p_ch,
            p_dis=p_dis,
            enablement=enablement,
            soc=soc,
            dt=self.dt,
            params=p,
            energy_revenue=energy_revenue,
            fcess_revenue=fcess_revenue,
            throughput_cost=throughput_cost,
        )


def _mask_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous ``True`` runs in ``mask`` as ``(start_index, length)`` pairs."""
    runs = []
    start = None
    for index, flag in enumerate(mask):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            runs.append((start, index - start))
            start = None
    if start is not None:
        runs.append((start, len(mask) - start))
    return runs


def solve_dispatch(
    energy_price: np.ndarray,
    params: BatteryParams,
    *,
    fcess_prices: np.ndarray | None = None,
    dt: float = DT_HOURS,
    esroi_mask: np.ndarray | None = None,
    fcess_cap: np.ndarray | None = None,
    soc_initial_mwh: float | None = None,
    cyclic: bool | None = None,
) -> DispatchResult:
    """Build and solve a dispatch LP in one call.

    Convenient for single windows and for the tests. The rolling-horizon strategy uses
    :class:`DispatchLP` directly so the model is built once and re-solved.
    """
    lp = DispatchLP(
        len(energy_price), params, dt=dt, esroi_mask=esroi_mask, fcess_cap=fcess_cap
    )
    return lp.solve(energy_price, fcess_prices, soc_initial_mwh=soc_initial_mwh, cyclic=cyclic)


def settle(
    p_ch: np.ndarray,
    p_dis: np.ndarray,
    enablement: dict[str, np.ndarray],
    energy_price: np.ndarray,
    fcess_prices: np.ndarray,
    params: BatteryParams,
    *,
    dt: float = DT_HOURS,
) -> dict:
    """Revenue of a given dispatch at given prices.

    Used to settle a forecast-driven dispatch on actual prices, which is the whole point
    of the realisable-versus-theoretical comparison: the strategy decides on forecasts
    but is paid on actuals.
    """
    energy_revenue = float((energy_price * (p_dis - p_ch)).sum() * dt * params.loss_factor)
    fcess_revenue = {
        service: float(
            (fcess_prices[index] * enablement.get(service, np.zeros_like(p_ch))).sum()
            * dt
            * params.performance_factor
        )
        for index, service in enumerate(SERVICES)
    }
    throughput_cost = float((p_ch + p_dis).sum() * dt / 2.0 * params.c_thru)

    return {
        "energy_revenue": energy_revenue,
        "fcess_revenue": fcess_revenue,
        "total_fcess_revenue": float(sum(fcess_revenue.values())),
        "throughput_cost": throughput_cost,
        "total_revenue": energy_revenue + float(sum(fcess_revenue.values())) - throughput_cost,
        "discharge_mwh": float(p_dis.sum() * dt),
        "charge_mwh": float(p_ch.sum() * dt),
    }


def with_duration(params: BatteryParams, duration_h: float) -> BatteryParams:
    """Copy of ``params`` with a different duration. Keeps the headline cases readable."""
    return replace(params, duration_h=duration_h)
