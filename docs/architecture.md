# IntelliWarm Architecture

## Working Architecture

The current repo should be treated as an incremental architecture, not a greenfield rewrite.

## Core Product Concept

IntelliWarm's primary value is **zone-level hybrid heating cost minimization**. For each zone at each cycle the system decides:

> Run the gas furnace for the whole zone, or run individual electric space heaters per room?

This decision is made by `HybridController` using live energy prices and per-room heat needs:

```
electric_cost  = Σ (heater_kW × room_demand × electricity_$/kWh)  per room needing heat
furnace_cost   = zone_demand × zone.hourly_gas_cost(gas_$/therm)

if zone.has_furnace and furnace_cost < electric_cost → gas furnace (zone-wide)
else                                                 → electric heaters (per room)
```

Gas furnace = zone-level actuator. Cannot selectively heat individual rooms. Cheaper when multiple rooms need heat.
Electric heaters = room-level actuators. Independent per-room control. Cheaper when only one or two rooms need heat.

## Main Boundaries

### Application Layer

- `app.py`: Flask entrypoint
- `intelliwarm/routes/`: thin Flask route modules split by dashboard and demo concerns
- `intelliwarm/services/runtime.py`: runtime orchestration, zone-aware hybrid optimization dispatch, dashboard view-model assembly, and room state management
- `intelliwarm/services/forecast_bundle.py`: aligned occupancy, weather, and pricing forecasts for controllers

### Domain And Simulation Layer

- `intelliwarm/data/models.py`: shared domain contracts:
	- `HeatSourceType` — `ELECTRIC` or `GAS_FURNACE`
	- `ZoneConfig` — zone identity, furnace specs (`btu_per_hour`, `efficiency` AFUE), `hourly_gas_cost()`
	- `HybridHeatingDecision` — zone-level decision with full cost breakdown, per-room normalized-demand map, rationale
	- `RoomConfig` — room identity, thermal parameters, `heat_source` field
	- `HeatingAction` — legacy label bands used for compatibility and reporting around continuous demand
	- `SimulationState`, `ControlDecision`, `ForecastBundle`, `ForecastStep`
- `intelliwarm/models/thermal_model.py`: room thermal dynamics with `step()` and `simulate()`
- `intelliwarm/models/simulator.py`: deterministic `HouseSimulator`
- `intelliwarm/prediction/occupancy_model.py`: schedule and timestamp-based occupancy prediction
- `intelliwarm/learning/gym_env.py`: Gym-compatible deterministic room and multi-room environments for future ML policy training
- `intelliwarm/learning/scenario_generator.py`: deterministic multi-room, multi-zone scenario library for training and evaluation
- `intelliwarm/learning/evaluation.py`: deterministic policy rollout and metric aggregation across the scenario library
- `intelliwarm/learning/policy_catalog.py`: named heuristic baselines for consistent evaluation runs
- `scripts/evaluate_policies.py`: thin CLI wrapper for scenario-batch policy comparisons

### Control Layer

- **`intelliwarm/control/hybrid_controller.py`: zone-level hybrid cost decision engine — primary actuator decision point for all zone heating** (see interface below)
- `intelliwarm/control/intent_resolver.py`: shared deterministic translation from room intents / zone source modes into actual normalized heat demand
- `intelliwarm/control/baseline_controller.py`: per-room explainable rule-based controller; infers room intents from occupancy and temperature, then resolves them through the shared intent logic before `HybridController` compares sources
- `intelliwarm/optimizer/mpc_controller.py`: continuous MPC (action contracts should converge with the shared normalized-demand runtime contract)

#### HybridController Interface

```python
class HybridController:
		def __init__(self, zone_config: ZoneConfig, room_configs: Dict[str, RoomConfig], ...)

		def decide(
				self,
				room_temperatures: Dict[str, float],
				occupancy_forecasts: Dict[str, Sequence[float]],
				electricity_price: float,   # $/kWh — must be live in production
				gas_price: float,           # $/therm — must be live in production
				outside_temp: float,
				room_intents: Optional[Dict[str, object]] = None,
				zone_source_preference: Optional[object] = None,
		) -> HybridHeatingDecision
```

`HybridHeatingDecision` carries: `furnace_on`, `heat_source`, `per_room_actions`, `per_room_action_labels`, `electric_hourly_cost`, `furnace_hourly_cost`, `chosen_hourly_cost`, `rationale`.

For learned control, PPO should choose high-level room intents and optional zone source modes, while the shared intent resolver computes the actual actuator command. This keeps learned policies aligned with runtime and hardware semantics.

### Integration Layer

- `intelliwarm/sensors/sensor_manager.py`: sensor backend boundary with hardware-ready fallback adapters
- `intelliwarm/control/device_controller.py`: HVAC actuation backend boundary with simulated fallback for both per-room electric heaters and zone furnaces
- `intelliwarm/pricing/energy_price_fetcher.py`: provider-based pricing service with offline-safe time-of-use fallback and a boundary for future live vendor integration

### Persistence Layer

- `intelliwarm/storage/database.py`: SQLite storage for rooms, logs, optimization runs, and reporting queries
- `intelliwarm/services/reporting.py`: report assembly on top of the current SQLite layer

## Current Architectural Rules

- extend the current package layout in place
- keep route handlers thin
- keep simulation deterministic
- push orchestration into services
- **route all zone heating decisions through `HybridController` — do not call `BaselineController` directly from services**
- share `HybridHeatingDecision` and `ForecastBundle` contracts across runtime, dashboard, and reporting
- keep integrations optional and offline-safe
- energy prices passed to `HybridController` must come from the live provider in production; use config static values only as fallback

## Hardware Actuation Contracts

When `HybridHeatingDecision.furnace_on is True`:
- `DeviceController` must emit a zone-level furnace relay or thermostat signal
- Per-room electric heaters in the same zone must be deactivated (set to `0.0` demand) to prevent double-heating costs

When `HybridHeatingDecision.furnace_on is False`:
- `DeviceController` emits per-room electric heater commands from `per_room_actions` as normalized demand levels
- Furnace relay (if present) must be set to OFF

These actuation semantics must be honored in both simulation and live hardware modes.

## Immediate Architecture Targets

1. Dashboard should continue expanding from runtime cards into deeper operator tooling for cost, comfort, and controller comparisons.
2. Live energy price provider must supply both gas and electricity prices for accurate decisions in production.
3. Build operator and CLI comparison workflows on top of the implemented learned-policy evaluation utilities and training environments.
4. Converge MPC, hybrid, and future learned policies on the same runtime action/reporting contracts.
5. Enrich hardware adapters with vendor-specific telemetry while preserving the current offline-safe fallback behavior.
