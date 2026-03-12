# IntelliWarm Roadmap

## Completed Or In Progress

### Foundation

- package layout established under `intelliwarm/`
- runtime orchestration extracted into `intelliwarm/services/runtime.py`
- focused runtime regression tests added

### Simulation Foundation

- typed room, occupancy, action, and simulation models added
- thermal `step()` and `simulate()` APIs added
- deterministic `HouseSimulator` added
- timestamp-aware occupancy prediction added
- simulation tests added

### Control Foundation

- explainable baseline controller added under `intelliwarm/control/baseline_controller.py`
- shared runtime support added for discrete `OFF`, `ECO`, `COMFORT`, and `PREHEAT` actions
- baseline controller exposed through runtime and optimization API selection

### Forecast Foundation

- aligned forecast bundle service added under `intelliwarm/services/forecast_bundle.py`
- occupancy, deterministic weather fallback, and energy pricing now share one horizon contract
- runtime optimization responses include the forecast bundle used for controller inputs

### Hardware Foundation

- sensor and device managers now delegate through hardware-ready backends with simulated fallback
- application bootstrap wires adapter boundaries without pushing hardware I/O into routes or optimizers
- focused tests cover hardware status reporting and offline-safe fallback behavior

### Route Foundation

- Flask routes are now split into thin modules under `intelliwarm/routes/`
- dashboard/form handlers remain thin and delegate room/config behavior to runtime services
- demo routes are isolated from dashboard routes while preserving existing endpoints

### Config Foundation

- `SystemConfig` now validates core comfort, runtime, device, weather, and room settings on load
- structured occupancy schedules are preserved through typed room config assembly
- the existing `weather_api` section is now represented by typed config state instead of being ignored

### Persistence Foundation

- SQLite optimization runs now persist controller type and action labels for reporting
- room and portfolio reports are assembled through `intelliwarm/services/reporting.py`
- reporting stays on top of the existing SQLite workflow without a storage migration framework

### Runtime Safety Foundation

- runtime safety overrides now guard against overheat and freeze-risk conditions before actuation
- runtime events are persisted for recent operational visibility and cycle summaries
- `IntelliWarmRuntime.get_runtime_status()` exposes mode, cycle summaries, and recent events for operator-facing monitoring

### Hybrid Heating Model (Cost Minimization Engine)

- `HeatSourceType` enum (`ELECTRIC`, `GAS_FURNACE`) added to domain models
- `ZoneConfig` dataclass with furnace specs and `hourly_gas_cost()` method
- `HybridHeatingDecision` frozen dataclass with full cost breakdown, per-room action map, and `to_dict()`
- `RoomConfig.heat_source` field; `SimulationState.heat_sources` dict
- `HybridController` implemented in `intelliwarm/control/hybrid_controller.py`
	- calls `BaselineController` per room to determine heat needs
	- computes electric vs. furnace hourly cost
	- selects the cheaper source; when furnace wins, all zone rooms get the max demanded action
	- returns `HybridHeatingDecision` with rationale for every decision
- `configs/config.yaml` updated: rooms have `heat_source`; `Residential` zone has furnace specs; `Work` zone has no furnace
- `intelliwarm/core/config.py` updated: `RoomSettings.heat_source`, `ZoneSettings` furnace fields
- 7 tests in `tests/test_hybrid_controller.py` covering all decision branches
- All exported from `intelliwarm/control/__init__` and `intelliwarm/data/__init__`

### Hybrid Runtime And Dashboard Integration

- `IntelliWarmRuntime.optimize_heating_plan()` now routes baseline/hybrid requests through `HybridController.decide()` per zone
- runtime assembles `ZoneConfig` objects from current config/runtime state and builds zone-level `HybridController` instances
- room-level runtime responses now include `hybrid_decision`, active heat source, requested mode, and applied mode metadata
- electric room commands are suppressed when the hybrid decision selects the zone furnace, preventing simultaneous electric+gas heating in the same zone
- dashboard/runtime view models expose zone and room heat-source choices, costs, rationale, and simulation/hardware source metadata
- Flask optimization endpoint now accepts `hybrid` explicitly and routes `baseline` requests through the hybrid decision path for compatibility

### Furnace Actuation Boundary

- `DeviceController` now models both per-room electric heaters and zone-level furnaces
- simulated and hardware-ready backends expose furnace registration, actuation, and status queries
- runtime now activates the zone furnace when hybrid gas heat wins and turns off room electric heaters in the same zone
- runtime turns the zone furnace off when hybrid or MPC paths choose electric/off room commands
- regression coverage verifies furnace status reporting, hardware fallback, and no simultaneous furnace+electric room heating

### Pricing Provider Boundary

- `EnergyPriceService` now delegates current and forecast pricing through explicit provider interfaces
- offline-safe `TimeOfUsePriceProvider` remains the default runtime behavior
- `StaticPriceProvider` supports deterministic tests and simple deployments
- `CallbackPriceProvider` creates a narrow seam for future live vendor integrations without changing runtime callers

### Gym-Compatible Learning Boundary

- `intelliwarm/learning/gym_env.py` adds a deterministic room-level environment with Gym-compatible `reset()` / `step()` APIs
- the environment reuses IntelliWarm's discrete `OFF`, `ECO`, `COMFORT`, and `PREHEAT` action space
- observations include current temperature, target band, occupancy probability, outdoor temperature, and current gas/electricity prices
- the environment keeps ML experimentation offline-safe by using the same simulator and pricing contracts as the rest of the platform
- regression coverage verifies deterministic rollouts and action-label mapping

### Multi-Room Training Environment And Scenario Library

- `IntelliWarmMultiRoomEnv` adds a padded `MultiDiscrete` action space spanning zone heat-source choices plus per-room heating modes
- the environment reuses the deterministic `HouseSimulator` and applies gas-furnace vs. electric semantics at zone scope
- `SyntheticScenarioGenerator` builds deterministic multi-room, multi-zone scenarios with varied schedules, weather, and price profiles
- scenario resets can cycle through different training situations without introducing randomness
- regression coverage verifies scenario generation, zone furnace propagation, and deterministic multi-room rollouts

### Deterministic Policy Evaluation Utilities

- `intelliwarm/learning/evaluation.py` adds reusable helpers for rolling a policy across one or more named scenarios
- evaluation summaries aggregate total reward, energy cost, comfort violation, and final zone heat-source decisions per scenario
- constant-policy helpers provide a simple baseline contract for future learned-policy or heuristic comparisons
- regression coverage verifies action-layout handling and multi-scenario metric aggregation

---

## Active Next Priority Order

These are the highest-impact integration tasks. Each should be done as a single bounded slice.

### 1. Live Energy Price Integration

The hybrid cost decision is only as good as the energy prices it uses.

- Implement a concrete live gas/electricity provider on top of the pricing provider boundary
- Keep the static fallback for offline/simulation mode
- Ensure `HybridController.decide()` always receives current prices, not stale cached values

### 2. Dashboard — Historical And Comparative Views

The dashboard now exposes the live hybrid heating choice to operators. The next step is deeper operator tooling.

- Add historical and scenario comparison panels for hybrid vs. MPC vs. future learned policies
- Visualize comfort drift, savings, and control source trends over time
- Keep using runtime/dashboard view models rather than reading controller internals directly in templates

### 3. RL Evaluation And Policy Comparison

Future ML control work should now build on the implemented room and multi-room training environments plus deterministic evaluation helpers.

- Add CLI/scripts that compare learned policies against hybrid and MPC baselines on the deterministic scenario library
- Reuse `OFF`, `ECO`, `COMFORT`, and `PREHEAT` actions so trained policies stay compatible with the live runtime
- Support both pure simulation training and offline evaluation against recorded scenarios
- Keep learned-policy execution behind the same runtime and safety boundaries as other controllers

### 4. Hardware Telemetry And Vendor Adapters

- Replace the shared hardware stubs with vendor-specific heater/furnace integrations
- Feed richer actuator telemetry into runtime status and operator views
- Preserve the simulated fallback path for offline development and deterministic testing

---

## Deferred Until Later

- live provider integrations as defaults
- production deployment of RL/learned policies
- broad package renames
- full UI redesign before control outputs stabilize
