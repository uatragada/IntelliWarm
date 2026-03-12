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

## Next Priority Order

- current delivery-order slices are implemented
- next work should deepen live provider integrations, dashboards, and operational hardening on the existing boundaries

## Deferred Until Later

- live provider integrations as defaults
- RL and research environment work
- broad package renames
- full UI redesign before control outputs stabilize
