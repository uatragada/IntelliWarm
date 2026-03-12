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

## Next Priority Order

### 1. Typed Config Evolution

Layer typed config validation on top of `configs/config.yaml` without breaking the current runtime.

### 2. Persistence And Reporting

Extend the current SQLite layer with richer repositories and report-generation support.

## Deferred Until Later

- live provider integrations as defaults
- RL and research environment work
- broad package renames
- full UI redesign before control outputs stabilize
