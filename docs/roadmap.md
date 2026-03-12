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

## Next Priority Order

### 1. Forecast Bundle Service

Add a shared service that aligns occupancy, outdoor temperature, and energy price horizons.

### 2. Route Modularization

Refactor `app.py` into thin Flask route modules that delegate to services.

### 3. Typed Config Evolution

Layer typed config validation on top of `configs/config.yaml` without breaking the current runtime.

### 4. Persistence And Reporting

Extend the current SQLite layer with richer repositories and report-generation support.

## Deferred Until Later

- live provider integrations as defaults
- RL and research environment work
- broad package renames
- full UI redesign before control outputs stabilize
