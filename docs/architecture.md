# IntelliWarm Architecture

## Working Architecture

The current repo should be treated as an incremental architecture, not a greenfield rewrite.

## Main Boundaries

### Application Layer

- `app.py`: Flask entrypoint
- `intelliwarm/services/runtime.py`: runtime orchestration, demo loading, optimization dispatch, and room state management
- `intelliwarm/services/forecast_bundle.py`: aligned occupancy, weather, and pricing forecasts for controllers

### Domain And Simulation Layer

- `intelliwarm/data/models.py`: shared `RoomConfig`, `OccupancyWindow`, `HeatingAction`, `SimulationState`, `ControlDecision`, and `ForecastBundle` contracts
- `intelliwarm/models/thermal_model.py`: room thermal dynamics with `step()` and `simulate()`
- `intelliwarm/models/simulator.py`: deterministic `HouseSimulator`
- `intelliwarm/prediction/occupancy_model.py`: schedule and timestamp-based occupancy prediction

### Control Layer

- `intelliwarm/optimizer/mpc_controller.py`: current MPC implementation
- `intelliwarm/control/baseline_controller.py`: explainable rule-based controller using `OFF`, `ECO`, `COMFORT`, and `PREHEAT`

### Integration Layer

- `intelliwarm/sensors/sensor_manager.py`: sensor backend boundary with hardware-ready fallback adapters
- `intelliwarm/control/device_controller.py`: HVAC actuation backend boundary with simulated fallback
- `intelliwarm/pricing/energy_price_fetcher.py`: static and time-of-use pricing service

### Persistence Layer

- `intelliwarm/storage/database.py`: SQLite storage for rooms, logs, optimization runs, and model parameters

## Current Architectural Rules

- extend the current package layout in place
- keep route handlers thin
- keep simulation deterministic
- push orchestration into services
- share action and forecast contracts across baseline and MPC
- keep integrations optional and offline-safe

## Immediate Architecture Targets

1. Add hardware adapter boundaries for sensors and HVAC devices with simulation fallback.
2. Break route logic out of `app.py` into dashboard-focused modules.
3. Move config handling toward typed models without breaking `configs/config.yaml`.
4. Extend persistence and reporting on top of the current SQLite workflow.
