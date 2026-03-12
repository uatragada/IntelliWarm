# IntelliWarm Architecture

## Working Architecture

The current repo should be treated as an incremental architecture, not a greenfield rewrite.

## Main Boundaries

### Application Layer

- `app.py`: Flask entrypoint
- `intelliwarm/services/runtime.py`: runtime orchestration, demo loading, optimization dispatch, and room state management

### Domain And Simulation Layer

- `intelliwarm/data/models.py`: `RoomConfig`, `OccupancyWindow`, `HeatingAction`, `SimulationState`
- `intelliwarm/models/thermal_model.py`: room thermal dynamics with `step()` and `simulate()`
- `intelliwarm/models/simulator.py`: deterministic `HouseSimulator`
- `intelliwarm/prediction/occupancy_model.py`: schedule and timestamp-based occupancy prediction

### Control Layer

- `intelliwarm/optimizer/mpc_controller.py`: current MPC implementation
- `intelliwarm/control/baseline_controller.py`: explainable rule-based controller using `OFF`, `ECO`, `COMFORT`, and `PREHEAT`

### Integration Layer

- `intelliwarm/sensors/sensor_manager.py`: simulated sensor registry
- `intelliwarm/control/device_controller.py`: simulated heater/device control
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

1. Add a baseline controller with shared discrete actions and explanation outputs.
2. Add a forecast bundle service consumed by both simulator and controllers.
3. Break route logic out of `app.py` into dashboard-focused modules.
4. Move config handling toward typed models without breaking `configs/config.yaml`.
