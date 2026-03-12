# IntelliWarm

IntelliWarm is an intelligent HVAC optimization platform built for real-world deployment. It minimizes energy costs in homes and offices by deciding in real time whether to run a **gas furnace** (zone-level) or **electric space heaters** (room-level) based on live energy prices — and controls real hardware through explicit integration boundaries.

Simulation is a first-class capability for safe development and offline validation, but every controller, adapter, and cost model is designed to run with real hardware.

## The Core Idea: Hybrid Heating Cost Minimization

The key cost-saving mechanism is `HybridController` in `intelliwarm/control/hybrid_controller.py`. For each zone at each timestep it answers:

> *Is it cheaper right now to run the gas furnace for the whole zone, or let each room run its own electric heater?*

```
electric_cost  = Σ (heater_kW × action.power_level × electricity_$/kWh)  per room needing heat
furnace_cost   = (furnace_btu_per_hour / 100_000) / efficiency × gas_$/therm

if zone.has_furnace and furnace_cost < electric_cost:
	→ activate furnace at the highest demanded action level for the whole zone
else:
	→ each room runs its electric heater at its individual baseline action level
```

The furnace heats an entire zone uniformly and is economical when many rooms need heat simultaneously. Electric heaters allow per-room precision and are cheaper when only one or two rooms need heat. The controller automatically switches based on real prices.

## Current Focus

The repo is being advanced in bounded slices so GitHub Copilot CLI autopilot can continue implementation with high alignment and low drift.

**Completed foundations:**

- **Hybrid heating cost engine** — `intelliwarm/control/hybrid_controller.py` ← primary cost saver
- `HeatSourceType`, `ZoneConfig`, `HybridHeatingDecision` domain types in `intelliwarm/data/models.py`
- Explainable baseline controller (`OFF`, `ECO`, `COMFORT`, `PREHEAT`) driving per-room needs
- Aligned forecast bundle service — occupancy, weather, and pricing share one horizon contract
- Runtime orchestration with safety overrides and recent-event observability
- Hardware-ready sensor and actuator backends with simulation fallback
- Typed YAML config loading with `INTELLIWARM_*` environment overrides
- Modular Flask routes, SQLite-backed reporting, and deterministic multi-room simulation
- Hybrid runtime integration in `IntelliWarmRuntime.optimize_heating_plan()`
- Dashboard/runtime view models exposing active heat source, requested mode, applied mode, cost, and rationale
- Zone furnace actuation through `DeviceController`, with room electric heaters forced off when gas heat is selected
- Pricing provider boundary (`TimeOfUsePriceProvider`, `StaticPriceProvider`, `CallbackPriceProvider`) with offline-safe fallback forecasts
- Gym-compatible training environment in `intelliwarm/learning/gym_env.py` using the same `OFF`/`ECO`/`COMFORT`/`PREHEAT` action contract
- 50+ regression tests across all modules

**Immediate next work:**

- Connect the pricing provider boundary to a concrete live vendor/provider for accurate production prices
- Expand operator-facing comparisons between hybrid, baseline, MPC, and future learned policies
- Add multi-room and zone-aware training environments on top of the new Gym-compatible room environment

## Repository Map

```text
IntelliWarm/
├── AGENTS.md
├── .github/copilot-instructions.md
├── app.py
├── configs/
├── docs/
├── intelliwarm/
│   ├── core/
│   ├── control/
│   ├── data/
│   ├── learning/
│   ├── models/
│   ├── optimizer/
│   ├── prediction/
│   ├── pricing/
│   ├── sensors/
│   ├── services/
│   └── storage/
├── templates/
└── tests/
```

## Architecture Summary

- `app.py` / `intelliwarm/services/application.py`: Flask entrypoint and bootstrap
- `intelliwarm/services/application.py`: Flask app/bootstrap wiring and route registration
- `intelliwarm/services/runtime.py`: application orchestration, zone-aware hybrid decisions, and dashboard/runtime state
- `intelliwarm/core/config.py`: typed config loading for YAML plus `INTELLIWARM_*` environment overrides
- `intelliwarm/control/hybrid_controller.py`: **zone-level hybrid heating cost engine** — the primary actuator decision point
- `intelliwarm/control/baseline_controller.py`: per-room explainable rule-based baseline (`OFF`/`ECO`/`COMFORT`/`PREHEAT`)
- `intelliwarm/models/thermal_model.py`: room thermal dynamics
- `intelliwarm/models/simulator.py`: deterministic multi-room simulation
- `intelliwarm/prediction/occupancy_model.py`: schedule and timestamp-based occupancy prediction
- `intelliwarm/optimizer/mpc_controller.py`: current MPC implementation
- `intelliwarm/control/device_controller.py`: HVAC actuator backend (hardware-ready with simulation fallback)
- `intelliwarm/sensors/sensor_manager.py`: sensor backend (hardware-ready with simulation fallback)
- `intelliwarm/learning/gym_env.py`: Gym-compatible deterministic training environment for future ML control work
- `intelliwarm/storage/database.py`: SQLite persistence layer

See `docs/architecture.md` for the working architecture and `docs/roadmap.md` for current sequencing.

## Quick Start

```bash
pip install -r requirements.txt
pytest tests/test_simulation.py tests/test_runtime_service.py tests/test_modules.py
python app.py
```

Then open `http://localhost:5000`.

Configuration is loaded from `configs/config.yaml`. You can override common values with `INTELLIWARM_*` environment variables, and `${ENV_NAME}` placeholders in the YAML are resolved at load time.

## Deployment Intent

IntelliWarm should support two operating modes:

- Simulation mode for deterministic testing, policy comparison, and offline development
- Live mode for real hardware integration through controlled adapters, safety checks, and runtime guardrails

All new control features must run in both modes using the shared `HybridHeatingDecision`, `ControlDecision`, and `ForecastBundle` contracts. The hybrid cost calculation must use real-time energy prices in live mode.

### Real-World Hardware Expectations

| Component | Live Mode Target | Simulation Fallback |
|-----------|-----------------|---------------------|
| Gas furnace | Zone relay/thermostat signal via `DeviceController` | `SimulatedFurnace` zone actuator |
| Electric heaters | Per-room relay or smart plug via `DeviceController` | `SimulatedHeater` per room |
| Temperature sensors | GPIO/MQTT/REST via `SensorManager` hardware backend | Thermal model state |
| Energy prices | Live gas and electricity provider APIs | `configs/config.yaml` static values |
| Occupancy | PIR sensors or presence detection | Schedule-based prediction |

## Copilot Autopilot

For high-alignment Copilot CLI runs, keep the work bounded and let the agent use the repo guidance files in this order:

1. `AGENTS.md`
2. `.github/copilot-instructions.md`
3. `docs/roadmap.md`
4. `docs/architecture.md`
5. `docs/srs.md`

Recommended command:

```bash
copilot --autopilot --yolo --max-autopilot-continues 20
```

The repo now includes explicit instructions and prompts for that flow.

## Current Priority Queue

The current delivery-order implementation slices are complete, including hybrid runtime integration. Next work should deepen live integrations, operator-facing comparisons, and future ML training support on the existing service and adapter boundaries.

## Testing

Focused verification for shared model and runtime work:

```bash
pytest tests/test_simulation.py tests/test_runtime_service.py tests/test_modules.py
```

Full suite:

```bash
pytest tests/
```

## Documentation Index

- `QUICKSTART.md`: local setup and execution
- `CONTRIBUTING.md`: development workflow
- `docs/architecture.md`: current working architecture
- `docs/roadmap.md`: milestone sequence and active next work
- `docs/srs.md`: repo-grounded product and system requirements

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Backend | Python 3.9+, Flask |
| ML/Optimization | NumPy, SciPy, scikit-learn |
| Database | SQLite |
| Configuration | YAML |
| Frontend | HTML/CSS/JavaScript (Plotly) |

## Contributing

Contributions are welcome. The current highest-impact work areas:

1. **Live energy price integration** — replace the current static price stub with a real gas and electricity provider API so cost decisions are accurate.
2. **Historical and scenario dashboards** — extend the current dashboard from live status cards into richer comparisons and operator reports.
3. **Multi-room RL environments** — extend the current room-level Gym-compatible environment into zone-aware and house-level training tasks.
4. **Hybrid/MPC contract convergence** — align controller outputs so learned or optimized policies can slot into the same runtime surfaces.
5. **Hardware command enrichment** — evolve furnace and room-heater adapters from shared stubs into vendor-specific integrations with explicit telemetry.

Before making non-trivial changes, read:

1. `AGENTS.md`
2. `.github/copilot-instructions.md`
3. `docs/roadmap.md`
4. `docs/architecture.md`
5. `docs/srs.md`

See `CONTRIBUTING.md` for the development workflow.

## License

No standalone license file is currently included in the repository.

## Author

**Uday Atragada** — [GitHub](https://github.com/uatragada)

## FAQ

**Q: Does IntelliWarm control real HVAC systems?**  
A: That is a core product goal. The platform uses simulation as a safe development and validation layer, while architecture and roadmap decisions are expected to support real hardware integrations.

**Q: Can I use this in production?**  
A: Production readiness is in progress. You should expect staged rollout practices, adapter-level safety controls, and deployment hardening before broad live operation.

**Q: How is occupancy predicted today?**  
A: The current codebase includes schedule-based and timestamp-aware occupancy prediction in `intelliwarm/prediction/occupancy_model.py`.
