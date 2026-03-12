# IntelliWarm

IntelliWarm is a Python and Flask project for intelligent HVAC optimization with a real-world deployment target. Simulation is a core capability for validation and safe iteration, but the platform is being built to control real HVAC hardware through explicit integration boundaries.

## Current Focus

The repo is being advanced in bounded slices so GitHub Copilot CLI autopilot can continue implementation with high alignment and low drift.

Current completed foundations:

- runtime orchestration extracted to `intelliwarm/services/runtime.py`
- Flask bootstrap and route wiring centralized in `intelliwarm/services/application.py`
- typed YAML config loading with environment overrides in `intelliwarm/core/config.py`
- typed simulation primitives in `intelliwarm/data/models.py`
- explainable baseline controller in `intelliwarm/control/baseline_controller.py`
- aligned forecast bundle service in `intelliwarm/services/forecast_bundle.py`
- hardware-ready sensor and actuator backends with simulation fallback
- modular Flask routes under `intelliwarm/routes/`
- typed config validation and preserved structured schedule support in `intelliwarm/core/config.py`
- deterministic multi-room simulation in `intelliwarm/models/simulator.py`
- thermal step and simulate APIs in `intelliwarm/models/thermal_model.py`
- timestamp-aware occupancy prediction in `intelliwarm/prediction/occupancy_model.py`
- focused regression tests for runtime, simulation, config, and app bootstrap

Real-world direction now explicitly includes:

- hardware adapter interfaces for sensors and device actuation
- deployment-safe runtime behavior with fallback simulation mode
- observability and persistence for field diagnostics and reporting
- incremental hardening for production usage (safety, reliability, and operability)

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

- `app.py`: Flask entrypoint
- `intelliwarm/services/application.py`: Flask app/bootstrap wiring and route registration
- `intelliwarm/services/runtime.py`: application orchestration and demo/runtime state
- `intelliwarm/core/config.py`: typed config loading for YAML plus `INTELLIWARM_*` environment overrides
- `intelliwarm/models/thermal_model.py`: room thermal dynamics
- `intelliwarm/models/simulator.py`: deterministic multi-room simulation
- `intelliwarm/prediction/occupancy_model.py`: schedule and timestamp-based occupancy prediction
- `intelliwarm/optimizer/mpc_controller.py`: current MPC implementation
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

All new control features should be designed to run in both modes using shared action and forecast contracts.

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

1. Persistence and reporting improvements on top of SQLite.
2. Runtime safety and observability improvements for live deployments.

## Testing

Focused verification for shared model and runtime work:

```bash
pytest tests/test_simulation.py tests/test_runtime_service.py tests/test_modules.py tests/test_config_loading.py tests/test_app_bootstrap.py
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
- `docs/copilot-prompts.md`: reusable prompts for bounded Copilot tasks
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

Contributions are welcome, especially in the current bounded work areas:

1. Baseline controller with explainable discrete actions.
2. Forecast bundle service for aligned occupancy, outdoor temperature, and pricing horizons.
3. Flask route modularization into thin service-backed modules.
4. Typed config validation layered on `configs/config.yaml`.
5. Persistence and reporting improvements on top of the current SQLite workflow.

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
