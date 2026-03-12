# IntelliWarm Copilot Instructions

Use these instructions for GitHub Copilot CLI autopilot and chat-based implementation.

## Project Intent

IntelliWarm is a Python and Flask project for intelligent HVAC optimization targeting real-world deployment. Simulation remains a required capability for deterministic validation, but implementation slices should move the system toward safe operation with real hardware.

## Source Of Truth

- Runtime orchestration: `intelliwarm/services/runtime.py`
- Thermal model: `intelliwarm/models/thermal_model.py`
- Simulation primitives: `intelliwarm/data/models.py`
- Multi-room simulator: `intelliwarm/models/simulator.py`
- Occupancy prediction: `intelliwarm/prediction/occupancy_model.py`
- Current storage layer: `intelliwarm/storage/database.py`
- Current plan and sequencing: `docs/roadmap.md`

## Required Working Style

- Make the smallest coherent change that completes the current slice.
- Preserve existing behavior unless the task explicitly changes it.
- Prefer typed Python and explicit data contracts.
- Keep route handlers thin and business logic in services or domain modules.
- Keep simulation deterministic.
- Keep live hardware integrations behind interfaces with simulation fallback.
- Keep new integrations offline-safe when hardware is unavailable.
- Update docs when architecture or workflow changes.

## Architecture Constraints

- Extend the current package layout; do not perform a broad package rename.
- Reuse existing modules before creating new ones.
- Baseline and MPC should converge on shared action and forecast contracts.
- Dashboard logic should consume service outputs, not internal model details.
- Hardware-specific code should remain in adapter/integration boundaries, not optimizer internals.
- Storage changes should remain compatible with the current SQLite workflow unless migration is explicitly requested.

## Testing Rules

- Add tests for every new module or public behavior change.
- Run focused tests first.
- If touching shared model or runtime code, run:

```bash
pytest tests/test_simulation.py tests/test_runtime_service.py tests/test_modules.py
```

## Priority Order

1. Baseline controller and explanation model.
2. Forecast bundle service.
3. Hardware integration boundary for sensors and device control.
4. Route modularization.
5. Typed config and validation.
6. Persistence/reporting improvements.
7. Live runtime safety and observability.

## Avoid

- Global rewrites.
- Hidden randomness in simulation.
- Hardcoded room-specific logic in controllers.
- Live API calls inside optimizer or simulator code.
- Direct hardware I/O in route handlers.
- Unbounded tasks that mix architecture, UI, storage, and control changes together.

## Preferred Task Pattern

1. Read `docs/roadmap.md` and the relevant module.
2. Propose or execute a single bounded subsystem change.
3. Add tests.
4. Run focused verification.
5. Update docs if needed.