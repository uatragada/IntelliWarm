# IntelliWarm Copilot Instructions

Use these instructions for GitHub Copilot CLI autopilot and chat-based implementation.

## Project Intent

IntelliWarm is a real-world HVAC cost minimization platform for homes and buildings with hybrid heating systems — a gas furnace that heats entire zones and electric space heaters that heat individual rooms. The platform decides in real time which source is cheaper, controls the appropriate hardware, and records the cost outcome.

Simulation is a required capability for deterministic validation and offline development. Every feature must run in both simulation mode and live hardware mode using shared normalized-demand action and forecast contracts.

**The central cost-saving mechanism is `HybridController` in `intelliwarm/control/hybrid_controller.py`. All new zone heating work must route through it.**

## Source Of Truth

- **Hybrid cost engine**: `intelliwarm/control/hybrid_controller.py`
- **Hybrid domain types**: `intelliwarm/data/models.py` (`HeatSourceType`, `ZoneConfig`, `HybridHeatingDecision`)
- **Runtime orchestration**: `intelliwarm/services/runtime.py`
- **Per-room baseline**: `intelliwarm/control/baseline_controller.py`
- **Device actuation boundary**: `intelliwarm/control/device_controller.py`
- **Thermal model**: `intelliwarm/models/thermal_model.py`
- **Simulation primitives**: `intelliwarm/data/models.py`
- **Multi-room simulator**: `intelliwarm/models/simulator.py`
- **Occupancy prediction**: `intelliwarm/prediction/occupancy_model.py`
- **Current storage layer**: `intelliwarm/storage/database.py`
- **Current plan and sequencing**: `docs/roadmap.md`

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
- **Zone heating decisions must go through `HybridController.decide()` — do not call `BaselineController` directly from services or routes.**
- Baseline generates per-room heat needs as inputs to `HybridController`; it is not a standalone zone controller.
- MPC and baseline should converge on shared normalized-demand, `HybridHeatingDecision`, and `ForecastBundle` contracts.
- Dashboard logic should consume service outputs, not internal model details.
- Hardware-specific code should remain in adapter/integration boundaries, not optimizer internals.
- Storage changes should remain compatible with the current SQLite workflow unless migration is explicitly requested.
- Energy prices passed to `HybridController` must be live values in production mode; static config prices are fallback only.
- When `furnace_on is True`, the `DeviceController` must suppress all per-room electric commands for that zone.

## Testing Rules

- Add tests for every new module or public behavior change.
- Run focused tests first.
- If touching shared model or runtime code, run:

```bash
pytest tests/test_simulation.py tests/test_runtime_service.py tests/test_modules.py
```

## Priority Order

1. Replace the static energy-price stub with live gas and electricity provider APIs.
2. Deepen dashboard and reporting comparisons across hybrid, baseline, MPC, and learned policies.
3. Expand deterministic learned-policy evaluation and checkpoint comparison tooling.
4. Continue converging MPC and learned policies on the shared normalized-demand runtime contract.
5. Enrich hardware adapters with vendor-specific telemetry while preserving simulation fallback.

## Avoid

- Global rewrites.
- Hidden randomness in simulation.
- Hardcoded room-specific logic in controllers.
- Live API calls inside optimizer or simulator code.
- Direct hardware I/O in route handlers.
- Unbounded tasks that mix architecture, UI, storage, and control changes together.
- Bypassing `HybridController` by calling `BaselineController` per room in services — this defeats the core cost-saving logic.
- Hard-coding energy prices in `HybridController` — prices must be injected at call time.
- Activating both furnace and electric heaters for the same zone simultaneously.

## Preferred Task Pattern

1. Read `docs/roadmap.md` and the relevant module.
2. Propose or execute a single bounded subsystem change.
3. Add tests.
4. Run focused verification.
5. Update docs if needed.
