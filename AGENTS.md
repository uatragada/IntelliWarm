# IntelliWarm Agent Guide

Read this file before making non-trivial changes.

## Mission

Build IntelliWarm as a real-world HVAC cost minimization platform for hybrid heating systems. The core product mechanism is already implemented: `HybridController` decides per zone whether the gas furnace or electric space heaters are cheaper. All remaining work integrates this decision into the running system, surfaces it to operators, and connects it to real hardware.

Keep simulation as a first-class validation mode. Every new feature must work in both modes.

## Read Order

1. `README.md`
2. `.github/copilot-instructions.md`
3. `docs/roadmap.md`
4. `docs/architecture.md`
5. `docs/srs.md`

## Current State

- `app.py` is still the Flask entrypoint.
- `intelliwarm/services/runtime.py` now owns runtime orchestration.
- Typed simulation primitives exist in `intelliwarm/data/models.py`.
- Deterministic multi-room simulation exists in `intelliwarm/models/simulator.py`.
- Existing MPC remains continuous/scalar and should converge on shared hybrid contracts over time.
- **`HybridController` in `intelliwarm/control/hybrid_controller.py` is implemented and tested but not yet wired into the runtime.**
- `HeatSourceType`, `ZoneConfig`, and `HybridHeatingDecision` exist in `intelliwarm/data/models.py`.
- `configs/config.yaml` has `heat_source` per room and furnace specs per zone.
- `IntelliWarmRuntime.optimize_heating_plan()` still calls `BaselineController` per room — this is the highest-priority gap.

## Operating Rules

- Keep changes bounded to one subsystem at a time.
- Prefer extending current modules over renaming or reorganizing the whole repo.
- Keep route handlers thin; move behavior into services.
- Keep simulation deterministic and offline-safe.
- Add or update tests for every behavior change.
- Keep live integrations behind interfaces with simulation fallbacks.
- Keep hardware interaction out of optimizer and route internals.
- Do not replace the current storage layer unless a task explicitly requires it.

## Immediate Priorities

1. **Wire `HybridController` into `IntelliWarmRuntime.optimize_heating_plan()`** — group rooms by zone, build one `HybridController` per zone on startup, replace the per-room baseline loop with zone-level `decide()` calls, aggregate `per_room_actions` into the existing response shape.
2. **`DeviceController` furnace actuation** — add a `FurnaceActuator` interface; when `furnace_on is True`, emit a zone-level furnace signal and suppress per-room electric commands for that zone.
3. **Dashboard hybrid display** — show furnace vs. electric choice, cost comparison, rooms needing heat, and rationale from `HybridHeatingDecision.to_dict()`.
4. **Load `ZoneConfig` from `configs/config.yaml`** — parse the `zones` section into typed `ZoneConfig` instances at startup; pass them through runtime to `HybridController` construction.
5. **Live energy price integration** — replace the static price stub in `energy_price_fetcher.py` with a real gas + electricity provider; keep static fallback for offline mode.
6. **Reporting** — persist `HybridHeatingDecision` fields alongside each optimization run so operators can review cost decisions historically.

## Non-Goals For Immediate Work

- No full repo rename to the idealized target tree.
- No RL work in production code paths.
- No direct hardware coupling inside optimizer/simulator internals.
- No large UI redesign before controller outputs stabilize.
- Do not call `BaselineController` directly from services for zone heating — always go through `HybridController`.
- Do not activate furnace and electric heaters simultaneously for the same zone.

## Done Criteria For Each Slice

- Behavior implemented in the intended module boundary.
- Tests added or updated.
- Existing focused suites still pass.
- Documentation updated if workflows or architecture changed.