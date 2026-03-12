# IntelliWarm Agent Guide

Read this file before making non-trivial changes.

## Mission

Build IntelliWarm as a real-world HVAC optimization platform by extending the current codebase in small, test-backed slices. Keep simulation as a first-class validation mode, not the final deployment target.

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
- Existing MPC remains continuous/scalar and should be evolved in place.

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

1. Baseline controller with explainable `OFF`, `ECO`, `COMFORT`, `PREHEAT` actions.
2. Forecast bundle service aligning occupancy, outdoor temperature, and pricing horizons.
3. Hardware adapter boundary for real sensors and HVAC actuators.
4. Flask route modularization into dashboard-focused modules.
5. Typed config evolution on top of `configs/config.yaml`.
6. Runtime safety and observability for live deployments.

## Non-Goals For Immediate Work

- No full repo rename to the idealized target tree.
- No RL work in production code paths.
- No direct hardware coupling inside optimizer/simulator internals.
- No large UI redesign before controller outputs stabilize.

## Done Criteria For Each Slice

- Behavior implemented in the intended module boundary.
- Tests added or updated.
- Existing focused suites still pass.
- Documentation updated if workflows or architecture changed.