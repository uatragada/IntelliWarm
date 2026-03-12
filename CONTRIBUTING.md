# Contributing to IntelliWarm

## Development Principles

- keep changes bounded to one subsystem at a time
- prefer extending the current package layout over broad renames
- keep route handlers thin and business logic in services or domain modules
- keep simulation deterministic and offline-safe
- add or update tests for every behavior change
- update documentation when workflows, architecture, or priorities change

## Setup

```bash
pip install -r requirements.txt
```

## Required Read Order Before Non-Trivial Changes

1. `AGENTS.md`
2. `.github/copilot-instructions.md`
3. `docs/roadmap.md`
4. `docs/architecture.md`
5. `docs/srs.md`

## Current High-Priority Work

1. Baseline controller with explanations and discrete actions.
2. Forecast bundle service.
3. Flask route modularization.
4. Typed config and validation improvements.

## Testing

Run this focused suite when touching runtime, simulation, occupancy, or model code:

```bash
pytest tests/test_simulation.py tests/test_runtime_service.py tests/test_modules.py
```

Run the full suite before finishing a broader slice:

```bash
pytest tests/
```

## Pull Request Expectations

Each PR should include:

- purpose
- modules changed
- tests added or updated
- screenshots if the dashboard changed
- assumptions or follow-up work

## Coding Expectations

- use type hints for public APIs
- keep public class and function docstrings concise
- do not hardcode room-specific logic in controllers
- do not call live APIs directly from the optimizer or simulator
- do not mix UI, storage, and control changes in one PR unless the slice requires it

## Project Layout

```text
intelliwarm/
├── core/
├── control/
├── data/
├── learning/
├── models/
├── optimizer/
├── prediction/
├── pricing/
├── sensors/
├── services/
└── storage/
```

## Documentation To Update When Needed

- `README.md`
- `QUICKSTART.md`
- `docs/architecture.md`
- `docs/roadmap.md`
- `docs/srs.md`
- `docs/copilot-prompts.md`
