# Quick Start

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Run Focused Verification First

```bash
pytest tests/test_simulation.py tests/test_runtime_service.py tests/test_modules.py
```

This verifies the current foundations:

- runtime orchestration
- thermal model behavior
- occupancy prediction
- deterministic simulation
- current MPC path

## 3. Start The App

```bash
python app.py
```

Then open `http://localhost:5000`.

## 4. Use The Repo As A Simulation Platform

Run examples without the web UI:

```bash
python examples.py
```

## 5. Key Files To Know

- `configs/config.yaml`: current configuration source
- `intelliwarm/services/runtime.py`: runtime orchestration
- `intelliwarm/data/models.py`: typed room, action, and simulation models
- `intelliwarm/models/thermal_model.py`: thermal step and simulate APIs
- `intelliwarm/models/simulator.py`: deterministic house simulator
- `intelliwarm/prediction/occupancy_model.py`: schedule and timestamp prediction

## 6. Copilot CLI Autopilot

Use this from the repo root:

```bash
copilot --autopilot --yolo --max-autopilot-continues 20
```

Before doing that, ensure the agent can read:

1. `AGENTS.md`
2. `.github/copilot-instructions.md`
3. `docs/roadmap.md`
4. `docs/architecture.md`
5. `docs/srs.md`

## 7. Troubleshooting

### Import problems

Run commands from the repo root.

### Database reset

Delete `intelliwarm.db` and start the app again if you need a clean local database.

### Test failures after model changes

Re-run the focused suite above before broader work.
