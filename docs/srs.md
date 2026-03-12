# IntelliWarm — Copilot-Optimized Project Plan

Version: 1.0
Target repo: `uatragada/IntelliWarm`
Primary stack: Python, Flask, Jupyter, HTML templates
Planning date: March 11, 2026

Implementation status update: March 12, 2026

- runtime orchestration moved into `intelliwarm/services/runtime.py`
- typed simulation models added in `intelliwarm/data/models.py`
- deterministic multi-room simulation added in `intelliwarm/models/simulator.py`
- occupancy prediction supports timestamp-based schedules
- this document remains the product and architecture target, but implementation should follow `docs/roadmap.md` and the current package layout

---

## 1. Goal

Turn the current IntelliWarm repository into a modular, testable, simulation-first intelligent HVAC optimization platform that GitHub Copilot can implement incrementally with minimal ambiguity.

This plan is designed so you can:

1. create GitHub issues directly from sections,
2. let Copilot generate code module-by-module,
3. keep implementation aligned with a strong architecture,
4. avoid Copilot drifting into vague or inconsistent designs.

---

## 2. Current repo snapshot

Observed top-level items in the working repository:

* `app.py`
* `sample_data.py`
* `templates/`
* `docs/`
* `intelliwarm/`
* `tests/`
* `data-analytics/`
* `README.md`

The repository has moved beyond the original exploratory prototype, but still needs incremental architecture hardening. The codebase should continue to evolve in phases:

* stabilize the existing app,
* modularize the logic,
* add simulation and forecasting,
* add optimization,
* add persistence, testing, and deployment support.

---

## 3. Project north star

IntelliWarm should become a system that can:

* ingest room temperatures, occupancy schedules, weather, and utility prices,
* model room thermal behavior,
* forecast future temperatures and occupancy,
* optimize heating decisions over a planning horizon,
* recommend or execute heating actions,
* explain savings, comfort tradeoffs, and decisions in the dashboard.

Primary optimization target:

**Minimize heating cost while maintaining room comfort constraints.**

---

## 4. Product scope

### In scope for MVP

* multi-room heating simulation,
* manual room configuration,
* occupancy schedules,
* weather input abstraction,
* utility price input abstraction,
* thermal model estimation,
* rule-based baseline controller,
* MPC-style optimizer,
* dashboard visualizations,
* logging and testing.

### Out of scope for MVP

* direct physical furnace control in production,
* mobile app,
* advanced RL deployment,
* enterprise multi-building support,
* full smart-home hardware integration across many vendors.

---

## 5. Target architecture

```text
Inputs
 ├── Room temperatures
 ├── Outdoor weather
 ├── Utility prices
 ├── Occupancy schedules / predictions
 └── Device states

Data Layer
 ├── Input adapters
 ├── Validation
 ├── Time-series logging
 └── Config loading

Model Layer
 ├── Thermal model
 ├── Occupancy model
 ├── Forecast utilities
 └── Scenario generation

Control Layer
 ├── Baseline rule controller
 ├── MPC planner
 ├── Action scoring
 └── Safety constraints

Application Layer
 ├── Flask API
 ├── Dashboard routes
 ├── Service orchestration
 └── Reports / analytics

Persistence + Testing
 ├── SQLite/Postgres
 ├── Repositories
 ├── Unit tests
 ├── Integration tests
 └── Simulation tests
```

---

## 6. Recommended repo structure

The target structure below is still directionally useful, but implementation should extend the current package layout in place rather than performing a broad rename. New work should prefer the existing `core`, `control`, `data`, `models`, `optimizer`, `prediction`, `pricing`, `services`, and `storage` modules.

```text
IntelliWarm/
├── app.py
├── README.md
├── requirements.txt
├── pyproject.toml
├── .env.example
├── config/
│   ├── default.yaml
│   ├── rooms.example.yaml
│   └── pricing.example.yaml
├── intelliwarm/
│   ├── __init__.py
│   ├── core/
│   │   ├── config.py
│   │   ├── scheduler.py
│   │   ├── exceptions.py
│   │   └── constants.py
│   ├── data/
│   │   ├── schemas.py
│   │   ├── validators.py
│   │   ├── loaders.py
│   │   └── sample_data.py
│   ├── sensors/
│   │   ├── base.py
│   │   ├── simulated.py
│   │   ├── room_temperature.py
│   │   └── outdoor_temperature.py
│   ├── pricing/
│   │   ├── base.py
│   │   ├── static_pricing.py
│   │   └── utility_service.py
│   ├── weather/
│   │   ├── base.py
│   │   ├── static_weather.py
│   │   └── weather_service.py
│   ├── occupancy/
│   │   ├── base.py
│   │   ├── schedule_model.py
│   │   └── feature_builder.py
│   ├── thermal/
│   │   ├── room_model.py
│   │   ├── estimator.py
│   │   └── simulator.py
│   ├── control/
│   │   ├── actions.py
│   │   ├── constraints.py
│   │   ├── baseline_controller.py
│   │   ├── mpc_controller.py
│   │   └── objective.py
│   ├── services/
│   │   ├── forecast_service.py
│   │   ├── optimization_service.py
│   │   ├── report_service.py
│   │   └── orchestration_service.py
│   ├── storage/
│   │   ├── db.py
│   │   ├── models.py
│   │   ├── repositories.py
│   │   └── migrations/
│   ├── dashboard/
│   │   ├── routes.py
│   │   ├── viewmodels.py
│   │   └── chart_builders.py
│   └── utils/
│       ├── time.py
│       ├── math.py
│       └── logging.py
├── templates/
├── static/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── simulation/
├── notebooks/
└── docs/
    ├── architecture.md
    ├── srs.md
    ├── roadmap.md
    └── copilot-prompts.md
```

---

## 7. Implementation strategy

### Phase 0 — Repo hardening

Purpose: make the repo easy for Copilot to work on without chaos.

#### Deliverables

* standard package layout under `intelliwarm/`
* `requirements.txt` or `pyproject.toml`
* configuration files under `config/`
* test scaffold under `tests/`
* linting and formatting setup
* improved README with architecture and setup steps

#### Copilot-ready tasks

1. Keep the existing `intelliwarm/` package layout and extend it in place.
2. Continue moving prototype logic out of `app.py` into services and route modules.
3. Add `requirements.txt` with Flask, pandas, numpy, scipy, pydantic, pytest.
4. Add a configuration loader that reads YAML and environment variables.
5. Add logging utilities and centralized exception handling.
6. Add pytest scaffolding and sample smoke tests.

#### Definition of done

* app still runs,
* imports are modular,
* tests run successfully,
* configs are no longer hardcoded in route logic.

---

### Phase 1 — Domain model and simulation foundation

Status: partially implemented.

Purpose: create a solid simulation-first foundation so Copilot can implement the system safely before real control logic.

#### Deliverables

* room data models
* occupancy schedule format
* simulated sensors
* weather abstraction
* pricing abstraction
* room thermal simulator

#### Core requirements

Each room must define:

* room id,
* display name,
* target comfort range,
* thermal coefficients,
* heater capacity,
* heat loss factor,
* occupancy schedule.

The simulator must support:

* per-room temperature evolution,
* external temperature effects,
* heating action effects,
* time-stepped simulation,
* deterministic replay for testing.

#### Copilot-ready tasks

1. Create `RoomConfig`, `OccupancyWindow`, `HeatingAction`, and `SimulationState` data models.
2. Implement a YAML room config schema and parser.
3. Build `SimulatedTemperatureSensor` and `SimulatedOutdoorSensor` classes.
4. Implement `ThermalRoomModel.step()` to compute next temperature.
5. Implement a `HouseSimulator` that steps all rooms forward over time.
6. Create sample schedules for six bedrooms and shared areas.
7. Add simulation unit tests for heating, cooling, and occupancy transitions.

#### Definition of done

* a simulation can run for 24 hours,
* room temperatures evolve consistently,
* sample schedules are loaded from config,
* outputs can be graphed in the dashboard or notebook.

---

### Phase 2 — Baseline control and explainability

Status: implemented with runtime integration; future work should focus on actuator boundaries and learned-policy support.

Purpose: create a reliable baseline before advanced optimization.

#### Deliverables

* rule-based controller
* action recommendation engine
* comfort and cost scoring
* explanation strings for dashboard output

#### Functional behavior

The baseline controller should:

* heat occupied rooms toward target temperature,
* use eco mode for unoccupied rooms,
* optionally preheat before scheduled occupancy,
* avoid rapid toggling,
* generate human-readable decision reasons.

#### Copilot-ready tasks

1. Create `BaselineController` with configurable thresholds.
2. Add occupancy-aware setpoint selection.
3. Add a minimum runtime / cooldown constraint to prevent chattering.
4. Create a `DecisionExplanation` model.
5. Implement cost estimation for each action per timestep.
6. Add dashboard cards showing recommended actions and reasons.
7. Add tests verifying baseline behavior under cold weather and changing schedules.

#### Definition of done

* controller produces sensible recommendations,
* costs are estimated,
* decisions are explainable in plain English,
* dashboard can display room-by-room action rationale.

---

### Phase 3 — Forecasting and thermal parameter estimation

Purpose: enable the system to learn room behavior and look ahead.

#### Deliverables

* thermal parameter estimation from historical data
* occupancy probability estimation
* forecast service for weather, prices, and occupancy

#### Functional behavior

Thermal estimation should infer at minimum:

* heating effectiveness,
* cooling / leakage coefficient,
* response lag.

Occupancy forecasting should support:

* deterministic schedule-based prediction for MVP,
* optional probabilistic output,
* future extensibility toward calendar / sensor fusion.

#### Copilot-ready tasks

1. Implement `ThermalParameterEstimator` using least squares on simulated or logged data.
2. Add a persistence layer for storing room coefficients.
3. Create `ScheduleOccupancyModel.predict_probability()`.
4. Build `ForecastService` that returns horizon-aligned forecasts.
5. Add a feature builder for time of day, day of week, occupancy, price, and weather.
6. Create evaluation tests comparing predicted vs actual temperature trajectories.
7. Add notebook examples for thermal parameter fitting.

#### Definition of done

* system can estimate room coefficients from historical data,
* forecasts are aligned on common timesteps,
* thermal prediction accuracy is measurable.

---

### Phase 4 — MPC optimizer

Status: existing continuous MPC is present; the planned work is to converge it on the shared normalized-demand and explanation contracts used by runtime, hybrid control, and learning.

Purpose: implement the intelligent core of IntelliWarm.

#### Deliverables

* action horizon planner
* MPC objective function
* comfort constraints
* switching penalties
* recommended plan visualization

#### Optimization objective

For each timestep over a finite horizon, minimize:

* energy cost,
* comfort violation penalty,
* switching penalty,
* optional preheat inefficiency penalty.

#### Constraints

* occupied rooms should remain within comfort bands when possible,
* rooms cannot heat infinitely fast,
* heater power cannot exceed room capacity,
* controller should respect minimum on/off intervals,
* optimization must finish quickly enough for interactive use.

#### Copilot-ready tasks

1. Define normalized room-demand action space for each room (`0.0..1.0`) with compatibility labels such as `OFF`, `ECO`, `COMFORT`, and `PREHEAT`.
2. Create `ObjectiveWeights` config structure.
3. Implement temperature rollout over a finite planning horizon.
4. Implement `score_plan()` for energy, comfort, and switching.
5. Implement `MPCController.plan()` returning the best first action and full candidate plan.
6. Add pruning or heuristics to keep plan search computationally manageable.
7. Add comparison tests: MPC vs baseline controller over 24-hour simulations.
8. Add dashboard visualization for predicted temperature trajectory and selected actions.

#### Definition of done

* MPC consistently outperforms or matches baseline in simulation,
* plan outputs are explainable,
* runtime is acceptable for small multi-room scenarios.

---

### Phase 5 — Persistence, analytics, and reporting

Status: SQLite persistence and reporting are implemented; richer historical analytics remain future work.

Purpose: convert prototype logic into a durable application.

#### Deliverables

* database schema
* logging repositories
* optimization history
* savings reports
* historical room analytics

#### Data to store

* room temperature readings,
* outdoor temperature,
* occupancy predictions,
* action recommendations,
* estimated costs,
* optimization plans,
* learned thermal coefficients.

#### Copilot-ready tasks

1. Add SQLAlchemy or lightweight repository-based persistence.
2. Create tables for rooms, readings, forecasts, plans, and model parameters.
3. Build repository methods for saving and retrieving time-series data.
4. Add a report service that computes daily and weekly savings.
5. Create historical charts for temperature, cost, and comfort violations.
6. Add integration tests for persistence and report generation.

#### Definition of done

* optimization runs are stored,
* reports are reproducible,
* dashboard can show historical performance.

---

### Phase 6 — Web app polish and user workflows

Status: route modularization and runtime-backed dashboard view models are implemented; richer comparative UX remains future work.

Purpose: make the product usable and legible.

#### Deliverables

* improved dashboard IA
* room configuration editor
* scenario comparison views
* simulation run controls
* cost/comfort summary panels

#### Copilot-ready tasks

1. Refactor Flask routes into blueprint-style modules.
2. Create pages for dashboard, rooms, simulation runs, and reports.
3. Add forms for editing room target temperatures and schedule windows.
4. Add charts for current temp vs target vs forecast.
5. Add scenario comparison: baseline vs optimized.
6. Add status cards for projected cost today and comfort risk.
7. Improve templates and styling consistency.

#### Definition of done

* app supports core workflows without notebook use,
* major system outputs are visible in the browser,
* user can compare controllers and simulation outcomes.

---

### Phase 7 — External integrations

Purpose: make IntelliWarm ready for real data sources.

Status: provider boundaries for pricing and device control are implemented with offline-safe fallbacks; concrete live vendor integrations remain future work.

#### Deliverables

* weather service adapter
* utility pricing adapter
* optional Home Assistant / MQTT adapter
* safe execution mode for commands

#### Copilot-ready tasks

1. Create weather provider interface and a static implementation for tests.
2. Extend the implemented energy price provider boundary with concrete live vendor integrations.
3. Add optional live provider stubs behind feature flags.
4. Extend the implemented device controller boundary with vendor-specific furnace/heater integrations.
5. Add command audit logging for all outbound control decisions.
6. Add safety mode that recommends actions without executing them.

#### Definition of done

* external services are abstracted cleanly,
* live integrations are optional and isolated,
* system is still testable offline.

---

### Phase 8 — Advanced intelligence layer

Purpose: prepare for future RL and adaptive control without breaking the core.

Status: deterministic Gym-compatible room and multi-room environments, reusable learned-policy evaluation utilities, and a basic policy-comparison CLI are implemented; richer house-level training objectives and production policy-selection workflows remain future work.

#### Deliverables

* richer probabilistic occupancy models
* uncertainty-aware planning hooks
* reinforcement-learning-compatible environment

#### Copilot-ready tasks

1. Refine state, action, reward, and episode logic for house-level control.
2. Add reward terms for cost, comfort, and switching.
3. Extend the evaluation script/CLI workflow to compare RL policy to hybrid and MPC baselines.
4. Expand the deterministic scenario library for broader seasonal and occupancy coverage.
5. Keep RL isolated from production control pathways.

#### Definition of done

* simulator is usable for research,
* RL can be experimented with safely,
* MPC remains the primary production strategy.

---

## 8. Detailed module specs for Copilot

### 8.1 Core config module

#### Responsibilities

* load YAML config,
* merge env vars,
* validate room schemas,
* expose typed config objects.

#### Key classes

* `AppConfig`
* `RoomConfig`
* `PricingConfig`
* `WeatherConfig`

#### Prompt for Copilot

> Create a typed configuration loader for IntelliWarm using Pydantic or dataclasses. It should load YAML from `config/default.yaml`, validate room definitions, and expose an `AppConfig` object to the Flask app and simulation services.

---

### 8.2 Thermal model module

#### Responsibilities

* predict room temperature changes,
* estimate room thermal coefficients,
* simulate future trajectories.

#### Required API

* `ThermalRoomModel.step(current_temp, outdoor_temp, heating_power, dt_minutes)`
* `ThermalRoomModel.simulate(initial_temp, forecast_inputs)`
* `ThermalParameterEstimator.fit(history_df)`

#### Prompt for Copilot

> Implement a room-level thermal model for IntelliWarm. Use a simple first-order heat equation with heating input and environmental loss. Include both single-step prediction and multi-step simulation methods. Add a parameter estimator using least squares on historical temperature data.

---

### 8.3 Occupancy module

#### Responsibilities

* represent schedules,
* estimate occupancy probability by time,
* expose occupancy forecasts for optimizer.

#### Required API

* `ScheduleOccupancyModel.predict(timestamp, room_id)`
* `ScheduleOccupancyModel.predict_horizon(start_time, horizon_steps, step_minutes)`

#### Prompt for Copilot

> Implement a schedule-based occupancy predictor for IntelliWarm. It should support day-of-week schedule windows, room-specific schedules, and horizon forecasts returning occupancy probability values from 0 to 1.

---

### 8.4 Baseline controller

#### Responsibilities

* provide stable fallback behavior,
* produce safe actions,
* generate human-readable rationale.

#### Required API

* `BaselineController.recommend(state)`
* `BaselineController.recommend_horizon(state, forecast)`

#### Prompt for Copilot

> Create a baseline HVAC controller for IntelliWarm that uses occupancy-aware temperature bands. The controller should recommend continuous normalized heat demand from 0 to 1, include compatibility labels such as OFF/ECO/COMFORT/PREHEAT for reporting, and explain each decision in plain language.

---

### 8.5 MPC controller

#### Responsibilities

* search candidate plans,
* evaluate rollouts,
* choose the best action sequence,
* return both the current action and forecast plan.

#### Required API

* `MPCController.plan(current_state, forecast_bundle)`
* `score_plan(plan, rollout)`

#### Prompt for Copilot

> Implement a model predictive control planner for IntelliWarm using a finite horizon and continuous normalized demand action space. Simulate future room temperatures, calculate energy cost and comfort penalties, penalize switching, and return the best plan plus explanation metadata.

---

### 8.6 Storage module

#### Responsibilities

* persist runs,
* store time-series data,
* support analytics and reports.

#### Key tables

* `rooms`
* `temperature_readings`
* `outdoor_readings`
* `occupancy_forecasts`
* `optimization_runs`
* `optimization_actions`
* `thermal_parameters`

#### Prompt for Copilot

> Build a persistence layer for IntelliWarm using SQLAlchemy. Include ORM models and repository methods for room configs, temperature readings, forecast data, optimization runs, and learned thermal parameters.

---

### 8.7 Dashboard module

#### Responsibilities

* expose routes,
* transform domain outputs to view models,
* render graphs and summaries.

#### Pages

* `/` dashboard overview,
* `/rooms` room status,
* `/simulate` run a simulation,
* `/reports` historical analytics,
* `/optimizer` current plan details.

#### Prompt for Copilot

> Refactor the IntelliWarm Flask UI into route modules and view models. Add pages for dashboard overview, room details, simulation runs, optimizer output, and reports. Keep route handlers thin and push business logic into services.

---

## 9. GitHub issue plan

Use these issue epics.

### Epic 1 — Foundation

* Initialize package layout
* Add typed config system
* Add linting, formatting, and tests
* Refactor app into modules

### Epic 2 — Simulation

* Add room domain models
* Add thermal simulator
* Add occupancy schedule model
* Add simulated sensors

### Epic 3 — Baseline control

* Add action enum and controller state
* Implement baseline controller
* Add decision explanations
* Add baseline dashboard cards

### Epic 4 — Forecasting

* Add thermal parameter estimator
* Add occupancy horizon prediction
* Add forecast bundle service
* Add evaluation notebooks

### Epic 5 — Optimization

* Add objective function
* Implement MPC planner
* Add runtime safeguards
* Compare baseline vs MPC

### Epic 6 — Persistence and reports

* Add DB schema
* Save optimization runs
* Add cost/savings reports
* Add historical charts

### Epic 7 — Integrations

* Add weather provider interface
* Add price provider interface
* Add simulated/live adapters
* Add safe device controller interface

### Epic 8 — Research and advanced control

* Add Gym wrapper
* Add RL experiment scripts
* Add uncertainty hooks
* Add benchmark suite

---

## 10. Suggested milestone order

### Milestone 1 — Make the repo real

Goal: package layout, configs, tests, modular app.

### Milestone 2 — Make the physics real

Goal: simulator, schedules, thermal model.

### Milestone 3 — Make the controls real

Goal: baseline controller plus explainability.

### Milestone 4 — Make it intelligent

Goal: forecasts and MPC optimization.

### Milestone 5 — Make it persistent

Goal: database, reports, analytics.

### Milestone 6 — Make it product-shaped

Goal: polished dashboard and integrations.

### Milestone 7 — Make it research-grade

Goal: RL environment and advanced experimentation.

---

## 11. Copilot operating rules

To get better output from Copilot, keep these constraints in the repo docs.

### Coding rules

* prefer typed Python,
* keep route handlers thin,
* isolate business logic in services,
* do not hardcode room names in controllers,
* require docstrings for public classes,
* add tests for every new module,
* prefer deterministic simulation over hidden randomness,
* avoid introducing live APIs unless behind interfaces.

### Architecture rules

* simulator must work offline,
* optimization must consume forecast bundles rather than direct API calls,
* dashboard must depend on services, not internal model details,
* every new provider should implement a base interface,
* RL experiments must not alter production control code.

### Pull request rules

Each PR should include:

* purpose,
* modules changed,
* tests added,
* screenshots if dashboard changed,
* notes on assumptions.

---

## 12. Copilot prompts file to include in the repo

`docs/copilot-prompts.md` is now part of the repo and should be kept aligned with the current roadmap.

### Prompt 1 — Refactor module

> Refactor the IntelliWarm Flask prototype into a modular service architecture. Keep route handlers thin, move business logic into `intelliwarm/services`, and preserve existing behavior.

### Prompt 2 — Build simulator

> Implement a deterministic multi-room thermal simulator for IntelliWarm. Each room should have temperature, target comfort band, heater capacity, occupancy schedule, and thermal coefficients. The simulator should step forward in fixed intervals and return time-series outputs.

### Prompt 3 — Build optimizer

> Implement a finite-horizon MPC optimizer for IntelliWarm using continuous room-demand control. Score plans based on energy cost, comfort violations, and switching penalties. Return the best current action and a structured explanation of the plan.

### Prompt 4 — Add tests

> Write pytest tests for the IntelliWarm thermal model and controllers. Cover heating, cooling, occupancy transitions, comfort-band logic, and baseline vs optimizer comparisons.

### Prompt 5 — Add dashboard page

> Add a Flask dashboard page for IntelliWarm that shows room temperatures, forecast temperatures, recommended actions, estimated cost, and comfort risk using thin routes and a view-model layer.

---

## 13. Acceptance criteria for the full MVP

The MVP is complete when:

1. the app can simulate a day across multiple rooms,
2. room schedules are configurable from YAML,
3. the baseline controller works consistently,
4. the MPC controller produces a better or comparable cost/comfort score,
5. optimization outputs are visible in the web UI,
6. simulation and optimization runs are stored,
7. tests cover major domain modules,
8. the repo is structured enough for Copilot to build features without ambiguity.

---

## 14. First 15 issues to create immediately

1. Set up `intelliwarm/` package structure
2. Add typed YAML configuration loader
3. Refactor `app.py` into route and service layers
4. Create room and schedule domain models
5. Implement deterministic thermal room model
6. Implement multi-room house simulator
7. Add simulated temperature and weather providers
8. Implement schedule-based occupancy predictor
9. Create heating action enum and controller state models
10. Implement baseline controller with explanations
11. Add objective function and cost calculator
12. Implement horizon forecast bundle service
13. Implement MPC planner with continuous normalized demand control
14. Add dashboard views for room temps and recommended actions
15. Add pytest unit tests for simulator and controllers

---

## 15. Recommended branch and workflow strategy

* `main` for stable code only
* `feature/foundation-*`
* `feature/simulator-*`
* `feature/controller-*`
* `feature/mpc-*`
* `feature/dashboard-*`
* `feature/storage-*`

For each issue:

1. paste the issue scope into Copilot,
2. ask it to propose a file-by-file implementation plan,
3. ask it to generate code only for the scoped files,
4. run tests,
5. then move to the next issue.

---

## 16. Final recommendation

The highest leverage move is **not** asking Copilot to “build IntelliWarm.”
The highest leverage move is asking Copilot to build **one bounded subsystem at a time** with typed models, explicit APIs, and tests.

That is how you turn Copilot from a code generator into a productive implementation partner for IntelliWarm.
