# Copilot Prompts

Use these prompts for bounded implementation slices.

## 1. Baseline Controller

> Implement a baseline HVAC controller for IntelliWarm using the existing `HeatingAction` enum in `intelliwarm/data/models.py`. Keep the logic deterministic, occupancy-aware, and valid for both simulation and live execution contracts. Add explanation outputs, add tests, and do not change the Flask UI yet.

## 2. Forecast Bundle Service

> Add a forecast bundle service to IntelliWarm that aligns occupancy, outdoor temperature, and energy prices on a shared horizon contract. Reuse existing occupancy and pricing modules. Keep the service independent from Flask routes, make outputs usable by both simulator and hardware adapters, and add focused tests.

## 3. Hardware Adapter Boundary

> Add or refine an adapter boundary for IntelliWarm real-world integrations, including sensor input and HVAC device actuation interfaces. Keep hardware-specific logic out of route handlers and optimizer internals, provide simulation fallback behavior when hardware is unavailable, and add tests using fakes/mocks.

## 4. Route Modularization

> Refactor the Flask entrypoint in IntelliWarm into thin route modules while preserving behavior. Keep orchestration in `intelliwarm/services/runtime.py`, avoid changing the simulation or optimizer behavior, and add route-level tests for any moved behavior.

## 5. Typed Config Evolution

> Introduce typed configuration models for IntelliWarm on top of `configs/config.yaml`. Preserve current behavior, avoid a broad config rewrite, include deployment-oriented settings for hardware mode/safety toggles where relevant, and add tests that validate room configuration parsing and defaults.

## 6. Persistence Upgrade

> Extend IntelliWarm persistence so optimization runs, forecasts, learned parameters, and live runtime telemetry can be queried for reporting. Keep the current SQLite workflow unless a migration is explicitly requested. Add integration tests for the new repository methods.

## 7. MPC Evolution

> Evolve the existing MPC implementation in IntelliWarm toward discrete, explainable actions while preserving the current optimizer module. Reuse `HeatingAction`, add rollout scoring, ensure compatibility with live adapter contracts, and compare output against the baseline controller in tests.