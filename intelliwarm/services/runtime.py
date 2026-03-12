"""
Application service layer for IntelliWarm runtime orchestration.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from intelliwarm.control import BaselineController, HybridController
from intelliwarm.data import ForecastBundle, HeatingAction, HybridHeatingDecision, RoomConfig, ZoneConfig
from intelliwarm.models import RoomThermalModel
from intelliwarm.optimizer import CostFunction, MPCController
from intelliwarm.prediction import OccupancyPredictor

from .forecast_bundle import ForecastBundleService
from .reporting import ReportService


class IntelliWarmRuntime:
    """Owns runtime state and orchestration for the Flask application."""

    def __init__(
        self,
        config,
        database,
        sensor_manager,
        device_controller,
        energy_service,
        forecast_service: Optional[ForecastBundleService] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.config = config
        self.database = database
        self.sensor_manager = sensor_manager
        self.device_controller = device_controller
        self.energy_service = energy_service
        self.forecast_service = forecast_service or ForecastBundleService(energy_service=energy_service)
        self.report_service = ReportService(database)
        self.logger = logger or logging.getLogger("IntelliWarm.Runtime")

        self.thermal_models: Dict[str, RoomThermalModel] = {}
        self.occupancy_predictors: Dict[str, OccupancyPredictor] = {}
        self.mpc_controllers: Dict[str, MPCController] = {}
        self.baseline_controllers: Dict[str, BaselineController] = {}
        self.hybrid_controllers: Dict[str, HybridController] = {}
        self.room_configs: Dict[str, Dict[str, Any]] = {}
        self.typed_room_configs: Dict[str, RoomConfig] = {}
        self.zone_configs: Dict[str, ZoneConfig] = {}
        self.last_room_plans: Dict[str, Dict[str, Any]] = {}

        self.rooms: List[Dict[str, Any]] = []
        self.zones: List[Dict[str, str]] = []
        self.utility_rates = {
            "electricity_price": self.config.electricity_price,
            "gas_price": self.config.gas_price,
        }
        self.last_cycle_summary: Dict[str, Any] = {
            "rooms_processed": 0,
            "successful_rooms": 0,
            "failed_rooms": 0,
            "safety_overrides": 0,
        }

        self.demo_data = pd.DataFrame()
        self.demo_timestamps: List[pd.Timestamp] = []
        self.demo_loaded = False

    @property
    def room_names(self) -> List[str]:
        """Return currently initialized room names."""
        return list(self.thermal_models.keys())

    def bootstrap_from_config(self):
        """Initialize configured zones and rooms from the typed config."""
        for zone_name, zone_config in self.config.zones.items():
            self.add_zone(zone_name, zone_config.get("description", ""))

        for room_name, room_config in self.config.rooms.items():
            zone_name = str(room_config.get("zone", "Unassigned"))
            if zone_name and not any(zone["name"] == zone_name for zone in self.zones):
                zone_details = self.config.zones.get(zone_name, {})
                self.add_zone(zone_name, zone_details.get("description", ""))

            initial_sensor_temp = room_config.get("initial_sensor_temp")
            if initial_sensor_temp is None:
                initial_sensor_temp = float(
                    room_config.get("target_temp", self.config.default_target_temp)
                )

            self.add_room(
                name=room_name,
                room_size=float(room_config.get("room_size", 150.0)),
                zone=zone_name,
                room_config=room_config,
                initial_sensor_temp=float(initial_sensor_temp),
                initial_occupancy=bool(room_config.get("initial_occupancy", False)),
                display_temp_f=room_config.get("display_temp_f"),
                humidity=float(room_config.get("humidity", 45.0)),
                heating_source=str(room_config.get("heating_source", "Off")),
            )

    def initialize_room_models(self, room_name: str, room_config: Dict[str, Any]):
        """Initialize room-specific predictive and control models."""
        resolved_room_config = self._resolve_room_config(room_name, room_config.get("zone"), room_config)
        typed_room_config = RoomConfig.from_legacy_config(
            room_name,
            resolved_room_config,
            default_name=room_name,
        )
        self.room_configs[room_name] = dict(resolved_room_config)

        self.thermal_models[room_name] = RoomThermalModel(
            room_name,
            alpha=resolved_room_config.get("heating_efficiency", 0.1),
            beta=resolved_room_config.get("thermal_mass", 0.05),
        )

        self.occupancy_predictors[room_name] = OccupancyPredictor(
            room_name,
            schedule=resolved_room_config.get("occupancy_schedule", ""),
        )

        self.mpc_controllers[room_name] = MPCController(
            self.config,
            self.thermal_models[room_name],
            CostFunction(self.config),
        )
        self.baseline_controllers[room_name] = BaselineController(
            room_config=typed_room_config,
            min_temperature=self.config.min_temperature,
            max_temperature=self.config.max_temperature,
        )
        self.typed_room_configs[room_name] = typed_room_config
        self._rebuild_zone_controller(typed_room_config.zone)

        self.device_controller.register_device(room_name)
        self.logger.info("Initialized models for room: %s", room_name)

    def _resolve_room_config(
        self,
        room_name: Optional[str],
        zone: Optional[str],
        room_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        overrides = room_config or {}
        if hasattr(self.config, "build_room_config"):
            return self.config.build_room_config(
                room_name=room_name,
                zone=zone,
                overrides=overrides,
            )

        base_config = {}
        if hasattr(self.config, "get_room_config") and room_name:
            base_config = dict(self.config.get_room_config(room_name) or {})

        resolved = {
            "zone": zone or base_config.get("zone", "Unassigned"),
            "room_size": float(base_config.get("room_size", 150.0)),
            "target_temp": float(base_config.get("target_temp", self.config.default_target_temp)),
            "heater_power": float(base_config.get("heater_power", 1500.0)),
            "thermal_mass": float(base_config.get("thermal_mass", 0.05)),
            "heating_efficiency": float(base_config.get("heating_efficiency", 0.85)),
            "occupancy_schedule": base_config.get("occupancy_schedule", ""),
            "initial_sensor_temp": base_config.get("initial_sensor_temp"),
            "initial_occupancy": bool(base_config.get("initial_occupancy", False)),
            "display_temp_f": base_config.get("display_temp_f"),
            "humidity": float(base_config.get("humidity", 45.0)),
            "heating_source": str(base_config.get("heating_source", "Off")),
            "heat_source": str(base_config.get("heat_source", "electric")),
        }
        resolved.update(dict(overrides))
        return resolved

    def _build_zone_config(self, zone_name: str) -> ZoneConfig:
        config_zones = getattr(self.config, "zones", {}) or {}
        zone_details = dict(config_zones.get(zone_name, {})) if isinstance(config_zones, dict) else {}
        runtime_zone = next((zone for zone in self.zones if zone["name"] == zone_name), {})
        return ZoneConfig(
            zone_id=zone_name,
            description=str(zone_details.get("description", runtime_zone.get("description", ""))),
            priority=int(zone_details.get("priority", 0)),
            has_furnace=bool(zone_details.get("has_furnace", False)),
            furnace_btu_per_hour=float(zone_details.get("furnace_btu_per_hour", 60000.0)),
            furnace_efficiency=float(zone_details.get("furnace_efficiency", 0.80)),
        )

    def _rebuild_zone_controller(self, zone_name: str):
        zone_room_configs = {
            room_id: room_config
            for room_id, room_config in self.typed_room_configs.items()
            if room_config.zone == zone_name
        }
        if not zone_room_configs:
            self.hybrid_controllers.pop(zone_name, None)
            self.zone_configs.pop(zone_name, None)
            return

        zone_config = self._build_zone_config(zone_name)
        self.zone_configs[zone_name] = zone_config
        self.hybrid_controllers[zone_name] = HybridController(
            zone_config=zone_config,
            room_configs=zone_room_configs,
            min_temperature=self.config.min_temperature,
            max_temperature=self.config.max_temperature,
        )

    def add_zone(self, name: str, description: str = ""):
        """Add a zone if it is not already present."""
        if any(zone["name"] == name for zone in self.zones):
            return

        self.zones.append({"name": name, "description": description})
        self.device_controller.register_furnace(name)
        self.logger.info("Zone added: %s", name)

    def update_utility_rates(
        self,
        electricity_price: Optional[float] = None,
        gas_price: Optional[float] = None,
    ):
        """Update in-memory utility rates and the pricing service."""
        if electricity_price is not None:
            self.energy_service.set_electricity_price(electricity_price)
            self.utility_rates["electricity_price"] = electricity_price

        if gas_price is not None:
            self.energy_service.set_gas_price(gas_price)
            self.utility_rates["gas_price"] = gas_price

    def add_room(
        self,
        name: str,
        room_size: float,
        zone: str,
        room_config: Optional[Dict[str, Any]] = None,
        initial_sensor_temp: float = 20.0,
        initial_occupancy: bool = False,
        display_temp_f: Optional[float] = None,
        humidity: float = 45.0,
        heating_source: str = "Off",
    ):
        """Create a room across persistence, sensors, control, and UI state."""
        resolved_room_config = self._resolve_room_config(name, zone, room_config)
        resolved_room_size = float(room_size)
        if zone and not any(existing_zone["name"] == zone for existing_zone in self.zones):
            zone_details = getattr(self.config, "zones", {}).get(zone, {}) if hasattr(self.config, "zones") else {}
            self.add_zone(zone, str(zone_details.get("description", "")))

        self.database.add_room(
            name,
            zone,
            resolved_room_size,
            resolved_room_config.get("target_temp", self.config.default_target_temp),
            resolved_room_config.get("heater_power", 1500),
        )

        self.sensor_manager.register_temperature_sensor(name, initial_sensor_temp)
        self.sensor_manager.register_occupancy_sensor(name, initial_occupancy)
        self.initialize_room_models(name, resolved_room_config)

        self.rooms = [room for room in self.rooms if room.get("name") != name]
        self.rooms.append(
            {
                "name": name,
                "room_size": resolved_room_size,
                "zone": zone,
                "temperature": display_temp_f if display_temp_f is not None else initial_sensor_temp,
                "humidity": humidity,
                "occupancy": initial_occupancy,
                "heating_source": heating_source,
            }
        )
        self.last_room_plans.pop(name, None)
        self.logger.info("Room added: %s", name)

    def load_demo_dataset(self, file_path: str = "data-analytics/roommate_data/roommates_occupancy.csv") -> bool:
        """Load demo CSV data and initialize runtime state."""
        self.rooms.clear()
        self.zones.clear()
        self.thermal_models.clear()
        self.occupancy_predictors.clear()
        self.mpc_controllers.clear()
        self.baseline_controllers.clear()
        self.hybrid_controllers.clear()
        self.room_configs.clear()
        self.typed_room_configs.clear()
        self.zone_configs.clear()
        self.last_room_plans.clear()

        if not pd.io.common.file_exists(file_path):
            self.logger.error("Demo data file not found: %s", file_path)
            self.demo_loaded = False
            return False

        try:
            data = pd.read_csv(file_path)
            data["timestamp"] = pd.to_datetime(data["timestamp"])
            data["occupied"] = data["occupied"].astype(int)
            data["zone"] = data["zone"].astype(str)

            self.demo_data = data
            self.demo_timestamps = sorted(data["timestamp"].dropna().drop_duplicates().tolist())

            for zone in sorted(data["zone"].unique().tolist()):
                self.add_zone(f"Zone {zone}", f"Zone {zone} description")

            first_timestamp = data["timestamp"].min()
            first_snapshot = data[data["timestamp"] == first_timestamp]
            for _, row in first_snapshot.iterrows():
                occupied = bool(int(row["occupied"]))
                self.add_room(
                    name=row["room"],
                    room_size=150,
                    zone=f"Zone {row['zone']}",
                    room_config={
                        "zone": f"Zone {row['zone']}",
                        "target_temp": 21,
                        "heating_efficiency": 0.85,
                        "thermal_mass": 0.05,
                        "occupancy_schedule": "9-18",
                    },
                    initial_sensor_temp=21.0 if occupied else 18.5,
                    initial_occupancy=occupied,
                    display_temp_f=70 if occupied else 66,
                    humidity=44,
                    heating_source="Furnace" if occupied else "Off",
                )

            self.demo_loaded = True
            self.logger.info(
                "Demo loaded: %s rooms, %s timeline points",
                len(self.rooms),
                len(self.demo_timestamps),
            )
            return True
        except Exception as exc:
            self.logger.error("Demo loading failed: %s", exc)
            self.demo_loaded = False
            return False

    @staticmethod
    def _heat_source_label(heat_source: Optional[str]) -> str:
        labels = {
            "electric": "Electric Heater",
            "gas_furnace": "Gas Furnace",
        }
        return labels.get(str(heat_source or "electric"), str(heat_source or "electric").replace("_", " ").title())

    def _available_heat_sources(self, zone_name: str, room_config: Dict[str, Any]) -> List[str]:
        zone_config = self.zone_configs.get(zone_name) or self._build_zone_config(zone_name)
        sources = ["electric"]
        if zone_config.has_furnace:
            sources.append("gas_furnace")

        configured_source = str(room_config.get("heat_source", "electric"))
        if configured_source not in sources:
            sources.append(configured_source)
        return sources

    def get_rooms_api_data(self) -> List[Dict[str, Any]]:
        """Build room API payload from the current runtime state."""
        rooms_data: List[Dict[str, Any]] = []
        for room_name in self.room_names:
            state = self.sensor_manager.get_room_state(room_name)
            device_status = self.device_controller.get_device_status(room_name) or {}
            room_config = self.room_configs.get(room_name, self.config.get_room_config(room_name))
            room_snapshot = next((room for room in self.rooms if room.get("name") == room_name), {})
            latest_plan = dict(self.last_room_plans.get(room_name, {}))
            zone_name = str(room_config.get("zone", "Unknown"))
            configured_heat_source = str(room_config.get("heat_source", "electric"))
            available_heat_sources = self._available_heat_sources(zone_name, room_config)
            active_heat_source = str(latest_plan.get("heat_source", configured_heat_source))

            rooms_data.append(
                {
                    **state,
                    "name": room_name,
                    "room": room_name,
                    "zone": zone_name,
                    "temperature": state.get("temperature"),
                    "target_temp": float(room_config.get("target_temp", self.config.default_target_temp)),
                    "humidity": room_snapshot.get("humidity", room_config.get("humidity", 45.0)),
                    "device_status": device_status,
                    "sensor_source": state.get("sensor_source", "simulated"),
                    "control_source": device_status.get("control_source", "simulated"),
                    "configured_heat_source": configured_heat_source,
                    "configured_heat_source_label": self._heat_source_label(configured_heat_source),
                    "available_heat_sources": available_heat_sources,
                    "available_heat_source_labels": [self._heat_source_label(source) for source in available_heat_sources],
                    "active_heat_source": active_heat_source,
                    "active_heat_source_label": self._heat_source_label(active_heat_source),
                    "controller": latest_plan.get("controller"),
                    "recommended_mode": latest_plan.get("hybrid_action_label", latest_plan.get("next_action_label")),
                    "applied_mode": latest_plan.get("next_action_label"),
                    "furnace_on": bool(latest_plan.get("furnace_on", False)),
                    "total_cost": latest_plan.get("total_cost"),
                    "explanation": latest_plan.get("explanation"),
                    "latest_plan": latest_plan or None,
                }
            )

        return rooms_data

    def get_zone_status_data(self) -> List[Dict[str, Any]]:
        """Build zone-level status for dashboard/operator surfaces."""
        zone_to_rooms: Dict[str, List[str]] = {
            str(zone.get("name", "Unknown")): []
            for zone in self.zones
        }
        for room_name in self.room_names:
            zone_name = str(self.room_configs.get(room_name, {}).get("zone", "Unknown"))
            zone_to_rooms.setdefault(zone_name, []).append(room_name)

        zone_data: List[Dict[str, Any]] = []
        for zone_name in sorted(zone_to_rooms):
            zone_config = self.zone_configs.get(zone_name) or self._build_zone_config(zone_name)
            zone_plan = next(
                (
                    self.last_room_plans[room_name]
                    for room_name in zone_to_rooms[zone_name]
                    if room_name in self.last_room_plans
                ),
                {},
            )
            hybrid_decision = zone_plan.get("hybrid_decision", {})
            active_heat_source = zone_plan.get("heat_source")
            furnace_status = self.device_controller.get_zone_furnace_status(zone_name)
            zone_data.append(
                {
                    "name": zone_name,
                    "description": zone_config.description,
                    "room_names": list(zone_to_rooms[zone_name]),
                    "room_count": len(zone_to_rooms[zone_name]),
                    "has_furnace": zone_config.has_furnace,
                    "available_heat_sources": ["electric", "gas_furnace"] if zone_config.has_furnace else ["electric"],
                    "active_heat_source": active_heat_source,
                    "active_heat_source_label": self._heat_source_label(active_heat_source) if active_heat_source else "No plan yet",
                    "recommended_mode": zone_plan.get("hybrid_action_label"),
                    "applied_mode": zone_plan.get("next_action_label"),
                    "furnace_on": bool(zone_plan.get("furnace_on", False)),
                    "hourly_cost": zone_plan.get("total_cost"),
                    "rooms_needing_heat": hybrid_decision.get("rooms_needing_heat", []),
                    "rationale": hybrid_decision.get("rationale", zone_plan.get("explanation")),
                    "hybrid_decision": hybrid_decision or None,
                    "furnace_status": furnace_status,
                }
            )

        return zone_data

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Build a dashboard-oriented runtime view model."""
        rooms = self.get_rooms_api_data()
        zones = self.get_zone_status_data()
        runtime_status = self.get_runtime_status(event_limit=5)
        return {
            "runtime": {
                "mode": runtime_status["mode"],
                "hardware_control_enabled": runtime_status["hardware_control_enabled"],
                "room_count": len(rooms),
                "zone_count": len(zones),
                "optimized_room_count": sum(1 for room in rooms if room.get("latest_plan")),
                "last_cycle_summary": runtime_status["last_cycle_summary"],
            },
            "zones": zones,
            "rooms": rooms,
        }

    def _room_names_for_zone(self, zone_name: str) -> List[str]:
        return [
            room_name
            for room_name, room_config in self.room_configs.items()
            if str(room_config.get("zone", "Unassigned")) == zone_name
        ]

    def build_forecast_bundle(
        self,
        room_name: str,
        occupancy_override: Optional[List[float]] = None,
        outdoor_temp_override: Optional[List[float]] = None,
        start_time=None,
    ) -> Optional[ForecastBundle]:
        """Build an aligned forecast bundle for a room."""
        if room_name not in self.occupancy_predictors:
            self.logger.warning("Forecast requested for uninitialized room: %s", room_name)
            return None

        bundle = self.forecast_service.build_bundle(
            room_name=room_name,
            occupancy_predictor=self.occupancy_predictors[room_name],
            horizon_steps=self.config.optimization_horizon,
            start_time=start_time,
        )
        if occupancy_override is not None or outdoor_temp_override is not None:
            bundle = self.forecast_service.override_bundle(
                bundle,
                occupancy_probabilities=occupancy_override,
                outdoor_temperatures=outdoor_temp_override,
            )
        return bundle

    def _record_runtime_event(
        self,
        event_type: str,
        severity: str,
        message: str,
        room_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.database.record_runtime_event(
            event_type=event_type,
            severity=severity,
            message=message,
            room_name=room_name,
            details=details,
        )
        log_method = getattr(self.logger, severity.lower(), self.logger.info)
        log_method("%s%s", f"[{room_name}] " if room_name else "", message)

    def _apply_safety_constraints(
        self,
        room_name: str,
        current_temp: float,
        plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        adjusted_plan = dict(plan)
        proposed_action = HeatingAction.from_value(adjusted_plan["next_action"])

        if current_temp >= self.config.max_temperature:
            adjusted_plan["next_action"] = HeatingAction.OFF.power_level
            adjusted_plan["next_action_label"] = HeatingAction.OFF.name
            adjusted_plan["safety_override"] = "overheat_protection"
            self._record_runtime_event(
                event_type="safety_override",
                severity="warning",
                room_name=room_name,
                message="Overheat protection forced heater OFF.",
                details={
                    "current_temp": current_temp,
                    "max_temperature": self.config.max_temperature,
                },
            )
            return adjusted_plan

        if (
            current_temp <= self.config.min_temperature - 1.0
            and proposed_action == HeatingAction.OFF
            and not adjusted_plan.get("furnace_on", False)
        ):
            adjusted_plan["next_action"] = HeatingAction.ECO.power_level
            adjusted_plan["next_action_label"] = HeatingAction.ECO.name
            adjusted_plan["safety_override"] = "freeze_protection"
            self._record_runtime_event(
                event_type="safety_override",
                severity="warning",
                room_name=room_name,
                message="Freeze protection raised heater output to ECO.",
                details={
                    "current_temp": current_temp,
                    "min_temperature": self.config.min_temperature,
                },
            )

        return adjusted_plan

    def _build_hybrid_plan(
        self,
        room_name: str,
        room_config: Dict[str, Any],
        forecast_bundle: ForecastBundle,
        target_temp_override: Optional[float],
        current_action_override: Optional[float],
    ) -> Optional[Dict[str, Any]]:
        zone_name = str(room_config.get("zone", "Unassigned"))
        hybrid_controller = self.hybrid_controllers.get(zone_name)
        if hybrid_controller is None:
            self._rebuild_zone_controller(zone_name)
            hybrid_controller = self.hybrid_controllers.get(zone_name)
        if hybrid_controller is None:
            self.logger.warning("Zone controller not initialized: %s", zone_name)
            return None

        room_temperatures: Dict[str, float] = {}
        occupancy_forecasts: Dict[str, List[float]] = {}
        current_actions: Dict[str, float] = {}
        target_temps: Dict[str, float] = {}

        electricity_price = forecast_bundle.electricity_prices[0] if forecast_bundle.steps else self.energy_service.electricity_price
        gas_price = forecast_bundle.gas_prices[0] if forecast_bundle.steps else self.energy_service.gas_price
        outside_temp = forecast_bundle.outdoor_temperatures[0] if forecast_bundle.steps else 5.0

        for zone_room_name in hybrid_controller.room_ids():
            zone_room_config = self.room_configs.get(zone_room_name, self.config.get_room_config(zone_room_name))
            zone_bundle = forecast_bundle if zone_room_name == room_name else self.build_forecast_bundle(zone_room_name)
            if zone_bundle is None:
                return None

            current_temp = self.sensor_manager.get_temperature(zone_room_name)
            room_temperatures[zone_room_name] = (
                float(current_temp)
                if current_temp is not None
                else float(zone_room_config.get("target_temp", self.config.default_target_temp))
            )
            occupancy_forecasts[zone_room_name] = zone_bundle.occupancy_probabilities
            zone_device_status = self.device_controller.get_device_status(zone_room_name)
            current_actions[zone_room_name] = (
                float(current_action_override)
                if zone_room_name == room_name and current_action_override is not None
                else float(zone_device_status["power_level"]) if zone_device_status else 0.0
            )
            target_temps[zone_room_name] = (
                float(target_temp_override)
                if zone_room_name == room_name and target_temp_override is not None
                else float(zone_room_config.get("target_temp", self.config.default_target_temp))
            )

        hybrid_decision = hybrid_controller.decide(
            room_temperatures=room_temperatures,
            occupancy_forecasts=occupancy_forecasts,
            electricity_price=electricity_price,
            gas_price=gas_price,
            outside_temp=outside_temp,
            current_actions=current_actions,
            target_temps=target_temps,
        )
        return self._hybrid_plan_to_runtime_plan(
            room_name=room_name,
            hybrid_decision=hybrid_decision,
            forecast_bundle=forecast_bundle,
            current_temp=room_temperatures[room_name],
        )

    def _hybrid_plan_to_runtime_plan(
        self,
        room_name: str,
        hybrid_decision: HybridHeatingDecision,
        forecast_bundle: ForecastBundle,
        current_temp: float,
    ) -> Dict[str, Any]:
        requested_action = hybrid_decision.per_room_actions[room_name]
        applied_action = HeatingAction.OFF if hybrid_decision.furnace_on else requested_action
        plan = {
            "room": room_name,
            "controller": "hybrid",
            "zone": hybrid_decision.zone,
            "next_action": applied_action.power_level,
            "next_action_label": applied_action.name,
            "hybrid_action": requested_action.power_level,
            "hybrid_action_label": requested_action.name,
            "furnace_on": hybrid_decision.furnace_on,
            "heat_source": hybrid_decision.heat_source.value,
            "total_cost": hybrid_decision.chosen_hourly_cost,
            "optimal_actions": [applied_action.power_level] * len(forecast_bundle.steps),
            "predicted_temperatures": [current_temp],
            "horizon": len(forecast_bundle.steps),
            "forecast_bundle": forecast_bundle.to_dict(),
            "hybrid_decision": hybrid_decision.to_dict(),
            "metadata": {
                "electric_command_suppressed": hybrid_decision.furnace_on,
            },
        }
        if hybrid_decision.furnace_on:
            plan["explanation"] = (
                "Hybrid controller selected the zone furnace; per-room electric heaters were "
                "suppressed while the zone furnace actuator was enabled."
            )
        else:
            plan["explanation"] = hybrid_decision.rationale
        return plan

    def _apply_device_plan(self, room_name: str, plan: Dict[str, Any]):
        zone_name = str(plan.get("zone", self.room_configs.get(room_name, {}).get("zone", "Unassigned")))
        if plan.get("controller") == "hybrid":
            if plan.get("furnace_on", False):
                furnace_level = float(plan.get("hybrid_action", 1.0))
                self.device_controller.set_zone_furnace(zone_name, furnace_level)
                for zone_room_name in self._room_names_for_zone(zone_name):
                    self.device_controller.turn_off(zone_room_name)
            else:
                self.device_controller.turn_off_zone_furnace(zone_name)
                self.device_controller.set_heater(room_name, plan["next_action"])
            return

        self.device_controller.turn_off_zone_furnace(zone_name)
        self.device_controller.set_heater(room_name, plan["next_action"])

    def optimize_heating_plan(
        self,
        room_name: str,
        occupancy_override: Optional[List[float]] = None,
        target_temp_override: Optional[float] = None,
        current_action_override: Optional[float] = None,
        controller_type: str = "hybrid",
    ) -> Optional[Dict[str, Any]]:
        """Compute and apply the next heating action for a room."""
        if room_name not in self.thermal_models:
            self.logger.warning("Room not initialized: %s", room_name)
            return None

        try:
            current_temp = self.sensor_manager.get_temperature(room_name)
            if current_temp is None:
                self._record_runtime_event(
                    event_type="sensor_error",
                    severity="error",
                    room_name=room_name,
                    message="Temperature reading unavailable; optimization skipped.",
                )
                return None

            room_config = self.room_configs.get(room_name, self.config.get_room_config(room_name))
            target_temp = (
                target_temp_override
                if target_temp_override is not None
                else room_config.get("target_temp", self.config.default_target_temp)
            )
            forecast_bundle = self.build_forecast_bundle(
                room_name,
                occupancy_override=occupancy_override,
            )
            if forecast_bundle is None:
                return None

            occupancy_probs = forecast_bundle.occupancy_probabilities
            outside_temp = forecast_bundle.outdoor_temperatures[0] if forecast_bundle.steps else 5.0
            energy_prices = forecast_bundle.electricity_prices

            device_status = self.device_controller.get_device_status(room_name)
            current_action = (
                current_action_override
                if current_action_override is not None
                else (device_status["power_level"] if device_status else 0.0)
            )

            if controller_type in {"baseline", "hybrid"}:
                plan = self._build_hybrid_plan(
                    room_name=room_name,
                    room_config=room_config,
                    forecast_bundle=forecast_bundle,
                    target_temp_override=target_temp_override,
                    current_action_override=current_action_override,
                )
            else:
                plan = self.mpc_controllers[room_name].compute_optimal_plan(
                    room_name,
                    current_temp,
                    outside_temp,
                    target_temp,
                    energy_prices,
                    occupancy_probs,
                    current_action,
                )

            if plan and "next_action" in plan:
                if plan.get("controller") == "hybrid":
                    controller_type = "hybrid"
                plan.setdefault("controller", controller_type)
                plan.setdefault("next_action_label", HeatingAction.from_value(plan["next_action"]).name)
                plan = self._apply_safety_constraints(room_name, current_temp, plan)
                self._apply_device_plan(room_name, plan)
                self.database.record_optimization(
                    room_name,
                    plan["next_action"],
                    plan["total_cost"],
                    controller_type=plan["controller"],
                    action_label=plan["next_action_label"],
                )
                plan.setdefault("forecast_bundle", forecast_bundle.to_dict())
                self.last_room_plans[room_name] = dict(plan)

            return plan
        except Exception as exc:
            self.logger.error("Optimization failed for %s: %s", room_name, exc)
            return None

    def get_demo_timeline_meta(self) -> Optional[Dict[str, Any]]:
        """Return summary metadata for the demo timeline."""
        if not self.demo_loaded and not self.load_demo_dataset():
            return None

        room_names = sorted(self.demo_data["room"].unique().tolist()) if not self.demo_data.empty else []
        return {
            "total_points": len(self.demo_timestamps),
            "start": str(self.demo_timestamps[0]) if self.demo_timestamps else None,
            "end": str(self.demo_timestamps[-1]) if self.demo_timestamps else None,
            "rooms": room_names,
        }

    def get_demo_timeline_point(self, index: int) -> Optional[Dict[str, Any]]:
        """Return a single demo timeline point with simulated actions and costs."""
        if not self.demo_loaded and not self.load_demo_dataset():
            return None

        if not self.demo_timestamps:
            return None

        bounded_index = max(0, min(index, len(self.demo_timestamps) - 1))
        timestamp = self.demo_timestamps[bounded_index]
        frame = self.demo_data[self.demo_data["timestamp"] == timestamp]
        zone_occupied_map = frame.groupby("zone")["occupied"].max().to_dict()

        rooms_state: List[Dict[str, Any]] = []
        actions: List[float] = []
        costs: List[float] = []

        for _, row in frame.iterrows():
            room_name = row["room"]
            occupied = bool(int(row["occupied"]))
            zone_name = row["zone"]
            zone_occupied = bool(int(zone_occupied_map.get(zone_name, 0)))

            current_temp_c = 21.0 if occupied else 18.5
            self.sensor_manager.set_occupancy(room_name, occupied)
            self.sensor_manager.set_temperature(room_name, current_temp_c)

            occupancy_horizon = [1.0 if occupied else 0.0] * self.config.optimization_horizon
            target_temp = 21.0 if occupied else self.config.min_temperature

            plan = self.optimize_heating_plan(
                room_name,
                occupancy_override=occupancy_horizon,
                target_temp_override=target_temp,
                current_action_override=0.0,
            )
            next_action = plan["next_action"] if plan else 0.0
            total_cost = float(plan["total_cost"]) if plan else 0.0
            heat_source = self._heat_source_label(plan.get("heat_source")) if plan else "Electric Heater"
            mode = plan.get("hybrid_action_label", plan.get("next_action_label", "OFF")) if plan else "OFF"

            if not zone_occupied:
                next_action = 0.0
                heat_source = "Off"
                mode = "OFF"

            actions.append(next_action)
            costs.append(total_cost)

            current_temp_f = round((current_temp_c * 9 / 5) + 32, 1)
            rooms_state.append(
                {
                    "room": room_name,
                    "zone": f"Zone {zone_name}",
                    "occupied": occupied,
                    "temperature_f": current_temp_f,
                    "next_action": round(next_action, 3),
                    "heating_source": heat_source,
                    "mode": mode,
                    "predicted_cost": round(total_cost, 3),
                }
            )

        avg_action = float(sum(actions) / len(actions)) if actions else 0.0
        total_cost = float(sum(costs)) if costs else 0.0

        return {
            "index": bounded_index,
            "timestamp": str(timestamp),
            "summary": {
                "occupied_rooms": int(sum(1 for room in rooms_state if room["occupied"])),
                "total_rooms": len(rooms_state),
                "avg_heating_action": round(avg_action, 3),
                "total_predicted_cost": round(total_cost, 3),
            },
            "rooms": rooms_state,
        }

    def run_optimization_cycle(self):
        """Run one optimization pass across all initialized rooms."""
        successful_rooms = 0
        failed_rooms = 0
        safety_overrides = 0
        for room_name in self.room_names:
            plan = self.optimize_heating_plan(room_name)
            if plan is None:
                failed_rooms += 1
                continue

            successful_rooms += 1
            if plan.get("safety_override"):
                safety_overrides += 1

        self.last_cycle_summary = {
            "rooms_processed": len(self.room_names),
            "successful_rooms": successful_rooms,
            "failed_rooms": failed_rooms,
            "safety_overrides": safety_overrides,
        }
        self._record_runtime_event(
            event_type="optimization_cycle",
            severity="info",
            message="Completed optimization cycle.",
            details=self.last_cycle_summary,
        )

    def get_room_report(self, room_name: str, limit: int = 10) -> Optional[Dict[str, Any]]:
        """Return a persisted report for a single room."""
        return self.report_service.build_room_report(room_name, limit=limit)

    def get_portfolio_report(self, limit_per_room: int = 5) -> Dict[str, Any]:
        """Return an aggregated report across all rooms."""
        return self.report_service.build_portfolio_report(limit_per_room=limit_per_room)

    def get_runtime_status(self, event_limit: int = 10) -> Dict[str, Any]:
        """Return deployment-oriented runtime status and recent events."""
        hardware_control_enabled = bool(getattr(self.config, "enable_device_control", False))
        return {
            "mode": "live" if hardware_control_enabled else "simulation",
            "hardware_control_enabled": hardware_control_enabled,
            "room_count": len(self.room_names),
            "last_cycle_summary": dict(self.last_cycle_summary),
            "recent_events": self.database.get_recent_runtime_events(limit=event_limit),
        }
