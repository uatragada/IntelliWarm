"""
Application service layer for IntelliWarm runtime orchestration.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from intelliwarm.control import BaselineController
from intelliwarm.data import ForecastBundle, HeatingAction, RoomConfig
from intelliwarm.models import RoomThermalModel
from intelliwarm.optimizer import CostFunction, MPCController
from intelliwarm.prediction import OccupancyPredictor

from .forecast_bundle import ForecastBundleService


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
        self.logger = logger or logging.getLogger("IntelliWarm.Runtime")

        self.thermal_models: Dict[str, RoomThermalModel] = {}
        self.occupancy_predictors: Dict[str, OccupancyPredictor] = {}
        self.mpc_controllers: Dict[str, MPCController] = {}
        self.baseline_controllers: Dict[str, BaselineController] = {}
        self.room_configs: Dict[str, Dict[str, Any]] = {}

        self.rooms: List[Dict[str, Any]] = []
        self.zones: List[Dict[str, str]] = []
        self.utility_rates = {
            "electricity_price": self.config.electricity_price,
            "gas_price": self.config.gas_price,
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
        }
        resolved.update(dict(overrides))
        return resolved

    def add_zone(self, name: str, description: str = ""):
        """Add a zone if it is not already present."""
        if any(zone["name"] == name for zone in self.zones):
            return

        self.zones.append({"name": name, "description": description})
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
        self.logger.info("Room added: %s", name)

    def load_demo_dataset(self, file_path: str = "data-analytics/roommate_data/roommates_occupancy.csv") -> bool:
        """Load demo CSV data and initialize runtime state."""
        self.rooms.clear()
        self.zones.clear()
        self.thermal_models.clear()
        self.occupancy_predictors.clear()
        self.mpc_controllers.clear()
        self.room_configs.clear()

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

    def get_rooms_api_data(self) -> List[Dict[str, Any]]:
        """Build room API payload from the current runtime state."""
        rooms_data: List[Dict[str, Any]] = []
        for room_name in self.room_names:
            state = self.sensor_manager.get_room_state(room_name)
            device_status = self.device_controller.get_device_status(room_name)
            room_config = self.room_configs.get(room_name, self.config.get_room_config(room_name))

            rooms_data.append(
                {
                    **state,
                    "zone": room_config.get("zone", "Unknown"),
                    "target_temp": room_config.get("target_temp", self.config.default_target_temp),
                    "device_status": device_status,
                }
            )

        return rooms_data

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

    def optimize_heating_plan(
        self,
        room_name: str,
        occupancy_override: Optional[List[float]] = None,
        target_temp_override: Optional[float] = None,
        current_action_override: Optional[float] = None,
        controller_type: str = "mpc",
    ) -> Optional[Dict[str, Any]]:
        """Compute and apply the next heating action for a room."""
        if room_name not in self.thermal_models:
            self.logger.warning("Room not initialized: %s", room_name)
            return None

        try:
            current_temp = self.sensor_manager.get_temperature(room_name)
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

            if controller_type == "baseline":
                decision = self.baseline_controllers[room_name].compute_decision(
                    current_temp=current_temp,
                    occupancy_forecast=occupancy_probs,
                    energy_prices=energy_prices,
                    current_action=current_action,
                    outside_temp=outside_temp,
                    target_temp=target_temp,
                )
                plan = decision.to_dict()
                plan["optimal_actions"] = [decision.action.power_level] * len(occupancy_probs)
                plan["predicted_temperatures"] = [current_temp]
                plan["horizon"] = len(occupancy_probs)
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
                self.device_controller.set_heater(room_name, plan["next_action"])
                self.database.record_optimization(room_name, plan["next_action"], plan["total_cost"])
                plan["forecast_bundle"] = forecast_bundle.to_dict()

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

            if not zone_occupied:
                next_action = 0.0

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
                    "heating_source": "Furnace" if next_action >= 0.35 else "Off",
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
        for room_name in self.room_names:
            self.optimize_heating_plan(room_name)
