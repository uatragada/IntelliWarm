"""
Typed system configuration loading for IntelliWarm.
"""

from __future__ import annotations

import copy
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml


ENV_PLACEHOLDER_PATTERN = re.compile(r"\$\{([^}]+)\}")
ENV_OVERRIDE_PATHS = {
    "INTELLIWARM_DEBUG": ("system", "debug"),
    "INTELLIWARM_LOGGING_LEVEL": ("system", "logging_level"),
    "INTELLIWARM_POLL_INTERVAL": ("system", "poll_interval"),
    "INTELLIWARM_OPTIMIZATION_HORIZON": ("system", "optimization_horizon"),
    "INTELLIWARM_MAX_OPTIMIZATION_TIME": ("system", "max_optimization_time"),
    "INTELLIWARM_MIN_TEMPERATURE": ("comfort", "min_temperature"),
    "INTELLIWARM_MAX_TEMPERATURE": ("comfort", "max_temperature"),
    "INTELLIWARM_DEFAULT_TARGET_TEMP": ("comfort", "default_target"),
    "INTELLIWARM_COMFORT_WEIGHT": ("optimization", "comfort_weight"),
    "INTELLIWARM_SWITCHING_WEIGHT": ("optimization", "switching_weight"),
    "INTELLIWARM_ENERGY_WEIGHT": ("optimization", "energy_weight"),
    "INTELLIWARM_ELECTRICITY_PRICE": ("energy", "electricity_price"),
    "INTELLIWARM_GAS_PRICE": ("energy", "gas_price"),
    "INTELLIWARM_DATABASE_PATH": ("database", "path"),
    "INTELLIWARM_ENABLE_DEVICE_CONTROL": ("devices", "enable_control"),
}


def _parse_scalar(value: str) -> Any:
    if value == "":
        return ""

    parsed = yaml.safe_load(value)
    if isinstance(parsed, (dict, list)):
        return value
    return parsed


def _resolve_env_placeholders(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _resolve_env_placeholders(item) for key, item in value.items()}

    if isinstance(value, list):
        return [_resolve_env_placeholders(item) for item in value]

    if not isinstance(value, str):
        return value

    matches = list(ENV_PLACEHOLDER_PATTERN.finditer(value))
    if not matches:
        return value

    if len(matches) == 1 and matches[0].span() == (0, len(value)):
        env_name = matches[0].group(1)
        env_value = os.getenv(env_name)
        return _parse_scalar(env_value) if env_value is not None else value

    return ENV_PLACEHOLDER_PATTERN.sub(
        lambda match: os.getenv(match.group(1), match.group(0)),
        value,
    )


def _apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    updated = copy.deepcopy(config)

    for env_name, path in ENV_OVERRIDE_PATHS.items():
        env_value = os.getenv(env_name)
        if env_value is None:
            continue

        target = updated
        for key in path[:-1]:
            target = target.setdefault(key, {})
        target[path[-1]] = _parse_scalar(env_value)

    return updated


@dataclass(frozen=True)
class SystemSettings:
    debug: bool = False
    logging_level: str = "INFO"
    poll_interval: int = 30
    optimization_horizon: int = 24
    max_optimization_time: float = 2.0


@dataclass(frozen=True)
class ComfortSettings:
    min_temperature: float = 18.0
    max_temperature: float = 24.0
    default_target: float = 21.0
    discomfort_penalty: float = 50.0


@dataclass(frozen=True)
class OptimizationSettings:
    comfort_weight: float = 1.0
    switching_weight: float = 0.5
    energy_weight: float = 1.0


@dataclass(frozen=True)
class EnergySettings:
    electricity_price: float = 0.12
    gas_price: float = 5.0


@dataclass(frozen=True)
class RoomSettings:
    zone: str = "Unassigned"
    room_size: float = 150.0
    target_temp: float = 21.0
    heater_power: float = 1500.0
    thermal_mass: float = 0.05
    heating_efficiency: float = 0.85
    occupancy_schedule: str = ""
    initial_sensor_temp: Optional[float] = None
    initial_occupancy: bool = False
    display_temp_f: Optional[float] = None
    humidity: float = 45.0
    heating_source: str = "Off"


@dataclass(frozen=True)
class ZoneSettings:
    description: str = ""
    priority: int = 0


@dataclass(frozen=True)
class DatabaseSettings:
    type: str = "sqlite"
    path: str = "intelliwarm.db"
    host: Optional[str] = None
    port: Optional[int] = None
    user: Optional[str] = None
    password: Optional[str] = None
    database: Optional[str] = None


@dataclass(frozen=True)
class DeviceSettings:
    enable_control: bool = False
    smart_plug_id: str = ""
    thermostat_id: str = ""


@dataclass(frozen=True)
class LoggingSettings:
    file: str = "logs/intelliwarm.log"
    max_size: str = "10MB"
    backup_count: int = 5


@dataclass(frozen=True)
class ConfigState:
    system: SystemSettings = field(default_factory=SystemSettings)
    comfort: ComfortSettings = field(default_factory=ComfortSettings)
    optimization: OptimizationSettings = field(default_factory=OptimizationSettings)
    energy: EnergySettings = field(default_factory=EnergySettings)
    rooms: Dict[str, RoomSettings] = field(default_factory=dict)
    zones: Dict[str, ZoneSettings] = field(default_factory=dict)
    database: DatabaseSettings = field(default_factory=DatabaseSettings)
    devices: DeviceSettings = field(default_factory=DeviceSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)


class SystemConfig:
    """Manages IntelliWarm system configuration."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config_path = str(Path(config_path))
        self.config: Dict[str, Any] = {}
        self.state = ConfigState()
        self.reload()

    def _load_config(self) -> Dict[str, Any]:
        config_file = Path(self.config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with config_file.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}

        config = _resolve_env_placeholders(config)
        return _apply_env_overrides(config)

    def _build_state(self, config: Dict[str, Any]) -> ConfigState:
        rooms = {
            room_name: RoomSettings(**room_config)
            for room_name, room_config in config.get("rooms", {}).items()
        }
        zones = {
            zone_name: ZoneSettings(**zone_config)
            for zone_name, zone_config in config.get("zones", {}).items()
        }

        return ConfigState(
            system=SystemSettings(**config.get("system", {})),
            comfort=ComfortSettings(**config.get("comfort", {})),
            optimization=OptimizationSettings(**config.get("optimization", {})),
            energy=EnergySettings(**config.get("energy", {})),
            rooms=rooms,
            zones=zones,
            database=DatabaseSettings(**config.get("database", {})),
            devices=DeviceSettings(**config.get("devices", {})),
            logging=LoggingSettings(**config.get("logging", {})),
        )

    @property
    def debug(self) -> bool:
        return self.state.system.debug

    @property
    def logging_level(self) -> str:
        return self.state.system.logging_level

    @property
    def poll_interval(self) -> int:
        return self.state.system.poll_interval

    @property
    def optimization_horizon(self) -> int:
        return self.state.system.optimization_horizon

    @property
    def max_optimization_time(self) -> float:
        return self.state.system.max_optimization_time

    @property
    def min_temperature(self) -> float:
        return self.state.comfort.min_temperature

    @property
    def max_temperature(self) -> float:
        return self.state.comfort.max_temperature

    @property
    def default_target_temp(self) -> float:
        return self.state.comfort.default_target

    @property
    def comfort_weight(self) -> float:
        return self.state.optimization.comfort_weight

    @property
    def switching_weight(self) -> float:
        return self.state.optimization.switching_weight

    @property
    def energy_weight(self) -> float:
        return self.state.optimization.energy_weight

    @property
    def electricity_price(self) -> float:
        return self.state.energy.electricity_price

    @property
    def gas_price(self) -> float:
        return self.state.energy.gas_price

    @property
    def rooms(self) -> Dict[str, Dict[str, Any]]:
        return {room_name: asdict(room) for room_name, room in self.state.rooms.items()}

    @property
    def zones(self) -> Dict[str, Dict[str, Any]]:
        return {zone_name: asdict(zone) for zone_name, zone in self.state.zones.items()}

    def get_room_config(self, room_name: str) -> Dict[str, Any]:
        room = self.state.rooms.get(room_name)
        return asdict(room) if room else {}

    def build_room_config(
        self,
        room_name: Optional[str] = None,
        zone: Optional[str] = None,
        overrides: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        room_config = self.get_room_config(room_name) if room_name else {}
        resolved = {
            "zone": zone or room_config.get("zone", "Unassigned"),
            "room_size": float(room_config.get("room_size", 150.0)),
            "target_temp": float(room_config.get("target_temp", self.default_target_temp)),
            "heater_power": float(room_config.get("heater_power", 1500.0)),
            "thermal_mass": float(room_config.get("thermal_mass", 0.05)),
            "heating_efficiency": float(room_config.get("heating_efficiency", 0.85)),
            "occupancy_schedule": str(room_config.get("occupancy_schedule", "")),
            "initial_sensor_temp": room_config.get("initial_sensor_temp"),
            "initial_occupancy": bool(room_config.get("initial_occupancy", False)),
            "display_temp_f": room_config.get("display_temp_f"),
            "humidity": float(room_config.get("humidity", 45.0)),
            "heating_source": str(room_config.get("heating_source", "Off")),
        }
        if overrides:
            resolved.update(dict(overrides))
        return resolved

    @property
    def database_config(self) -> Dict[str, Any]:
        return asdict(self.state.database)

    @property
    def database_path(self) -> str:
        return self.state.database.path

    @property
    def enable_device_control(self) -> bool:
        return self.state.devices.enable_control

    @property
    def logging_config(self) -> Dict[str, Any]:
        return asdict(self.state.logging)

    def reload(self):
        self.config = self._load_config()
        self.state = self._build_state(self.config)
