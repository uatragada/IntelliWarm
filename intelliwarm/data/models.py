"""
Typed domain models used by simulation and controller layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class HeatingAction(Enum):
    """Discrete heating actions used by controllers and simulation."""

    OFF = 0.0
    ECO = 0.35
    COMFORT = 0.7
    PREHEAT = 1.0

    @property
    def power_level(self) -> float:
        """Return normalized heater power for the action."""
        return float(self.value)

    @classmethod
    def from_value(cls, value: Any) -> "HeatingAction":
        """Resolve an enum member from an enum instance, name, or numeric power."""
        if isinstance(value, cls):
            return value

        if isinstance(value, str):
            normalized = value.strip().upper()
            if normalized in cls.__members__:
                return cls[normalized]

        numeric_value = float(value)
        return min(cls, key=lambda action: abs(action.power_level - numeric_value))


@dataclass(frozen=True)
class OccupancyWindow:
    """Occupancy interval for a specific day of the week."""

    day_of_week: int
    start_hour: int
    end_hour: int
    probability: float = 1.0

    def contains(self, timestamp: datetime) -> bool:
        """Return whether the timestamp falls inside the occupancy window."""
        if timestamp.weekday() != self.day_of_week:
            return False

        hour = timestamp.hour + (timestamp.minute / 60.0)
        return self.start_hour <= hour < self.end_hour


@dataclass
class RoomConfig:
    """Room-level configuration used by simulation and control layers."""

    room_id: str
    display_name: str
    zone: str
    target_min_temp: float
    target_max_temp: float
    heater_capacity: float
    heat_loss_factor: float
    heating_efficiency: float
    occupancy_schedule: List[OccupancyWindow] = field(default_factory=list)

    @classmethod
    def from_legacy_config(
        cls,
        room_id: str,
        config: Dict[str, Any],
        default_name: Optional[str] = None,
    ) -> "RoomConfig":
        """Build a typed room config from the legacy YAML structure."""
        target_temp = float(config.get("target_temp", 21.0))
        comfort_delta = float(config.get("comfort_delta", 1.0))
        schedule = cls.parse_schedule(config.get("occupancy_schedule", ""))
        return cls(
            room_id=room_id,
            display_name=default_name or str(config.get("display_name", room_id)),
            zone=str(config.get("zone", "Unknown")),
            target_min_temp=float(config.get("target_min_temp", target_temp - comfort_delta)),
            target_max_temp=float(config.get("target_max_temp", target_temp + comfort_delta)),
            heater_capacity=float(config.get("heater_power", 1500.0)),
            heat_loss_factor=float(config.get("heat_loss_factor", config.get("thermal_mass", 0.05))),
            heating_efficiency=float(config.get("heating_efficiency", 0.1)),
            occupancy_schedule=schedule,
        )

    @staticmethod
    def parse_schedule(schedule: Any) -> List[OccupancyWindow]:
        """Parse legacy schedule formats into occupancy windows."""
        if not schedule:
            return []

        if isinstance(schedule, list):
            windows: List[OccupancyWindow] = []
            for item in schedule:
                if isinstance(item, OccupancyWindow):
                    windows.append(item)
                    continue
                if isinstance(item, dict):
                    windows.append(
                        OccupancyWindow(
                            day_of_week=int(item["day_of_week"]),
                            start_hour=int(item["start_hour"]),
                            end_hour=int(item["end_hour"]),
                            probability=float(item.get("probability", 1.0)),
                        )
                    )
            return windows

        if isinstance(schedule, str) and "-" in schedule:
            start_hour, end_hour = map(int, schedule.split("-", maxsplit=1))
            return [
                OccupancyWindow(
                    day_of_week=day_of_week,
                    start_hour=start_hour,
                    end_hour=end_hour,
                    probability=0.8,
                )
                for day_of_week in range(7)
            ]

        return []


@dataclass
class SimulationState:
    """Snapshot of multi-room simulation state for a single timestep."""

    timestamp: datetime
    outdoor_temp: float
    room_temperatures: Dict[str, float]
    heating_actions: Dict[str, HeatingAction]
    occupancy: Dict[str, float]


@dataclass(frozen=True)
class ForecastStep:
    """Aligned per-step forecast inputs for control and simulation."""

    timestamp: datetime
    occupancy_probability: float
    outdoor_temp: float
    electricity_price: float
    gas_price: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize a forecast step for API and service responses."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "occupancy_probability": self.occupancy_probability,
            "outdoor_temp": self.outdoor_temp,
            "electricity_price": self.electricity_price,
            "gas_price": self.gas_price,
        }


@dataclass(frozen=True)
class ForecastBundle:
    """Aligned forecast horizon shared across controllers and services."""

    room_id: str
    start_time: datetime
    step_minutes: int
    steps: List[ForecastStep] = field(default_factory=list)
    source: str = "deterministic"

    @property
    def occupancy_probabilities(self) -> List[float]:
        return [step.occupancy_probability for step in self.steps]

    @property
    def outdoor_temperatures(self) -> List[float]:
        return [step.outdoor_temp for step in self.steps]

    @property
    def electricity_prices(self) -> List[float]:
        return [step.electricity_price for step in self.steps]

    @property
    def gas_prices(self) -> List[float]:
        return [step.gas_price for step in self.steps]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the bundle for runtime responses."""
        return {
            "room": self.room_id,
            "start_time": self.start_time.isoformat(),
            "step_minutes": self.step_minutes,
            "source": self.source,
            "steps": [step.to_dict() for step in self.steps],
        }


@dataclass(frozen=True)
class ControlDecision:
    """Shared control decision contract for baseline and runtime consumers."""

    room_id: str
    action: HeatingAction
    source: str
    rationale: str
    reasons: List[str] = field(default_factory=list)
    target_temp: float = 0.0
    current_temp: float = 0.0
    occupancy_probability: float = 0.0
    next_occupied_within_steps: Optional[int] = None
    projected_cost: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the decision for API and runtime responses."""
        return {
            "room": self.room_id,
            "controller": self.source,
            "action": self.action.name,
            "next_action": self.action.power_level,
            "next_action_label": self.action.name,
            "total_cost": self.projected_cost,
            "target_temp": self.target_temp,
            "current_temp": self.current_temp,
            "occupancy_probability": self.occupancy_probability,
            "next_occupied_within_steps": self.next_occupied_within_steps,
            "explanation": self.rationale,
            "reasons": list(self.reasons),
            "metadata": dict(self.metadata),
        }
