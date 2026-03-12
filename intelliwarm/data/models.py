"""
Typed domain models used by simulation and controller layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class HeatSourceType(Enum):
    """Physical heat source installed in a room or zone."""

    ELECTRIC = "electric"    # room-level electric space heater
    GAS_FURNACE = "gas_furnace"  # zone-level gas furnace

    @classmethod
    def from_str(cls, value: str) -> "HeatSourceType":
        normalized = str(value).strip().lower().replace(" ", "_")
        for member in cls:
            if member.value == normalized:
                return member
        return cls.ELECTRIC


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
    heat_source: HeatSourceType = HeatSourceType.ELECTRIC

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
        raw_source = config.get("heat_source", "electric")
        heat_source = HeatSourceType.from_str(str(raw_source)) if raw_source else HeatSourceType.ELECTRIC
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
            heat_source=heat_source,
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
    heat_sources: Dict[str, HeatSourceType] = field(default_factory=dict)


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
        heat_source = self.metadata.get("heat_source", HeatSourceType.ELECTRIC.value)
        if isinstance(heat_source, HeatSourceType):
            heat_source = heat_source.value
        return {
            "room": self.room_id,
            "controller": self.source,
            "action": self.action.name,
            "next_action": self.action.power_level,
            "next_action_label": self.action.name,
            "total_cost": self.projected_cost,
            "heat_source": heat_source,
            "target_temp": self.target_temp,
            "current_temp": self.current_temp,
            "occupancy_probability": self.occupancy_probability,
            "next_occupied_within_steps": self.next_occupied_within_steps,
            "explanation": self.rationale,
            "reasons": list(self.reasons),
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# Hybrid heating domain models
# ---------------------------------------------------------------------------


@dataclass
class ZoneConfig:
    """Zone-level configuration, including furnace details."""

    zone_id: str
    description: str = ""
    priority: int = 0
    has_furnace: bool = False
    furnace_btu_per_hour: float = 60_000.0  # typical residential furnace
    furnace_efficiency: float = 0.80  # AFUE (0-1)

    @property
    def furnace_therms_per_hour(self) -> float:
        """Gas consumption rate in therms/hour at rated *input* capacity.

        ``furnace_btu_per_hour`` is the INPUT BTU/hr (gas burned), following
        the standard HVAC/AHRI convention where a "20,000 BTU furnace" refers
        to gas consumption, not heat delivered.  Heat delivered = input × AFUE.
        """
        return self.furnace_btu_per_hour / 100_000.0

    def hourly_gas_cost(self, gas_price_per_therm: float) -> float:
        """Cost in $/hr to run the furnace at rated input capacity.

        Since ``furnace_btu_per_hour`` is already the INPUT (gas burned),
        the cost is simply therms_consumed × price.  No AFUE division is
        needed here — AFUE only affects *heat output*, which is handled by
        the thermal model (``from_room_config`` multiplies by AFUE to obtain
        the heat delivered to the zone).
        """
        return self.furnace_therms_per_hour * gas_price_per_therm


@dataclass(frozen=True)
class HybridHeatingDecision:
    """
    Zone-level decision: run the gas furnace for the whole zone, or
    let each room use its own electric space heater.

    The controller selects the source with the lower projected hourly cost
    whenever at least one room needs heat.
    """

    zone: str
    heat_source: HeatSourceType
    furnace_on: bool
    per_room_actions: Dict[str, HeatingAction]
    rooms_needing_heat: List[str]
    electric_hourly_cost: float
    furnace_hourly_cost: float
    chosen_hourly_cost: float
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "zone": self.zone,
            "heat_source": self.heat_source.value,
            "furnace_on": self.furnace_on,
            "rooms_needing_heat": list(self.rooms_needing_heat),
            "per_room_actions": {
                room: action.name for room, action in self.per_room_actions.items()
            },
            "electric_hourly_cost": round(self.electric_hourly_cost, 4),
            "furnace_hourly_cost": round(self.furnace_hourly_cost, 4),
            "chosen_hourly_cost": round(self.chosen_hourly_cost, 4),
            "rationale": self.rationale,
        }
