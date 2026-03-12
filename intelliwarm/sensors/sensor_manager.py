"""
Sensor Data Module
Manages temperature, occupancy, and device state sensors
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Callable, Dict, Optional


class Sensor(ABC):
    """Abstract base class for all sensors"""
    
    @abstractmethod
    def read(self) -> float:
        """Read sensor value"""
        pass


class TemperatureSensor(Sensor):
    """Temperature sensor implementation"""
    
    def __init__(self, room_name: str, value: float = 20.0):
        self.room_name = room_name
        self.current_value = value
    
    def read(self) -> float:
        """Return current temperature"""
        return self.current_value
    
    def set_value(self, value: float):
        """Update temperature (for simulation)"""
        self.current_value = value


class OccupancySensor(Sensor):
    """Occupancy sensor implementation"""
    
    def __init__(self, room_name: str, occupied: bool = False):
        self.room_name = room_name
        self.occupied = occupied
    
    def read(self) -> bool:
        """Return occupancy status"""
        return self.occupied
    
    def set_occupied(self, occupied: bool):
        """Update occupancy status"""
        self.occupied = occupied


class SensorBackend(ABC):
    """Backend contract for room sensor I/O."""

    @abstractmethod
    def register_temperature_sensor(self, room_name: str, initial_temp: float = 20.0):
        """Register or attach a temperature sensor."""

    @abstractmethod
    def register_occupancy_sensor(self, room_name: str, initial_occupied: bool = False):
        """Register or attach an occupancy sensor."""

    @abstractmethod
    def get_temperature(self, room_name: str) -> Optional[float]:
        """Read temperature for a room."""

    @abstractmethod
    def get_occupancy(self, room_name: str) -> Optional[bool]:
        """Read occupancy for a room."""

    @abstractmethod
    def set_temperature(self, room_name: str, temp: float):
        """Update the simulated fallback temperature."""

    @abstractmethod
    def set_occupancy(self, room_name: str, occupied: bool):
        """Update the simulated fallback occupancy."""

    @abstractmethod
    def get_room_state(self, room_name: str) -> Dict:
        """Return the combined room state."""


class SimulatedSensorBackend(SensorBackend):
    """Purely simulated room sensor backend."""

    def __init__(self):
        self.sensors: Dict[str, Dict[str, Sensor]] = {}
        self.logger = logging.getLogger("IntelliWarm.Sensors")

    def register_temperature_sensor(self, room_name: str, initial_temp: float = 20.0):
        if room_name not in self.sensors:
            self.sensors[room_name] = {}

        self.sensors[room_name]["temperature"] = TemperatureSensor(room_name, initial_temp)
        self.logger.info("Temperature sensor registered: %s", room_name)

    def register_occupancy_sensor(self, room_name: str, initial_occupied: bool = False):
        if room_name not in self.sensors:
            self.sensors[room_name] = {}

        self.sensors[room_name]["occupancy"] = OccupancySensor(room_name, initial_occupied)
        self.logger.info("Occupancy sensor registered: %s", room_name)

    def get_temperature(self, room_name: str) -> Optional[float]:
        if room_name in self.sensors and "temperature" in self.sensors[room_name]:
            return self.sensors[room_name]["temperature"].read()
        return None

    def get_occupancy(self, room_name: str) -> Optional[bool]:
        if room_name in self.sensors and "occupancy" in self.sensors[room_name]:
            return self.sensors[room_name]["occupancy"].read()
        return None

    def set_temperature(self, room_name: str, temp: float):
        if room_name in self.sensors and "temperature" in self.sensors[room_name]:
            self.sensors[room_name]["temperature"].set_value(temp)

    def set_occupancy(self, room_name: str, occupied: bool):
        if room_name in self.sensors and "occupancy" in self.sensors[room_name]:
            self.sensors[room_name]["occupancy"].set_occupied(occupied)

    def get_room_state(self, room_name: str) -> Dict:
        return {
            "room": room_name,
            "temperature": self.get_temperature(room_name),
            "occupancy": self.get_occupancy(room_name),
            "timestamp": datetime.now().isoformat(),
            "sensor_source": "simulated",
        }


class HardwareSensorBackend(SimulatedSensorBackend):
    """Hardware-ready backend with simulated fallback when hardware is unavailable."""

    def __init__(
        self,
        temperature_reader: Optional[Callable[[str], Optional[float]]] = None,
        occupancy_reader: Optional[Callable[[str], Optional[bool]]] = None,
    ):
        super().__init__()
        self.temperature_reader = temperature_reader
        self.occupancy_reader = occupancy_reader
        self.read_sources: Dict[str, str] = {}

    def get_temperature(self, room_name: str) -> Optional[float]:
        fallback_value = super().get_temperature(room_name)
        if self.temperature_reader is None:
            self.read_sources[room_name] = "simulated"
            return fallback_value

        try:
            value = self.temperature_reader(room_name)
        except Exception as exc:
            self.logger.warning("Hardware temperature read failed for %s: %s", room_name, exc)
            self.read_sources[room_name] = "simulated_fallback"
            return fallback_value

        if value is None:
            self.read_sources[room_name] = "simulated_fallback"
            return fallback_value

        resolved_value = float(value)
        super().set_temperature(room_name, resolved_value)
        self.read_sources[room_name] = "hardware"
        return resolved_value

    def get_occupancy(self, room_name: str) -> Optional[bool]:
        fallback_value = super().get_occupancy(room_name)
        if self.occupancy_reader is None:
            self.read_sources.setdefault(room_name, "simulated")
            return fallback_value

        try:
            value = self.occupancy_reader(room_name)
        except Exception as exc:
            self.logger.warning("Hardware occupancy read failed for %s: %s", room_name, exc)
            self.read_sources[room_name] = "simulated_fallback"
            return fallback_value

        if value is None:
            self.read_sources[room_name] = "simulated_fallback"
            return fallback_value

        resolved_value = bool(value)
        super().set_occupancy(room_name, resolved_value)
        self.read_sources[room_name] = "hardware"
        return resolved_value

    def get_room_state(self, room_name: str) -> Dict:
        return {
            "room": room_name,
            "temperature": self.get_temperature(room_name),
            "occupancy": self.get_occupancy(room_name),
            "timestamp": datetime.now().isoformat(),
            "sensor_source": self.read_sources.get(room_name, "simulated"),
        }


class SensorManager:
    """Manages all sensors across rooms"""

    def __init__(self, backend: Optional[SensorBackend] = None):
        self.backend = backend or SimulatedSensorBackend()
        self.logger = logging.getLogger("IntelliWarm.Sensors")

    @classmethod
    def with_hardware_fallback(
        cls,
        temperature_reader: Optional[Callable[[str], Optional[float]]] = None,
        occupancy_reader: Optional[Callable[[str], Optional[bool]]] = None,
    ) -> "SensorManager":
        """Create a hardware-ready manager that falls back to simulation."""
        return cls(
            backend=HardwareSensorBackend(
                temperature_reader=temperature_reader,
                occupancy_reader=occupancy_reader,
            )
        )

    def register_temperature_sensor(self, room_name: str, initial_temp: float = 20.0):
        """Register temperature sensor for a room"""
        self.backend.register_temperature_sensor(room_name, initial_temp)

    def register_occupancy_sensor(self, room_name: str, initial_occupied: bool = False):
        """Register occupancy sensor for a room"""
        self.backend.register_occupancy_sensor(room_name, initial_occupied)

    def get_temperature(self, room_name: str) -> Optional[float]:
        """Get room temperature"""
        return self.backend.get_temperature(room_name)

    def get_occupancy(self, room_name: str) -> Optional[bool]:
        """Get room occupancy status"""
        return self.backend.get_occupancy(room_name)

    def set_temperature(self, room_name: str, temp: float):
        """Update temperature (simulation)"""
        self.backend.set_temperature(room_name, temp)

    def set_occupancy(self, room_name: str, occupied: bool):
        """Update occupancy status (simulation)"""
        self.backend.set_occupancy(room_name, occupied)

    def get_room_state(self, room_name: str) -> Dict:
        """Get all sensor readings for a room"""
        return self.backend.get_room_state(room_name)
