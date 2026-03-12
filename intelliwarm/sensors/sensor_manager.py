"""
Sensor Data Module
Manages temperature, occupancy, and device state sensors
"""

import logging
from typing import Dict, Optional
from datetime import datetime
from abc import ABC, abstractmethod


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


class SensorManager:
    """Manages all sensors across rooms"""
    
    def __init__(self):
        self.sensors: Dict[str, Dict[str, Sensor]] = {}
        self.logger = logging.getLogger("IntelliWarm.Sensors")
    
    def register_temperature_sensor(self, room_name: str, initial_temp: float = 20.0):
        """Register temperature sensor for a room"""
        if room_name not in self.sensors:
            self.sensors[room_name] = {}
        
        self.sensors[room_name]["temperature"] = TemperatureSensor(room_name, initial_temp)
        self.logger.info(f"Temperature sensor registered: {room_name}")
    
    def register_occupancy_sensor(self, room_name: str, initial_occupied: bool = False):
        """Register occupancy sensor for a room"""
        if room_name not in self.sensors:
            self.sensors[room_name] = {}
        
        self.sensors[room_name]["occupancy"] = OccupancySensor(room_name, initial_occupied)
        self.logger.info(f"Occupancy sensor registered: {room_name}")
    
    def get_temperature(self, room_name: str) -> Optional[float]:
        """Get room temperature"""
        if room_name in self.sensors and "temperature" in self.sensors[room_name]:
            return self.sensors[room_name]["temperature"].read()
        return None
    
    def get_occupancy(self, room_name: str) -> Optional[bool]:
        """Get room occupancy status"""
        if room_name in self.sensors and "occupancy" in self.sensors[room_name]:
            return self.sensors[room_name]["occupancy"].read()
        return None
    
    def set_temperature(self, room_name: str, temp: float):
        """Update temperature (simulation)"""
        if room_name in self.sensors and "temperature" in self.sensors[room_name]:
            self.sensors[room_name]["temperature"].set_value(temp)
    
    def set_occupancy(self, room_name: str, occupied: bool):
        """Update occupancy status (simulation)"""
        if room_name in self.sensors and "occupancy" in self.sensors[room_name]:
            self.sensors[room_name]["occupancy"].set_occupied(occupied)
    
    def get_room_state(self, room_name: str) -> Dict:
        """Get all sensor readings for a room"""
        return {
            "room": room_name,
            "temperature": self.get_temperature(room_name),
            "occupancy": self.get_occupancy(room_name),
            "timestamp": datetime.now().isoformat()
        }
