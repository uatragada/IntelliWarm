"""
Device Control Module
Manages heating device control (thermostats, smart plugs, relays)
"""

import logging
from typing import Dict
from abc import ABC, abstractmethod


class DeviceInterface(ABC):
    """Abstract interface for controllable heating devices"""
    
    @abstractmethod
    def set_power(self, level: float):
        """Set power level (0-1)"""
        pass
    
    @abstractmethod
    def turn_on(self):
        """Turn device on"""
        pass
    
    @abstractmethod
    def turn_off(self):
        """Turn device off"""
        pass
    
    @abstractmethod
    def get_status(self) -> Dict:
        """Get current device status"""
        pass


class SimulatedHeater(DeviceInterface):
    """Simulated heater for testing"""
    
    def __init__(self, room_name: str, power_watts: float = 1500):
        self.room_name = room_name
        self.max_power = power_watts
        self.power_level = 0.0
        self.is_on = False
        self.logger = logging.getLogger("IntelliWarm.Control")
    
    def set_power(self, level: float):
        """Set power level 0-1"""
        self.power_level = max(0.0, min(1.0, level))
        self.is_on = self.power_level > 0
        self.logger.debug(f"[{self.room_name}] Power set to {self.power_level:.2f}")
    
    def turn_on(self):
        """Turn device on"""
        self.set_power(1.0)
    
    def turn_off(self):
        """Turn device off"""
        self.set_power(0.0)
    
    def get_status(self) -> Dict:
        """Get device status"""
        return {
            "room": self.room_name,
            "is_on": self.is_on,
            "power_level": self.power_level,
            "power_watts": self.power_level * self.max_power
        }


class DeviceController:
    """Manages all heating devices"""
    
    def __init__(self):
        self.devices: Dict[str, DeviceInterface] = {}
        self.logger = logging.getLogger("IntelliWarm.DeviceController")
    
    def register_device(self, room_name: str, device: DeviceInterface = None):
        """Register a heating device"""
        if device is None:
            device = SimulatedHeater(room_name)
        
        self.devices[room_name] = device
        self.logger.info(f"Device registered for {room_name}")
    
    def set_heater(self, room_name: str, power_level: float):
        """Set heater power level"""
        if room_name not in self.devices:
            self.logger.warning(f"Device not found: {room_name}")
            return
        
        self.devices[room_name].set_power(power_level)
    
    def turn_off(self, room_name: str):
        """Turn off heater"""
        if room_name not in self.devices:
            self.logger.warning(f"Device not found: {room_name}")
            return
        
        self.devices[room_name].turn_off()
    
    def get_device_status(self, room_name: str) -> Dict:
        """Get device status"""
        if room_name not in self.devices:
            return None
        
        return self.devices[room_name].get_status()
    
    def get_all_device_status(self) -> Dict[str, Dict]:
        """Get status of all devices"""
        return {room: device.get_status() for room, device in self.devices.items()}
