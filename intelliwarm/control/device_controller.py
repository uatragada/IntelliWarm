"""
Device Control Module
Manages heating device control (thermostats, smart plugs, relays)
"""

import logging
from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional


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
            "power_watts": self.power_level * self.max_power,
        }


class DeviceBackend(ABC):
    """Backend contract for heater/device control."""

    @abstractmethod
    def register_device(self, room_name: str, device: Optional[DeviceInterface] = None):
        """Register a device backend for a room."""

    @abstractmethod
    def set_heater(self, room_name: str, power_level: float):
        """Apply a heater power level."""

    @abstractmethod
    def turn_off(self, room_name: str):
        """Turn off a room device."""

    @abstractmethod
    def get_device_status(self, room_name: str) -> Optional[Dict]:
        """Get device status for a room."""

    @abstractmethod
    def get_all_device_status(self) -> Dict[str, Dict]:
        """Get all device statuses."""


class SimulatedDeviceBackend(DeviceBackend):
    """Simulated heater backend."""

    def __init__(self):
        self.devices: Dict[str, DeviceInterface] = {}
        self.logger = logging.getLogger("IntelliWarm.DeviceController")

    def register_device(self, room_name: str, device: DeviceInterface = None):
        """Register a heating device"""
        if device is None:
            device = SimulatedHeater(room_name)

        self.devices[room_name] = device
        self.logger.info("Device registered for %s", room_name)

    def set_heater(self, room_name: str, power_level: float):
        """Set heater power level"""
        if room_name not in self.devices:
            self.logger.warning("Device not found: %s", room_name)
            return

        self.devices[room_name].set_power(power_level)

    def turn_off(self, room_name: str):
        """Turn off heater"""
        if room_name not in self.devices:
            self.logger.warning("Device not found: %s", room_name)
            return

        self.devices[room_name].turn_off()

    def get_device_status(self, room_name: str) -> Optional[Dict]:
        """Get device status"""
        if room_name not in self.devices:
            return None

        return self.devices[room_name].get_status()

    def get_all_device_status(self) -> Dict[str, Dict]:
        """Get status of all devices"""
        return {room: device.get_status() for room, device in self.devices.items()}


class HardwareDeviceBackend(SimulatedDeviceBackend):
    """Hardware-ready backend with simulated fallback for offline-safe control."""

    def __init__(
        self,
        command_writer: Optional[Callable[[str, float], None]] = None,
        status_reader: Optional[Callable[[str], Optional[Dict]]] = None,
        device_ids: Optional[Dict[str, str]] = None,
        enable_hardware: bool = False,
    ):
        super().__init__()
        self.command_writer = command_writer
        self.status_reader = status_reader
        self.device_ids = dict(device_ids or {})
        self.enable_hardware = bool(enable_hardware)
        self.control_sources: Dict[str, str] = {}

    def _device_id(self, room_name: str) -> Optional[str]:
        return self.device_ids.get(room_name) or self.device_ids.get("*")

    def register_device(self, room_name: str, device: DeviceInterface = None):
        super().register_device(room_name, device)
        self.control_sources.setdefault(room_name, "simulated")

    def set_heater(self, room_name: str, power_level: float):
        device_id = self._device_id(room_name)
        if self.enable_hardware and self.command_writer and device_id:
            try:
                self.command_writer(device_id, power_level)
                self.control_sources[room_name] = "hardware"
            except Exception as exc:
                self.logger.warning("Hardware device command failed for %s: %s", room_name, exc)
                self.control_sources[room_name] = "simulated_fallback"
        elif device_id:
            self.control_sources[room_name] = "simulated_fallback"
        else:
            self.control_sources[room_name] = "simulated"

        super().set_heater(room_name, power_level)

    def turn_off(self, room_name: str):
        self.set_heater(room_name, 0.0)

    def get_device_status(self, room_name: str) -> Optional[Dict]:
        status = super().get_device_status(room_name)
        if status is None:
            return None

        device_id = self._device_id(room_name)
        if self.enable_hardware and self.status_reader and device_id:
            try:
                hardware_status = self.status_reader(device_id)
            except Exception as exc:
                self.logger.warning("Hardware device status read failed for %s: %s", room_name, exc)
                hardware_status = None

            if hardware_status is not None:
                merged = dict(status)
                merged.update(dict(hardware_status))
                merged["control_source"] = "hardware"
                merged["device_id"] = device_id
                return merged

        status = dict(status)
        status["control_source"] = self.control_sources.get(room_name, "simulated")
        status["device_id"] = device_id
        return status


class DeviceController:
    """Manages all heating devices."""

    def __init__(self, backend: Optional[DeviceBackend] = None):
        self.backend = backend or SimulatedDeviceBackend()
        self.logger = logging.getLogger("IntelliWarm.DeviceController")

    @classmethod
    def with_hardware_fallback(
        cls,
        enable_hardware: bool = False,
        default_device_id: str = "",
        command_writer: Optional[Callable[[str, float], None]] = None,
        status_reader: Optional[Callable[[str], Optional[Dict]]] = None,
    ) -> "DeviceController":
        """Create a controller that can use hardware commands with simulated fallback."""
        device_ids = {"*": default_device_id} if default_device_id else {}
        return cls(
            backend=HardwareDeviceBackend(
                command_writer=command_writer,
                status_reader=status_reader,
                device_ids=device_ids,
                enable_hardware=enable_hardware,
            )
        )

    def register_device(self, room_name: str, device: DeviceInterface = None):
        self.backend.register_device(room_name, device)

    def set_heater(self, room_name: str, power_level: float):
        self.backend.set_heater(room_name, power_level)

    def turn_off(self, room_name: str):
        self.backend.turn_off(room_name)

    def get_device_status(self, room_name: str) -> Optional[Dict]:
        return self.backend.get_device_status(room_name)

    def get_all_device_status(self) -> Dict[str, Dict]:
        return self.backend.get_all_device_status()
