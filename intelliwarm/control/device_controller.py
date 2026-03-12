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


class FurnaceInterface(ABC):
    """Abstract interface for zone-wide furnace actuators."""

    @abstractmethod
    def set_power(self, level: float):
        """Set furnace output level (0-1)."""

    @abstractmethod
    def turn_on(self):
        """Turn furnace on."""

    @abstractmethod
    def turn_off(self):
        """Turn furnace off."""

    @abstractmethod
    def get_status(self) -> Dict:
        """Get furnace status."""


class SimulatedFurnace(FurnaceInterface):
    """Simulated zone furnace for testing and offline-safe runtime behavior."""

    def __init__(self, zone_name: str):
        self.zone_name = zone_name
        self.power_level = 0.0
        self.is_on = False
        self.logger = logging.getLogger("IntelliWarm.Control")

    def set_power(self, level: float):
        self.power_level = max(0.0, min(1.0, level))
        self.is_on = self.power_level > 0.0
        self.logger.debug("[%s] Furnace power set to %.2f", self.zone_name, self.power_level)

    def turn_on(self):
        self.set_power(1.0)

    def turn_off(self):
        self.set_power(0.0)

    def get_status(self) -> Dict:
        return {
            "zone": self.zone_name,
            "is_on": self.is_on,
            "power_level": self.power_level,
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

    @abstractmethod
    def register_furnace(self, zone_name: str, furnace: Optional[FurnaceInterface] = None):
        """Register a furnace backend for a zone."""

    @abstractmethod
    def set_zone_furnace(self, zone_name: str, power_level: float):
        """Apply a zone furnace power level."""

    @abstractmethod
    def turn_off_zone_furnace(self, zone_name: str):
        """Turn off a zone furnace."""

    @abstractmethod
    def get_zone_furnace_status(self, zone_name: str) -> Optional[Dict]:
        """Get furnace status for a zone."""

    @abstractmethod
    def get_all_zone_furnace_status(self) -> Dict[str, Dict]:
        """Get all furnace statuses."""


class SimulatedDeviceBackend(DeviceBackend):
    """Simulated heater backend."""

    def __init__(self):
        self.devices: Dict[str, DeviceInterface] = {}
        self.furnaces: Dict[str, FurnaceInterface] = {}
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

    def register_furnace(self, zone_name: str, furnace: FurnaceInterface = None):
        if furnace is None:
            furnace = SimulatedFurnace(zone_name)
        self.furnaces[zone_name] = furnace
        self.logger.info("Furnace registered for zone %s", zone_name)

    def set_zone_furnace(self, zone_name: str, power_level: float):
        if zone_name not in self.furnaces:
            self.logger.warning("Furnace not found for zone: %s", zone_name)
            return
        self.furnaces[zone_name].set_power(power_level)

    def turn_off_zone_furnace(self, zone_name: str):
        if zone_name not in self.furnaces:
            self.logger.warning("Furnace not found for zone: %s", zone_name)
            return
        self.furnaces[zone_name].turn_off()

    def get_zone_furnace_status(self, zone_name: str) -> Optional[Dict]:
        if zone_name not in self.furnaces:
            return None
        return self.furnaces[zone_name].get_status()

    def get_all_zone_furnace_status(self) -> Dict[str, Dict]:
        return {zone: furnace.get_status() for zone, furnace in self.furnaces.items()}


class HardwareDeviceBackend(SimulatedDeviceBackend):
    """Hardware-ready backend with simulated fallback for offline-safe control."""

    def __init__(
        self,
        command_writer: Optional[Callable[[str, float], None]] = None,
        status_reader: Optional[Callable[[str], Optional[Dict]]] = None,
        device_ids: Optional[Dict[str, str]] = None,
        furnace_command_writer: Optional[Callable[[str, float], None]] = None,
        furnace_status_reader: Optional[Callable[[str], Optional[Dict]]] = None,
        furnace_ids: Optional[Dict[str, str]] = None,
        enable_hardware: bool = False,
    ):
        super().__init__()
        self.command_writer = command_writer
        self.status_reader = status_reader
        self.device_ids = dict(device_ids or {})
        self.furnace_command_writer = furnace_command_writer or command_writer
        self.furnace_status_reader = furnace_status_reader or status_reader
        self.furnace_ids = dict(furnace_ids or {})
        self.enable_hardware = bool(enable_hardware)
        self.control_sources: Dict[str, str] = {}
        self.furnace_control_sources: Dict[str, str] = {}

    def _device_id(self, room_name: str) -> Optional[str]:
        return self.device_ids.get(room_name) or self.device_ids.get("*")

    def register_device(self, room_name: str, device: DeviceInterface = None):
        super().register_device(room_name, device)
        self.control_sources.setdefault(room_name, "simulated")

    def _furnace_id(self, zone_name: str) -> Optional[str]:
        return self.furnace_ids.get(zone_name) or self.furnace_ids.get("*")

    def register_furnace(self, zone_name: str, furnace: FurnaceInterface = None):
        super().register_furnace(zone_name, furnace)
        self.furnace_control_sources.setdefault(zone_name, "simulated")

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

    def set_zone_furnace(self, zone_name: str, power_level: float):
        furnace_id = self._furnace_id(zone_name)
        if self.enable_hardware and self.furnace_command_writer and furnace_id:
            try:
                self.furnace_command_writer(furnace_id, power_level)
                self.furnace_control_sources[zone_name] = "hardware"
            except Exception as exc:
                self.logger.warning("Hardware furnace command failed for %s: %s", zone_name, exc)
                self.furnace_control_sources[zone_name] = "simulated_fallback"
        elif furnace_id:
            self.furnace_control_sources[zone_name] = "simulated_fallback"
        else:
            self.furnace_control_sources[zone_name] = "simulated"

        super().set_zone_furnace(zone_name, power_level)

    def turn_off_zone_furnace(self, zone_name: str):
        self.set_zone_furnace(zone_name, 0.0)

    def get_zone_furnace_status(self, zone_name: str) -> Optional[Dict]:
        status = super().get_zone_furnace_status(zone_name)
        if status is None:
            return None

        furnace_id = self._furnace_id(zone_name)
        if self.enable_hardware and self.furnace_status_reader and furnace_id:
            try:
                hardware_status = self.furnace_status_reader(furnace_id)
            except Exception as exc:
                self.logger.warning("Hardware furnace status read failed for %s: %s", zone_name, exc)
                hardware_status = None

            if hardware_status is not None:
                merged = dict(status)
                merged.update(dict(hardware_status))
                merged["control_source"] = "hardware"
                merged["device_id"] = furnace_id
                return merged

        status = dict(status)
        status["control_source"] = self.furnace_control_sources.get(zone_name, "simulated")
        status["device_id"] = furnace_id
        return status

    def get_all_zone_furnace_status(self) -> Dict[str, Dict]:
        return {
            zone: self.get_zone_furnace_status(zone)
            for zone in self.furnaces
        }


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
        default_furnace_id: str = "",
        command_writer: Optional[Callable[[str, float], None]] = None,
        status_reader: Optional[Callable[[str], Optional[Dict]]] = None,
    ) -> "DeviceController":
        """Create a controller that can use hardware commands with simulated fallback."""
        device_ids = {"*": default_device_id} if default_device_id else {}
        furnace_ids = {"*": default_furnace_id} if default_furnace_id else {}
        return cls(
            backend=HardwareDeviceBackend(
                command_writer=command_writer,
                status_reader=status_reader,
                device_ids=device_ids,
                furnace_ids=furnace_ids,
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

    def register_furnace(self, zone_name: str, furnace: FurnaceInterface = None):
        self.backend.register_furnace(zone_name, furnace)

    def set_zone_furnace(self, zone_name: str, power_level: float):
        self.backend.set_zone_furnace(zone_name, power_level)

    def turn_off_zone_furnace(self, zone_name: str):
        self.backend.turn_off_zone_furnace(zone_name)

    def get_zone_furnace_status(self, zone_name: str) -> Optional[Dict]:
        return self.backend.get_zone_furnace_status(zone_name)

    def get_all_zone_furnace_status(self) -> Dict[str, Dict]:
        return self.backend.get_all_zone_furnace_status()
