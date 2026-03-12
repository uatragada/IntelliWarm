"""
System Configuration Manager
Loads and validates configuration from YAML
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any


class SystemConfig:
    """Manages IntelliWarm system configuration"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Load configuration from YAML file
        
        Args:
            config_path: Path to config.yaml
        """
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    # System settings
    @property
    def debug(self) -> bool:
        return self.config.get("system", {}).get("debug", False)
    
    @property
    def poll_interval(self) -> int:
        return self.config.get("system", {}).get("poll_interval", 30)
    
    @property
    def optimization_horizon(self) -> int:
        return self.config.get("system", {}).get("optimization_horizon", 24)
    
    @property
    def max_optimization_time(self) -> float:
        return self.config.get("system", {}).get("max_optimization_time", 2.0)
    
    # Comfort settings
    @property
    def min_temperature(self) -> float:
        return self.config.get("comfort", {}).get("min_temperature", 18)
    
    @property
    def max_temperature(self) -> float:
        return self.config.get("comfort", {}).get("max_temperature", 24)
    
    @property
    def default_target_temp(self) -> float:
        return self.config.get("comfort", {}).get("default_target", 21)
    
    # Optimization weights
    @property
    def comfort_weight(self) -> float:
        return self.config.get("optimization", {}).get("comfort_weight", 1.0)
    
    @property
    def switching_weight(self) -> float:
        return self.config.get("optimization", {}).get("switching_weight", 0.5)
    
    @property
    def energy_weight(self) -> float:
        return self.config.get("optimization", {}).get("energy_weight", 1.0)
    
    # Energy prices
    @property
    def electricity_price(self) -> float:
        return self.config.get("energy", {}).get("electricity_price", 0.12)
    
    @property
    def gas_price(self) -> float:
        return self.config.get("energy", {}).get("gas_price", 5.0)
    
    # Rooms
    @property
    def rooms(self) -> Dict[str, Dict]:
        return self.config.get("rooms", {})
    
    def get_room_config(self, room_name: str) -> Dict[str, Any]:
        """Get configuration for specific room"""
        return self.rooms.get(room_name, {})
    
    # Database
    @property
    def database_config(self) -> Dict[str, Any]:
        return self.config.get("database", {})
    
    # Device control
    @property
    def enable_device_control(self) -> bool:
        return self.config.get("devices", {}).get("enable_control", False)
    
    def reload(self):
        """Reload configuration from file"""
        self.config = self._load_config()
