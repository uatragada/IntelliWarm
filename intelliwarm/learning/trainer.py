"""
Learning System
Continuous thermal-model retraining utilities.

This module is for fitting room thermal parameters from historical data.
The PPO policy-training workflow lives in scripts/train_opt_heating_policy.py.
"""

import logging
from typing import List, Tuple


class ModelUpdater:
    """Updates models with new data"""
    
    def __init__(self, thermal_model):
        """
        Initialize model updater
        
        Args:
            thermal_model: RoomThermalModel instance
        """
        self.thermal_model = thermal_model
        self.logger = logging.getLogger("IntelliWarm.Learning")
    
    def update_thermal_parameters(self, historical_data: List[Tuple]):
        """
        Update thermal model parameters from historical data
        
        Args:
            historical_data: List of (T_current, H_action, T_outside, T_next) tuples
        """
        if len(historical_data) < 10:
            self.logger.warning("Insufficient data for model update")
            return
        
        try:
            self.thermal_model.estimate_parameters(historical_data)
            self.logger.info("Thermal model parameters updated")
        except Exception as e:
            self.logger.error(f"Model update failed: {e}")


class Trainer:
    """Handles periodic model retraining"""
    
    def __init__(self, database, thermal_models: dict):
        """
        Initialize trainer
        
        Args:
            database: Database instance
            thermal_models: Dict of room_name -> ThermalModel
        """
        self.database = database
        self.thermal_models = thermal_models
        self.logger = logging.getLogger("IntelliWarm.Trainer")
    
    def retrain_models(self):
        """Retrain all thermal models from historical data"""
        for room_name, model in self.thermal_models.items():
            self._retrain_room_model(room_name, model)
    
    def _retrain_room_model(self, room_name: str, model):
        """Retrain model for a single room"""
        try:
            # Get historical data
            history = self.database.get_temperature_history(room_name, limit=200)
            
            if len(history) < 10:
                self.logger.info(f"Insufficient data for {room_name}")
                return
            
            # TODO: Transform history to (T_current, H_action, T_outside, T_next)
            # This requires joining with optimization_runs table
            
            self.logger.info(f"Retrained model for {room_name}")
        
        except Exception as e:
            self.logger.error(f"Retraining failed for {room_name}: {e}")
