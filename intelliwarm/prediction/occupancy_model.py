"""
Occupancy Prediction Module
Predicts room occupancy probability
"""

import logging
from typing import Dict, List
from datetime import datetime


class OccupancyPredictor:
    """
    Predicts room occupancy probability
    
    Initial implementation uses rule-based schedule
    Future: Bayesian model, ML classifier
    """
    
    def __init__(self, room_name: str, schedule: str = ""):
        """
        Initialize occupancy predictor
        
        Args:
            room_name: Name of the room
            schedule: Occupancy schedule (e.g., "9-18" = 9am to 6pm)
        """
        self.room_name = room_name
        self.schedule = schedule
        self.logger = logging.getLogger("IntelliWarm.OccupancyPrediction")
    
    def predict_occupancy(self, hour: int) -> float:
        """
        Predict occupancy probability for a given hour
        
        Args:
            hour: Hour of day (0-23)
            
        Returns:
            Occupancy probability (0-1)
        """
        if not self.schedule or "-" not in self.schedule:
            return 0.5  # Default: assume 50% chance
        
        try:
            start_hour, end_hour = map(int, self.schedule.split("-"))
            
            if start_hour <= hour < end_hour:
                return 0.8  # High probability during scheduled hours
            else:
                return 0.1  # Low probability outside scheduled hours
        
        except Exception as e:
            self.logger.error(f"Schedule parsing failed: {e}")
            return 0.5
    
    def predict_occupancy_horizon(self, hours: int = 24) -> List[float]:
        """
        Predict occupancy probability for next N hours
        
        Args:
            hours: Number of hours to predict
            
        Returns:
            List of occupancy probabilities
        """
        current_hour = datetime.now().hour
        predictions = []
        
        for h in range(hours):
            hour = (current_hour + h) % 24
            prob = self.predict_occupancy(hour)
            predictions.append(prob)
        
        return predictions
    
    def update_schedule(self, schedule: str):
        """Update occupancy schedule"""
        self.schedule = schedule
        self.logger.info(f"Schedule updated: {schedule}")
