"""
Occupancy prediction module.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional

from intelliwarm.data import OccupancyWindow, RoomConfig


class OccupancyPredictor:
    """
    Predicts room occupancy probability
    
    Initial implementation uses rule-based schedule
    Future: Bayesian model, ML classifier
    """
    
    def __init__(self, room_name: str, schedule: object = ""):
        """
        Initialize occupancy predictor
        
        Args:
            room_name: Name of the room
            schedule: Occupancy schedule (e.g., "9-18" = 9am to 6pm)
        """
        self.room_name = room_name
        self.schedule = schedule
        self.windows = RoomConfig.parse_schedule(schedule)
        self.logger = logging.getLogger("IntelliWarm.OccupancyPrediction")

    def predict(self, timestamp: datetime) -> float:
        """Predict occupancy probability for a timestamp."""
        if not self.windows:
            return 0.5

        matching_windows = [window for window in self.windows if window.contains(timestamp)]
        if not matching_windows:
            return 0.1

        return max(window.probability for window in matching_windows)
    
    def predict_occupancy(self, hour: int) -> float:
        """
        Predict occupancy probability for a given hour
        
        Args:
            hour: Hour of day (0-23)
            
        Returns:
            Occupancy probability (0-1)
        """
        reference_time = datetime.now().replace(hour=hour % 24, minute=0, second=0, microsecond=0)
        try:
            return self.predict(reference_time)
        except Exception as e:
            self.logger.error(f"Schedule parsing failed: {e}")
            return 0.5

    def predict_probability(self, timestamp: datetime) -> float:
        """Compatibility alias for timestamp-based occupancy prediction."""
        return self.predict(timestamp)

    def predict_horizon(
        self,
        start_time: datetime,
        horizon_steps: int,
        step_minutes: int,
    ) -> List[float]:
        """Predict occupancy probability over a future horizon."""
        return [
            self.predict(start_time + timedelta(minutes=step_minutes * step_index))
            for step_index in range(horizon_steps)
        ]
    
    def predict_occupancy_horizon(
        self,
        hours: int = 24,
        start_time: Optional[datetime] = None,
    ) -> List[float]:
        """
        Predict occupancy probability for next N hours
        
        Args:
            hours: Number of hours to predict
            
        Returns:
            List of occupancy probabilities
        """
        return self.predict_horizon(start_time or datetime.now(), hours, 60)
    
    def update_schedule(self, schedule: str):
        """Update occupancy schedule"""
        self.schedule = schedule
        self.windows = RoomConfig.parse_schedule(schedule)
        self.logger.info(f"Schedule updated: {schedule}")
