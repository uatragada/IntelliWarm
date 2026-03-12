"""
Thermal Model Module
Predicts room temperature based on heating actions
"""

import logging
from typing import List, Tuple
import numpy as np


class RoomThermalModel:
    """
    Models room thermal dynamics
    
    Equation:
    T(t+1) = T(t) + α*H(t) - β*(T(t) - T_outside)
    
    Where:
    - T(t) = room temperature
    - H(t) = heating action (0-1)
    - α = heating efficiency coefficient
    - β = heat loss coefficient
    """
    
    def __init__(self, room_name: str, alpha: float = 0.1, beta: float = 0.05):
        """
        Initialize thermal model
        
        Args:
            room_name: Name of the room
            alpha: Heating efficiency (rate of temp increase per unit power)
            beta: Heat loss coefficient (rate of cooling relative to outside)
        """
        self.room_name = room_name
        self.alpha = alpha
        self.beta = beta
        self.logger = logging.getLogger("IntelliWarm.ThermalModel")
    
    def predict_temperature(
        self,
        current_temp: float,
        outside_temp: float,
        heating_actions: List[float],
        hours: int = 24
    ) -> List[float]:
        """
        Predict future temperatures
        
        Args:
            current_temp: Current room temperature (°C)
            outside_temp: Outside temperature (°C)
            heating_actions: List of heating actions [0-1] for each hour
            hours: Number of hours to predict
            
        Returns:
            List of predicted temperatures
        """
        predictions = [current_temp]
        
        for h in range(min(hours, len(heating_actions))):
            T_t = predictions[-1]
            H_t = heating_actions[h]
            
            # Heating effect + cooling effect
            dT = self.alpha * H_t - self.beta * (T_t - outside_temp)
            T_next = T_t + dT
            
            predictions.append(T_next)
        
        return predictions[:-1]  # Return predictions for each hour
    
    def estimate_parameters(self, historical_data: List[Tuple[float, float, float]]):
        """
        Estimate α and β from historical data using least squares
        
        Args:
            historical_data: List of (T_current, H_action, outside_temp, T_next) tuples
        """
        if len(historical_data) < 10:
            self.logger.warning("Insufficient data for parameter estimation")
            return
        
        try:
            # Extract data
            T_current = np.array([d[0] for d in historical_data])
            H_action = np.array([d[1] for d in historical_data])
            T_outside = np.array([d[2] for d in historical_data])
            T_next = np.array([d[3] for d in historical_data])
            
            # Construct the system: T(t+1) - T(t) = α*H - β*(T - T_out)
            # Rearrange: dT = α*H - β*T + β*T_out
            dT = T_next - T_current
            
            # Design matrix: [H, -(T - T_outside)]
            A = np.column_stack([H_action, -(T_current - T_outside)])
            
            # Solve least squares: A * [α, β] = dT
            solution = np.linalg.lstsq(A, dT, rcond=None)[0]
            
            new_alpha = max(0, solution[0])
            new_beta = max(0, solution[1])
            
            self.alpha = new_alpha
            self.beta = new_beta
            
            self.logger.info(f"Parameters updated: α={self.alpha:.4f}, β={self.beta:.4f}")
        
        except Exception as e:
            self.logger.error(f"Parameter estimation failed: {e}")
    
    def get_parameters(self) -> Tuple[float, float]:
        """Return current model parameters"""
        return self.alpha, self.beta
