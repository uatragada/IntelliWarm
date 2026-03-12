"""
Model Predictive Control (MPC) Optimizer
Computes optimal heating actions to minimize cost while maintaining comfort
"""

import logging
from typing import Dict, List, Tuple
import numpy as np
from scipy.optimize import minimize


class CostFunction:
    """Defines the optimization cost function"""
    
    def __init__(self, config):
        """
        Initialize cost function
        
        Args:
            config: SystemConfig instance
        """
        self.comfort_weight = config.comfort_weight
        self.switching_weight = config.switching_weight
        self.energy_weight = config.energy_weight
        self.min_temp = config.min_temperature
        self.max_temp = config.max_temperature
        self.logger = logging.getLogger("IntelliWarm.CostFunction")
    
    def compute_cost(
        self,
        heating_actions: List[float],
        predicted_temps: List[float],
        target_temp: float,
        energy_prices: List[float],
        occupancy_probs: List[float],
        current_action: float = 0.0
    ) -> float:
        """
        Compute total cost for a sequence of heating actions
        
        Cost = Energy Cost + Discomfort Penalty + Switching Penalty
        
        Args:
            heating_actions: List of heating actions [0-1]
            predicted_temps: Predicted temperatures from thermal model
            target_temp: Target comfort temperature
            energy_prices: Energy prices per kWh for each hour
            occupancy_probs: Occupancy probability for each hour
            current_action: Current heating action (for switching penalty)
            
        Returns:
            Total cost
        """
        total_cost = 0.0
        
        # Energy cost
        energy_cost = sum(
            h * price for h, price in zip(heating_actions, energy_prices)
        ) * self.energy_weight
        total_cost += energy_cost
        
        # Discomfort cost (only when occupied)
        comfort_cost = sum(
            occ * self._discomfort_penalty(T, target_temp)
            for T, occ in zip(predicted_temps, occupancy_probs)
        ) * self.comfort_weight
        total_cost += comfort_cost
        
        # Switching cost (penalize frequent on/off)
        switching_cost = 0.0
        prev_action = current_action
        for action in heating_actions:
            if abs(action - prev_action) > 0.1:  # Threshold for switching
                switching_cost += 1.0
            prev_action = action
        total_cost += switching_cost * self.switching_weight
        
        return total_cost
    
    def _discomfort_penalty(self, current_temp: float, target_temp: float) -> float:
        """Compute discomfort penalty for temperature deviation"""
        if current_temp < self.min_temp:
            # Too cold
            return ((self.min_temp - current_temp) ** 2) * 10
        elif current_temp > self.max_temp:
            # Too hot
            return ((current_temp - self.max_temp) ** 2) * 10
        else:
            # Within acceptable range, penalize deviation from target
            return (abs(current_temp - target_temp) ** 2) * 0.5


class MPCController:
    """Model Predictive Control optimizer"""
    
    def __init__(self, config, thermal_model, cost_function: CostFunction = None):
        """
        Initialize MPC controller
        
        Args:
            config: SystemConfig instance
            thermal_model: RoomThermalModel instance
            cost_function: CostFunction instance (default created if None)
        """
        self.config = config
        self.thermal_model = thermal_model
        self.cost_function = cost_function or CostFunction(config)
        self.horizon = config.optimization_horizon
        self.logger = logging.getLogger("IntelliWarm.MPCController")
    
    def compute_optimal_plan(
        self,
        room_name: str,
        current_temp: float,
        outside_temp: float,
        target_temp: float,
        energy_prices: List[float],
        occupancy_probs: List[float],
        current_action: float = 0.0
    ) -> Dict:
        """
        Compute optimal heating schedule
        
        Args:
            room_name: Name of room
            current_temp: Current room temperature
            outside_temp: Outside temperature
            target_temp: Target temperature
            energy_prices: Prices for next N hours
            occupancy_probs: Occupancy probabilities for next N hours
            current_action: Current heating action
            
        Returns:
            Dict with optimal actions and cost breakdown
        """
        # Initial guess: maintain current action
        initial_guess = np.full(min(self.horizon, len(energy_prices)), current_action)
        
        # Bounds for heating actions [0, 1]
        bounds = [(0.0, 1.0) for _ in initial_guess]
        
        # Objective function to minimize
        def objective(actions):
            # Predict temperatures with these actions
            predictions = self.thermal_model.predict_temperature(
                current_temp,
                outside_temp,
                actions.tolist(),
                len(actions)
            )
            
            # Pad predictions if needed
            while len(predictions) < len(actions):
                predictions.append(predictions[-1])
            
            return self.cost_function.compute_cost(
                actions.tolist(),
                predictions,
                target_temp,
                energy_prices[:len(actions)],
                occupancy_probs[:len(actions)],
                current_action
            )
        
        # Optimize
        try:
            result = minimize(
                objective,
                initial_guess,
                bounds=bounds,
                method="L-BFGS-B"
            )
            
            optimal_actions = result.x.tolist()
            optimal_cost = result.fun
            
        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            optimal_actions = initial_guess.tolist()
            optimal_cost = objective(initial_guess)
        
        # Get predicted temperatures
        predicted_temps = self.thermal_model.predict_temperature(
            current_temp,
            outside_temp,
            optimal_actions,
            len(optimal_actions)
        )
        
        return {
            "room": room_name,
            "optimal_actions": optimal_actions,
            "predicted_temperatures": predicted_temps,
            "total_cost": optimal_cost,
            "next_action": optimal_actions[0] if optimal_actions else 0.0,
            "horizon": len(optimal_actions)
        }
