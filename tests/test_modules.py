"""
Unit tests for IntelliWarm modules
Run with: pytest tests/
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from intelliwarm.data import HeatingAction, RoomConfig
from intelliwarm.models import RoomThermalModel
from intelliwarm.optimizer import MPCController, CostFunction
from intelliwarm.prediction import OccupancyPredictor
from intelliwarm.pricing import EnergyPriceService
from intelliwarm.sensors import SensorManager
from intelliwarm.control import BaselineController, DeviceController
from intelliwarm.core import SystemConfig


class MockConfig:
    """Mock configuration for testing"""
    debug = True
    poll_interval = 30
    optimization_horizon = 24
    max_optimization_time = 2.0
    min_temperature = 18
    max_temperature = 24
    default_target_temp = 21
    comfort_weight = 1.0
    switching_weight = 0.5
    energy_weight = 1.0
    electricity_price = 0.12
    gas_price = 5.0
    
    def get_room_config(self, room_name):
        return {"target_temp": 21}


# ============================================================================
# Thermal Model Tests
# ============================================================================

class TestThermalModel:
    """Test thermal dynamics model"""
    
    def test_initialization(self):
        """Test model initialization"""
        model = RoomThermalModel("bedroom", alpha=0.1, beta=0.05)
        assert model.room_name == "bedroom"
        assert model.alpha == 0.1
        assert model.beta == 0.05
    
    def test_temperature_prediction_heating(self):
        """Test temperature increases with heating"""
        model = RoomThermalModel("bedroom", alpha=1.8, beta=0.05)
        
        # Predict with heating
        predictions = model.predict_temperature(
            current_temp=20.0,
            outside_temp=5.0,
            heating_actions=[0.5] * 3,  # 50% heating
            hours=3
        )
        
        # Temperature should increase
        assert len(predictions) == 3
        assert predictions[0] > 20.0  # First hour should be warmer
        assert all(predictions[i] <= predictions[i+1] for i in range(len(predictions)-1))
    
    def test_temperature_prediction_no_heating(self):
        """Test temperature decreases without heating (cooling to outside)"""
        model = RoomThermalModel("bedroom", alpha=0.1, beta=0.1)
        
        # Predict with no heating (cool room vs warm outside)
        predictions = model.predict_temperature(
            current_temp=25.0,
            outside_temp=20.0,
            heating_actions=[0.0] * 3,
            hours=3
        )
        
        # Temperature should decrease toward outside temp
        assert len(predictions) == 3
        assert predictions[-1] < 25.0
    
    def test_parameter_estimation(self):
        """Test parameter learning from data"""
        model = RoomThermalModel("bedroom", alpha=0.05, beta=0.02)
        
        # Generate synthetic data
        historical_data = [
            (20.0, 0.5, 5.0, 20.2),
            (20.2, 0.5, 5.0, 20.4),
            (20.4, 0.5, 5.0, 20.6),
            (20.6, 0.0, 5.0, 20.3),
            (20.3, 0.0, 5.0, 20.0),
        ] * 3
        
        # Estimate (should improve from initial values)
        model.estimate_parameters(historical_data)
        alpha, beta = model.get_parameters()
        
        # Parameters should be positive and reasonable
        assert alpha > 0
        assert beta > 0


# ============================================================================
# Optimizer Tests
# ============================================================================

class TestOptimizer:
    """Test MPC optimizer"""
    
    def test_cost_function_energy_term(self):
        """Test energy cost calculation"""
        config = MockConfig()
        cost_fn = CostFunction(config)
        
        # More heating = higher cost
        actions1 = [0.5] * 6
        actions2 = [0.1] * 6
        temps = [20.0] * 6
        prices = [0.12] * 6
        occ = [0.5] * 6
        
        cost1 = cost_fn.compute_cost(actions1, temps, 21, prices, occ)
        cost2 = cost_fn.compute_cost(actions2, temps, 21, prices, occ)
        
        assert cost1 > cost2  # More heating is more expensive
    
    def test_cost_function_comfort_penalty(self):
        """Test discomfort penalty"""
        config = MockConfig()
        cost_fn = CostFunction(config)
        
        # Cold room = higher penalty
        actions = [0.0] * 6
        cold_temps = [15.0] * 6  # Below min (18°C)
        warm_temps = [22.0] * 6  # Within comfort (18-24°C)
        prices = [0.12] * 6
        occ = [0.8] * 6
        
        cost_cold = cost_fn.compute_cost(actions, cold_temps, 21, prices, occ)
        cost_warm = cost_fn.compute_cost(actions, warm_temps, 21, prices, occ)
        
        assert cost_cold > cost_warm
    
    def test_mpc_optimization(self):
        """Test MPC optimization produces valid actions"""
        config = MockConfig()
        model = RoomThermalModel("bedroom", alpha=0.1, beta=0.05)
        optimizer = MPCController(config, model)
        
        plan = optimizer.compute_optimal_plan(
            room_name="bedroom",
            current_temp=18.0,
            outside_temp=5.0,
            target_temp=21.0,
            energy_prices=[0.12] * 24,
            occupancy_probs=[0.8] * 12 + [0.1] * 12,
            current_action=0.0
        )
        
        # Check output structure
        assert "optimal_actions" in plan
        assert "predicted_temperatures" in plan
        assert "total_cost" in plan
        assert "next_action" in plan
        
        # Actions should be normalized [0, 1]
        assert all(0 <= a <= 1 for a in plan["optimal_actions"])
        assert plan["total_cost"] >= 0


class TestBaselineController:
    """Test explainable baseline control decisions."""

    def _controller(self):
        return BaselineController(
            room_config=RoomConfig.from_legacy_config(
                "bedroom",
                {
                    "zone": "Residential",
                    "target_temp": 21,
                    "target_min_temp": 20,
                    "target_max_temp": 22,
                    "heater_power": 1500,
                    "thermal_mass": 0.05,
                    "heating_efficiency": 0.9,
                    "occupancy_schedule": "9-18",
                },
            ),
            min_temperature=18,
            max_temperature=24,
        )

    def test_baseline_selects_preheat_for_cold_occupied_room(self):
        controller = self._controller()

        decision = controller.compute_decision(
            current_temp=18.8,
            occupancy_forecast=[0.9] * 6,
            energy_prices=[0.12] * 6,
            current_action=0.0,
            outside_temp=0.0,
            target_temp=21.0,
        )

        assert decision.action == HeatingAction.PREHEAT
        assert "occupied" in decision.rationale.lower()
        assert decision.to_dict()["next_action_label"] == "PREHEAT"

    def test_baseline_selects_off_when_unoccupied_and_warm(self):
        controller = self._controller()

        decision = controller.compute_decision(
            current_temp=21.5,
            occupancy_forecast=[0.1] * 6,
            energy_prices=[0.12] * 6,
            current_action=0.0,
            outside_temp=6.0,
            target_temp=21.0,
        )

        assert decision.action == HeatingAction.OFF
        assert any("unoccupied" in reason.lower() for reason in decision.reasons)

    def test_baseline_holds_previous_action_to_reduce_chatter(self):
        controller = self._controller()

        decision = controller.compute_decision(
            current_temp=20.05,
            occupancy_forecast=[0.1] * 6,
            energy_prices=[0.12] * 6,
            current_action=HeatingAction.ECO.power_level,
            outside_temp=1.0,
            target_temp=21.0,
        )

        assert decision.action == HeatingAction.ECO
        assert any("chatter" in reason.lower() for reason in decision.reasons)


# ============================================================================
# Prediction Tests
# ============================================================================

class TestOccupancyPrediction:
    """Test occupancy prediction"""
    
    def test_occupancy_schedule_based(self):
        """Test rule-based occupancy prediction"""
        predictor = OccupancyPredictor("bedroom", schedule="9-18")
        
        # During schedule
        assert predictor.predict_occupancy(10) > 0.5
        
        # Outside schedule
        assert predictor.predict_occupancy(20) < 0.5
    
    def test_occupancy_forecast_horizon(self):
        """Test occupancy forecast"""
        predictor = OccupancyPredictor("bedroom", schedule="9-18")
        
        forecast = predictor.predict_occupancy_horizon(24)
        
        assert len(forecast) == 24
        assert all(0 <= p <= 1 for p in forecast)


# ============================================================================
# Pricing Tests
# ============================================================================

class TestEnergyPricing:
    """Test energy pricing service"""
    
    def test_price_initialization(self):
        """Test price service setup"""
        service = EnergyPriceService(0.15, 5.5)
        
        assert service.get_current_electricity_price() == 0.15
        assert service.get_current_gas_price() == 5.5
    
    def test_price_update(self):
        """Test price updates"""
        service = EnergyPriceService(0.12, 5.0)
        
        service.set_electricity_price(0.14)
        assert service.get_current_electricity_price() == 0.14
    
    def test_price_forecast(self):
        """Test price forecasting"""
        service = EnergyPriceService(0.12, 5.0)
        
        forecast = service.get_price_forecast(24)
        
        assert len(forecast) == 24
        assert all("electricity" in p and "gas" in p for p in forecast)
        assert all(p["electricity"] > 0 for p in forecast)


# ============================================================================
# Sensor Tests
# ============================================================================

class TestSensorManager:
    """Test sensor management"""
    
    def test_temperature_sensor(self):
        """Test temperature sensor registration and reading"""
        manager = SensorManager()
        manager.register_temperature_sensor("bedroom", 20.0)
        
        assert manager.get_temperature("bedroom") == 20.0
    
    def test_occupancy_sensor(self):
        """Test occupancy sensor"""
        manager = SensorManager()
        manager.register_occupancy_sensor("bedroom", True)
        
        assert manager.get_occupancy("bedroom") == True
    
    def test_room_state(self):
        """Test room state aggregation"""
        manager = SensorManager()
        manager.register_temperature_sensor("bedroom", 21.5)
        manager.register_occupancy_sensor("bedroom", False)
        
        state = manager.get_room_state("bedroom")
        
        assert state["room"] == "bedroom"
        assert state["temperature"] == 21.5
        assert state["occupancy"] == False


# ============================================================================
# Device Control Tests
# ============================================================================

class TestDeviceController:
    """Test device control"""
    
    def test_device_registration(self):
        """Test device registration"""
        controller = DeviceController()
        controller.register_device("bedroom")
        
        status = controller.get_device_status("bedroom")
        assert status["room"] == "bedroom"
        assert status["is_on"] == False
    
    def test_heater_control(self):
        """Test heater power control"""
        controller = DeviceController()
        controller.register_device("bedroom")
        
        # Turn on
        controller.set_heater("bedroom", 0.5)
        status = controller.get_device_status("bedroom")
        assert status["power_level"] == 0.5
        assert status["is_on"] == True
        
        # Turn off
        controller.turn_off("bedroom")
        status = controller.get_device_status("bedroom")
        assert status["power_level"] == 0.0
        assert status["is_on"] == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
