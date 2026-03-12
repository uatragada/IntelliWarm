"""
Unit tests for IntelliWarm modules
Run with: pytest tests/
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from intelliwarm.data import HeatingAction, RoomConfig
from datetime import datetime

from intelliwarm.models import RoomThermalModel
from intelliwarm.optimizer import MPCController, CostFunction
from intelliwarm.prediction import OccupancyPredictor
from intelliwarm.pricing import CallbackPriceProvider, EnergyPriceService, StaticPriceProvider
from intelliwarm.sensors import HardwareSensorBackend, SensorManager
from intelliwarm.control import BaselineController, DeviceController, HardwareDeviceBackend
from intelliwarm.core import SystemConfig
from intelliwarm.services import ForecastBundleService


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

        forecast = service.get_price_forecast(24, start_time=datetime(2026, 3, 11, 8, 0))

        assert len(forecast) == 24
        assert all("electricity" in p and "gas" in p for p in forecast)
        assert all(p["electricity"] > 0 for p in forecast)

    def test_static_price_provider_returns_constant_forecast(self):
        service = EnergyPriceService(0.12, 5.0, provider=StaticPriceProvider())

        forecast = service.get_price_forecast(3, start_time=datetime(2026, 3, 11, 8, 0))

        assert [point["electricity"] for point in forecast] == [0.12, 0.12, 0.12]
        assert [point["gas"] for point in forecast] == [5.0, 5.0, 5.0]

    def test_callback_price_provider_can_override_current_and_forecast_prices(self):
        provider = CallbackPriceProvider(
            current_reader=lambda: {"electricity": 0.22, "gas": 1.75},
            forecast_reader=lambda hours, start_time: [
                {
                    "hour": (start_time.hour + offset) % 24,
                    "electricity": 0.20 + offset,
                    "gas": 1.50,
                }
                for offset in range(hours)
            ],
        )
        service = EnergyPriceService(0.12, 5.0, provider=provider)

        assert service.get_current_electricity_price() == 0.22
        assert service.get_current_gas_price() == 1.75

        forecast = service.get_price_forecast(2, start_time=datetime(2026, 3, 11, 8, 0))
        assert forecast[0]["electricity"] == 0.20
        assert forecast[1]["electricity"] == 1.20
        assert forecast[0]["gas"] == 1.50


class TestForecastBundleService:
    """Test aligned forecast bundle generation."""

    def test_build_bundle_aligns_forecast_horizons(self):
        service = ForecastBundleService(EnergyPriceService(0.12, 5.0))
        predictor = OccupancyPredictor("office", schedule="9-18")

        bundle = service.build_bundle(
            room_name="office",
            occupancy_predictor=predictor,
            horizon_steps=6,
            start_time=datetime(2026, 3, 11, 8, 0),
        )

        assert bundle.room_id == "office"
        assert len(bundle.steps) == 6
        assert bundle.steps[0].timestamp.hour == 8
        assert bundle.steps[1].timestamp.hour == 9
        assert bundle.occupancy_probabilities[1] > bundle.occupancy_probabilities[0]
        assert len(bundle.electricity_prices) == 6
        assert len(bundle.outdoor_temperatures) == 6

    def test_override_bundle_replaces_aligned_values(self):
        service = ForecastBundleService(EnergyPriceService(0.12, 5.0))
        predictor = OccupancyPredictor("office", schedule="9-18")
        bundle = service.build_bundle(
            room_name="office",
            occupancy_predictor=predictor,
            horizon_steps=3,
            start_time=datetime(2026, 3, 11, 8, 0),
        )

        updated = service.override_bundle(
            bundle,
            occupancy_probabilities=[0.2, 0.4, 0.6],
            outdoor_temperatures=[3.0, 4.0, 5.0],
        )

        assert updated.occupancy_probabilities == [0.2, 0.4, 0.6]
        assert updated.outdoor_temperatures == [3.0, 4.0, 5.0]


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

    def test_hardware_sensor_backend_falls_back_to_simulation(self):
        backend = HardwareSensorBackend(
            temperature_reader=lambda room_name: None,
            occupancy_reader=lambda room_name: (_ for _ in ()).throw(RuntimeError("offline")),
        )
        manager = SensorManager(backend=backend)
        manager.register_temperature_sensor("bedroom", 20.0)
        manager.register_occupancy_sensor("bedroom", True)

        state = manager.get_room_state("bedroom")

        assert state["temperature"] == 20.0
        assert state["occupancy"] is True
        assert state["sensor_source"] == "simulated_fallback"


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

    def test_hardware_device_backend_reports_hardware_status(self):
        commands = []
        controller = DeviceController.with_hardware_fallback(
            enable_hardware=True,
            default_device_id="thermostat-1",
            command_writer=lambda device_id, level: commands.append((device_id, level)),
            status_reader=lambda device_id: {
                "is_on": True,
                "power_level": 0.5,
                "power_watts": 750.0,
            },
        )
        controller.register_device("bedroom")

        controller.set_heater("bedroom", 0.5)
        status = controller.get_device_status("bedroom")

        assert commands == [("thermostat-1", 0.5)]
        assert status["control_source"] == "hardware"
        assert status["device_id"] == "thermostat-1"

    def test_hardware_device_backend_falls_back_to_simulation(self):
        controller = DeviceController.with_hardware_fallback(
            enable_hardware=True,
            default_device_id="thermostat-1",
        )
        controller.register_device("bedroom")

        controller.set_heater("bedroom", 0.4)
        status = controller.get_device_status("bedroom")

        assert status["power_level"] == 0.4
        assert status["control_source"] == "simulated_fallback"

    def test_zone_furnace_control(self):
        controller = DeviceController()
        controller.register_furnace("Residential")

        controller.set_zone_furnace("Residential", 0.7)
        status = controller.get_zone_furnace_status("Residential")

        assert status["zone"] == "Residential"
        assert status["is_on"] is True
        assert status["power_level"] == 0.7

        controller.turn_off_zone_furnace("Residential")
        status = controller.get_zone_furnace_status("Residential")
        assert status["power_level"] == 0.0
        assert status["is_on"] is False

    def test_hardware_furnace_backend_reports_hardware_status(self):
        commands = []
        controller = DeviceController.with_hardware_fallback(
            enable_hardware=True,
            default_furnace_id="furnace-1",
            command_writer=lambda device_id, level: commands.append((device_id, level)),
            status_reader=lambda device_id: {
                "is_on": True,
                "power_level": 0.8,
            },
        )
        controller.register_furnace("Residential")

        controller.set_zone_furnace("Residential", 0.8)
        status = controller.get_zone_furnace_status("Residential")

        assert commands == [("furnace-1", 0.8)]
        assert status["control_source"] == "hardware"
        assert status["device_id"] == "furnace-1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
