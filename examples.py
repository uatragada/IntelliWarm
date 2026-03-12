"""
IntelliWarm Example Usage
Demonstrates how to use the core modules
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from intelliwarm.core import SystemConfig
from intelliwarm.sensors import SensorManager
from intelliwarm.models import RoomThermalModel
from intelliwarm.prediction import OccupancyPredictor
from intelliwarm.pricing import EnergyPriceService
from intelliwarm.optimizer import MPCController, CostFunction
from intelliwarm.control import DeviceController
from intelliwarm.storage import Database


def example_basic_optimization():
    """Example 1: Basic optimization for a single room"""
    print("=" * 60)
    print("Example 1: Basic Single-Room Optimization")
    print("=" * 60)
    
    # Load configuration
    config = SystemConfig("configs/config.yaml")
    print(f"✓ Config loaded: {config.optimization_horizon}h horizon")
    
    # Initialize components
    sensors = SensorManager()
    db = Database("intelliwarm.db")
    
    # Register sensors for bedroom
    sensors.register_temperature_sensor("bedroom", initial_temp=18.0)
    sensors.register_occupancy_sensor("bedroom", initial_occupied=True)
    
    # Initialize thermal model
    thermal_model = RoomThermalModel("bedroom", alpha=0.1, beta=0.05)
    
    # Initialize occupancy predictor
    occupancy = OccupancyPredictor("bedroom", schedule="9-18")
    
    # Initialize optimizer
    cost_func = CostFunction(config)
    optimizer = MPCController(config, thermal_model, cost_func)
    
    # Initialize device controller
    devices = DeviceController()
    devices.register_device("bedroom")
    
    # Get current state
    print(f"\nCurrent State:")
    print(f"  Temperature: {sensors.get_temperature('bedroom'):.1f}°C")
    print(f"  Target: 21°C")
    print(f"  Outside: 5°C")
    
    # Get forecasts
    occupancy_probs = occupancy.predict_occupancy_horizon(24)
    print(f"\nOccupancy Prediction (next 24h):")
    print(f"  {occupancy_probs[:12]}")
    
    # Energy prices
    energy_svc = EnergyPriceService(0.12, 5.0)
    prices_forecast = energy_svc.get_price_forecast(24)
    prices = [p["electricity"] for p in prices_forecast]
    print(f"\nElectricity Prices (next 6h): {prices[:6]}")
    
    # Compute optimal plan
    plan = optimizer.compute_optimal_plan(
        room_name="bedroom",
        current_temp=18.0,
        outside_temp=5.0,
        target_temp=21.0,
        energy_prices=prices,
        occupancy_probs=occupancy_probs,
        current_action=0.0
    )
    
    print(f"\n✓ Optimization Complete:")
    print(f"  Total Cost: ${plan['total_cost']:.2f}")
    print(f"  Next Action: {plan['next_action']:.2f} (0=off, 1=full)")
    print(f"  Predicted Temps (24h): {[f'{t:.1f}' for t in plan['predicted_temperatures'][:6]]}...")
    
    # Execute heating action
    devices.set_heater("bedroom", plan["next_action"])
    status = devices.get_device_status("bedroom")
    print(f"\n  Device set to {status['power_level']:.0%} power ({status['power_watts']:.0f}W)")
    
    # Log to database
    db.record_optimization("bedroom", plan["next_action"], plan["total_cost"])
    print(f"  ✓ Logged to database")


def example_multi_room():
    """Example 2: Multi-room optimization"""
    print("\n" + "=" * 60)
    print("Example 2: Multi-Room System")
    print("=" * 60)
    
    config = SystemConfig("configs/config.yaml")
    sensors = SensorManager()
    devices = DeviceController()
    
    # Setup multiple rooms
    rooms = ["bedroom", "living_room", "office"]
    temps = [18.0, 19.0, 20.0]
    
    for room, temp in zip(rooms, temps):
        sensors.register_temperature_sensor(room, temp)
        sensors.register_occupancy_sensor(room, False)
        devices.register_device(room)
    
    print(f"\n✓ Registered {len(rooms)} rooms:")
    for room, temp in zip(rooms, temps):
        state = sensors.get_room_state(room)
        print(f"  - {room}: {state['temperature']:.1f}°C")


def example_thermal_learning():
    """Example 3: Thermal model parameter learning"""
    print("\n" + "=" * 60)
    print("Example 3: Thermal Model Learning")
    print("=" * 60)
    
    # Create model with initial parameters
    model = RoomThermalModel("bedroom", alpha=0.08, beta=0.04)
    alpha_0, beta_0 = model.get_parameters()
    print(f"\nInitial parameters: α={alpha_0:.4f}, β={beta_0:.4f}")
    
    # Simulate historical data: (T_current, H_action, T_outside, T_next)
    historical_data = [
        (20.0, 0.5, 5.0, 20.3),
        (20.3, 0.5, 5.0, 20.6),
        (20.6, 0.3, 5.0, 20.5),
        (20.5, 0.0, 5.0, 20.2),
        (20.2, 0.8, 5.0, 20.8),
    ] * 5  # Repeat to have enough data
    
    # Estimate parameters from historical data
    model.estimate_parameters(historical_data)
    alpha_new, beta_new = model.get_parameters()
    print(f"Learned parameters:  α={alpha_new:.4f}, β={beta_new:.4f}")
    
    # Predict with new parameters
    temps = model.predict_temperature(
        current_temp=20.0,
        outside_temp=5.0,
        heating_actions=[0.5] * 8,
        hours=8
    )
    print(f"\nPredicted temps (8h with 50% heating): {[f'{t:.1f}' for t in temps]}")


def example_cost_analysis():
    """Example 4: Cost analysis for different strategies"""
    print("\n" + "=" * 60)
    print("Example 4: Cost Sensitivity Analysis")
    print("=" * 60)
    
    config = SystemConfig("configs/config.yaml")
    model = RoomThermalModel("bedroom", alpha=0.1, beta=0.05)
    cost_fn = CostFunction(config)
    
    # Scenario: Room needs to heat from 18°C to 21°C
    current_temp = 18.0
    target_temp = 21.0
    outside_temp = 5.0
    
    # Compare strategies
    strategies = {
        "Aggressive (100% power)": [1.0] * 6,
        "Moderate (50% power)": [0.5] * 6,
        "Gentle (25% power)": [0.25] * 6,
        "Minimal (10% power)": [0.1] * 6,
    }
    
    energy_prices = [0.12] * 6
    occ_probs = [0.8] * 6
    
    print(f"\nCost comparison (target: {target_temp}°C, current: {current_temp}°C):\n")
    
    for name, actions in strategies.items():
        temps = model.predict_temperature(current_temp, outside_temp, actions, 6)
        cost = cost_fn.compute_cost(actions, temps, target_temp, energy_prices, occ_probs)
        final_temp = temps[-1]
        
        print(f"  {name:30} → Final: {final_temp:.1f}°C | Cost: ${cost:.2f}")


if __name__ == "__main__":
    print("\n" + "🌡️  IntelliWarm Examples\n")
    
    try:
        example_basic_optimization()
        example_multi_room()
        example_thermal_learning()
        example_cost_analysis()
        
        print("\n" + "=" * 60)
        print("✓ All examples completed successfully!")
        print("=" * 60 + "\n")
    
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()
