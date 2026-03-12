"""
IntelliWarm Flask Application
Integrates with SRS architecture modules
"""

from flask import Flask, render_template, request, redirect, url_for, jsonify
import threading
import time
import pandas as pd
import logging
import os
import sys

# Add intelliwarm package to path
sys.path.insert(0, os.path.dirname(__file__))

from intelliwarm.core import SystemConfig, SystemScheduler
from intelliwarm.sensors import SensorManager
from intelliwarm.models import RoomThermalModel
from intelliwarm.prediction import OccupancyPredictor
from intelliwarm.pricing import EnergyPriceService
from intelliwarm.optimizer import MPCController, CostFunction
from intelliwarm.control import DeviceController
from intelliwarm.storage import Database
from intelliwarm.learning import Trainer

# Initialize Flask
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IntelliWarm.App")

# Initialize core components
config = SystemConfig("configs/config.yaml")
database = Database(os.path.join(os.path.dirname(__file__), "intelliwarm.db"))
sensor_manager = SensorManager()
device_controller = DeviceController()
energy_service = EnergyPriceService(
    config.electricity_price,
    config.gas_price
)
scheduler = SystemScheduler()

# Initialize room-specific models
thermal_models = {}
occupancy_predictors = {}
mpc_controllers = {}

# Legacy data structures
rooms = []
zones = []
utility_rates = {
    "electricity_price": config.electricity_price,
    "gas_price": config.gas_price
}

# Demo timeline state
demo_data = pd.DataFrame()
demo_timestamps = []
demo_loaded = False


def _load_demo_dataset() -> bool:
    """Load CSV demo dataset and initialize demo rooms/models."""
    global rooms, zones, demo_data, demo_timestamps, demo_loaded

    rooms.clear()
    zones.clear()
    thermal_models.clear()
    occupancy_predictors.clear()
    mpc_controllers.clear()

    file_path = "data-analytics/roommate_data/roommates_occupancy.csv"
    if not os.path.exists(file_path):
        logger.error(f"Demo data file not found: {file_path}")
        demo_loaded = False
        return False

    try:
        data = pd.read_csv(file_path)
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        data["occupied"] = data["occupied"].astype(int)
        data["zone"] = data["zone"].astype(str)

        demo_data = data
        demo_timestamps = sorted(data["timestamp"].dropna().unique().tolist())

        unique_zones = sorted(data["zone"].unique().tolist())
        for zone in unique_zones:
            zones.append({"name": f"Zone {zone}", "description": f"Zone {zone} description"})

        # Initialize rooms from first timestamp snapshot.
        first_ts = demo_timestamps[0]
        first_snapshot = data[data["timestamp"] == first_ts]
        for _, row in first_snapshot.iterrows():
            room_name = row["room"]
            occupied = bool(int(row["occupied"]))

            database.add_room(room_name, f"Zone {row['zone']}", 150, 21, 1500)
            sensor_manager.register_temperature_sensor(room_name, 21.0 if occupied else 18.5)
            sensor_manager.register_occupancy_sensor(room_name, occupied)

            room_config = {
                "zone": f"Zone {row['zone']}",
                "target_temp": 21,
                "heating_efficiency": 0.85,
                "thermal_mass": 0.05,
                "occupancy_schedule": "9-18"
            }
            _initialize_room_models(room_name, room_config)

            rooms.append({
                "name": room_name,
                "zone": f"Zone {row['zone']}",
                "temperature": 70 if occupied else 66,
                "humidity": 44,
                "occupancy": occupied,
                "heating_source": "Furnace" if occupied else "Off"
            })

        demo_loaded = True
        logger.info(f"Demo loaded: {len(rooms)} rooms, {len(demo_timestamps)} timeline points")
        return True

    except Exception as e:
        logger.error(f"Demo loading failed: {e}")
        demo_loaded = False
        return False


def _initialize_room_models(room_name: str, room_config: dict):
    """Initialize thermal model, occupancy predictor, and MPC controller for a room"""
    # Thermal model
    thermal_models[room_name] = RoomThermalModel(
        room_name,
        alpha=room_config.get("heating_efficiency", 0.1),
        beta=room_config.get("thermal_mass", 0.05)
    )
    
    # Occupancy predictor
    occupancy_predictors[room_name] = OccupancyPredictor(
        room_name,
        schedule=room_config.get("occupancy_schedule", "")
    )
    
    # MPC controller
    cost_func = CostFunction(config)
    mpc_controllers[room_name] = MPCController(
        config,
        thermal_models[room_name],
        cost_func
    )
    
    # Register device
    device_controller.register_device(room_name)
    
    logger.info(f"Initialized models for room: {room_name}")


def _optimize_heating_plan(
    room_name: str,
    occupancy_override: list = None,
    target_temp_override: float = None,
    current_action_override: float = None
):
    """Compute optimal heating plan for a room"""
    if room_name not in thermal_models:
        logger.warning(f"Room not initialized: {room_name}")
        return None
    
    try:
        # Get current state
        current_temp = sensor_manager.get_temperature(room_name)
        room_config = config.get_room_config(room_name)
        target_temp = target_temp_override if target_temp_override is not None else room_config.get("target_temp", 21)
        outside_temp = 5.0  # TODO: Get from weather API
        
        # Get forecasts
        occupancy_probs = occupancy_override if occupancy_override is not None else occupancy_predictors[room_name].predict_occupancy_horizon(
            config.optimization_horizon
        )
        energy_forecast = energy_service.get_price_forecast(config.optimization_horizon)
        energy_prices = [p["electricity"] for p in energy_forecast]
        
        # Current heating action
        device_status = device_controller.get_device_status(room_name)
        current_action = current_action_override if current_action_override is not None else (device_status["power_level"] if device_status else 0.0)
        
        # Compute optimal plan
        plan = mpc_controllers[room_name].compute_optimal_plan(
            room_name,
            current_temp,
            outside_temp,
            target_temp,
            energy_prices,
            occupancy_probs,
            current_action
        )
        
        # Execute first action
        if plan and "next_action" in plan:
            device_controller.set_heater(room_name, plan["next_action"])
            database.record_optimization(room_name, plan["next_action"], plan["total_cost"])
        
        return plan
    
    except Exception as e:
        logger.error(f"Optimization failed for {room_name}: {e}")
        return None


@app.route("/")
def dashboard():
    return render_template("dashboard.html", rooms=rooms)


@app.route("/api/rooms", methods=["GET"])
def api_get_rooms():
    """Get all rooms with current state"""
    rooms_data = []
    for room_name in thermal_models.keys():
        state = sensor_manager.get_room_state(room_name)
        device_status = device_controller.get_device_status(room_name)
        room_config = config.get_room_config(room_name)
        
        rooms_data.append({
            **state,
            "zone": room_config.get("zone", "Unknown"),
            "target_temp": room_config.get("target_temp", 21),
            "device_status": device_status
        })
    
    return jsonify(rooms_data)


@app.route("/api/optimization/<room_name>", methods=["GET"])
def api_get_optimization(room_name: str):
    """Get optimization plan for a room"""
    plan = _optimize_heating_plan(room_name)
    return jsonify(plan) if plan else jsonify({"error": "Optimization failed"}), 500




@app.route('/add_room', methods=['GET', 'POST'])
def add_room():
    if request.method == 'POST':
        name = request.form.get('roomName')
        room_size = request.form.get('roomSize')
        zone = request.form.get('zone')

        if name:
            from random import uniform
            
            # Add to database
            database.add_room(
                name,
                zone,
                float(room_size),
                config.default_target_temp,
                1500  # Default heater power
            )
            
            # Initialize sensors
            sensor_manager.register_temperature_sensor(name, uniform(18, 22))
            sensor_manager.register_occupancy_sensor(name, False)
            
            # Initialize models
            room_config = {
                "zone": zone,
                "target_temp": config.default_target_temp,
                "heating_efficiency": 0.85,
                "thermal_mass": 0.05,
                "occupancy_schedule": "9-18"
            }
            _initialize_room_models(name, room_config)
            
            # Add to legacy list
            room = {
                'name': name,
                'room_size': int(room_size),
                'zone': zone,
                'temperature': round(uniform(65, 75), 1),
                'humidity': round(uniform(35, 55), 1),
                'occupancy': False,
                'heating_source': 'Off'
            }
            rooms.append(room)
            logger.info(f"Room added: {name}")
            return redirect(url_for('add_room'))

    return render_template('add_room.html', zones=[z["name"] for z in zones])


@app.route('/config_home', methods=['GET', 'POST'])
def config_home():
    global utility_rates

    if request.method == 'POST':
        zone_name = request.form.get('zoneName')
        zone_desc = request.form.get('zoneDescription')

        if zone_name:
            zones.append({'name': zone_name, 'description': zone_desc})
            logger.info(f"Zone added: {zone_name}")

        elec_price = request.form.get('electricityPrice')
        gas_price = request.form.get('gasPrice')

        if elec_price:
            price = float(elec_price)
            energy_service.set_electricity_price(price)
            utility_rates['electricity_price'] = price

        if gas_price:
            price = float(gas_price)
            energy_service.set_gas_price(price)
            utility_rates['gas_price'] = price

        return redirect(url_for('config_home'))

    return render_template('config_home.html', zones=zones, rates=utility_rates)


@app.route('/demo')
def demo():
    success = _load_demo_dataset()
    if not success:
        return redirect(url_for('dashboard'))
    return redirect(url_for('demo_timeline'))


@app.route('/demo_timeline')
def demo_timeline():
    if not demo_loaded:
        _load_demo_dataset()
    return render_template('demo_timeline.html')


@app.route('/api/demo/timeline/meta', methods=['GET'])
def api_demo_timeline_meta():
    if not demo_loaded:
        if not _load_demo_dataset():
            return jsonify({"error": "Demo dataset not available"}), 500

    room_names = sorted(demo_data["room"].unique().tolist()) if not demo_data.empty else []
    return jsonify({
        "total_points": len(demo_timestamps),
        "start": str(demo_timestamps[0]) if demo_timestamps else None,
        "end": str(demo_timestamps[-1]) if demo_timestamps else None,
        "rooms": room_names
    })


@app.route('/api/demo/timeline/point', methods=['GET'])
def api_demo_timeline_point():
    if not demo_loaded:
        if not _load_demo_dataset():
            return jsonify({"error": "Demo dataset not available"}), 500

    if not demo_timestamps:
        return jsonify({"error": "No timeline points found"}), 500

    index = request.args.get('index', default=0, type=int)
    index = max(0, min(index, len(demo_timestamps) - 1))

    ts = demo_timestamps[index]
    frame = demo_data[demo_data["timestamp"] == ts]

    # If no room in a zone is occupied at this timestamp, keep the whole zone off in demo mode.
    zone_occupied_map = frame.groupby("zone")["occupied"].max().to_dict()

    rooms_state = []
    actions = []
    costs = []

    for _, row in frame.iterrows():
        room_name = row["room"]
        occupied = bool(int(row["occupied"]))
        zone_name = row["zone"]
        zone_occupied = bool(int(zone_occupied_map.get(zone_name, 0)))

        # Simulated current temperature in Celsius for MPC input.
        current_temp_c = 21.0 if occupied else 18.5
        sensor_manager.set_occupancy(room_name, occupied)
        sensor_manager.set_temperature(room_name, current_temp_c)

        # Demo uses dataset occupancy at this timestamp rather than schedule-based occupancy.
        occupancy_horizon = [1.0 if occupied else 0.0] * config.optimization_horizon
        target_temp = 21.0 if occupied else config.min_temperature

        plan = _optimize_heating_plan(
            room_name,
            occupancy_override=occupancy_horizon,
            target_temp_override=target_temp,
            current_action_override=0.0
        )
        next_action = plan["next_action"] if plan else 0.0
        total_cost = float(plan["total_cost"]) if plan else 0.0

        if not zone_occupied:
            next_action = 0.0

        actions.append(next_action)
        costs.append(total_cost)

        # UI uses Fahrenheit for consistency with existing dashboard cards.
        current_temp_f = round((current_temp_c * 9 / 5) + 32, 1)
        rooms_state.append({
            "room": room_name,
            "zone": f"Zone {zone_name}",
            "occupied": occupied,
            "temperature_f": current_temp_f,
            "next_action": round(next_action, 3),
            "heating_source": "Furnace" if next_action >= 0.35 else "Off",
            "predicted_cost": round(total_cost, 3)
        })

    avg_action = float(sum(actions) / len(actions)) if actions else 0.0
    total_cost = float(sum(costs)) if costs else 0.0

    return jsonify({
        "index": index,
        "timestamp": str(ts),
        "summary": {
            "occupied_rooms": int(sum(1 for r in rooms_state if r["occupied"])),
            "total_rooms": len(rooms_state),
            "avg_heating_action": round(avg_action, 3),
            "total_predicted_cost": round(total_cost, 3)
        },
        "rooms": rooms_state
    })


def run_optimization_loop():
    """Background task: run optimization every poll_interval seconds"""
    while True:
        try:
            for room_name in thermal_models.keys():
                _optimize_heating_plan(room_name)
            time.sleep(config.poll_interval)
        except Exception as e:
            logger.error(f"Optimization loop error: {e}")
            time.sleep(5)


if __name__ == '__main__':
    # Start optimization thread
    opt_thread = threading.Thread(target=run_optimization_loop, daemon=True)
    opt_thread.start()
    logger.info("Started optimization loop")
    
    # Start Flask
    app.run(debug=config.debug, port=5000)
