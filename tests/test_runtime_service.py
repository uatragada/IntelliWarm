"""
Tests for the application runtime service.
"""

import logging
import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from intelliwarm.control import DeviceController
from intelliwarm.pricing import EnergyPriceService
from intelliwarm.sensors import SensorManager
from intelliwarm.services import IntelliWarmRuntime
from intelliwarm.storage import Database


class RuntimeTestConfig:
    """Minimal config contract for runtime service tests."""

    optimization_horizon = 6
    min_temperature = 18
    max_temperature = 24
    default_target_temp = 21
    comfort_weight = 1.0
    switching_weight = 0.5
    energy_weight = 1.0
    electricity_price = 0.12
    gas_price = 5.0

    def __init__(self):
        self._rooms = {
            "bedroom": {
                "zone": "Residential",
                "target_temp": 21,
            }
        }

    def get_room_config(self, room_name):
        return self._rooms.get(room_name, {})


def create_runtime(tmp_path):
    """Build a runtime instance backed by a temporary database."""
    config = RuntimeTestConfig()
    database = Database(str(tmp_path / "runtime.db"))
    return IntelliWarmRuntime(
        config=config,
        database=database,
        sensor_manager=SensorManager(),
        device_controller=DeviceController(),
        energy_service=EnergyPriceService(config.electricity_price, config.gas_price),
        logger=logging.getLogger("tests.runtime"),
    )


def test_load_demo_dataset_initializes_rooms(tmp_path):
    runtime = create_runtime(tmp_path)
    csv_path = tmp_path / "demo.csv"
    csv_path.write_text(
        "timestamp,room,occupied,zone\n"
        "2026-03-11T08:00:00,Room A,1,1\n"
        "2026-03-11T08:00:00,Room B,0,1\n"
        "2026-03-11T09:00:00,Room A,0,1\n",
        encoding="ascii",
    )

    loaded = runtime.load_demo_dataset(str(csv_path))

    assert loaded is True
    assert runtime.demo_loaded is True
    assert len(runtime.rooms) == 2
    assert len(runtime.demo_timestamps) == 2
    assert sorted(room["name"] for room in runtime.rooms) == ["Room A", "Room B"]
    assert runtime.room_names == ["Room A", "Room B"]


def test_optimize_heating_plan_updates_device_and_database(tmp_path):
    runtime = create_runtime(tmp_path)
    runtime.add_room(
        name="bedroom",
        room_size=150,
        zone="Residential",
        room_config={
            "zone": "Residential",
            "target_temp": 21,
            "heating_efficiency": 0.1,
            "thermal_mass": 0.05,
            "occupancy_schedule": "9-18",
        },
        initial_sensor_temp=19.0,
        initial_occupancy=True,
    )

    plan = runtime.optimize_heating_plan(
        "bedroom",
        occupancy_override=[1.0] * runtime.config.optimization_horizon,
        current_action_override=0.0,
    )

    assert plan is not None
    assert 0.0 <= plan["next_action"] <= 1.0

    device_status = runtime.device_controller.get_device_status("bedroom")
    assert device_status is not None
    assert device_status["power_level"] == plan["next_action"]

    connection = sqlite3.connect(runtime.database.db_path)
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM optimization_runs")
        recorded_rows = cursor.fetchone()[0]
    finally:
        connection.close()

    assert recorded_rows == 1


def test_optimize_heating_plan_supports_baseline_controller(tmp_path):
    runtime = create_runtime(tmp_path)
    runtime.add_room(
        name="bedroom",
        room_size=150,
        zone="Residential",
        room_config={
            "zone": "Residential",
            "target_temp": 21,
            "target_min_temp": 20,
            "target_max_temp": 22,
            "heating_efficiency": 0.1,
            "thermal_mass": 0.05,
            "occupancy_schedule": "9-18",
        },
        initial_sensor_temp=18.8,
        initial_occupancy=True,
    )

    plan = runtime.optimize_heating_plan(
        "bedroom",
        occupancy_override=[1.0] * runtime.config.optimization_horizon,
        current_action_override=0.0,
        controller_type="baseline",
    )

    assert plan is not None
    assert plan["controller"] == "baseline"
    assert plan["next_action_label"] == "PREHEAT"
    assert "explanation" in plan

    device_status = runtime.device_controller.get_device_status("bedroom")
    assert device_status["power_level"] == plan["next_action"]
