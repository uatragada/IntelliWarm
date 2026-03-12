"""
Simulation and typed domain model tests.
"""

from datetime import datetime
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from intelliwarm.data import HeatingAction, OccupancyWindow, RoomConfig
from intelliwarm.models import HouseSimulator, RoomThermalModel
from intelliwarm.prediction import OccupancyPredictor


def test_room_config_parses_legacy_schedule():
    config = RoomConfig.from_legacy_config(
        "bedroom1",
        {
            "zone": "Residential",
            "target_temp": 21,
            "heater_power": 1500,
            "thermal_mass": 0.07,
            "heating_efficiency": 0.9,
            "occupancy_schedule": "9-18",
        },
    )

    assert config.room_id == "bedroom1"
    assert config.target_min_temp == 20.0
    assert config.target_max_temp == 22.0
    assert config.heat_loss_factor == 0.07
    assert len(config.occupancy_schedule) == 7


def test_occupancy_predictor_supports_timestamp_windows():
    predictor = OccupancyPredictor(
        "office",
        schedule=[OccupancyWindow(day_of_week=2, start_hour=9, end_hour=17, probability=0.9)],
    )

    occupied_time = datetime(2026, 3, 11, 10, 0)
    empty_time = datetime(2026, 3, 11, 20, 0)

    assert predictor.predict(occupied_time) == 0.9
    assert predictor.predict(empty_time) == 0.1


def test_thermal_model_step_matches_equation():
    model = RoomThermalModel("bedroom", alpha=0.8, beta=0.1)

    next_temp = model.step(
        current_temp=20.0,
        outside_temp=10.0,
        heating_power=0.5,
        dt_minutes=60,
    )

    assert next_temp == 19.4


def test_house_simulator_runs_deterministic_multi_room_day():
    room_configs = {
        "bedroom": RoomConfig.from_legacy_config(
            "bedroom",
            {
                "zone": "Residential",
                "target_temp": 21,
                "heater_power": 1500,
                "thermal_mass": 0.05,
                "heating_efficiency": 1.6,
                "occupancy_schedule": "9-18",
            },
        ),
        "office": RoomConfig.from_legacy_config(
            "office",
            {
                "zone": "Work",
                "target_temp": 21,
                "heater_power": 1000,
                "thermal_mass": 0.06,
                "heating_efficiency": 1.2,
                "occupancy_schedule": [
                    {
                        "day_of_week": 2,
                        "start_hour": 8,
                        "end_hour": 17,
                        "probability": 0.95,
                    }
                ],
            },
        ),
    }
    thermal_models = {
        "bedroom": RoomThermalModel("bedroom", alpha=1.6, beta=0.05),
        "office": RoomThermalModel("office", alpha=1.2, beta=0.06),
    }
    occupancy_predictors = {
        room_name: OccupancyPredictor(room_name, room_config.occupancy_schedule)
        for room_name, room_config in room_configs.items()
    }
    simulator = HouseSimulator(room_configs, thermal_models, occupancy_predictors)

    start_time = datetime(2026, 3, 11, 8, 0)
    outdoor_temps = [8.0] * 24
    heating_plan = []
    for step_index in range(24):
        hour = (start_time.hour + step_index) % 24
        if 8 <= hour < 18:
            heating_plan.append({"bedroom": HeatingAction.COMFORT, "office": HeatingAction.PREHEAT})
        else:
            heating_plan.append({"bedroom": HeatingAction.OFF, "office": HeatingAction.ECO})

    states = simulator.simulate(
        start_time=start_time,
        initial_temperatures={"bedroom": 19.0, "office": 18.5},
        outdoor_temperatures=outdoor_temps,
        heating_plan=heating_plan,
        dt_minutes=60,
    )

    assert len(states) == 25
    assert states[0].timestamp == start_time
    assert states[-1].timestamp.hour == 8
    assert states[1].room_temperatures["bedroom"] > states[0].room_temperatures["bedroom"]
    assert states[1].occupancy["bedroom"] >= 0.8
    assert states[10].occupancy["bedroom"] == 0.1