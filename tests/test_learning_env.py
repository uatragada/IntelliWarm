"""
Tests for the Gym-compatible IntelliWarm training environment.
"""

from datetime import datetime
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from intelliwarm.data import RoomConfig
from intelliwarm.data import HeatSourceType, HeatingAction
from intelliwarm.learning import (
    IntelliWarmMultiRoomEnv,
    IntelliWarmRoomEnv,
    SyntheticScenarioGenerator,
    build_policy_catalog,
    constant_policy,
    evaluate_named_policies,
    evaluate_policy,
)
from intelliwarm.models import RoomThermalModel
from intelliwarm.prediction import OccupancyPredictor
from intelliwarm.pricing import EnergyPriceService, StaticPriceProvider
from scripts.evaluate_policies import main as evaluate_policies_main


def _build_env():
    room_config = RoomConfig.from_legacy_config(
        "bedroom",
        {
            "zone": "Residential",
            "target_temp": 21,
            "target_min_temp": 20,
            "target_max_temp": 22,
            "heater_power": 1500,
            "thermal_mass": 0.05,
            "heating_efficiency": 1.5,
            "occupancy_schedule": "9-18",
        },
    )
    thermal_model = RoomThermalModel("bedroom", alpha=1.5, beta=0.05)
    predictor = OccupancyPredictor("bedroom", schedule="9-18")
    pricing = EnergyPriceService(0.12, 5.0, provider=StaticPriceProvider())
    return IntelliWarmRoomEnv(
        room_config=room_config,
        thermal_model=thermal_model,
        occupancy_predictor=predictor,
        energy_service=pricing,
        horizon_steps=3,
        start_time=datetime(2026, 3, 11, 9, 0),
        initial_temp=19.0,
        outside_temperature_profile=[5.0, 5.0, 5.0],
    )


def test_learning_env_reset_returns_observation_and_info():
    env = _build_env()

    observation, info = env.reset()

    assert observation.shape == (7,)
    assert info["action_label"] == "OFF"
    assert observation[0] == 19.0


def test_learning_env_step_uses_discrete_action_labels():
    env = _build_env()
    env.reset()

    observation, reward, terminated, truncated, info = env.step(3)

    assert observation.shape == (7,)
    assert reward < 0
    assert terminated is False
    assert truncated is False
    assert info["action_label"] == "PREHEAT"
    assert info["energy_cost"] > 0


def test_learning_env_is_deterministic_for_same_action_sequence():
    first_env = _build_env()
    second_env = _build_env()

    first_env.reset()
    second_env.reset()

    first_rollout = [first_env.step(action) for action in (0, 2, 3)]
    second_rollout = [second_env.step(action) for action in (0, 2, 3)]

    first_rewards = [step[1] for step in first_rollout]
    second_rewards = [step[1] for step in second_rollout]
    first_temps = [float(step[0][0]) for step in first_rollout]
    second_temps = [float(step[0][0]) for step in second_rollout]

    assert first_rewards == second_rewards
    assert first_temps == second_temps
    assert first_rollout[-1][2] is True


def test_synthetic_scenario_generator_produces_multiple_zones_and_schedules():
    generator = SyntheticScenarioGenerator()

    scenarios = generator.default_scenarios()

    assert len(scenarios) >= 3
    assert any(len(scenario.zone_configs) > 1 for scenario in scenarios)
    assert any(
        any(room_config.occupancy_schedule for room_config in scenario.room_configs.values())
        for scenario in scenarios
    )


def test_multi_room_env_reset_returns_padded_multi_room_observation():
    generator = SyntheticScenarioGenerator()
    env = IntelliWarmMultiRoomEnv(generator.default_scenarios())

    observation, info = env.reset(options={"scenario_name": "winter-workday"})

    assert observation.shape == (34,)
    assert info["scenario_name"] == "winter-workday"
    assert info["zone_names"] == ["Residential", "Work"]
    assert info["room_names"] == ["bedroom", "living_room", "office"]
    assert info["active_zones"] == 2
    assert info["max_zones"] == 3
    assert info["active_rooms"] == 3
    assert info["max_rooms"] == 3


def test_multi_room_env_applies_zone_furnace_to_all_rooms_in_zone():
    generator = SyntheticScenarioGenerator()
    env = IntelliWarmMultiRoomEnv(generator.default_scenarios())
    env.reset(options={"scenario_name": "winter-workday"})

    action = [1, 0, 0, 0, 3, 2]
    observation, reward, terminated, truncated, info = env.step(action)

    assert observation.shape == (34,)
    assert reward < 0
    assert terminated is False
    assert truncated is False
    assert info["zone_heat_sources"]["Residential"] == "gas_furnace"
    assert info["effective_room_actions"]["bedroom"] == "PREHEAT"
    assert info["effective_room_actions"]["living_room"] == "PREHEAT"
    assert info["effective_room_actions"]["office"] == "COMFORT"


def test_multi_room_env_is_deterministic_for_same_scenario_and_actions():
    scenarios = SyntheticScenarioGenerator().default_scenarios()
    first_env = IntelliWarmMultiRoomEnv(scenarios)
    second_env = IntelliWarmMultiRoomEnv(scenarios)

    first_env.reset(options={"scenario_name": "weekend-family"})
    second_env.reset(options={"scenario_name": "weekend-family"})

    actions = (
        [0, 1, 0, 2, 1, 3],
        [0, 1, 0, 1, 2, 3],
    )
    first_rollout = [first_env.step(action) for action in actions]
    second_rollout = [second_env.step(action) for action in actions]

    assert [step[1] for step in first_rollout] == [step[1] for step in second_rollout]
    assert [step[4]["total_cost"] for step in first_rollout] == [step[4]["total_cost"] for step in second_rollout]


def test_constant_policy_uses_env_action_layout():
    env = IntelliWarmMultiRoomEnv(SyntheticScenarioGenerator().default_scenarios())
    _, info = env.reset(options={"scenario_name": "weekend-family"})

    policy = constant_policy(
        room_action=HeatingAction.COMFORT,
        zone_source=HeatSourceType.GAS_FURNACE,
    )
    action = policy(None, info)

    assert action == [1, 1, 1, 2, 2, 2]


def test_evaluate_policy_rolls_up_metrics_across_scenarios():
    scenarios = SyntheticScenarioGenerator().default_scenarios()
    env = IntelliWarmMultiRoomEnv(scenarios)

    summary = evaluate_policy(
        env,
        constant_policy(room_action=HeatingAction.ECO, zone_source=HeatSourceType.ELECTRIC),
        scenario_names=["winter-workday", "weekend-family"],
        max_steps=2,
    )

    assert summary.scenario_count == 2
    assert all(result.steps == 2 for result in summary.scenario_results)
    assert summary.total_cost > 0
    assert summary.total_reward < 0
    assert summary.total_comfort_violation >= 0
    assert [result.scenario_name for result in summary.scenario_results] == [
        "winter-workday",
        "weekend-family",
    ]


def test_policy_catalog_exposes_named_furnace_and_electric_policies():
    catalog = build_policy_catalog()

    assert "eco-electric" in catalog
    assert "comfort-furnace" in catalog
    assert "preheat-furnace" in catalog


def test_evaluate_named_policies_returns_summaries_for_selected_policies():
    results = evaluate_named_policies(
        ["eco-electric", "comfort-furnace"],
        scenario_names=["winter-workday"],
        max_steps=1,
    )

    assert sorted(results.keys()) == ["comfort-furnace", "eco-electric"]
    assert all(summary.scenario_count == 1 for summary in results.values())
    assert all(summary.scenario_results[0].steps == 1 for summary in results.values())


def test_evaluate_policies_script_emits_json_summary(capsys):
    exit_code = evaluate_policies_main(
        ["--policy", "eco-electric", "--scenario", "winter-workday", "--max-steps", "1", "--json"]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert '"eco-electric"' in captured.out
    assert '"winter-workday"' in captured.out
