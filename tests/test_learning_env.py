"""
Tests for the Gym-compatible IntelliWarm training environment.
"""

from datetime import datetime
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from intelliwarm.control import (
    RoomHeatingIntent,
    ZoneSourceMode,
    room_intent_index,
    zone_source_mode_index,
)
from intelliwarm.data import OccupancyWindow, RoomConfig, ZoneConfig
from intelliwarm.data import HeatSourceType
from intelliwarm.learning import (
    IntelliWarmMultiRoomEnv,
    IntelliWarmRoomEnv,
    SyntheticScenarioGenerator,
    build_policy_catalog,
    constant_policy,
    evaluate_named_policies,
    evaluate_policy,
)
from intelliwarm.models import PhysicsRoomThermalModel, RoomThermalModel, HouseSimulator
from intelliwarm.learning.gym_env import ScenarioBoundPriceService
from intelliwarm.prediction import OccupancyPredictor
from intelliwarm.pricing import EnergyPriceService, StaticPriceProvider
from scripts.evaluate_policies import main as evaluate_policies_main
import scripts.train_opt_heating_policy as train_opt_heating_policy_script


def _env_action(
    zone_modes,
    room_intents,
):
    return [zone_source_mode_index(mode) for mode in zone_modes] + [
        room_intent_index(intent) for intent in room_intents
    ]


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
    assert info["requested_intent"] == "preheat"
    assert info["action_level"] > 0.5
    assert info["energy_cost"] > 0
    assert info["raw_reward"] == pytest.approx(reward)


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

    assert observation.shape == (52,)
    assert info["scenario_name"] == "winter-workday"
    assert info["zone_names"] == ["Residential", "Work"]
    assert info["room_names"] == ["bedroom", "living_room", "office"]
    assert info["active_zones"] == 2
    assert info["max_zones"] == 3
    assert info["active_rooms"] == 3
    assert info["max_rooms"] == 3
    assert info["occupancy_forecast_horizon_steps"] == 6


def test_multi_room_env_observation_includes_per_room_occupancy_forecast():
    generator = SyntheticScenarioGenerator()
    env = IntelliWarmMultiRoomEnv(generator.default_scenarios())

    observation, info = env.reset(options={"scenario_name": "winter-workday"})

    forecast_horizon = info["occupancy_forecast_horizon_steps"]
    room_feature_block = info["max_rooms"] * 6
    bedroom_forecast = observation[room_feature_block : room_feature_block + forecast_horizon]
    living_room_forecast = observation[
        room_feature_block + forecast_horizon : room_feature_block + 2 * forecast_horizon
    ]
    office_forecast = observation[
        room_feature_block + 2 * forecast_horizon : room_feature_block + 3 * forecast_horizon
    ]

    assert list(bedroom_forecast) == pytest.approx([0.8, 0.1, 0.1, 0.1, 0.1, 0.1])
    assert list(living_room_forecast) == pytest.approx([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    assert list(office_forecast) == pytest.approx([0.1, 0.1, 0.8, 0.8, 0.8, 0.8])


def test_multi_room_env_applies_zone_furnace_to_all_rooms_in_zone():
    generator = SyntheticScenarioGenerator()
    env = IntelliWarmMultiRoomEnv(generator.default_scenarios())
    env.reset(options={"scenario_name": "winter-workday"})

    action = _env_action(
        [ZoneSourceMode.GAS_FURNACE, ZoneSourceMode.ELECTRIC, ZoneSourceMode.AUTO],
        [RoomHeatingIntent.PREHEAT, RoomHeatingIntent.PREHEAT, RoomHeatingIntent.MAINTAIN],
    )
    observation, reward, terminated, truncated, info = env.step(action)

    assert observation.shape == (52,)
    assert reward < 0
    assert terminated is False
    assert truncated is False
    assert info["requested_zone_source_modes"]["Residential"] == "gas_furnace"
    assert info["zone_heat_sources"]["Residential"] == "gas_furnace"
    assert info["requested_room_intents"]["bedroom"] == "preheat"
    assert info["effective_room_power_levels"]["bedroom"] == pytest.approx(
        info["effective_room_power_levels"]["living_room"]
    )
    assert info["effective_room_power_levels"]["bedroom"] > 0.5
    assert info["effective_room_power_levels"]["office"] >= 0.0


def test_multi_room_env_applies_zone_source_on_same_step():
    class _PhysicsMultiRoomEnv(IntelliWarmMultiRoomEnv):
        def _build_simulator(self, scenario):
            thermal_models = {
                room_name: PhysicsRoomThermalModel.from_room_config(
                    room_config,
                    zone_config=scenario.zone_configs[room_config.zone],
                    num_zone_rooms=1,
                )
                for room_name, room_config in scenario.room_configs.items()
            }
            occupancy_predictors = {
                room_name: OccupancyPredictor(room_name, room_config.occupancy_schedule)
                for room_name, room_config in scenario.room_configs.items()
            }
            return HouseSimulator(
                room_configs=scenario.room_configs,
                thermal_models=thermal_models,
                occupancy_predictors=occupancy_predictors,
            )

    zone_config = ZoneConfig(
        zone_id="Residential",
        has_furnace=True,
        furnace_btu_per_hour=80000.0,
        furnace_efficiency=0.80,
    )
    room_configs = {
        "bedroom": RoomConfig(
            room_id="bedroom",
            display_name="Bedroom",
            zone="Residential",
            target_min_temp=20.0,
            target_max_temp=22.0,
            heater_capacity=500.0,
            heat_loss_factor=0.02,
            heating_efficiency=0.2,
        ),
    }
    scenario = SyntheticScenarioGenerator().build_scenario(
        name="source-lag-check",
        start_time=datetime(2026, 1, 5, 6, 0),
        room_configs=room_configs,
        zone_configs={"Residential": zone_config},
        initial_temperatures={"bedroom": 18.0},
        outdoor_temperatures=[5.0],
        electricity_prices=[0.20],
        gas_prices=[1.20],
    )
    env = _PhysicsMultiRoomEnv([scenario])

    obs_electric, _ = env.reset(options={"scenario_name": "source-lag-check"})
    electric_step, _, _, _, electric_info = env.step(
        _env_action([ZoneSourceMode.ELECTRIC], [RoomHeatingIntent.RECOVER])
    )

    obs_furnace, _ = env.reset(options={"scenario_name": "source-lag-check"})
    furnace_step, _, _, _, furnace_info = env.step(
        _env_action([ZoneSourceMode.GAS_FURNACE], [RoomHeatingIntent.RECOVER])
    )

    assert obs_electric[0] == obs_furnace[0]
    assert electric_info["zone_heat_sources"]["Residential"] == "electric"
    assert furnace_info["zone_heat_sources"]["Residential"] == "gas_furnace"
    assert furnace_step[0] > electric_step[0]


def test_multi_room_env_is_deterministic_for_same_scenario_and_actions():
    scenarios = SyntheticScenarioGenerator().default_scenarios()
    first_env = IntelliWarmMultiRoomEnv(scenarios)
    second_env = IntelliWarmMultiRoomEnv(scenarios)

    first_env.reset(options={"scenario_name": "weekend-family"})
    second_env.reset(options={"scenario_name": "weekend-family"})

    actions = (
        _env_action(
            [ZoneSourceMode.AUTO, ZoneSourceMode.GAS_FURNACE, ZoneSourceMode.AUTO],
            [RoomHeatingIntent.MAINTAIN, RoomHeatingIntent.PROTECT, RoomHeatingIntent.PREHEAT],
        ),
        _env_action(
            [ZoneSourceMode.AUTO, ZoneSourceMode.GAS_FURNACE, ZoneSourceMode.AUTO],
            [RoomHeatingIntent.PROTECT, RoomHeatingIntent.MAINTAIN, RoomHeatingIntent.RECOVER],
        ),
    )
    first_rollout = [first_env.step(action) for action in actions]
    second_rollout = [second_env.step(action) for action in actions]

    assert [step[1] for step in first_rollout] == [step[1] for step in second_rollout]
    assert [step[4]["total_cost"] for step in first_rollout] == [step[4]["total_cost"] for step in second_rollout]


def test_constant_policy_uses_env_action_layout():
    env = IntelliWarmMultiRoomEnv(SyntheticScenarioGenerator().default_scenarios())
    _, info = env.reset(options={"scenario_name": "weekend-family"})

    policy = constant_policy(
        room_intent=RoomHeatingIntent.MAINTAIN,
        zone_source=ZoneSourceMode.GAS_FURNACE,
    )
    action = policy(None, info)

    assert action == _env_action(
        [ZoneSourceMode.GAS_FURNACE, ZoneSourceMode.GAS_FURNACE, ZoneSourceMode.ELECTRIC],
        [RoomHeatingIntent.MAINTAIN, RoomHeatingIntent.MAINTAIN, RoomHeatingIntent.MAINTAIN],
    )


def test_evaluate_policy_rolls_up_metrics_across_scenarios():
    scenarios = SyntheticScenarioGenerator().default_scenarios()
    env = IntelliWarmMultiRoomEnv(scenarios)

    summary = evaluate_policy(
        env,
        constant_policy(room_intent=RoomHeatingIntent.PROTECT, zone_source=ZoneSourceMode.ELECTRIC),
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


def test_train_opt_heating_policy_script_emits_json_summary(tmp_path, monkeypatch, capsys):
    def _fake_train_opt_policy(**kwargs):
        assert kwargs["progress_interval"] == 5000
        return {
            "total_timesteps": kwargs["total_timesteps"],
            "n_envs": kwargs["n_envs"],
            "vec_env_type": "SubprocVecEnv",
            "device": "cpu",
            "parameter_count": 414000,
            "elapsed_seconds": 12.5,
            "episode_rewards": [1.0, 2.0],
            "episode_costs": [0.1, 0.2],
            "episode_violations": [0.3, 0.4],
            "model_path": str(tmp_path / "opt_policy_ppo.zip"),
            "summary_path": str(tmp_path / "opt_policy_training_summary.json"),
        }

    monkeypatch.setattr(train_opt_heating_policy_script, "train_opt_policy", _fake_train_opt_policy)

    exit_code = train_opt_heating_policy_script.main(
        ["--timesteps", "123", "--n-envs", "2", "--output-dir", str(tmp_path), "--json"]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert '"total_timesteps": 123' in captured.out
    assert '"n_envs": 2' in captured.out
    assert '"parameter_count": 414000' in captured.out


def test_train_opt_heating_policy_script_prints_text_summary(tmp_path, monkeypatch, capsys):
    def _fake_train_opt_policy(**kwargs):
        assert kwargs["text_progress"] is True
        assert kwargs["progress_bar"] is False
        assert kwargs["progress_interval"] == 250
        return {
            "total_timesteps": kwargs["total_timesteps"],
            "n_envs": kwargs["n_envs"],
            "vec_env_type": "DummyVecEnv",
            "device": "cpu",
            "parameter_count": 414000,
            "elapsed_seconds": 9.5,
            "episode_rewards": [1.0],
            "episode_costs": [0.1],
            "episode_violations": [0.3],
            "model_path": str(tmp_path / "opt_policy_ppo.zip"),
            "summary_path": str(tmp_path / "opt_policy_training_summary.json"),
        }

    monkeypatch.setattr(train_opt_heating_policy_script, "train_opt_policy", _fake_train_opt_policy)

    exit_code = train_opt_heating_policy_script.main(
        ["--timesteps", "500", "--n-envs", "1", "--output-dir", str(tmp_path), "--no-progress-bar", "--progress-interval", "250"]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Training complete in 9.5s using DummyVecEnv on cpu" in captured.out
    assert "Recorded 1 completed episodes" in captured.out


def test_multi_room_env_penalizes_invalid_furnace_request():
    env = IntelliWarmMultiRoomEnv(SyntheticScenarioGenerator().default_scenarios(), invalid_source_penalty=3.5)
    env.reset(options={"scenario_name": "winter-workday"})

    _, _, _, _, info = env.step(
        _env_action(
            [ZoneSourceMode.GAS_FURNACE, ZoneSourceMode.GAS_FURNACE, ZoneSourceMode.AUTO],
            [RoomHeatingIntent.OFF, RoomHeatingIntent.OFF, RoomHeatingIntent.OFF],
        )
    )

    assert info["requested_zone_source_modes"]["Work"] == "gas_furnace"
    assert info["zone_heat_sources"]["Work"] == "electric"
    assert info["invalid_source_penalty"] == pytest.approx(3.5)


def test_multi_room_env_increases_resolved_room_power_for_more_aggressive_intent():
    env = IntelliWarmMultiRoomEnv(SyntheticScenarioGenerator().default_scenarios())
    env.reset(options={"scenario_name": "weekend-family"})

    _, _, _, _, low_info = env.step(
        _env_action(
            [ZoneSourceMode.GAS_FURNACE, ZoneSourceMode.GAS_FURNACE, ZoneSourceMode.AUTO],
            [RoomHeatingIntent.MAINTAIN, RoomHeatingIntent.MAINTAIN, RoomHeatingIntent.MAINTAIN],
        )
    )
    env.reset(options={"scenario_name": "weekend-family"})
    _, _, _, _, high_info = env.step(
        _env_action(
            [ZoneSourceMode.GAS_FURNACE, ZoneSourceMode.GAS_FURNACE, ZoneSourceMode.AUTO],
            [RoomHeatingIntent.RECOVER, RoomHeatingIntent.RECOVER, RoomHeatingIntent.RECOVER],
        )
    )

    assert low_info["gas_cost"] > 0.0
    assert max(high_info["effective_room_power_levels"].values()) >= max(low_info["effective_room_power_levels"].values())


def test_scenario_bound_price_service_uses_scenario_time_offset():
    start_time = datetime(2026, 3, 11, 8, 0)
    service = ScenarioBoundPriceService(
        electricity_prices=[0.10, 0.20, 0.30],
        gas_prices=[1.0, 1.1, 1.2],
        start_time=start_time,
    )

    first = service.get_price_forecast(1, start_time=start_time)[0]
    second = service.get_price_forecast(1, start_time=start_time.replace(hour=9))[0]

    assert first["electricity"] == pytest.approx(0.10)
    assert second["electricity"] == pytest.approx(0.20)


def test_scenario_bound_price_service_uses_subhour_step_offsets():
    start_time = datetime(2026, 3, 11, 8, 0)
    service = ScenarioBoundPriceService(
        electricity_prices=[0.10, 0.11, 0.12],
        gas_prices=[1.0, 1.0, 1.0],
        start_time=start_time,
        step_minutes=5,
    )

    first = service.get_price_forecast(1, start_time=start_time)[0]
    second = service.get_price_forecast(1, start_time=start_time.replace(minute=5))[0]
    third = service.get_price_forecast(1, start_time=start_time.replace(minute=10))[0]

    assert first["electricity"] == pytest.approx(0.10)
    assert second["electricity"] == pytest.approx(0.11)
    assert third["electricity"] == pytest.approx(0.12)


def test_max_forecast_steps_caps_observation_dimension():
    scenarios = SyntheticScenarioGenerator().default_scenarios()
    env_full = IntelliWarmMultiRoomEnv(scenarios)
    env_capped = IntelliWarmMultiRoomEnv(scenarios, max_forecast_steps=2)

    obs_full, info_full = env_full.reset(options={"scenario_name": "winter-workday"})
    obs_capped, info_capped = env_capped.reset(options={"scenario_name": "winter-workday"})

    assert info_full["occupancy_forecast_horizon_steps"] == 6
    assert info_capped["occupancy_forecast_horizon_steps"] == 2
    expected_diff = info_full["max_rooms"] * (6 - 2)
    assert obs_full.shape[0] - obs_capped.shape[0] == expected_diff


def test_max_forecast_steps_none_preserves_default_behavior():
    scenarios = SyntheticScenarioGenerator().default_scenarios()
    env_default = IntelliWarmMultiRoomEnv(scenarios)
    env_none = IntelliWarmMultiRoomEnv(scenarios, max_forecast_steps=None)

    obs_default, _ = env_default.reset(options={"scenario_name": "winter-workday"})
    obs_none, _ = env_none.reset(options={"scenario_name": "winter-workday"})

    assert obs_default.shape == obs_none.shape


def test_comfort_warmup_scales_penalty_during_early_steps():
    scenarios = SyntheticScenarioGenerator().default_scenarios()
    env_no_warmup = IntelliWarmMultiRoomEnv(scenarios, comfort_penalty_weight=10.0, comfort_warmup_steps=0)
    env_warmup = IntelliWarmMultiRoomEnv(scenarios, comfort_penalty_weight=10.0, comfort_warmup_steps=4)

    env_no_warmup.reset(options={"scenario_name": "winter-workday"})
    env_warmup.reset(options={"scenario_name": "winter-workday"})

    action = _env_action(
        [ZoneSourceMode.AUTO, ZoneSourceMode.AUTO, ZoneSourceMode.AUTO],
        [RoomHeatingIntent.OFF, RoomHeatingIntent.OFF, RoomHeatingIntent.OFF],
    )
    _, reward_no_warmup, _, _, info_no = env_no_warmup.step(action)
    _, reward_warmup, _, _, info_wu = env_warmup.step(action)

    if info_no["comfort_violation"] > 0:
        assert reward_warmup > reward_no_warmup


def test_multi_room_env_global_occupancy_features_cover_true_1h_and_2h_windows():
    room_configs = {
        "office": RoomConfig(
            room_id="office",
            display_name="Office",
            zone="Office",
            target_min_temp=20.0,
            target_max_temp=22.0,
            heater_capacity=1000.0,
            heat_loss_factor=0.01,
            heating_efficiency=0.9,
            occupancy_schedule=[OccupancyWindow(0, 1, 2, 1.0)],
            heat_source=HeatSourceType.ELECTRIC,
        ),
    }
    zone_configs = {"Office": ZoneConfig(zone_id="Office", has_furnace=False)}
    scenario = SyntheticScenarioGenerator().build_scenario(
        name="occupancy-lookahead",
        start_time=datetime(2026, 1, 5, 0, 0),
        room_configs=room_configs,
        zone_configs=zone_configs,
        initial_temperatures={"office": 18.0},
        outdoor_temperatures=[5.0] * 24,
        electricity_prices=[0.10] * 24,
        gas_prices=[1.0] * 24,
        step_minutes=5,
    )
    env = IntelliWarmMultiRoomEnv([scenario], max_forecast_steps=24)

    observation, _ = env.reset(options={"scenario_name": "occupancy-lookahead"})

    assert observation[-2] == pytest.approx(1.0)
    assert observation[-1] == pytest.approx(1.0)


def test_multi_room_env_reports_preoccupancy_penalty_before_occupancy_begins():
    room_configs = {
        "office": RoomConfig(
            room_id="office",
            display_name="Office",
            zone="Office",
            target_min_temp=20.0,
            target_max_temp=22.0,
            heater_capacity=1000.0,
            heat_loss_factor=0.01,
            heating_efficiency=0.9,
            occupancy_schedule=[OccupancyWindow(0, 1, 2, 1.0)],
            heat_source=HeatSourceType.ELECTRIC,
        ),
    }
    zone_configs = {"Office": ZoneConfig(zone_id="Office", has_furnace=False)}
    scenario = SyntheticScenarioGenerator().build_scenario(
        name="preoccupancy-penalty",
        start_time=datetime(2026, 1, 5, 0, 0),
        room_configs=room_configs,
        zone_configs=zone_configs,
        initial_temperatures={"office": 16.0},
        outdoor_temperatures=[5.0] * 24,
        electricity_prices=[0.10] * 24,
        gas_prices=[1.0] * 24,
        step_minutes=5,
    )
    env = IntelliWarmMultiRoomEnv(
        [scenario],
        max_forecast_steps=24,
        preoccupancy_penalty_weight=30.0,
        preoccupancy_lookahead_steps=24,
    )
    env.reset(options={"scenario_name": "preoccupancy-penalty"})

    _, reward, _, _, info = env.step(
        _env_action([ZoneSourceMode.AUTO], [RoomHeatingIntent.OFF])
    )

    assert info["preoccupancy_penalty"] > 0.0
    assert info["preoccupancy_penalty"] > info["comfort_violation"]
    assert info["raw_reward"] == pytest.approx(reward)


def test_train_env_workers_start_on_staggered_scenarios():
    envs = [
        train_opt_heating_policy_script._make_train_env(seed=index, scenario_index=index)()
        for index in range(3)
    ]

    scenario_names = [env._scenario.name for env in envs]

    assert scenario_names == ["winter_workday", "winter_weekend", "spring_workday"]

    for env in envs:
        env.close()
