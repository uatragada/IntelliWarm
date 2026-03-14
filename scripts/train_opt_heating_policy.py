import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import gymnasium as gym
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from intelliwarm.control import (
    BaselineController,
    HybridController,
    RoomHeatingIntent,
    ZoneSourceMode,
    room_intent_feature_value,
    zone_source_mode_feature_value,
)
from intelliwarm.data import (
    HeatSourceType,
    OccupancyWindow,
    RoomConfig,
    ZoneConfig,
)
from intelliwarm.learning.gym_env import IntelliWarmMultiRoomEnv
from intelliwarm.learning.scenario_generator import SyntheticScenarioGenerator, TrainingScenario
from intelliwarm.models import HouseSimulator, PhysicsRoomThermalModel
from intelliwarm.prediction import OccupancyPredictor


STEP_MINUTES = 5
STEPS_PER_HOUR = 60 // STEP_MINUTES
ENV_KWARGS = dict(
    comfort_penalty_weight=5.0,
    energy_weight=1.0,
    switching_weight=0.05,
    invalid_source_penalty=2.0,
    preoccupancy_penalty_weight=30.0,
    preoccupancy_lookahead_steps=24,
    max_forecast_steps=24,
    comfort_warmup_steps=24,
)


def _build_zones() -> Dict[str, ZoneConfig]:
    return {
        "Main": ZoneConfig(
            zone_id="Main",
            description="Living areas - living room and kitchen",
            has_furnace=True,
            furnace_btu_per_hour=20_000.0,
            furnace_efficiency=0.80,
        ),
        "Sleeping": ZoneConfig(
            zone_id="Sleeping",
            description="Bedrooms - master and kids",
            has_furnace=True,
            furnace_btu_per_hour=15_000.0,
            furnace_efficiency=0.85,
        ),
        "Office": ZoneConfig(
            zone_id="Office",
            description="Home office - electric only",
            has_furnace=False,
        ),
    }


def _build_rooms() -> Dict[str, RoomConfig]:
    weekdays = list(range(5))
    weekends = [5, 6]
    return {
        "living_room": RoomConfig(
            room_id="living_room",
            display_name="Living Room",
            zone="Main",
            target_min_temp=19.5,
            target_max_temp=22.0,
            heater_capacity=2000.0,
            heat_loss_factor=0.003,
            heating_efficiency=0.80,
            occupancy_schedule=(
                [OccupancyWindow(day, 17, 22, 0.90) for day in weekdays]
                + [OccupancyWindow(day, 8, 22, 0.95) for day in weekends]
            ),
            heat_source=HeatSourceType.GAS_FURNACE,
        ),
        "kitchen": RoomConfig(
            room_id="kitchen",
            display_name="Kitchen",
            zone="Main",
            target_min_temp=18.5,
            target_max_temp=22.0,
            heater_capacity=1500.0,
            heat_loss_factor=0.005,
            heating_efficiency=0.80,
            occupancy_schedule=(
                [OccupancyWindow(day, 6, 8, 0.90) for day in weekdays]
                + [OccupancyWindow(day, 17, 19, 0.85) for day in weekdays]
                + [OccupancyWindow(day, 7, 19, 0.75) for day in weekends]
            ),
            heat_source=HeatSourceType.GAS_FURNACE,
        ),
        "master_bedroom": RoomConfig(
            room_id="master_bedroom",
            display_name="Master Bedroom",
            zone="Sleeping",
            target_min_temp=18.0,
            target_max_temp=21.0,
            heater_capacity=1500.0,
            heat_loss_factor=0.004,
            heating_efficiency=0.85,
            occupancy_schedule=(
                [OccupancyWindow(day, 5, 8, 0.85) for day in weekdays]
                + [OccupancyWindow(day, 21, 24, 0.95) for day in range(7)]
            ),
            heat_source=HeatSourceType.GAS_FURNACE,
        ),
        "kids_bedroom": RoomConfig(
            room_id="kids_bedroom",
            display_name="Kids Bedroom",
            zone="Sleeping",
            target_min_temp=19.0,
            target_max_temp=22.0,
            heater_capacity=1000.0,
            heat_loss_factor=0.005,
            heating_efficiency=0.85,
            occupancy_schedule=(
                [OccupancyWindow(day, 6, 8, 0.85) for day in weekdays]
                + [OccupancyWindow(day, 19, 24, 0.95) for day in range(7)]
                + [OccupancyWindow(day, 8, 19, 0.70) for day in weekends]
            ),
            heat_source=HeatSourceType.GAS_FURNACE,
        ),
        "home_office": RoomConfig(
            room_id="home_office",
            display_name="Home Office",
            zone="Office",
            target_min_temp=20.0,
            target_max_temp=23.0,
            heater_capacity=1200.0,
            heat_loss_factor=0.006,
            heating_efficiency=0.90,
            occupancy_schedule=[OccupancyWindow(day, 9, 17, 0.90) for day in weekdays],
            heat_source=HeatSourceType.ELECTRIC,
        ),
    }


def _expand_hourly_profile(values: Sequence[float]) -> List[float]:
    return [float(value) for value in values for _ in range(STEPS_PER_HOUR)]


def build_scenarios() -> List[TrainingScenario]:
    generator = SyntheticScenarioGenerator()
    zones = _build_zones()
    rooms = _build_rooms()
    winter_workday = generator.build_scenario(
        name="winter_workday",
        start_time=datetime(2026, 1, 5, 0, 0),
        room_configs=rooms,
        zone_configs=zones,
        initial_temperatures={room_name: 15.0 for room_name in rooms},
        outdoor_temperatures=_expand_hourly_profile(
            [-12, -13, -13, -14, -14, -13, -11, -9, -7, -5, -3, -1, 0, 1, 2, 1, 0, -1, -3, -5, -7, -9, -10, -11]
        ),
        electricity_prices=_expand_hourly_profile(
            [0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.12, 0.22, 0.28, 0.26, 0.22, 0.16, 0.14, 0.14, 0.14, 0.14, 0.16, 0.20, 0.26, 0.28, 0.28, 0.24, 0.16, 0.10]
        ),
        gas_prices=[1.20] * (24 * STEPS_PER_HOUR),
        step_minutes=STEP_MINUTES,
        description="Cold January weekday: sharp morning/evening pricing peaks",
    )
    winter_weekend = generator.build_scenario(
        name="winter_weekend",
        start_time=datetime(2026, 1, 10, 0, 0),
        room_configs=rooms,
        zone_configs=zones,
        initial_temperatures={room_name: 16.0 for room_name in rooms},
        outdoor_temperatures=_expand_hourly_profile(
            [-8, -9, -9, -10, -9, -8, -6, -4, -2, 0, 2, 3, 4, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6]
        ),
        electricity_prices=_expand_hourly_profile(
            [0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.10, 0.12, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.16, 0.18, 0.18, 0.16, 0.14, 0.12, 0.10]
        ),
        gas_prices=[1.20] * (24 * STEPS_PER_HOUR),
        step_minutes=STEP_MINUTES,
        description="Cold January Saturday: family home all day, flat electricity",
    )
    spring_workday = generator.build_scenario(
        name="spring_workday",
        start_time=datetime(2026, 3, 16, 0, 0),
        room_configs=rooms,
        zone_configs=zones,
        initial_temperatures={room_name: 17.5 for room_name in rooms},
        outdoor_temperatures=_expand_hourly_profile(
            [4, 3, 3, 3, 4, 5, 7, 9, 11, 12, 13, 14, 15, 15, 14, 13, 11, 9, 8, 7, 6, 6, 5, 5]
        ),
        electricity_prices=_expand_hourly_profile(
            [0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.10, 0.16, 0.20, 0.18, 0.14, 0.12, 0.12, 0.12, 0.12, 0.12, 0.14, 0.18, 0.22, 0.22, 0.18, 0.14, 0.10, 0.08]
        ),
        gas_prices=[1.20] * (24 * STEPS_PER_HOUR),
        step_minutes=STEP_MINUTES,
        description="Mild March weekday: less heating needed, electric sometimes cheaper",
    )
    return [winter_workday, winter_weekend, spring_workday]


class PhysicsMultiRoomEnv(IntelliWarmMultiRoomEnv):
    def __init__(
        self,
        *args,
        latitude_deg: float = 43.7,
        cloud_cover: float = 0.30,
        infiltration_ach: float = 0.50,
        albedo: float = 0.20,
        include_smart_tou_features: bool = False,
        smart_tou_feature_fn=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._latitude_deg = latitude_deg
        self._cloud_cover = cloud_cover
        self._infiltration_ach = infiltration_ach
        self._albedo = albedo
        self._include_smart_tou_features = include_smart_tou_features
        self._smart_tou_feature_fn = smart_tou_feature_fn
        self._smart_tou_feature_dim = self.max_zones + self.max_rooms
        if self._include_smart_tou_features:
            base_dim = self.observation_space.shape[0]
            self.observation_space = gym.spaces.Box(
                low=-1000.0,
                high=1000.0,
                shape=(base_dim + self._smart_tou_feature_dim,),
                dtype=np.float32,
            )

    def _build_simulator(self, scenario: TrainingScenario) -> HouseSimulator:
        thermal_models = {}
        for room_name, room_config in scenario.room_configs.items():
            zone_id = room_config.zone
            zone_config = scenario.zone_configs.get(zone_id)
            num_zone_rooms = sum(1 for config in scenario.room_configs.values() if config.zone == zone_id)
            thermal_models[room_name] = PhysicsRoomThermalModel.from_room_config(
                room_config,
                zone_config=zone_config,
                num_zone_rooms=num_zone_rooms,
                infiltration_ach=self._infiltration_ach,
            )
        occupancy_predictors = {
            room_name: OccupancyPredictor(room_name, room_config.occupancy_schedule)
            for room_name, room_config in scenario.room_configs.items()
        }
        return HouseSimulator(
            room_configs=scenario.room_configs,
            thermal_models=thermal_models,
            occupancy_predictors=occupancy_predictors,
            latitude_deg=self._latitude_deg,
            cloud_cover=self._cloud_cover,
            albedo=self._albedo,
        )

    def _augment_observation(self, obs: np.ndarray, info: dict):
        info = dict(info)
        info["base_observation_dim"] = int(obs.shape[0])
        info["smart_tou_feature_dim"] = self._smart_tou_feature_dim if self._include_smart_tou_features else 0
        if not self._include_smart_tou_features:
            return obs, info
        if self._smart_tou_feature_fn is None:
            raise RuntimeError("smart_tou teacher features requested without a feature function.")
        smart_tou_features = np.asarray(self._smart_tou_feature_fn(obs, info), dtype=np.float32)
        info["smart_tou_zone_mode_offset"] = int(obs.shape[0])
        info["smart_tou_room_intent_offset"] = int(obs.shape[0] + self.max_zones)
        return np.concatenate([obs.astype(np.float32), smart_tou_features]), info

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        obs, info = super().reset(seed=seed, options=options)
        return self._augment_observation(obs, info)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        obs, info = self._augment_observation(obs, info)
        return obs, reward, terminated, truncated, info


def _active_zone_names(info: dict) -> List[str]:
    return list(info.get("zone_names", []))


def _room_vector(room_demands: Dict[str, float], info: dict, fill: float = 0.0) -> List[float]:
    values = [float(room_demands.get(room_name, fill)) for room_name in info.get("room_names", [])]
    values.extend([fill] * (info["max_rooms"] - len(values)))
    return values


def _zone_vector(zone_signals: Dict[str, float], info: dict) -> List[float]:
    values = [float(zone_signals.get(zone_name, 0.0)) for zone_name in _active_zone_names(info)]
    values.extend([0.0] * (info["max_zones"] - len(values)))
    return values


def _obs_context(obs, info):
    base_obs = obs[: info.get("base_observation_dim", len(obs))]
    max_rooms = info["max_rooms"]
    horizon = info["occupancy_forecast_horizon_steps"]
    room_feature_block = max_rooms * 6
    room_context = {}
    for idx, room_name in enumerate(info.get("room_names", [])):
        base = idx * 6
        forecast_base = room_feature_block + (idx * horizon)
        occ_now = float(base_obs[base + 3])
        future_occ = [float(base_obs[forecast_base + offset]) for offset in range(horizon)]
        room_context[room_name] = {
            "temp": float(base_obs[base + 0]),
            "target_min": float(base_obs[base + 1]),
            "target_max": float(base_obs[base + 2]),
            "forecast": [occ_now] + future_occ,
            "current_occ": occ_now,
            "last_action": float(base_obs[base + 4]),
            "target": 0.5 * (float(base_obs[base + 1]) + float(base_obs[base + 2])),
        }
    outside_temp = float(base_obs[-7])
    electricity_price = float(base_obs[-6])
    gas_price = float(base_obs[-5])
    return room_context, outside_temp, electricity_price, gas_price


ROOMS = _build_rooms()
ZONES = _build_zones()
SCENARIOS = build_scenarios()
BASELINE_CONTROLLERS = {
    room_name: BaselineController(
        room_config=room_config,
        min_temperature=18.0,
        max_temperature=24.0,
        preheat_lookahead_steps=2,
    )
    for room_name, room_config in ROOMS.items()
}
HYBRID_CONTROLLERS = {
    zone_name: HybridController(
        zone_config=zone_config,
        room_configs={room_name: room_config for room_name, room_config in ROOMS.items() if room_config.zone == zone_name},
        min_temperature=18.0,
        max_temperature=24.0,
        preheat_lookahead_steps=2,
    )
    for zone_name, zone_config in ZONES.items()
}


def _smart_tou_decision(obs, info):
    room_context, outside_temp, electricity_price, gas_price = _obs_context(obs, info)
    room_intents = {}
    for room_name, ctx in room_context.items():
        decision = BASELINE_CONTROLLERS[room_name].compute_decision(
            current_temp=ctx["temp"],
            occupancy_forecast=ctx["forecast"],
            energy_prices=[electricity_price],
            current_action=ctx["last_action"],
            outside_temp=outside_temp,
            target_temp=ctx["target"],
        )
        room_intents[room_name] = decision.metadata.get("room_intent", RoomHeatingIntent.OFF.value)

    zone_modes = {}
    for zone_name in _active_zone_names(info):
        zone_rooms = {room_name: ctx for room_name, ctx in room_context.items() if ROOMS[room_name].zone == zone_name}
        decision = HYBRID_CONTROLLERS[zone_name].decide(
            room_temperatures={room_name: ctx["temp"] for room_name, ctx in zone_rooms.items()},
            occupancy_forecasts={room_name: ctx["forecast"] for room_name, ctx in zone_rooms.items()},
            electricity_price=electricity_price,
            gas_price=gas_price,
            outside_temp=outside_temp,
            current_actions={room_name: ctx["last_action"] for room_name, ctx in zone_rooms.items()},
            target_temps={room_name: ctx["target"] for room_name, ctx in zone_rooms.items()},
            room_intents={
                room_name: room_intents[room_name]
                for room_name in zone_rooms
            },
            zone_source_preference=ZoneSourceMode.AUTO,
        )
        zone_modes[zone_name] = (
            ZoneSourceMode.GAS_FURNACE
            if decision.heat_source == HeatSourceType.GAS_FURNACE
            else ZoneSourceMode.ELECTRIC
        )
    return zone_modes, room_intents


def _smart_tou_feature_vector(obs, info):
    zone_modes, room_intents = _smart_tou_decision(obs, info)
    return _zone_vector(
        {
            zone_name: zone_source_mode_feature_value(mode)
            for zone_name, mode in zone_modes.items()
        },
        info,
    ) + _room_vector(
        {
            room_name: room_intent_feature_value(intent)
            for room_name, intent in room_intents.items()
        },
        info,
    )


def _make_train_env(seed: int = 0, scenario_index: int = 0):
    def _init():
        env = PhysicsMultiRoomEnv(
            scenarios=SCENARIOS,
            include_smart_tou_features=True,
            smart_tou_feature_fn=_smart_tou_feature_vector,
            **ENV_KWARGS,
        )
        env.reset(seed=seed, options={"scenario_index": scenario_index % len(SCENARIOS)})
        return env

    return _init


def _build_train_vec_env(n_envs: int, force_dummy_vec_env: bool = False):
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    env_fns = [
        _make_train_env(seed=index, scenario_index=index)
        for index in range(n_envs)
    ]
    if n_envs <= 1 or force_dummy_vec_env:
        return DummyVecEnv(env_fns), "DummyVecEnv"

    vec_env = None
    try:
        vec_env = SubprocVecEnv(env_fns, start_method="spawn")
        vec_env.reset()
        if hasattr(vec_env.action_space, "nvec"):
            test_actions = np.zeros((n_envs, len(vec_env.action_space.nvec)), dtype=np.int64)
        else:
            test_actions = np.zeros((n_envs,) + vec_env.action_space.shape, dtype=np.float32)
        vec_env.step(test_actions)
        vec_env.reset()
        return vec_env, "SubprocVecEnv"
    except Exception as exc:
        if vec_env is not None:
            try:
                vec_env.close()
            except Exception:
                pass
        print(f"Warning: SubprocVecEnv startup/check failed ({exc!r}); falling back to DummyVecEnv.")
        return DummyVecEnv(env_fns), "DummyVecEnv"


def _device_label():
    return "cpu", "cpu"


def train_opt_policy(
    *,
    total_timesteps: int = 100_000,
    n_envs: int = 4,
    output_dir: Path,
    force_dummy_vec_env: bool = False,
    progress_bar: bool = True,
    text_progress: bool = True,
    progress_interval: int = 5_000,
):
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback

    class TrainingLogger(BaseCallback):
        def __init__(
            self,
            n_envs: int,
            total_timesteps: int,
            text_progress: bool,
            progress_interval: int,
        ):
            super().__init__(verbose=0)
            self._n_envs = n_envs
            self._total_timesteps = max(1, int(total_timesteps))
            self._text_progress = bool(text_progress)
            self._progress_interval = max(1, int(progress_interval))
            self._next_progress = self._progress_interval
            self._started_at = time.time()
            self.ep_rewards: List[float] = []
            self.ep_rewards_normalized: List[float] = []
            self.ep_costs: List[float] = []
            self.ep_violations: List[float] = []
            self._ep_reward = [0.0] * n_envs
            self._ep_reward_normalized = [0.0] * n_envs
            self._ep_cost = [0.0] * n_envs
            self._ep_violation = [0.0] * n_envs

        def _print_progress(self):
            elapsed = time.time() - self._started_at
            percent = min(100.0, (self.num_timesteps / self._total_timesteps) * 100.0)
            mean_reward = (
                float(np.mean(self.ep_rewards[-20:]))
                if self.ep_rewards
                else float(np.mean(self._ep_reward))
            )
            print(
                f"[train] {self.num_timesteps:>7,d}/{self._total_timesteps:,} "
                f"({percent:5.1f}%) | episodes={len(self.ep_rewards):>4} "
                f"| mean_reward={mean_reward:8.2f} | elapsed={elapsed:7.1f}s",
                flush=True,
            )

        def _on_step(self) -> bool:
            rewards = self.locals.get("rewards", [0.0] * self._n_envs)
            dones = self.locals.get("dones", [False] * self._n_envs)
            infos = self.locals.get("infos", [{}] * self._n_envs)
            for idx in range(self._n_envs):
                normalized_reward = float(rewards[idx])
                raw_reward = float(infos[idx].get("raw_reward", normalized_reward))
                self._ep_reward[idx] += raw_reward
                self._ep_reward_normalized[idx] += normalized_reward
                self._ep_cost[idx] += float(infos[idx].get("total_cost", 0.0))
                self._ep_violation[idx] += float(
                    infos[idx].get("comfort_violation", 0.0)
                )
                if dones[idx]:
                    self.ep_rewards.append(self._ep_reward[idx])
                    self.ep_rewards_normalized.append(self._ep_reward_normalized[idx])
                    self.ep_costs.append(self._ep_cost[idx])
                    self.ep_violations.append(self._ep_violation[idx])
                    self._ep_reward[idx] = 0.0
                    self._ep_reward_normalized[idx] = 0.0
                    self._ep_cost[idx] = 0.0
                    self._ep_violation[idx] = 0.0
            if self._text_progress and self.num_timesteps >= self._next_progress:
                self._print_progress()
                while self.num_timesteps >= self._next_progress:
                    self._next_progress += self._progress_interval
            return True

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "opt_policy_ppo"
    normalize_path = output_dir / "opt_policy_vec_normalize.pkl"
    summary_path = output_dir / "opt_policy_training_summary.json"
    vec_env, vec_env_type = _build_train_vec_env(n_envs, force_dummy_vec_env=force_dummy_vec_env)

    from stable_baselines3.common.vec_env import VecNormalize
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

    logger = TrainingLogger(
        n_envs=n_envs,
        total_timesteps=total_timesteps,
        text_progress=text_progress and not progress_bar,
        progress_interval=progress_interval,
    )
    device, device_label = _device_label()
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=0,
        n_steps=288,
        batch_size=128,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.999,
        gae_lambda=0.95,
        ent_coef=0.02,
        clip_range=0.20,
        policy_kwargs=dict(net_arch=[256, 256]),
        seed=42,
        device=device,
    )
    parameter_count = int(sum(param.numel() for param in model.policy.parameters()))
    started = time.time()
    try:
        model.learn(total_timesteps=total_timesteps, callback=logger, progress_bar=progress_bar)
        elapsed_seconds = time.time() - started
        if text_progress and not progress_bar and logger.num_timesteps > 0:
            logger._print_progress()
        model.save(str(model_path))
        vec_env.save(str(normalize_path))
        summary = {
            "total_timesteps": int(total_timesteps),
            "n_envs": int(n_envs),
            "vec_env_type": vec_env_type,
            "device": device_label,
            "parameter_count": parameter_count,
            "elapsed_seconds": float(elapsed_seconds),
            "episode_rewards": [float(value) for value in logger.ep_rewards],
            "episode_rewards_normalized": [
                float(value) for value in logger.ep_rewards_normalized
            ],
            "episode_costs": [float(value) for value in logger.ep_costs],
            "episode_violations": [float(value) for value in logger.ep_violations],
            "model_path": str(model_path.with_suffix(".zip")),
            "normalize_path": str(normalize_path),
            "summary_path": str(summary_path),
        }
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        return summary
    finally:
        vec_env.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the IntelliWarm OPT PPO policy outside Jupyter and save model + metrics."
    )
    parser.add_argument("--timesteps", type=int, default=100_000, help="Total PPO timesteps to train.")
    parser.add_argument("--n-envs", type=int, default=4, help="Number of vectorized training environments.")
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "output"),
        help="Directory where the trained model and training summary JSON will be written.",
    )
    parser.add_argument(
        "--force-dummy-vec-env",
        action="store_true",
        help="Use DummyVecEnv instead of SubprocVecEnv.",
    )
    parser.add_argument(
        "--no-progress-bar",
        action="store_true",
        help="Disable the Stable-Baselines3 progress bar.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the final training summary as JSON.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=5_000,
        help="How many timesteps between plain-text progress updates when Rich progress is disabled.",
    )
    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    output_dir = Path(args.output_dir)
    summary = train_opt_policy(
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        output_dir=output_dir,
        force_dummy_vec_env=args.force_dummy_vec_env,
        progress_bar=not args.no_progress_bar,
        text_progress=True,
        progress_interval=args.progress_interval,
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    print(f"Training complete in {summary['elapsed_seconds']:.1f}s using {summary['vec_env_type']} on {summary['device']}")
    print(f"Model saved to {summary['model_path']}")
    print(f"Training summary saved to {summary['summary_path']}")
    print(f"Recorded {len(summary['episode_rewards'])} completed episodes at {summary['parameter_count']:,} parameters")
    return 0


if __name__ == "__main__":
    mp.freeze_support()

    raise SystemExit(main())
