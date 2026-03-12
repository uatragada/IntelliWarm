"""
Gym-compatible training environment for IntelliWarm.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

from intelliwarm.data import HeatingAction, RoomConfig
from intelliwarm.models import RoomThermalModel
from intelliwarm.prediction import OccupancyPredictor
from intelliwarm.pricing import EnergyPriceService

try:
    import gymnasium as gym  # type: ignore
    from gymnasium import spaces  # type: ignore
except ImportError:  # pragma: no cover - exercised through fallback-compatible behavior
    try:
        import gym  # type: ignore
        from gym import spaces  # type: ignore
    except ImportError:  # pragma: no cover
        class _Discrete:
            def __init__(self, n: int):
                self.n = n

            def contains(self, value: object) -> bool:
                return isinstance(value, int) and 0 <= value < self.n

        class _Box:
            def __init__(self, low, high, shape, dtype):
                self.low = low
                self.high = high
                self.shape = shape
                self.dtype = dtype

            def contains(self, value: object) -> bool:
                array = np.asarray(value, dtype=self.dtype)
                return tuple(array.shape) == tuple(self.shape)

        class _Spaces:
            Discrete = _Discrete
            Box = _Box

        class _Env:
            pass

        class _Gym:
            Env = _Env

        gym = _Gym()
        spaces = _Spaces()


class IntelliWarmRoomEnv(gym.Env):
    """Deterministic single-room environment with Gym-compatible reset/step APIs."""

    ACTIONS: List[HeatingAction] = [
        HeatingAction.OFF,
        HeatingAction.ECO,
        HeatingAction.COMFORT,
        HeatingAction.PREHEAT,
    ]

    def __init__(
        self,
        room_config: RoomConfig,
        thermal_model: RoomThermalModel,
        occupancy_predictor: OccupancyPredictor,
        energy_service: EnergyPriceService,
        horizon_steps: int = 24,
        step_minutes: int = 60,
        start_time: Optional[datetime] = None,
        initial_temp: Optional[float] = None,
        outside_temperature_profile: Optional[List[float]] = None,
        comfort_penalty_weight: float = 10.0,
        energy_weight: float = 1.0,
        switching_weight: float = 0.25,
    ):
        self.room_config = room_config
        self.thermal_model = thermal_model
        self.occupancy_predictor = occupancy_predictor
        self.energy_service = energy_service
        self.horizon_steps = int(horizon_steps)
        self.step_minutes = int(step_minutes)
        self.start_time = start_time or datetime(2026, 1, 1, 0, 0)
        self.initial_temp = (
            float(initial_temp)
            if initial_temp is not None
            else (self.room_config.target_min_temp + self.room_config.target_max_temp) / 2.0
        )
        self.outside_temperature_profile = list(outside_temperature_profile or [5.0] * self.horizon_steps)
        self.comfort_penalty_weight = float(comfort_penalty_weight)
        self.energy_weight = float(energy_weight)
        self.switching_weight = float(switching_weight)

        self.action_space = spaces.Discrete(len(self.ACTIONS))
        self.observation_space = spaces.Box(
            low=np.array([-50.0, -50.0, -50.0, 0.0, -50.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([60.0, 60.0, 60.0, 1.0, 60.0, 100.0, 100.0], dtype=np.float32),
            shape=(7,),
            dtype=np.float32,
        )

        self._current_temp = self.initial_temp
        self._current_time = self.start_time
        self._step_index = 0
        self._last_action = HeatingAction.OFF

    def _outside_temp(self, step_index: int) -> float:
        if not self.outside_temperature_profile:
            return 5.0
        bounded_index = min(step_index, len(self.outside_temperature_profile) - 1)
        return float(self.outside_temperature_profile[bounded_index])

    def _current_prices(self) -> Dict[str, float]:
        return self.energy_service.get_price_forecast(1, start_time=self._current_time)[0]

    def _current_occupancy(self) -> float:
        return float(self.occupancy_predictor.predict(self._current_time))

    def _observation(self) -> np.ndarray:
        prices = self._current_prices()
        occupancy = self._current_occupancy()
        return np.array(
            [
                self._current_temp,
                self.room_config.target_min_temp,
                self.room_config.target_max_temp,
                occupancy,
                self._outside_temp(self._step_index),
                float(prices["electricity"]),
                float(prices["gas"]),
            ],
            dtype=np.float32,
        )

    def _resolve_action(self, action: object) -> HeatingAction:
        if isinstance(action, int):
            return self.ACTIONS[action]
        return HeatingAction.from_value(action)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            np.random.seed(seed)

        self._current_temp = self.initial_temp
        self._current_time = self.start_time
        self._step_index = 0
        self._last_action = HeatingAction.OFF
        observation = self._observation()
        return observation, {
            "time": self._current_time.isoformat(),
            "action_label": self._last_action.name,
        }

    def step(self, action: object):
        resolved_action = self._resolve_action(action)
        prices = self._current_prices()
        outside_temp = self._outside_temp(self._step_index)
        occupancy = self._current_occupancy()

        next_temp = self.thermal_model.step(
            current_temp=self._current_temp,
            outside_temp=outside_temp,
            heating_power=resolved_action.power_level,
            dt_minutes=self.step_minutes,
        )
        energy_cost = (
            self.room_config.heater_capacity / 1000.0
        ) * resolved_action.power_level * float(prices["electricity"])
        comfort_violation = max(self.room_config.target_min_temp - next_temp, 0.0) + max(
            next_temp - self.room_config.target_max_temp,
            0.0,
        )
        switching_penalty = abs(resolved_action.power_level - self._last_action.power_level)
        reward = -(
            (self.energy_weight * energy_cost)
            + (self.comfort_penalty_weight * occupancy * comfort_violation)
            + (self.switching_weight * switching_penalty)
        )

        self._current_temp = float(next_temp)
        self._current_time = self._current_time + timedelta(minutes=self.step_minutes)
        self._step_index += 1
        self._last_action = resolved_action

        terminated = self._step_index >= self.horizon_steps
        truncated = False
        observation = self._observation()
        info = {
            "action_label": resolved_action.name,
            "energy_cost": energy_cost,
            "comfort_violation": comfort_violation,
            "occupancy_probability": occupancy,
            "outside_temp": outside_temp,
            "electricity_price": float(prices["electricity"]),
            "gas_price": float(prices["gas"]),
            "time": self._current_time.isoformat(),
        }
        return observation, reward, terminated, truncated, info
