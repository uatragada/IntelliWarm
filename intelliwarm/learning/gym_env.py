"""
Gym-compatible training environment for IntelliWarm.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from intelliwarm.data import (
    action_name_for_power_level,
    clamp_power_level,
    HeatSourceType,
    HeatingAction,
    RoomConfig,
    SimulationState,
    ZoneConfig,
)
from intelliwarm.models import HouseSimulator, RoomThermalModel
from intelliwarm.prediction import OccupancyPredictor
from intelliwarm.pricing import EnergyPriceService
from .scenario_generator import TrainingScenario

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

        class _MultiDiscrete:
            def __init__(self, nvec):
                self.nvec = list(nvec)

            def contains(self, value: object) -> bool:
                array = np.asarray(value, dtype=np.int64)
                if array.shape != (len(self.nvec),):
                    return False
                return all(0 <= int(item) < limit for item, limit in zip(array.tolist(), self.nvec))

        class _Spaces:
            Discrete = _Discrete
            Box = _Box
            MultiDiscrete = _MultiDiscrete

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

        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            shape=(1,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=np.array([-50.0, -50.0, -50.0, 0.0, -50.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([60.0, 60.0, 60.0, 1.0, 60.0, 100.0, 100.0], dtype=np.float32),
            shape=(7,),
            dtype=np.float32,
        )

        self._current_temp = self.initial_temp
        self._current_time = self.start_time
        self._step_index = 0
        self._last_action = 0.0

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

    def _resolve_action(self, action: object) -> float:
        if isinstance(action, int):
            return self.ACTIONS[action].power_level
        array = np.asarray(action, dtype=np.float32).reshape(-1)
        if array.size == 0:
            return 0.0
        return clamp_power_level(float(array[0]))

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            np.random.seed(seed)

        self._current_temp = self.initial_temp
        self._current_time = self.start_time
        self._step_index = 0
        self._last_action = 0.0
        observation = self._observation()
        return observation, {
            "time": self._current_time.isoformat(),
            "action_label": action_name_for_power_level(self._last_action),
        }

    def step(self, action: object):
        resolved_action = self._resolve_action(action)
        prices = self._current_prices()
        outside_temp = self._outside_temp(self._step_index)
        occupancy = self._current_occupancy()

        next_temp = self.thermal_model.step(
            current_temp=self._current_temp,
            outside_temp=outside_temp,
            heating_power=resolved_action,
            dt_minutes=self.step_minutes,
        )
        energy_cost = (
            self.room_config.heater_capacity / 1000.0
        ) * resolved_action * float(prices["electricity"])
        comfort_violation = max(self.room_config.target_min_temp - next_temp, 0.0) + max(
            next_temp - self.room_config.target_max_temp,
            0.0,
        )
        switching_penalty = abs(resolved_action - self._last_action)
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
            "action_level": resolved_action,
            "action_label": action_name_for_power_level(resolved_action),
            "energy_cost": energy_cost,
            "comfort_violation": comfort_violation,
            "occupancy_probability": occupancy,
            "outside_temp": outside_temp,
            "electricity_price": float(prices["electricity"]),
            "gas_price": float(prices["gas"]),
            "time": self._current_time.isoformat(),
        }
        return observation, reward, terminated, truncated, info


class ScenarioBoundPriceService(EnergyPriceService):
    """Price service backed by deterministic scenario profiles."""

    def __init__(
        self,
        electricity_prices: Sequence[float],
        gas_prices: Sequence[float],
        start_time: Optional[datetime] = None,
        step_minutes: int = 60,
    ):
        self._electricity_prices = list(electricity_prices)
        self._gas_prices = list(gas_prices)
        self._start_time = start_time
        self._step_minutes = max(1, int(step_minutes))
        super().__init__(
            electricity_price=float(self._electricity_prices[0]),
            gas_price=float(self._gas_prices[0]),
        )

    def get_price_forecast(self, hours: int = 24, start_time: datetime = None) -> List[Dict]:
        reference_time = start_time or datetime.now()
        forecast = []
        start_index = 0
        if self._start_time is not None and start_time is not None:
            delta_seconds = (start_time - self._start_time).total_seconds()
            delta_steps = int(delta_seconds // (self._step_minutes * 60))
            start_index = max(0, delta_steps)
        for offset in range(hours):
            index = min(start_index + offset, len(self._electricity_prices) - 1)
            forecast.append(
                {
                    "hour": (
                        reference_time.hour
                        + ((reference_time.minute + (offset * self._step_minutes)) / 60.0)
                    ) % 24,
                    "electricity": float(self._electricity_prices[index]),
                    "gas": float(self._gas_prices[index]),
                }
            )
        return forecast


class IntelliWarmMultiRoomEnv(gym.Env):
    """Deterministic multi-room, multi-zone environment for controller training."""

    ACTIONS: List[HeatingAction] = IntelliWarmRoomEnv.ACTIONS
    SOURCE_ACTIONS: List[HeatSourceType] = [
        HeatSourceType.ELECTRIC,
        HeatSourceType.GAS_FURNACE,
    ]

    def __init__(
        self,
        scenarios: List[TrainingScenario],
        comfort_penalty_weight: float = 10.0,
        energy_weight: float = 1.0,
        switching_weight: float = 0.25,
        invalid_source_penalty: float = 1.0,
        max_forecast_steps: Optional[int] = None,
        comfort_warmup_steps: int = 0,
    ):
        if not scenarios:
            raise ValueError("At least one training scenario is required")

        self.scenarios = list(scenarios)
        self.comfort_penalty_weight = float(comfort_penalty_weight)
        self.energy_weight = float(energy_weight)
        self.switching_weight = float(switching_weight)
        self.invalid_source_penalty = float(invalid_source_penalty)
        self.comfort_warmup_steps = max(0, int(comfort_warmup_steps))

        self.max_rooms = max(len(scenario.room_configs) for scenario in self.scenarios)
        self.max_zones = max(len(scenario.zone_configs) for scenario in self.scenarios)
        self.occupancy_forecast_horizon_steps = max(
            scenario.horizon_steps for scenario in self.scenarios
        )
        if max_forecast_steps is not None:
            self.occupancy_forecast_horizon_steps = min(
                self.occupancy_forecast_horizon_steps, int(max_forecast_steps)
            )
        # Global features: T_out, elec_price, gas_price, hour_sin, hour_cos,
        #                   next_1h_occ_max, next_2h_occ_max  (7 total)
        self._n_global_features = 7
        self.observation_space = spaces.Box(
            low=-1000.0,
            high=1000.0,
            shape=(
                self.max_rooms * (6 + self.occupancy_forecast_horizon_steps)
                + self.max_zones * 3
                + self._n_global_features,
            ),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.zeros(self.max_zones + self.max_rooms, dtype=np.float32),
            high=np.ones(self.max_zones + self.max_rooms, dtype=np.float32),
            shape=(self.max_zones + self.max_rooms,),
            dtype=np.float32,
        )

        self._scenario_index = -1
        self._scenario: Optional[TrainingScenario] = None
        self._room_names: List[str] = []
        self._zone_names: List[str] = []
        self._zone_rooms: Dict[str, List[str]] = {}
        self._simulator: Optional[HouseSimulator] = None
        self._current_state: Optional[SimulationState] = None
        self._price_service: Optional[ScenarioBoundPriceService] = None
        self._occupancy_predictors: Dict[str, OccupancyPredictor] = {}
        self._step_index = 0
        self._last_effective_actions: Dict[str, HeatingAction] = {}
        self._last_source_actions: Dict[str, HeatSourceType] = {}

    def _build_simulator(self, scenario: TrainingScenario) -> HouseSimulator:
        thermal_models = {
            room_name: RoomThermalModel(
                room_name,
                alpha=room_config.heating_efficiency,
                beta=room_config.heat_loss_factor,
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

    def _resolve_scenario(self, options: Optional[Dict]) -> TrainingScenario:
        if options and "scenario_name" in options:
            scenario_name = str(options["scenario_name"])
            for index, scenario in enumerate(self.scenarios):
                if scenario.name == scenario_name:
                    self._scenario_index = index
                    return scenario
            raise KeyError(f"Unknown scenario_name: {scenario_name}")

        if options and "scenario_index" in options:
            index = int(options["scenario_index"]) % len(self.scenarios)
            self._scenario_index = index
            return self.scenarios[index]

        self._scenario_index = (self._scenario_index + 1) % len(self.scenarios)
        return self.scenarios[self._scenario_index]

    def _outside_temp(self) -> float:
        assert self._scenario is not None
        index = min(self._step_index, len(self._scenario.outdoor_temperatures) - 1)
        return float(self._scenario.outdoor_temperatures[index])

    def _current_prices(self) -> Dict[str, float]:
        assert self._scenario is not None
        index = min(self._step_index, len(self._scenario.electricity_prices) - 1)
        return {
            "electricity": float(self._scenario.electricity_prices[index]),
            "gas": float(self._scenario.gas_prices[index]),
        }

    def _observation(self) -> np.ndarray:
        assert self._scenario is not None
        assert self._current_state is not None

        room_features: List[float] = []
        occupancy_forecast_features: List[float] = []
        step_td = timedelta(minutes=self._scenario.step_minutes)
        for room_name in self._room_names:
            room_config = self._scenario.room_configs[room_name]
            room_features.extend(
                [
                    float(self._current_state.room_temperatures[room_name]),
                    float(room_config.target_min_temp),
                    float(room_config.target_max_temp),
                    float(self._current_state.occupancy[room_name]),
                    float(self._last_effective_actions[room_name]),
                    1.0,
                ]
            )
            predictor = self._occupancy_predictors.get(room_name)
            occupancy_forecast_features.extend(
                [
                    float(
                        predictor.predict(self._current_state.timestamp + step_td * (forecast_step + 1))
                    )
                    if predictor is not None
                    else 0.0
                    for forecast_step in range(self.occupancy_forecast_horizon_steps)
                ]
            )
        for _ in range(self.max_rooms - len(self._room_names)):
            room_features.extend([0.0] * 6)
            occupancy_forecast_features.extend([0.0] * self.occupancy_forecast_horizon_steps)

        zone_features: List[float] = []
        for zone_name in self._zone_names:
            zone_config = self._scenario.zone_configs[zone_name]
            zone_features.extend(
                [
                    1.0 if zone_config.has_furnace else 0.0,
                    1.0 if self._last_source_actions[zone_name] == HeatSourceType.GAS_FURNACE else 0.0,
                    1.0,
                ]
            )
        for _ in range(self.max_zones - len(self._zone_names)):
            zone_features.extend([0.0] * 3)

        prices = self._current_prices()
        # Compute hour-of-day cyclical encoding so the policy can learn
        # time-dependent patterns (e.g. start preheating 2 h before 7 am).
        hour = self._current_state.timestamp.hour + self._current_state.timestamp.minute / 60.0
        hour_sin = math.sin(2 * math.pi * hour / 24.0)
        hour_cos = math.cos(2 * math.pi * hour / 24.0)
        # Lookahead occupancy: maximum probability across all rooms 1 and 2 h ahead.
        # This gives the policy an explicit "occupancy is coming" signal so it can
        # schedule preheating during unoccupied early-morning hours.
        next_1h_time = self._current_state.timestamp + step_td
        next_2h_time = self._current_state.timestamp + 2 * step_td
        if self._occupancy_predictors:
            next_1h_occ = max(p.predict(next_1h_time) for p in self._occupancy_predictors.values())
            next_2h_occ = max(p.predict(next_2h_time) for p in self._occupancy_predictors.values())
        else:
            next_1h_occ = 0.0
            next_2h_occ = 0.0
        global_features = [
            float(self._outside_temp()),
            float(prices["electricity"]),
            float(prices["gas"]),
            hour_sin,
            hour_cos,
            float(next_1h_occ),
            float(next_2h_occ),
        ]
        return np.array(
            room_features + occupancy_forecast_features + zone_features + global_features,
            dtype=np.float32,
        )

    def _normalize_action(self, action: Sequence[float]) -> np.ndarray:
        values = np.asarray(action, dtype=np.float32).reshape(-1)
        expected = self.max_zones + self.max_rooms
        if values.size != expected:
            raise ValueError(f"Expected action vector of length {expected}, got {values.size}")
        return np.clip(values, 0.0, 1.0)

    def _resolve_zone_sources(
        self,
        action: np.ndarray,
    ) -> Tuple[Dict[str, HeatSourceType], Dict[str, HeatSourceType], Dict[str, float]]:
        zone_source_indices = list(action[: self.max_zones])
        requested_sources: Dict[str, HeatSourceType] = {}
        effective_sources: Dict[str, HeatSourceType] = {}
        source_signals: Dict[str, float] = {}
        for zone_index, zone_name in enumerate(self._zone_names):
            source_signal = clamp_power_level(float(zone_source_indices[zone_index]))
            requested = (
                HeatSourceType.GAS_FURNACE
                if source_signal >= 0.5
                else HeatSourceType.ELECTRIC
            )
            zone_config = self._scenario.zone_configs[zone_name]
            requested_sources[zone_name] = requested
            source_signals[zone_name] = source_signal
            if requested == HeatSourceType.GAS_FURNACE and not zone_config.has_furnace:
                effective_sources[zone_name] = HeatSourceType.ELECTRIC
            else:
                effective_sources[zone_name] = requested
        return requested_sources, effective_sources, source_signals

    def _resolve_room_actions(self, action: np.ndarray) -> Dict[str, float]:
        room_action_indices = list(action[self.max_zones : self.max_zones + self.max_rooms])
        return {
            room_name: clamp_power_level(float(room_action_indices[index]))
            for index, room_name in enumerate(self._room_names)
        }

    def _effective_room_actions(
        self,
        zone_sources: Dict[str, HeatSourceType],
        requested_room_actions: Dict[str, float],
    ) -> Dict[str, float]:
        effective_actions: Dict[str, float] = {}
        for zone_name, room_names in self._zone_rooms.items():
            if zone_sources.get(zone_name) == HeatSourceType.GAS_FURNACE:
                zone_action = max(
                    (requested_room_actions[room_name] for room_name in room_names),
                )
                for room_name in room_names:
                    effective_actions[room_name] = zone_action
            else:
                for room_name in room_names:
                    effective_actions[room_name] = requested_room_actions[room_name]
        return effective_actions

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            np.random.seed(seed)

        self._scenario = self._resolve_scenario(options)
        self._room_names = sorted(self._scenario.room_configs.keys())
        self._zone_names = sorted(self._scenario.zone_configs.keys())
        self._zone_rooms = {
            zone_name: [
                room_name
                for room_name, room_config in self._scenario.room_configs.items()
                if room_config.zone == zone_name
            ]
            for zone_name in self._zone_names
        }
        self._simulator = self._build_simulator(self._scenario)
        self._price_service = ScenarioBoundPriceService(
            electricity_prices=self._scenario.electricity_prices,
            gas_prices=self._scenario.gas_prices,
            start_time=self._scenario.start_time,
            step_minutes=self._scenario.step_minutes,
        )
        # Build per-room occupancy predictors so _observation() can look
        # ahead by 1–2 hours to give the policy a preheat signal.
        self._occupancy_predictors = {
            room_name: OccupancyPredictor(room_name, room_config.occupancy_schedule)
            for room_name, room_config in self._scenario.room_configs.items()
        }
        self._step_index = 0
        self._last_effective_actions = {
            room_name: 0.0
            for room_name in self._room_names
        }
        self._last_source_actions = {
            zone_name: HeatSourceType.ELECTRIC
            for zone_name in self._zone_names
        }
        self._current_state = SimulationState(
            timestamp=self._scenario.start_time,
            outdoor_temp=self._outside_temp(),
            room_temperatures=dict(self._scenario.initial_temperatures),
            heating_actions=dict(self._last_effective_actions),
            occupancy={
                room_name: OccupancyPredictor(room_name, room_config.occupancy_schedule).predict(self._scenario.start_time)
                for room_name, room_config in self._scenario.room_configs.items()
            },
            # Keyed by room_name so simulator.step() can look up per-room source.
            heat_sources={room_name: HeatSourceType.ELECTRIC for room_name in self._room_names},
        )
        return self._observation(), {
            "scenario_name": self._scenario.name,
            "room_names": list(self._room_names),
            "zone_names": list(self._zone_names),
            "active_rooms": len(self._room_names),
            "active_zones": len(self._zone_names),
            "max_rooms": self.max_rooms,
            "max_zones": self.max_zones,
            "occupancy_forecast_horizon_steps": self.occupancy_forecast_horizon_steps,
            "zone_has_furnace": {
                zone_name: self._scenario.zone_configs[zone_name].has_furnace
                for zone_name in self._zone_names
            },
        }

    def step(self, action: Sequence[int]):
        assert self._scenario is not None
        assert self._simulator is not None
        assert self._current_state is not None
        action_values = self._normalize_action(action)

        current_state = SimulationState(
            timestamp=self._current_state.timestamp,
            outdoor_temp=self._outside_temp(),
            room_temperatures=dict(self._current_state.room_temperatures),
            heating_actions=dict(self._current_state.heating_actions),
            occupancy=dict(self._current_state.occupancy),
            heat_sources=dict(self._current_state.heat_sources),
        )
        requested_zone_sources, zone_sources, zone_source_signals = self._resolve_zone_sources(action_values)
        requested_room_actions = self._resolve_room_actions(action_values)
        effective_actions = self._effective_room_actions(zone_sources, requested_room_actions)
        current_state.heat_sources = {
            room_name: zone_sources[self._scenario.room_configs[room_name].zone]
            for room_name in self._room_names
        }

        next_state = self._simulator.step(
            current_state,
            heating_actions=effective_actions,
            next_timestamp=self._scenario.start_time + timedelta(minutes=self._scenario.step_minutes * (self._step_index + 1)),
            dt_minutes=self._scenario.step_minutes,
        )
        prices = self._current_prices()

        electric_cost = 0.0
        gas_cost = 0.0
        comfort_violation = 0.0
        switching_penalty = 0.0
        invalid_source_penalty = 0.0
        for zone_name, room_names in self._zone_rooms.items():
            zone_config = self._scenario.zone_configs[zone_name]
            zone_source = zone_sources[zone_name]
            if requested_zone_sources[zone_name] == HeatSourceType.GAS_FURNACE and not zone_config.has_furnace:
                invalid_source_penalty += self.invalid_source_penalty

            zone_power = max((effective_actions[room_name] for room_name in room_names), default=0.0)
            if zone_source == HeatSourceType.GAS_FURNACE and zone_power > 0.0:
                gas_cost += zone_config.hourly_gas_cost(float(prices["gas"])) * zone_power
            else:
                for room_name in room_names:
                    electric_cost += (
                        self._scenario.room_configs[room_name].heater_capacity / 1000.0
                    ) * effective_actions[room_name] * float(prices["electricity"])

            switching_penalty += abs(
                (1.0 if zone_source == HeatSourceType.GAS_FURNACE else 0.0)
                - (1.0 if self._last_source_actions[zone_name] == HeatSourceType.GAS_FURNACE else 0.0)
            )

        for room_name, room_action in effective_actions.items():
            room_config = self._scenario.room_configs[room_name]
            room_temp = float(next_state.room_temperatures[room_name])
            occupancy = float(next_state.occupancy[room_name])
            comfort_violation += occupancy * (
                max(room_config.target_min_temp - room_temp, 0.0)
                + max(room_temp - room_config.target_max_temp, 0.0)
            )
            switching_penalty += abs(room_action - self._last_effective_actions[room_name])

        total_cost = electric_cost + gas_cost
        warmup_scale = 1.0
        if self.comfort_warmup_steps > 0 and self._step_index < self.comfort_warmup_steps:
            warmup_scale = self._step_index / self.comfort_warmup_steps
        reward = -(
            (self.energy_weight * total_cost)
            + (warmup_scale * self.comfort_penalty_weight * comfort_violation)
            + (self.switching_weight * switching_penalty)
            + invalid_source_penalty
        )

        # Expand zone-level source decision to per-room keys so the simulator
        # can look up each room's active heat source by room_name.
        next_state.heat_sources = {
            room_name: zone_sources[self._scenario.room_configs[room_name].zone]
            for room_name in self._room_names
        }
        self._current_state = next_state
        self._last_effective_actions = dict(effective_actions)
        self._last_source_actions = dict(zone_sources)
        self._step_index += 1

        terminated = self._step_index >= self._scenario.horizon_steps
        truncated = False
        info = {
            "scenario_name": self._scenario.name,
            "requested_room_actions": {
                room_name: action_name_for_power_level(action_level)
                for room_name, action_level in requested_room_actions.items()
            },
            "requested_room_power_levels": dict(requested_room_actions),
            "effective_room_actions": {
                room_name: action_name_for_power_level(action_level)
                for room_name, action_level in effective_actions.items()
            },
            "effective_room_power_levels": dict(effective_actions),
            "requested_zone_heat_sources": {
                zone_name: source.value for zone_name, source in requested_zone_sources.items()
            },
            "zone_heat_sources": {zone_name: source.value for zone_name, source in zone_sources.items()},
            "zone_source_signals": zone_source_signals,
            "electric_cost": electric_cost,
            "gas_cost": gas_cost,
            "total_cost": total_cost,
            "comfort_violation": comfort_violation,
            "invalid_source_penalty": invalid_source_penalty,
            "room_names": list(self._room_names),
            "zone_names": list(self._zone_names),
            "active_rooms": len(self._room_names),
            "active_zones": len(self._zone_names),
            "max_rooms": self.max_rooms,
            "max_zones": self.max_zones,
            "occupancy_forecast_horizon_steps": self.occupancy_forecast_horizon_steps,
            "zone_has_furnace": {
                zone_name: self._scenario.zone_configs[zone_name].has_furnace
                for zone_name in self._zone_names
            },
        }
        return self._observation(), reward, terminated, truncated, info
