"""
Deterministic policy evaluation helpers for IntelliWarm learning environments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from intelliwarm.data import clamp_power_level, HeatSourceType, HeatingAction
from .gym_env import IntelliWarmMultiRoomEnv


PolicyFn = Callable[[np.ndarray, Dict], Sequence[float]]


@dataclass(frozen=True)
class ScenarioEvaluationResult:
    """Aggregate evaluation metrics for one scenario rollout."""

    scenario_name: str
    total_reward: float
    total_cost: float
    total_comfort_violation: float
    steps: int
    final_zone_heat_sources: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyEvaluationSummary:
    """Summary of a policy evaluated across multiple scenarios."""

    scenario_results: List[ScenarioEvaluationResult]

    @property
    def scenario_count(self) -> int:
        return len(self.scenario_results)

    @property
    def total_reward(self) -> float:
        return sum(result.total_reward for result in self.scenario_results)

    @property
    def total_cost(self) -> float:
        return sum(result.total_cost for result in self.scenario_results)

    @property
    def total_comfort_violation(self) -> float:
        return sum(result.total_comfort_violation for result in self.scenario_results)


def constant_policy(
    room_action: object = HeatingAction.ECO,
    zone_source: HeatSourceType = HeatSourceType.ELECTRIC,
) -> PolicyFn:
    """Build a constant policy compatible with the multi-room environment."""

    zone_value = 1.0 if zone_source == HeatSourceType.GAS_FURNACE else 0.0
    room_value = clamp_power_level(room_action)

    def _policy(_observation: np.ndarray, info: Dict) -> Sequence[float]:
        zone_names = list(info.get("zone_names", []))
        zone_has_furnace = dict(info.get("zone_has_furnace", {}))
        zone_values = []
        for zone_index in range(int(info["max_zones"])):
            zone_name = zone_names[zone_index] if zone_index < len(zone_names) else ""
            if zone_source == HeatSourceType.GAS_FURNACE and not zone_has_furnace.get(zone_name, False):
                zone_values.append(0.0)
            else:
                zone_values.append(zone_value)
        return zone_values + [room_value] * int(info["max_rooms"])

    return _policy


def evaluate_policy(
    env: IntelliWarmMultiRoomEnv,
    policy: PolicyFn,
    scenario_names: Optional[List[str]] = None,
    max_steps: Optional[int] = None,
) -> PolicyEvaluationSummary:
    """Evaluate a policy across one or more deterministic scenarios."""

    selected_scenarios = scenario_names or [scenario.name for scenario in env.scenarios]
    results: List[ScenarioEvaluationResult] = []

    for scenario_name in selected_scenarios:
        observation, info = env.reset(options={"scenario_name": scenario_name})
        terminated = False
        truncated = False
        steps = 0
        total_reward = 0.0
        total_cost = 0.0
        total_comfort_violation = 0.0
        last_info = dict(info)

        while not terminated and not truncated:
            if max_steps is not None and steps >= max_steps:
                break

            action = policy(observation, last_info)
            observation, reward, terminated, truncated, last_info = env.step(action)
            total_reward += float(reward)
            total_cost += float(last_info.get("total_cost", 0.0))
            total_comfort_violation += float(
                last_info.get(
                    "reported_comfort_violation",
                    last_info.get("comfort_violation", 0.0),
                )
            )
            steps += 1

        results.append(
            ScenarioEvaluationResult(
                scenario_name=scenario_name,
                total_reward=total_reward,
                total_cost=total_cost,
                total_comfort_violation=total_comfort_violation,
                steps=steps,
                final_zone_heat_sources=dict(last_info.get("zone_heat_sources", {})),
            )
        )

    return PolicyEvaluationSummary(scenario_results=results)
