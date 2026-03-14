"""
Named policy helpers for deterministic multi-room evaluation.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from intelliwarm.control import RoomHeatingIntent, ZoneSourceMode

from .evaluation import PolicyEvaluationSummary, constant_policy, evaluate_policy
from .gym_env import IntelliWarmMultiRoomEnv
from .scenario_generator import SyntheticScenarioGenerator


def build_policy_catalog() -> Dict[str, Callable]:
    """Return the built-in deterministic policy catalog."""

    return {
        "eco-electric": constant_policy(
            room_intent=RoomHeatingIntent.PROTECT,
            zone_source=ZoneSourceMode.ELECTRIC,
        ),
        "comfort-electric": constant_policy(
            room_intent=RoomHeatingIntent.MAINTAIN,
            zone_source=ZoneSourceMode.ELECTRIC,
        ),
        "preheat-electric": constant_policy(
            room_intent=RoomHeatingIntent.PREHEAT,
            zone_source=ZoneSourceMode.ELECTRIC,
        ),
        "comfort-furnace": constant_policy(
            room_intent=RoomHeatingIntent.MAINTAIN,
            zone_source=ZoneSourceMode.GAS_FURNACE,
        ),
        "preheat-furnace": constant_policy(
            room_intent=RoomHeatingIntent.PREHEAT,
            zone_source=ZoneSourceMode.GAS_FURNACE,
        ),
    }


def evaluate_named_policies(
    policy_names: List[str],
    scenario_names: Optional[List[str]] = None,
    max_steps: Optional[int] = None,
) -> Dict[str, PolicyEvaluationSummary]:
    """Evaluate named built-in policies across the default scenario library."""

    catalog = build_policy_catalog()
    unknown = [name for name in policy_names if name not in catalog]
    if unknown:
        raise KeyError(f"Unknown policy names: {', '.join(sorted(unknown))}")

    env = IntelliWarmMultiRoomEnv(SyntheticScenarioGenerator().default_scenarios())
    return {
        policy_name: evaluate_policy(
            env,
            catalog[policy_name],
            scenario_names=scenario_names,
            max_steps=max_steps,
        )
        for policy_name in policy_names
    }
