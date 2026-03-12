"""
Rule-based baseline controller with explainable discrete actions.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

from intelliwarm.data import ControlDecision, HeatingAction, RoomConfig


class BaselineController:
    """Choose an explainable discrete heating action for a single room."""

    def __init__(
        self,
        room_config: RoomConfig,
        min_temperature: float,
        max_temperature: float,
        preheat_lookahead_steps: int = 2,
    ):
        self.room_config = room_config
        self.min_temperature = float(min_temperature)
        self.max_temperature = float(max_temperature)
        self.preheat_lookahead_steps = max(1, int(preheat_lookahead_steps))
        self.logger = logging.getLogger("IntelliWarm.BaselineController")

    def compute_decision(
        self,
        current_temp: float,
        occupancy_forecast: Sequence[float],
        energy_prices: Sequence[float],
        current_action: float = 0.0,
        outside_temp: float = 5.0,
        target_temp: Optional[float] = None,
    ) -> ControlDecision:
        """Return the next explainable heating action."""
        forecast = list(occupancy_forecast) or [0.0]
        occupancy_now = float(forecast[0])
        target = float(target_temp if target_temp is not None else self._default_target_temp())
        next_occupied_step = self._next_occupied_step(forecast)
        base_action, reasons = self._select_action(
            current_temp=current_temp,
            occupancy_now=occupancy_now,
            next_occupied_step=next_occupied_step,
            outside_temp=outside_temp,
            target_temp=target,
        )
        action, hold_reason = self._apply_hysteresis(
            proposed_action=base_action,
            current_action=current_action,
            current_temp=current_temp,
            target_temp=target,
        )
        if hold_reason:
            reasons.append(hold_reason)

        projected_cost = round(action.power_level * float(energy_prices[0] if energy_prices else 0.0), 4)
        rationale = self._build_rationale(action, reasons, next_occupied_step)

        return ControlDecision(
            room_id=self.room_config.room_id,
            action=action,
            source="baseline",
            rationale=rationale,
            reasons=reasons,
            target_temp=target,
            current_temp=float(current_temp),
            occupancy_probability=occupancy_now,
            next_occupied_within_steps=next_occupied_step,
            projected_cost=projected_cost,
            metadata={
                "zone": self.room_config.zone,
                "outside_temp": float(outside_temp),
            },
        )

    def _default_target_temp(self) -> float:
        return (self.room_config.target_min_temp + self.room_config.target_max_temp) / 2.0

    def _comfort_floor(self) -> float:
        return max(self.min_temperature, self.room_config.target_min_temp)

    def _comfort_ceiling(self) -> float:
        return min(self.max_temperature, self.room_config.target_max_temp)

    def _next_occupied_step(self, occupancy_forecast: List[float]) -> Optional[int]:
        return next((index for index, value in enumerate(occupancy_forecast) if value >= 0.6), None)

    def _select_action(
        self,
        current_temp: float,
        occupancy_now: float,
        next_occupied_step: Optional[int],
        outside_temp: float,
        target_temp: float,
    ) -> Tuple[HeatingAction, List[str]]:
        comfort_floor = self._comfort_floor()
        comfort_ceiling = self._comfort_ceiling()
        occupied_now = occupancy_now >= 0.6
        occupied_soon = next_occupied_step is not None and 0 < next_occupied_step <= self.preheat_lookahead_steps

        if occupied_now:
            if current_temp < comfort_floor - 0.5:
                return HeatingAction.PREHEAT, [
                    "Room is occupied and below the comfort floor.",
                    "Applying maximum heat to recover comfort quickly.",
                ]
            if current_temp < target_temp - 0.3 or (outside_temp <= 0.0 and current_temp < target_temp):
                return HeatingAction.COMFORT, [
                    "Room is occupied and below the occupied target.",
                    "Maintaining comfort without overshooting.",
                ]
            if current_temp <= comfort_ceiling:
                return HeatingAction.ECO, [
                    "Room is occupied but already within the comfort band.",
                    "Holding a low, steady output to reduce cycling.",
                ]
            return HeatingAction.OFF, [
                "Room is occupied but already above the comfort ceiling.",
                "Turning heat off to avoid overheating.",
            ]

        if occupied_soon:
            if current_temp < comfort_floor - 0.5:
                return HeatingAction.PREHEAT, [
                    "Occupancy is expected soon and the room is too cold.",
                    "Preheating now reduces comfort lag at arrival.",
                ]
            if current_temp < target_temp - 1.0:
                return HeatingAction.COMFORT, [
                    "Occupancy is expected soon and the room is below target.",
                    "Starting comfort recovery ahead of arrival.",
                ]
            return HeatingAction.ECO, [
                "Occupancy is expected soon, but the room is not far from target.",
                "Using ECO to coast toward comfort efficiently.",
            ]

        if current_temp < comfort_floor:
            return HeatingAction.ECO, [
                "Room is unoccupied but below the minimum protection temperature.",
                "Using ECO to protect the room while saving energy.",
            ]

        if outside_temp < 0.0 and current_temp < comfort_floor + 0.5:
            return HeatingAction.ECO, [
                "Outdoor conditions are cold and the room is near the protection floor.",
                "Using ECO to reduce cold-soak risk.",
            ]

        return HeatingAction.OFF, [
            "Room is unoccupied and above the protection floor.",
            "Turning heat off to minimize cost.",
        ]

    def _apply_hysteresis(
        self,
        proposed_action: HeatingAction,
        current_action: float,
        current_temp: float,
        target_temp: float,
    ) -> Tuple[HeatingAction, Optional[str]]:
        current_action_enum = HeatingAction.from_value(current_action)

        if (
            proposed_action == HeatingAction.OFF
            and current_action_enum == HeatingAction.ECO
            and current_temp < self._comfort_floor() + 0.2
        ):
            return current_action_enum, "Holding ECO to avoid chatter near the protection floor."

        if (
            proposed_action == HeatingAction.ECO
            and current_action_enum == HeatingAction.COMFORT
            and current_temp < target_temp - 0.1
        ):
            return current_action_enum, "Holding COMFORT briefly to avoid oscillating around the target."

        if (
            proposed_action == HeatingAction.COMFORT
            and current_action_enum == HeatingAction.PREHEAT
            and current_temp < target_temp - 0.5
        ):
            return current_action_enum, "Holding PREHEAT until the room is closer to the occupied target."

        return proposed_action, None

    def _build_rationale(
        self,
        action: HeatingAction,
        reasons: Sequence[str],
        next_occupied_step: Optional[int],
    ) -> str:
        summary = f"{action.name} selected."
        if next_occupied_step is not None and next_occupied_step > 0:
            summary += f" Expected occupancy begins in {next_occupied_step} step(s)."
        if reasons:
            summary += " " + " ".join(reasons)
        return summary
