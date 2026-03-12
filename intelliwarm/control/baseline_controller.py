"""
Rule-based baseline controller with explainable continuous thermostat demand.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

from intelliwarm.data import (
    action_name_for_power_level,
    clamp_power_level,
    ControlDecision,
    RoomConfig,
)


class BaselineController:
    """Choose an explainable continuous heating demand for a single room."""

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
        """Return the next explainable heating demand."""
        forecast = list(occupancy_forecast) or [0.0]
        occupancy_now = float(forecast[0])
        target = float(target_temp if target_temp is not None else self._default_target_temp())
        next_occupied_step = self._next_occupied_step(forecast)
        base_power, reasons = self._select_power(
            current_temp=current_temp,
            occupancy_now=occupancy_now,
            next_occupied_step=next_occupied_step,
            outside_temp=outside_temp,
            target_temp=target,
        )
        action_power, hold_reason = self._apply_hysteresis(
            proposed_power=base_power,
            current_action=current_action,
            current_temp=current_temp,
            target_temp=target,
        )
        if hold_reason:
            reasons.append(hold_reason)

        projected_cost = round(
            action_power
            * (self.room_config.heater_capacity / 1000.0)
            * float(energy_prices[0] if energy_prices else 0.0),
            4,
        )
        rationale = self._build_rationale(action_power, reasons, next_occupied_step)

        return ControlDecision(
            room_id=self.room_config.room_id,
            action=action_power,
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

    def _cold_outdoor_boost(self, outside_temp: float) -> float:
        return min(0.20, max(0.0, (0.0 - float(outside_temp)) * 0.02))

    def _select_power(
        self,
        current_temp: float,
        occupancy_now: float,
        next_occupied_step: Optional[int],
        outside_temp: float,
        target_temp: float,
    ) -> Tuple[float, List[str]]:
        comfort_floor = self._comfort_floor()
        comfort_ceiling = self._comfort_ceiling()
        occupied_now = occupancy_now >= 0.6
        occupied_soon = next_occupied_step is not None and 0 < next_occupied_step <= self.preheat_lookahead_steps
        cold_boost = self._cold_outdoor_boost(outside_temp)

        if occupied_now:
            if current_temp >= comfort_ceiling:
                return 0.0, [
                    "Room is occupied but already above the comfort ceiling.",
                    "Turning heat off to avoid overheating.",
                ]
            error = max(target_temp - current_temp, 0.0)
            if current_temp < comfort_floor - 0.5:
                return clamp_power_level(0.55 + (0.25 * error) + cold_boost), [
                    "Room is occupied and below the comfort floor.",
                    "Applying a strong recovery demand to restore comfort quickly.",
                ]
            if current_temp < target_temp - 0.3 or (outside_temp <= 0.0 and current_temp < target_temp):
                return clamp_power_level(0.35 + (0.20 * error) + cold_boost + (0.10 * occupancy_now)), [
                    "Room is occupied and below the occupied target.",
                    "Ramping heat to maintain comfort without overshooting.",
                ]
            return clamp_power_level(0.12 + (0.10 * max(comfort_ceiling - current_temp, 0.0)) + (0.5 * cold_boost)), [
                "Room is occupied and already near the comfort target.",
                "Holding a light steady demand to reduce cycling.",
            ]

        if occupied_soon:
            urgency = (
                (self.preheat_lookahead_steps - next_occupied_step + 1) / self.preheat_lookahead_steps
                if next_occupied_step is not None
                else 0.0
            )
            target_gap = max(target_temp - current_temp, 0.0)
            if current_temp < comfort_floor - 0.5:
                return clamp_power_level(0.45 + (0.20 * target_gap) + (0.20 * urgency) + cold_boost), [
                    "Occupancy is expected soon and the room is too cold.",
                    "Starting an aggressive preheat so comfort is ready at arrival.",
                ]
            if current_temp < target_temp - 1.0:
                return clamp_power_level(0.25 + (0.18 * target_gap) + (0.25 * urgency) + cold_boost), [
                    "Occupancy is expected soon and the room is below target.",
                    "Starting comfort recovery ahead of arrival.",
                ]
            return clamp_power_level(0.10 + (0.12 * urgency) + (0.5 * cold_boost)), [
                "Occupancy is expected soon, but the room is not far from target.",
                "Using a light preheat demand to coast toward comfort efficiently.",
            ]

        if current_temp < comfort_floor:
            return clamp_power_level(0.15 + (0.20 * (comfort_floor - current_temp)) + cold_boost), [
                "Room is unoccupied but below the minimum protection temperature.",
                "Applying a modest protection demand while saving energy.",
            ]

        if outside_temp < 0.0 and current_temp < comfort_floor + 0.5:
            return clamp_power_level(0.08 + (0.15 * (comfort_floor + 0.5 - current_temp)) + (0.5 * cold_boost)), [
                "Outdoor conditions are cold and the room is near the protection floor.",
                "Adding a small hold demand to reduce cold-soak risk.",
            ]

        return 0.0, [
            "Room is unoccupied and above the protection floor.",
            "Turning heat off to minimize cost.",
        ]

    def _apply_hysteresis(
        self,
        proposed_power: float,
        current_action: float,
        current_temp: float,
        target_temp: float,
    ) -> Tuple[float, Optional[str]]:
        proposed = clamp_power_level(proposed_power)
        current = clamp_power_level(current_action)

        if (
            proposed <= 0.05
            and current >= 0.20
            and current_temp < self._comfort_floor() + 0.2
        ):
            return min(current, 0.35), "Holding some heat to avoid chatter near the protection floor."

        if (
            proposed < current
            and current >= 0.55
            and current_temp < target_temp - 0.1
        ):
            return max(proposed, current - 0.10), "Tapering heat gradually to avoid oscillating around the target."

        if (
            proposed < current
            and current >= 0.85
            and current_temp < target_temp - 0.5
        ):
            return max(proposed, current - 0.10), "Holding a stronger recovery demand until the room is closer to target."

        return proposed, None

    def _build_rationale(
        self,
        action: float,
        reasons: Sequence[str],
        next_occupied_step: Optional[int],
    ) -> str:
        action_label = action_name_for_power_level(action)
        summary = f"{action_label}-like demand ({clamp_power_level(action):.2f}) selected."
        if next_occupied_step is not None and next_occupied_step > 0:
            summary += f" Expected occupancy begins in {next_occupied_step} step(s)."
        if reasons:
            summary += " " + " ".join(reasons)
        return summary
