"""
Rule-based baseline controller with explainable continuous thermostat demand.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

from intelliwarm.data import (
    ControlDecision,
    RoomConfig,
)
from intelliwarm.control.intent_resolver import IntentCommandResolver


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
        self._intent_resolver = IntentCommandResolver(
            room_config=room_config,
            min_temperature=min_temperature,
            max_temperature=max_temperature,
            preheat_lookahead_steps=preheat_lookahead_steps,
        )

    def compute_decision(
        self,
        current_temp: float,
        occupancy_forecast: Sequence[float],
        energy_prices: Sequence[float],
        current_action: float = 0.0,
        outside_temp: float = 5.0,
        target_temp: Optional[float] = None,
        room_intent: Optional[object] = None,
    ) -> ControlDecision:
        """Return the next explainable heating demand."""
        resolved_command = self._intent_resolver.resolve(
            current_temp=current_temp,
            occupancy_forecast=occupancy_forecast,
            energy_prices=energy_prices,
            current_action=current_action,
            outside_temp=outside_temp,
            target_temp=target_temp,
            room_intent=room_intent,
        )

        return ControlDecision(
            room_id=self.room_config.room_id,
            action=resolved_command.action,
            source="baseline",
            rationale=resolved_command.rationale,
            reasons=resolved_command.reasons,
            target_temp=resolved_command.target_temp,
            current_temp=float(current_temp),
            occupancy_probability=resolved_command.occupancy_probability,
            next_occupied_within_steps=resolved_command.next_occupied_step,
            projected_cost=resolved_command.projected_cost,
            metadata={
                "zone": self.room_config.zone,
                "outside_temp": float(outside_temp),
                "room_intent": resolved_command.intent.value,
                "room_intent_inferred": room_intent is None,
            },
        )
