"""
Shared intent-to-command resolution for deterministic and learned controllers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional, Sequence, Tuple

from intelliwarm.data import RoomConfig, action_name_for_power_level, clamp_power_level


class RoomHeatingIntent(Enum):
    """High-level room intent chosen by a policy before deterministic actuation."""

    OFF = "off"
    PROTECT = "protect"
    MAINTAIN = "maintain"
    PREHEAT = "preheat"
    RECOVER = "recover"


class ZoneSourceMode(Enum):
    """High-level zone source preference chosen before deterministic dispatch."""

    AUTO = "auto"
    ELECTRIC = "electric"
    GAS_FURNACE = "gas_furnace"


ROOM_HEATING_INTENTS = list(RoomHeatingIntent)
ZONE_SOURCE_MODES = list(ZoneSourceMode)


def normalize_room_intent(value: Any) -> RoomHeatingIntent:
    """Resolve an intent enum from an enum instance, label, or discrete index."""
    if isinstance(value, RoomHeatingIntent):
        return value
    if isinstance(value, str):
        normalized = value.strip().upper().replace("-", "_").replace(" ", "_")
        legacy = {
            "ECO": "PROTECT",
            "COMFORT": "MAINTAIN",
        }
        normalized = legacy.get(normalized, normalized)
        if normalized in RoomHeatingIntent.__members__:
            return RoomHeatingIntent[normalized]
        value = int(float(value))
    index = int(value)
    if not 0 <= index < len(ROOM_HEATING_INTENTS):
        raise IndexError(f"Unknown room intent index: {index}")
    return ROOM_HEATING_INTENTS[index]


def normalize_zone_source_mode(value: Any) -> ZoneSourceMode:
    """Resolve a zone source mode from an enum instance, label, or discrete index."""
    if isinstance(value, ZoneSourceMode):
        return value
    if isinstance(value, str):
        normalized = value.strip().upper().replace("-", "_").replace(" ", "_")
        if normalized in ZoneSourceMode.__members__:
            return ZoneSourceMode[normalized]
        value = int(float(value))
    index = int(value)
    if not 0 <= index < len(ZONE_SOURCE_MODES):
        raise IndexError(f"Unknown zone source mode index: {index}")
    return ZONE_SOURCE_MODES[index]


def room_intent_index(value: Any) -> int:
    """Return the discrete index used by MultiDiscrete action spaces."""
    return ROOM_HEATING_INTENTS.index(normalize_room_intent(value))


def zone_source_mode_index(value: Any) -> int:
    """Return the discrete index used by MultiDiscrete action spaces."""
    return ZONE_SOURCE_MODES.index(normalize_zone_source_mode(value))


def room_intent_feature_value(value: Any) -> float:
    """Encode a room intent as a compact scalar feature in [0, 1]."""
    if len(ROOM_HEATING_INTENTS) <= 1:
        return 0.0
    return room_intent_index(value) / float(len(ROOM_HEATING_INTENTS) - 1)


def zone_source_mode_feature_value(value: Any) -> float:
    """Encode a zone source mode as a compact scalar feature in [0, 1]."""
    if len(ZONE_SOURCE_MODES) <= 1:
        return 0.0
    return zone_source_mode_index(value) / float(len(ZONE_SOURCE_MODES) - 1)


@dataclass(frozen=True)
class ResolvedRoomCommand:
    """Deterministic room command produced from a high-level intent."""

    intent: RoomHeatingIntent
    action: float
    reasons: List[str] = field(default_factory=list)
    rationale: str = ""
    target_temp: float = 0.0
    current_temp: float = 0.0
    occupancy_probability: float = 0.0
    next_occupied_step: Optional[int] = None
    projected_cost: float = 0.0


class IntentCommandResolver:
    """Convert high-level thermal intents into deterministic normalized heat demand."""

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

    def resolve(
        self,
        current_temp: float,
        occupancy_forecast: Sequence[float],
        energy_prices: Sequence[float],
        current_action: float = 0.0,
        outside_temp: float = 5.0,
        target_temp: Optional[float] = None,
        room_intent: Optional[object] = None,
    ) -> ResolvedRoomCommand:
        forecast = list(occupancy_forecast) or [0.0]
        occupancy_now = float(forecast[0])
        target = float(target_temp if target_temp is not None else self._default_target_temp())
        next_occupied_step = self._next_occupied_step(forecast)
        intent = (
            normalize_room_intent(room_intent)
            if room_intent is not None
            else self.infer_intent(
                current_temp=current_temp,
                occupancy_forecast=forecast,
                outside_temp=outside_temp,
                target_temp=target,
            )
        )
        base_power, reasons = self._power_for_intent(
            intent=intent,
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
        rationale = self._build_rationale(
            intent=intent,
            action=action_power,
            reasons=reasons,
            next_occupied_step=next_occupied_step,
        )
        return ResolvedRoomCommand(
            intent=intent,
            action=action_power,
            reasons=reasons,
            rationale=rationale,
            target_temp=target,
            current_temp=float(current_temp),
            occupancy_probability=occupancy_now,
            next_occupied_step=next_occupied_step,
            projected_cost=projected_cost,
        )

    def infer_intent(
        self,
        current_temp: float,
        occupancy_forecast: Sequence[float],
        outside_temp: float = 5.0,
        target_temp: Optional[float] = None,
    ) -> RoomHeatingIntent:
        forecast = list(occupancy_forecast) or [0.0]
        occupancy_now = float(forecast[0])
        target = float(target_temp if target_temp is not None else self._default_target_temp())
        comfort_floor = self._comfort_floor()
        comfort_ceiling = self._comfort_ceiling()
        next_occupied_step = self._next_occupied_step(forecast)
        occupied_now = occupancy_now >= 0.6
        occupied_soon = next_occupied_step is not None and 0 < next_occupied_step <= self.preheat_lookahead_steps

        if occupied_now:
            if current_temp >= comfort_ceiling:
                return RoomHeatingIntent.OFF
            if current_temp < comfort_floor - 0.5:
                return RoomHeatingIntent.RECOVER
            return RoomHeatingIntent.MAINTAIN

        if occupied_soon:
            return RoomHeatingIntent.PREHEAT

        if current_temp < comfort_floor:
            return RoomHeatingIntent.PROTECT

        if outside_temp < 0.0 and current_temp < comfort_floor + 0.5:
            return RoomHeatingIntent.PROTECT

        if current_temp < target - 0.2 and occupancy_now >= 0.3:
            return RoomHeatingIntent.MAINTAIN

        return RoomHeatingIntent.OFF

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

    def _urgency(self, next_occupied_step: Optional[int]) -> float:
        if next_occupied_step is None:
            return 1.0
        if next_occupied_step <= 0:
            return 1.0
        return max(
            0.0,
            min(
                1.0,
                (self.preheat_lookahead_steps - next_occupied_step + 1) / self.preheat_lookahead_steps,
            ),
        )

    def _power_for_intent(
        self,
        intent: RoomHeatingIntent,
        current_temp: float,
        occupancy_now: float,
        next_occupied_step: Optional[int],
        outside_temp: float,
        target_temp: float,
    ) -> Tuple[float, List[str]]:
        comfort_floor = self._comfort_floor()
        comfort_ceiling = self._comfort_ceiling()
        cold_boost = self._cold_outdoor_boost(outside_temp)
        target_gap = max(target_temp - current_temp, 0.0)
        urgency = self._urgency(next_occupied_step)

        if intent == RoomHeatingIntent.OFF:
            return 0.0, [
                "Intent is OFF.",
                "Turning heat off unless hysteresis keeps a small hold demand.",
            ]

        if intent == RoomHeatingIntent.PROTECT:
            if current_temp < comfort_floor:
                return clamp_power_level(0.15 + (0.20 * (comfort_floor - current_temp)) + cold_boost), [
                    "Intent is PROTECT and the room is below the protection floor.",
                    "Applying a modest hold demand to avoid cold-soak while saving energy.",
                ]
            if outside_temp < 0.0 and current_temp < comfort_floor + 0.5:
                return clamp_power_level(
                    0.08 + (0.15 * (comfort_floor + 0.5 - current_temp)) + (0.5 * cold_boost)
                ), [
                    "Intent is PROTECT and the room is near the protection floor in cold weather.",
                    "Adding a small hold demand to reduce freeze-risk and cold-soak.",
                ]
            return 0.0, [
                "Intent is PROTECT, but the room is already warm enough to coast.",
                "Leaving heat off to save energy.",
            ]

        if intent == RoomHeatingIntent.MAINTAIN:
            if current_temp >= comfort_ceiling:
                return 0.0, [
                    "Intent is MAINTAIN, but the room is already above the comfort ceiling.",
                    "Turning heat off to avoid overheating.",
                ]
            if current_temp < comfort_floor - 0.5:
                return clamp_power_level(0.40 + (0.20 * target_gap) + cold_boost + (0.05 * occupancy_now)), [
                    "Intent is MAINTAIN and the room has drifted well below the comfort floor.",
                    "Applying a stronger recovery demand before tapering back.",
                ]
            if current_temp < target_temp - 0.3 or (outside_temp <= 0.0 and current_temp < target_temp):
                return clamp_power_level(0.25 + (0.16 * target_gap) + cold_boost + (0.05 * occupancy_now)), [
                    "Intent is MAINTAIN and the room is below target.",
                    "Using a moderate demand to hold comfort without overshooting.",
                ]
            return clamp_power_level(0.10 + (0.08 * max(comfort_ceiling - current_temp, 0.0)) + (0.5 * cold_boost)), [
                "Intent is MAINTAIN and the room is already near target.",
                "Holding a light steady demand to reduce cycling.",
            ]

        if intent == RoomHeatingIntent.PREHEAT:
            if current_temp >= comfort_ceiling:
                return 0.0, [
                    "Intent is PREHEAT, but the room is already above the comfort ceiling.",
                    "Turning heat off to avoid wasting energy.",
                ]
            if current_temp < comfort_floor - 0.5:
                return clamp_power_level(0.45 + (0.20 * target_gap) + (0.20 * urgency) + cold_boost), [
                    "Intent is PREHEAT and the room is too cold for the upcoming occupancy window.",
                    "Starting an aggressive preheat so comfort is ready at arrival.",
                ]
            if current_temp < target_temp - 1.0:
                return clamp_power_level(0.25 + (0.18 * target_gap) + (0.25 * urgency) + cold_boost), [
                    "Intent is PREHEAT and the room is below the occupied target.",
                    "Ramping heat ahead of arrival rather than waiting for occupancy to begin.",
                ]
            return clamp_power_level(0.10 + (0.12 * urgency) + (0.5 * cold_boost)), [
                "Intent is PREHEAT and the room is already near target.",
                "Using a light preheat demand so the room coasts into comfort efficiently.",
            ]

        if current_temp >= comfort_ceiling:
            return 0.0, [
                "Intent is RECOVER, but the room is already above the comfort ceiling.",
                "Turning heat off to avoid overshooting.",
            ]
        return clamp_power_level(
            0.60
            + (0.22 * target_gap)
            + (0.10 * max(comfort_floor - current_temp, 0.0))
            + cold_boost
            + (0.05 * occupancy_now)
        ), [
            "Intent is RECOVER and the room is below comfort.",
            "Applying an aggressive demand to restore comfort quickly before tapering.",
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
        intent: RoomHeatingIntent,
        action: float,
        reasons: Sequence[str],
        next_occupied_step: Optional[int],
    ) -> str:
        intent_label = intent.value.replace("_", " ").title()
        summary = (
            f"{intent_label} intent resolved to "
            f"{action_name_for_power_level(action)}-like demand ({clamp_power_level(action):.2f})."
        )
        if next_occupied_step is not None and next_occupied_step > 0:
            summary += f" Expected occupancy begins in {next_occupied_step} step(s)."
        if reasons:
            summary += " " + " ".join(reasons)
        return summary
