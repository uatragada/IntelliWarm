"""
Hybrid heating controller.

Decides per zone whether to run the gas furnace (zone-wide) or individual
electric space heaters (per room) based on projected hourly cost.

Decision logic
--------------
For each zone at each timestep:
  1. Use the BaselineController to get a per-room action recommendation.
  2. Identify which rooms actually need heat (action != OFF).
  3. Compute:
       electric_cost  = sum of (heater_capacity_kw * action.power_level
                                * electricity_price) for each room needing heat
       furnace_cost   = (furnace_btu_per_hour / 100_000)
                        / furnace_efficiency * gas_price_per_therm
  4. If the zone has a furnace and furnace_cost < electric_cost:
       activate the furnace for the whole zone; all rooms get the max action
       level among the rooms that need heat.
     Else:
       keep per-room electric actions from the baseline.

The furnace heats the entire zone uniformly — you cannot target individual
rooms with a central gas furnace.  Individual electric heaters allow
room-level control and are cheaper when only one or two rooms need heat.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

from intelliwarm.data.models import (
    ControlDecision,
    HeatingAction,
    HeatSourceType,
    HybridHeatingDecision,
    RoomConfig,
    ZoneConfig,
)
from intelliwarm.control.baseline_controller import BaselineController


logger = logging.getLogger("IntelliWarm.HybridController")


# ---------------------------------------------------------------------------
# Cost helpers
# ---------------------------------------------------------------------------

def _electric_room_cost(
    action: HeatingAction,
    heater_capacity_watts: float,
    electricity_price_kwh: float,
) -> float:
    """$/hr for a single electric heater at the given action level."""
    heater_kw = heater_capacity_watts / 1000.0
    return action.power_level * heater_kw * electricity_price_kwh


def _furnace_hourly_cost(zone_config: ZoneConfig, gas_price_per_therm: float) -> float:
    """$/hr to run the zone furnace at full output."""
    return zone_config.hourly_gas_cost(gas_price_per_therm)


# ---------------------------------------------------------------------------
# HybridController
# ---------------------------------------------------------------------------

class HybridController:
    """
    Zone-level hybrid heating controller.

    Parameters
    ----------
    zone_config:
        Zone configuration including furnace details.
    room_configs:
        Mapping of room_id → RoomConfig for every room in the zone.
    min_temperature:
        System-wide freeze-protection floor (°C).
    max_temperature:
        System-wide overheat ceiling (°C).
    preheat_lookahead_steps:
        Passed through to BaselineController for each room.
    """

    def __init__(
        self,
        zone_config: ZoneConfig,
        room_configs: Dict[str, RoomConfig],
        min_temperature: float = 18.0,
        max_temperature: float = 24.0,
        preheat_lookahead_steps: int = 2,
    ):
        self.zone_config = zone_config
        self.room_configs = room_configs
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature

        self._room_controllers: Dict[str, BaselineController] = {
            room_id: BaselineController(
                room_config=cfg,
                min_temperature=min_temperature,
                max_temperature=max_temperature,
                preheat_lookahead_steps=preheat_lookahead_steps,
            )
            for room_id, cfg in room_configs.items()
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decide(
        self,
        room_temperatures: Dict[str, float],
        occupancy_forecasts: Dict[str, Sequence[float]],
        electricity_price: float,
        gas_price: float,
        outside_temp: float = 5.0,
        current_actions: Optional[Dict[str, float]] = None,
        target_temps: Optional[Dict[str, float]] = None,
    ) -> HybridHeatingDecision:
        """
        Return a zone-level heating decision for this timestep.

        Parameters
        ----------
        room_temperatures:
            Current temperature (°C) keyed by room_id.
        occupancy_forecasts:
            Occupancy probability horizon keyed by room_id.
        electricity_price:
            Current electricity price ($/kWh).
        gas_price:
            Current gas price ($/therm).
        outside_temp:
            Current outdoor temperature (°C).
        current_actions:
            Current power level (0–1) per room for hysteresis.
        target_temps:
            Optional per-room target temperature override.
        """
        current_actions = current_actions or {}
        target_temps = target_temps or {}

        # Step 1 — baseline per-room decisions (always computed for context)
        per_room_decisions: Dict[str, ControlDecision] = {}
        for room_id, room_cfg in self.room_configs.items():
            forecast = list(occupancy_forecasts.get(room_id, [0.5]))
            current_temp = room_temperatures.get(room_id, 20.0)
            decision = self._room_controllers[room_id].compute_decision(
                current_temp=current_temp,
                occupancy_forecast=forecast,
                energy_prices=[electricity_price],
                current_action=current_actions.get(room_id, 0.0),
                outside_temp=outside_temp,
                target_temp=target_temps.get(room_id),
            )
            per_room_decisions[room_id] = decision

        # Step 2 — rooms that actually need heat
        rooms_needing_heat = [
            room_id
            for room_id, decision in per_room_decisions.items()
            if decision.action != HeatingAction.OFF
        ]

        # Step 3 — cost comparison
        electric_cost = sum(
            _electric_room_cost(
                action=per_room_decisions[room_id].action,
                heater_capacity_watts=self.room_configs[room_id].heater_capacity,
                electricity_price_kwh=electricity_price,
            )
            for room_id in rooms_needing_heat
        )

        furnace_cost = (
            _furnace_hourly_cost(self.zone_config, gas_price)
            if self.zone_config.has_furnace
            else float("inf")
        )

        # Step 4 — choose source
        use_furnace = (
            self.zone_config.has_furnace
            and bool(rooms_needing_heat)
            and furnace_cost < electric_cost
        )

        if use_furnace:
            heat_source = HeatSourceType.GAS_FURNACE
            # Furnace heats the whole zone uniformly; apply the highest
            # demanded action level across all rooms that need heat.
            zone_action = max(
                (per_room_decisions[r].action for r in rooms_needing_heat),
                key=lambda a: a.power_level,
            )
            per_room_actions: Dict[str, HeatingAction] = {
                room_id: zone_action for room_id in self.room_configs
            }
            chosen_cost = furnace_cost
            rationale = (
                f"Gas furnace selected for zone '{self.zone_config.zone_id}'. "
                f"{len(rooms_needing_heat)} of {len(self.room_configs)} room(s) need heat. "
                f"Furnace cost (${furnace_cost:.4f}/hr) is lower than "
                f"electric total (${electric_cost:.4f}/hr). "
                f"All zone rooms set to {zone_action.name}."
            )
        elif rooms_needing_heat:
            heat_source = HeatSourceType.ELECTRIC
            per_room_actions = {
                room_id: per_room_decisions[room_id].action
                for room_id in self.room_configs
            }
            chosen_cost = electric_cost
            if self.zone_config.has_furnace:
                rationale = (
                    f"Electric heaters selected for zone '{self.zone_config.zone_id}'. "
                    f"{len(rooms_needing_heat)} of {len(self.room_configs)} room(s) need heat. "
                    f"Electric total (${electric_cost:.4f}/hr) is lower than "
                    f"furnace cost (${furnace_cost:.4f}/hr). "
                    f"Each room keeps its individual baseline action."
                )
            else:
                rationale = (
                    f"Electric heaters only — zone '{self.zone_config.zone_id}' has no furnace. "
                    f"{len(rooms_needing_heat)} room(s) heating."
                )
        else:
            # No rooms need heat
            heat_source = HeatSourceType.ELECTRIC
            per_room_actions = {
                room_id: HeatingAction.OFF for room_id in self.room_configs
            }
            chosen_cost = 0.0
            rationale = (
                f"No rooms in zone '{self.zone_config.zone_id}' need heat. "
                f"All devices set to OFF."
            )

        return HybridHeatingDecision(
            zone=self.zone_config.zone_id,
            heat_source=heat_source,
            furnace_on=use_furnace,
            per_room_actions=per_room_actions,
            rooms_needing_heat=rooms_needing_heat,
            electric_hourly_cost=electric_cost,
            furnace_hourly_cost=furnace_cost if self.zone_config.has_furnace else 0.0,
            chosen_hourly_cost=chosen_cost,
            rationale=rationale,
        )

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def zone_id(self) -> str:
        return self.zone_config.zone_id

    @property
    def has_furnace(self) -> bool:
        return self.zone_config.has_furnace

    def room_ids(self) -> List[str]:
        return list(self.room_configs.keys())
