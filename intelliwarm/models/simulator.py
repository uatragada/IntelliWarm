"""
Deterministic multi-room simulator.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional

from intelliwarm.data import HeatingAction, RoomConfig, SimulationState
from intelliwarm.models.thermal_model import solar_irradiance_wm2
from intelliwarm.prediction import OccupancyPredictor


class HouseSimulator:
    """Advance room temperatures for a collection of rooms over fixed timesteps.

    When the thermal model for a room accepts ``solar_irradiance_w_m2`` and
    ``occupancy`` keyword arguments (i.e. both :class:`RoomThermalModel` and
    :class:`PhysicsRoomThermalModel`), the simulator automatically provides
    them.  Solar irradiance is derived from the simulation timestamp using the
    built-in astronomical clear-sky model; occupancy comes from the per-room
    predictors or the room config schedule.

    Args:
        room_configs:          Mapping of room_id → :class:`RoomConfig`.
        thermal_models:        Mapping of room_id → thermal model instance.
        occupancy_predictors:  Optional per-room occupancy predictors.
        latitude_deg:          Site latitude for solar irradiance [°N].
        cloud_cover:           Constant fractional cloud cover [0–1].
    """

    def __init__(
        self,
        room_configs: Dict[str, RoomConfig],
        thermal_models: Dict[str, object],
        occupancy_predictors: Optional[Dict[str, OccupancyPredictor]] = None,
        latitude_deg: float = 40.0,
        cloud_cover: float = 0.0,
    ):
        self.room_configs = room_configs
        self.thermal_models = thermal_models
        self.occupancy_predictors = occupancy_predictors or {}
        self.latitude_deg = latitude_deg
        self.cloud_cover = cloud_cover

    def _resolve_action(self, action: object) -> HeatingAction:
        return HeatingAction.from_value(action)

    def _occupancy_for_timestamp(self, room_name: str, timestamp: datetime) -> float:
        predictor = self.occupancy_predictors.get(room_name)
        if predictor is not None:
            return predictor.predict(timestamp)

        room_config = self.room_configs.get(room_name)
        if room_config is None or not room_config.occupancy_schedule:
            return 0.5

        matching_windows = [window for window in room_config.occupancy_schedule if window.contains(timestamp)]
        if not matching_windows:
            return 0.1

        return max(window.probability for window in matching_windows)

    def step(
        self,
        state: SimulationState,
        heating_actions: Dict[str, object],
        next_timestamp: Optional[datetime] = None,
        dt_minutes: int = 60,
    ) -> SimulationState:
        """Advance the simulation by one step."""
        resolved_timestamp = next_timestamp or (state.timestamp + timedelta(minutes=dt_minutes))
        next_temperatures: Dict[str, float] = {}
        resolved_actions: Dict[str, HeatingAction] = {}

        solar_w_m2 = solar_irradiance_wm2(
            resolved_timestamp,
            latitude_deg=self.latitude_deg,
            cloud_cover=self.cloud_cover,
        )

        for room_name, current_temp in state.room_temperatures.items():
            action = self._resolve_action(heating_actions.get(room_name, HeatingAction.OFF))
            resolved_actions[room_name] = action
            occupancy_val = self._occupancy_for_timestamp(room_name, resolved_timestamp)
            next_temperatures[room_name] = self.thermal_models[room_name].step(
                current_temp=current_temp,
                outside_temp=state.outdoor_temp,
                heating_power=action.power_level,
                dt_minutes=dt_minutes,
                solar_irradiance_w_m2=solar_w_m2,
                occupancy=occupancy_val,
            )

        next_occupancy = {
            room_name: self._occupancy_for_timestamp(room_name, resolved_timestamp)
            for room_name in next_temperatures
        }

        return SimulationState(
            timestamp=resolved_timestamp,
            outdoor_temp=state.outdoor_temp,
            room_temperatures=next_temperatures,
            heating_actions=resolved_actions,
            occupancy=next_occupancy,
        )

    def simulate(
        self,
        start_time: datetime,
        initial_temperatures: Dict[str, float],
        outdoor_temperatures: List[float],
        heating_plan: List[Dict[str, object]],
        dt_minutes: int = 60,
    ) -> List[SimulationState]:
        """Run a deterministic multi-room simulation over a planning horizon."""
        if len(outdoor_temperatures) != len(heating_plan):
            raise ValueError("Outdoor temperatures and heating plan must have the same length")

        initial_occupancy = {
            room_name: self._occupancy_for_timestamp(room_name, start_time)
            for room_name in initial_temperatures
        }
        states = [
            SimulationState(
                timestamp=start_time,
                outdoor_temp=outdoor_temperatures[0] if outdoor_temperatures else 0.0,
                room_temperatures=dict(initial_temperatures),
                heating_actions={room_name: HeatingAction.OFF for room_name in initial_temperatures},
                occupancy=initial_occupancy,
            )
        ]

        current_state = states[0]
        for step_index, planned_actions in enumerate(heating_plan):
            current_state = SimulationState(
                timestamp=current_state.timestamp,
                outdoor_temp=outdoor_temperatures[step_index],
                room_temperatures=current_state.room_temperatures,
                heating_actions=current_state.heating_actions,
                occupancy=current_state.occupancy,
            )
            next_state = self.step(
                current_state,
                planned_actions,
                next_timestamp=start_time + timedelta(minutes=dt_minutes * (step_index + 1)),
                dt_minutes=dt_minutes,
            )
            states.append(next_state)
            current_state = next_state

        return states