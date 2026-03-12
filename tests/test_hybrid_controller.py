"""
Tests for HybridController zone-level heat source selection.

Scenarios covered:
  1. Multiple rooms need heat → furnace cheaper → furnace selected.
  2. Single room needs heat → electric heaters cheaper → electric selected.
  3. Zone has no furnace → always falls back to electric.
  4. No rooms need heat → all OFF, no cost incurred.
  5. Furnace comparatively expensive → electric selected despite multiple rooms.
  6. Zone action level lifted to highest required room when furnace is picked.
"""

import pytest

from intelliwarm.data.models import (
    HeatSourceType,
    HybridHeatingDecision,
    RoomConfig,
    ZoneConfig,
)
from intelliwarm.control.hybrid_controller import HybridController


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_room(
    room_id: str,
    zone: str = "Residential",
    heat_source: HeatSourceType = HeatSourceType.ELECTRIC,
    heater_capacity: float = 1500.0,
) -> RoomConfig:
    return RoomConfig(
        room_id=room_id,
        display_name=room_id.replace("_", " ").title(),
        zone=zone,
        target_min_temp=20.0,
        target_max_temp=22.0,
        heater_capacity=heater_capacity,
        heat_loss_factor=0.05,
        heating_efficiency=0.9,
        heat_source=heat_source,
    )


def _residential_zone(has_furnace: bool = True) -> ZoneConfig:
    return ZoneConfig(
        zone_id="Residential",
        description="Main living area",
        has_furnace=has_furnace,
        furnace_btu_per_hour=60_000.0,
        furnace_efficiency=0.80,
    )


def _work_zone() -> ZoneConfig:
    return ZoneConfig(
        zone_id="Work",
        description="Office area",
        has_furnace=False,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_controller(
    zone_config: ZoneConfig,
    room_ids: list,
    heater_capacity: float = 1500.0,
) -> HybridController:
    rooms = {
        r: _make_room(r, zone=zone_config.zone_id, heater_capacity=heater_capacity)
        for r in room_ids
    }
    return HybridController(
        zone_config=zone_config,
        room_configs=rooms,
        min_temperature=18.0,
        max_temperature=24.0,
    )


# ---------------------------------------------------------------------------
# Test 1: multiple rooms cold → furnace cheaper
# ---------------------------------------------------------------------------

def test_furnace_selected_when_multiple_rooms_need_heat_and_is_cheaper():
    """
    With 3 cold rooms (each needing COMFORT heat) at an electricity price high
    enough that the combined electric spend exceeds the furnace hourly cost,
    the controller should choose the gas furnace.
    """
    zone = _residential_zone(has_furnace=True)
    rooms = ["bedroom1", "bedroom2", "living_room"]
    ctrl = _build_controller(zone, rooms, heater_capacity=1500.0)

    # All rooms well below setpoint → should all need heat
    room_temps = {"bedroom1": 15.0, "bedroom2": 15.0, "living_room": 15.0}
    # 3 × 1.5 kW × 0.7 (COMFORT) × 0.30 $/kWh ≈ $0.945/hr electric
    # Furnace: (60000/100000)/0.80 × 1.20 $/therm ≈ $0.90/hr
    # → furnace wins
    result = ctrl.decide(
        room_temperatures=room_temps,
        occupancy_forecasts={r: [1.0] * 4 for r in rooms},
        electricity_price=0.30,
        gas_price=1.20,
    )

    assert isinstance(result, HybridHeatingDecision)
    assert result.furnace_on is True
    assert result.heat_source == HeatSourceType.GAS_FURNACE
    assert len(result.rooms_needing_heat) == 3
    assert result.chosen_hourly_cost == pytest.approx(result.furnace_hourly_cost, rel=1e-3)


# ---------------------------------------------------------------------------
# Test 2: single room cold → electric cheaper
# ---------------------------------------------------------------------------

def test_electric_selected_when_single_room_needs_heat():
    """
    With only one cold occupied room and cheap electricity, the per-room
    electric cost is well below the furnace hourly cost; electric is chosen.

    bedroom1 is cold and occupied (needs heat).
    bedroom2 and living_room are warm and unoccupied (action = OFF from baseline).
    """
    zone = _residential_zone(has_furnace=True)
    rooms = ["bedroom1", "bedroom2", "living_room"]
    ctrl = _build_controller(zone, rooms, heater_capacity=1500.0)

    # bedroom1 cold + occupied; others warm + unoccupied
    room_temps = {"bedroom1": 15.0, "bedroom2": 23.0, "living_room": 23.0}
    occupancy = {
        "bedroom1": [1.0] * 4,
        "bedroom2": [0.0] * 4,
        "living_room": [0.0] * 4,
    }
    # 1 × 1.5 kW at COMFORT/PREHEAT × 0.12 $/kWh ≈ $0.126–0.18/hr electric
    # Furnace: ≈ $0.90/hr at 1.20 $/therm → electric wins easily
    result = ctrl.decide(
        room_temperatures=room_temps,
        occupancy_forecasts=occupancy,
        electricity_price=0.12,
        gas_price=1.20,
    )

    assert result.furnace_on is False
    assert result.heat_source == HeatSourceType.ELECTRIC
    assert "bedroom1" in result.rooms_needing_heat
    assert result.electric_hourly_cost < result.furnace_hourly_cost
    assert result.chosen_hourly_cost == pytest.approx(result.electric_hourly_cost, rel=1e-3)


# ---------------------------------------------------------------------------
# Test 3: zone has no furnace → always electric
# ---------------------------------------------------------------------------

def test_electric_forced_when_zone_has_no_furnace():
    """
    A zone without a furnace must always use electric heaters regardless of
    how cold all rooms are or how high the electricity price is.
    """
    zone = _work_zone()  # has_furnace=False
    rooms = ["office1", "office2"]
    ctrl = _build_controller(zone, rooms, heater_capacity=2000.0)

    room_temps = {"office1": 15.0, "office2": 15.0}
    result = ctrl.decide(
        room_temperatures=room_temps,
        occupancy_forecasts={r: [1.0] * 4 for r in rooms},
        electricity_price=0.50,   # very expensive electricity
        gas_price=0.50,
    )

    assert result.furnace_on is False
    assert result.heat_source == HeatSourceType.ELECTRIC
    # Electric cost should be non-zero since rooms need heat
    assert result.electric_hourly_cost > 0.0


# ---------------------------------------------------------------------------
# Test 4: no rooms need heat → all OFF
# ---------------------------------------------------------------------------

def test_no_heat_when_all_rooms_at_setpoint():
    """
    If all rooms are at or above their target temperature and unoccupied,
    the decision should set every room to OFF with zero cost.
    """
    zone = _residential_zone(has_furnace=True)
    rooms = ["bedroom1", "living_room"]
    ctrl = _build_controller(zone, rooms, heater_capacity=1500.0)

    room_temps = {"bedroom1": 23.0, "living_room": 23.0}
    result = ctrl.decide(
        room_temperatures=room_temps,
        occupancy_forecasts={r: [0.0] * 4 for r in rooms},
        electricity_price=0.20,
        gas_price=1.00,
    )

    assert result.furnace_on is False
    assert result.rooms_needing_heat == []
    for action in result.per_room_actions.values():
        assert action <= 0.05
    assert result.chosen_hourly_cost == 0.0


# ---------------------------------------------------------------------------
# Test 5: expensive gas → electric preferred even with many rooms
# ---------------------------------------------------------------------------

def test_electric_preferred_when_gas_is_expensive():
    """
    With very expensive gas and cheap electricity, electric should still win
    even when multiple rooms need heat.
    """
    zone = _residential_zone(has_furnace=True)
    rooms = ["bedroom1", "bedroom2", "bedroom3"]
    ctrl = _build_controller(zone, rooms, heater_capacity=1000.0)

    room_temps = {r: 15.0 for r in rooms}
    # Electric: 3 × 1.0 kW × 0.7 × 0.10 = $0.21/hr
    # Furnace: (0.6/0.80) × 5.00 $/therm = $3.75/hr → very expensive
    result = ctrl.decide(
        room_temperatures=room_temps,
        occupancy_forecasts={r: [1.0] * 4 for r in rooms},
        electricity_price=0.10,
        gas_price=5.00,
    )

    assert result.furnace_on is False
    assert result.heat_source == HeatSourceType.ELECTRIC
    assert result.electric_hourly_cost < result.furnace_hourly_cost


# ---------------------------------------------------------------------------
# Test 6: furnace action level is the max required across rooms
# ---------------------------------------------------------------------------

def test_furnace_action_level_is_max_across_needing_rooms():
    """
    When the furnace is selected, all rooms must be set to at least the
    highest action level demanded by any single room that needs heat.
    """
    zone = _residential_zone(has_furnace=True)
    rooms = ["bedroom1", "bedroom2"]
    ctrl = _build_controller(zone, rooms, heater_capacity=1500.0)

    # Both rooms cold, occupied, furnace expected to win with expensive electricity
    room_temps = {"bedroom1": 15.0, "bedroom2": 15.0}
    result = ctrl.decide(
        room_temperatures=room_temps,
        occupancy_forecasts={r: [1.0] * 4 for r in rooms},
        electricity_price=0.40,   # expensive electricity → furnace wins
        gas_price=0.80,
    )

    if result.furnace_on:
        actions = list(result.per_room_actions.values())
        max_level = max(actions)
        # All rooms should share the same zone-wide action level
        for action in actions:
            assert action == pytest.approx(max_level)


# ---------------------------------------------------------------------------
# Test 7: to_dict serialization
# ---------------------------------------------------------------------------

def test_hybrid_decision_to_dict_structure():
    """HybridHeatingDecision.to_dict() should return a well-formed dict."""
    zone = _residential_zone(has_furnace=True)
    rooms = ["bedroom1"]
    ctrl = _build_controller(zone, rooms)

    room_temps = {"bedroom1": 15.0}
    result = ctrl.decide(
        room_temperatures=room_temps,
        occupancy_forecasts={"bedroom1": [1.0] * 4},
        electricity_price=0.15,
        gas_price=1.50,
    )
    d = result.to_dict()

    assert "zone" in d
    assert "heat_source" in d
    assert "furnace_on" in d
    assert "per_room_actions" in d
    assert "per_room_action_labels" in d
    assert "rooms_needing_heat" in d
    assert "electric_hourly_cost" in d
    assert "furnace_hourly_cost" in d
    assert "chosen_hourly_cost" in d
    assert "rationale" in d
    assert d["heat_source"] in ("electric", "gas_furnace")
    assert all(0.0 <= power <= 1.0 for power in d["per_room_actions"].values())
