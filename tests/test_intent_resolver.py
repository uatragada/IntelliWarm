"""
Tests for shared intent-resolution logic used by runtime and learning paths.
"""

from intelliwarm.control import IntentCommandResolver, RoomHeatingIntent, ZoneSourceMode
from intelliwarm.control.intent_resolver import (
    normalize_room_intent,
    normalize_zone_source_mode,
    room_intent_feature_value,
    zone_source_mode_feature_value,
)
from intelliwarm.data import RoomConfig


def _room_config() -> RoomConfig:
    return RoomConfig.from_legacy_config(
        "office",
        {
            "zone": "Office",
            "target_temp": 21,
            "target_min_temp": 20,
            "target_max_temp": 22,
            "heater_power": 1500,
            "thermal_mass": 0.05,
            "heating_efficiency": 0.9,
            "occupancy_schedule": "9-18",
        },
    )


def test_infer_preheat_when_room_is_cold_and_occupancy_is_soon():
    resolver = IntentCommandResolver(
        room_config=_room_config(),
        min_temperature=18.0,
        max_temperature=24.0,
        preheat_lookahead_steps=3,
    )

    command = resolver.resolve(
        current_temp=18.5,
        occupancy_forecast=[0.0, 0.0, 1.0, 1.0],
        energy_prices=[0.12],
        current_action=0.0,
        outside_temp=0.0,
        target_temp=21.0,
    )

    assert command.intent == RoomHeatingIntent.PREHEAT
    assert command.action > 0.5


def test_recover_intent_resolves_to_stronger_command_than_maintain():
    resolver = IntentCommandResolver(
        room_config=_room_config(),
        min_temperature=18.0,
        max_temperature=24.0,
    )

    maintain = resolver.resolve(
        current_temp=18.7,
        occupancy_forecast=[1.0, 1.0, 1.0],
        energy_prices=[0.12],
        current_action=0.0,
        outside_temp=0.0,
        target_temp=21.0,
        room_intent=RoomHeatingIntent.MAINTAIN,
    )
    recover = resolver.resolve(
        current_temp=18.7,
        occupancy_forecast=[1.0, 1.0, 1.0],
        energy_prices=[0.12],
        current_action=0.0,
        outside_temp=0.0,
        target_temp=21.0,
        room_intent=RoomHeatingIntent.RECOVER,
    )

    assert recover.action > maintain.action


def test_intent_and_source_helpers_accept_legacy_and_scalar_encodings():
    assert normalize_room_intent("comfort") == RoomHeatingIntent.MAINTAIN
    assert normalize_room_intent(3) == RoomHeatingIntent.PREHEAT
    assert normalize_zone_source_mode("gas_furnace") == ZoneSourceMode.GAS_FURNACE
    assert normalize_zone_source_mode(1) == ZoneSourceMode.ELECTRIC
    assert 0.0 <= room_intent_feature_value(RoomHeatingIntent.RECOVER) <= 1.0
    assert 0.0 <= zone_source_mode_feature_value(ZoneSourceMode.AUTO) <= 1.0
