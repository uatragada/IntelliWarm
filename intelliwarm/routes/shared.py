"""
Shared helpers for thin Flask route modules.
"""

from __future__ import annotations

from typing import Mapping

from flask import current_app

from intelliwarm.services.runtime import IntelliWarmRuntime


def current_bootstrap():
    """Return the active runtime bootstrap stored on the Flask app."""
    return current_app.extensions["intelliwarm_bootstrap"]


def _default_display_temp_f(temp_c: float) -> float:
    return round((temp_c * 9 / 5) + 32, 1)


def _heating_source_label(heat_source: str) -> str:
    return {
        "electric": "Electric Heater",
        "gas_furnace": "Gas Furnace",
    }.get(heat_source, heat_source.replace("_", " ").title())


def add_room_from_form(runtime: IntelliWarmRuntime, form: Mapping[str, str]) -> bool:
    """Parse add-room form data and create the room through runtime services."""
    name = (form.get("roomName") or "").strip()
    if not name:
        return False

    zone = (form.get("zone") or "Unassigned").strip()
    room_size_raw = (form.get("roomSize") or "").strip()
    target_temp_raw = (form.get("targetTemp") or "").strip()
    heat_source = (form.get("heatSource") or "electric").strip() or "electric"
    overrides = {
        "heat_source": heat_source,
        "heating_source": _heating_source_label(heat_source),
    }
    if target_temp_raw:
        overrides["target_temp"] = float(target_temp_raw)
    room_config = runtime.config.build_room_config(zone=zone, overrides=overrides)
    room_size = float(room_size_raw) if room_size_raw else float(room_config.get("room_size", 150.0))
    initial_sensor_temp = float(room_config.get("target_temp", runtime.config.default_target_temp))

    runtime.add_room(
        name=name,
        room_size=room_size,
        zone=zone,
        room_config=room_config,
        initial_sensor_temp=initial_sensor_temp,
        initial_occupancy=bool(room_config.get("initial_occupancy", False)),
        display_temp_f=room_config.get("display_temp_f") or _default_display_temp_f(initial_sensor_temp),
        humidity=float(room_config.get("humidity", 45.0)),
        heating_source=str(room_config.get("heating_source", _heating_source_label(heat_source))),
    )
    return True


def apply_home_configuration(runtime: IntelliWarmRuntime, form: Mapping[str, str]):
    """Apply home configuration form changes through runtime services."""
    zone_name = (form.get("zoneName") or "").strip()
    zone_desc = (form.get("zoneDescription") or "").strip()
    if zone_name:
        runtime.add_zone(zone_name, zone_desc)

    elec_price = (form.get("electricityPrice") or "").strip()
    gas_price = (form.get("gasPrice") or "").strip()

    if elec_price:
        runtime.update_utility_rates(electricity_price=float(elec_price))

    if gas_price:
        runtime.update_utility_rates(gas_price=float(gas_price))
