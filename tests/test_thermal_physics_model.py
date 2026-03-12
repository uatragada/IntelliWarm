"""
Tests for PhysicsRoomThermalModel and solar_irradiance_wm2.

Covers:
- Passive cooling toward outside temperature
- Active heating with HVAC
- Solar gain contribution
- Occupant metabolic heat gain
- Steady-state temperature accuracy (Q_in = Q_loss)
- RC time-constant property
- from_room_config() factory heuristics
- HouseSimulator integration with PhysicsRoomThermalModel
- solar_irradiance_wm2: day/night, seasonal variation, cloud correction
"""

import math
import os
import sys
from datetime import datetime

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from intelliwarm.data import HeatingAction, RoomConfig, SimulationState
from intelliwarm.models import HouseSimulator, PhysicsRoomThermalModel, solar_irradiance_wm2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_model(
    c_kj_k: float = 200.0,
    ua_w_k: float = 10.0,
    hvac_w: float = 1500.0,
    solar_ap: float = 1.0,
    occ_gain: float = 80.0,
) -> PhysicsRoomThermalModel:
    return PhysicsRoomThermalModel(
        room_name="test_room",
        thermal_capacitance_kj_k=c_kj_k,
        conductance_ua_w_k=ua_w_k,
        hvac_power_w=hvac_w,
        solar_aperture_m2=solar_ap,
        occupant_gain_w=occ_gain,
    )


# ---------------------------------------------------------------------------
# PhysicsRoomThermalModel — basic physics
# ---------------------------------------------------------------------------

class TestPhysicsRoomThermalModelCooling:
    """A room with no heat input must drift toward outdoor temperature."""

    def test_room_cools_toward_outside(self):
        model = make_model()
        T_start = 22.0
        T_out = 5.0
        T1 = model.step(T_start, T_out, heating_power=0.0, dt_minutes=60)
        assert T1 < T_start, "Room should lose heat to the cold outdoors"
        assert T1 > T_out, "Room should not cool below outdoor temperature in one step"

    def test_room_asymptotes_to_outside_over_many_steps(self):
        model = make_model(c_kj_k=50.0, ua_w_k=20.0)
        T = 22.0
        T_out = 0.0
        for _ in range(200):
            T = model.step(T, T_out, heating_power=0.0, dt_minutes=60)
        assert abs(T - T_out) < 0.1, f"Room should converge to outside temp, got {T:.2f}"

    def test_colder_outside_causes_faster_cooling(self):
        model = make_model()
        T_start = 20.0
        drop_mild = T_start - model.step(T_start, 15.0, heating_power=0.0)
        drop_cold = T_start - model.step(T_start, -5.0, heating_power=0.0)
        assert drop_cold > drop_mild, "Larger ΔT should produce greater heat loss"


class TestPhysicsRoomThermalModelHeating:
    """HVAC power must raise room temperature."""

    def test_room_heats_when_hvac_on(self):
        model = make_model()
        T_start = 15.0
        T_out = 5.0
        T1 = model.step(T_start, T_out, heating_power=1.0, dt_minutes=60)
        assert T1 > T_start, "Full HVAC power should raise room temperature"

    def test_more_power_means_more_heating(self):
        model = make_model()
        T_start = 15.0
        T_out = 5.0
        T_half = model.step(T_start, T_out, heating_power=0.5)
        T_full = model.step(T_start, T_out, heating_power=1.0)
        assert T_full > T_half, "Full power should heat more than half power"

    def test_zero_power_does_not_heat(self):
        model = make_model()
        T_start = 20.0
        T_out = 20.0  # no ΔT, no solar, no occupants
        T1 = model.step(T_start, T_out, heating_power=0.0,
                        solar_irradiance_w_m2=0.0, occupancy=0.0)
        assert abs(T1 - T_start) < 1e-9, "Zero inputs should leave temperature unchanged"


class TestPhysicsRoomThermalModelSteadyState:
    """At steady state the temperature rise above outdoor equals Q_hvac / UA."""

    def test_hvac_steady_state_temperature(self):
        ua = 10.0
        hvac = 1000.0
        model = make_model(c_kj_k=100.0, ua_w_k=ua, hvac_w=hvac,
                           solar_ap=0.0, occ_gain=0.0)
        T_out = 0.0
        T = T_out
        for _ in range(500):
            T = model.step(T, T_out, heating_power=1.0, dt_minutes=60)
        expected_delta = hvac / ua  # °C above outdoor
        assert abs(T - T_out - expected_delta) < 0.5, (
            f"Steady-state ΔT={T - T_out:.1f}°C, expected {expected_delta:.1f}°C"
        )

    def test_steady_state_delta_t_property(self):
        model = make_model(ua_w_k=10.0, hvac_w=1500.0)
        assert model.steady_state_delta_t == pytest.approx(150.0)


class TestPhysicsRoomThermalModelTimeConstant:
    """RC time constant should match C / UA (in hours)."""

    def test_time_constant_property(self):
        model = make_model(c_kj_k=360.0, ua_w_k=10.0)
        # τ = C / UA = 360_000 J/K / (10 W/K × 3600 s/hr) = 10 hr
        assert model.time_constant_hours == pytest.approx(10.0, rel=1e-6)

    def test_exponential_cooling_matches_time_constant(self):
        """After τ hours of free cooling the temperature gap should be ~37 % of initial."""
        c_kj_k = 360.0
        ua_w_k = 10.0
        tau_hours = (c_kj_k * 1000.0) / (ua_w_k * 3600.0)
        model = make_model(c_kj_k=c_kj_k, ua_w_k=ua_w_k,
                           hvac_w=0.0, solar_ap=0.0, occ_gain=0.0)
        T_out = 0.0
        delta_start = 20.0
        T = T_out + delta_start

        steps = int(tau_hours)  # simulate τ hours at 1-hr steps
        for _ in range(steps):
            T = model.step(T, T_out, heating_power=0.0, dt_minutes=60)

        ratio = (T - T_out) / delta_start
        # After τ hours ratio ≈ e^(−1) ≈ 0.368; allow ±8 % for step-size error
        assert 0.29 < ratio < 0.45, f"Cooling ratio after τ={tau_hours:.1f} hr: {ratio:.3f}"


class TestPhysicsRoomThermalModelSolarGain:
    """Solar irradiance must raise room temperature via the solar aperture."""

    def test_solar_gain_warms_room(self):
        model = make_model(solar_ap=2.0, occ_gain=0.0)
        T_out = T_start = 15.0
        T_solar = model.step(T_start, T_out, heating_power=0.0,
                             solar_irradiance_w_m2=500.0, occupancy=0.0)
        T_no_solar = model.step(T_start, T_out, heating_power=0.0,
                                solar_irradiance_w_m2=0.0, occupancy=0.0)
        assert T_solar > T_no_solar, "Solar irradiance should increase room temperature"

    def test_zero_aperture_ignores_solar(self):
        model = make_model(solar_ap=0.0)
        T_out = T_start = 15.0
        T_solar = model.step(T_start, T_out, heating_power=0.0,
                             solar_irradiance_w_m2=800.0, occupancy=0.0)
        T_none = model.step(T_start, T_out, heating_power=0.0,
                            solar_irradiance_w_m2=0.0, occupancy=0.0)
        assert T_solar == pytest.approx(T_none), "Zero aperture should ignore solar gain"


class TestPhysicsRoomThermalModelOccupancy:
    """Occupant metabolic heat should warm the room."""

    def test_occupancy_warms_room(self):
        model = make_model(solar_ap=0.0, occ_gain=100.0)
        T_out = T_start = 15.0
        T_occ = model.step(T_start, T_out, heating_power=0.0,
                           solar_irradiance_w_m2=0.0, occupancy=1.0)
        T_empty = model.step(T_start, T_out, heating_power=0.0,
                             solar_irradiance_w_m2=0.0, occupancy=0.0)
        assert T_occ > T_empty, "Occupancy should raise room temperature"

    def test_fractional_occupancy(self):
        model = make_model(solar_ap=0.0, occ_gain=100.0)
        T_out = T_start = 15.0
        T_half = model.step(T_start, T_out, heating_power=0.0,
                            solar_irradiance_w_m2=0.0, occupancy=0.5)
        T_full = model.step(T_start, T_out, heating_power=0.0,
                            solar_irradiance_w_m2=0.0, occupancy=1.0)
        assert T_full > T_half > T_out, "Higher occupancy should produce more heat gain"


# ---------------------------------------------------------------------------
# from_room_config factory
# ---------------------------------------------------------------------------

class TestPhysicsRoomThermalModelFromRoomConfig:

    def test_factory_produces_model(self):
        rc = RoomConfig.from_legacy_config(
            "bedroom1",
            {
                "zone": "Residential",
                "target_temp": 21,
                "heater_power": 1500,
                "thermal_mass": 0.07,
                "heating_efficiency": 0.85,
                "occupancy_schedule": "9-18",
            },
        )
        model = PhysicsRoomThermalModel.from_room_config(rc)
        assert model.room_name == "bedroom1"
        assert model.thermal_capacitance_kj_k > 0
        assert model.conductance_ua_w_k > 0
        assert model.hvac_power_w == pytest.approx(1500.0)
        assert model.solar_aperture_m2 > 0

    def test_factory_heats_room(self):
        rc = RoomConfig.from_legacy_config(
            "office",
            {
                "zone": "Work",
                "target_temp": 21,
                "heater_power": 1000,
                "thermal_mass": 0.05,
                "heating_efficiency": 0.85,
            },
        )
        model = PhysicsRoomThermalModel.from_room_config(rc)
        T_start = 15.0
        T_out = 5.0
        T1 = model.step(T_start, T_out, heating_power=1.0, dt_minutes=60)
        assert T1 > T_start, "Factory-built model should heat the room"

    def test_factory_time_constant_in_realistic_range(self):
        """Residential room time constant should be roughly 3–20 hours."""
        rc = RoomConfig.from_legacy_config(
            "living_room",
            {
                "zone": "Residential",
                "target_temp": 20,
                "heater_power": 2000,
                "thermal_mass": 0.05,
                "heating_efficiency": 0.85,
            },
        )
        model = PhysicsRoomThermalModel.from_room_config(rc)
        assert 2.0 < model.time_constant_hours < 25.0, (
            f"Unexpected time constant: {model.time_constant_hours:.1f} hr"
        )

    def test_factory_invalid_capacity_raises(self):
        with pytest.raises((ValueError, ZeroDivisionError)):
            PhysicsRoomThermalModel(
                room_name="bad",
                thermal_capacitance_kj_k=0.0,
                conductance_ua_w_k=10.0,
                hvac_power_w=1000.0,
            )


# ---------------------------------------------------------------------------
# HouseSimulator integration
# ---------------------------------------------------------------------------

class TestHouseSimulatorWithPhysicsModel:

    def _make_simulator(self):
        rc = RoomConfig.from_legacy_config(
            "bedroom",
            {
                "zone": "Residential",
                "target_temp": 21,
                "heater_power": 1500,
                "thermal_mass": 0.07,
                "heating_efficiency": 0.85,
                "occupancy_schedule": "9-18",
            },
        )
        model = PhysicsRoomThermalModel.from_room_config(rc)
        return HouseSimulator(
            room_configs={"bedroom": rc},
            thermal_models={"bedroom": model},
        )

    def test_simulator_step_returns_new_state(self):
        sim = self._make_simulator()
        state = SimulationState(
            timestamp=datetime(2026, 3, 12, 8, 0),
            outdoor_temp=5.0,
            room_temperatures={"bedroom": 15.0},
            heating_actions={"bedroom": HeatingAction.OFF},
            occupancy={"bedroom": 0.5},
        )
        next_state = sim.step(state, {"bedroom": HeatingAction.COMFORT})
        assert "bedroom" in next_state.room_temperatures
        assert next_state.room_temperatures["bedroom"] > 15.0, (
            "COMFORT heating should warm the room"
        )

    def test_simulator_deterministic(self):
        sim = self._make_simulator()
        state = SimulationState(
            timestamp=datetime(2026, 6, 21, 12, 0),  # summer noon
            outdoor_temp=20.0,
            room_temperatures={"bedroom": 18.0},
            heating_actions={"bedroom": HeatingAction.OFF},
            occupancy={"bedroom": 0.0},
        )
        s1 = sim.step(state, {"bedroom": HeatingAction.ECO})
        s2 = sim.step(state, {"bedroom": HeatingAction.ECO})
        assert s1.room_temperatures["bedroom"] == s2.room_temperatures["bedroom"], (
            "Simulator must be deterministic"
        )

    def test_simulator_solar_raises_noon_temp(self):
        """Summer noon (high solar) should produce higher temp than winter midnight."""
        rc = RoomConfig.from_legacy_config(
            "room",
            {"zone": "Z", "target_temp": 21, "heater_power": 1500,
             "thermal_mass": 0.05, "heating_efficiency": 0.85},
        )
        model = PhysicsRoomThermalModel.from_room_config(rc)
        sim = HouseSimulator(
            room_configs={"room": rc},
            thermal_models={"room": model},
            latitude_deg=40.0,
        )
        base_state = SimulationState(
            timestamp=datetime(2026, 6, 21, 11, 0),  # 1 hr before noon
            outdoor_temp=20.0,
            room_temperatures={"room": 20.0},
            heating_actions={"room": HeatingAction.OFF},
            occupancy={"room": 0.0},
        )
        midnight_state = SimulationState(
            timestamp=datetime(2026, 1, 15, 0, 0),
            outdoor_temp=20.0,
            room_temperatures={"room": 20.0},
            heating_actions={"room": HeatingAction.OFF},
            occupancy={"room": 0.0},
        )
        noon_next = sim.step(base_state, {"room": HeatingAction.OFF})
        midnight_next = sim.step(midnight_state, {"room": HeatingAction.OFF})
        assert noon_next.room_temperatures["room"] > midnight_next.room_temperatures["room"], (
            "Summer midday should have higher temperature due to solar gain"
        )


# ---------------------------------------------------------------------------
# solar_irradiance_wm2 utility
# ---------------------------------------------------------------------------

class TestSolarIrradianceWm2:

    def test_zero_at_midnight(self):
        midnight = datetime(2026, 6, 21, 0, 0)
        assert solar_irradiance_wm2(midnight) == pytest.approx(0.0)

    def test_zero_before_sunrise(self):
        before_dawn = datetime(2026, 6, 21, 3, 0)
        assert solar_irradiance_wm2(before_dawn) == pytest.approx(0.0)

    def test_positive_at_solar_noon(self):
        noon = datetime(2026, 6, 21, 12, 0)
        ghi = solar_irradiance_wm2(noon, latitude_deg=40.0)
        assert ghi > 500.0, f"Summer noon should exceed 500 W/m², got {ghi:.1f}"

    def test_summer_exceeds_winter(self):
        summer_noon = datetime(2026, 7, 4, 12, 0)
        winter_noon = datetime(2026, 1, 15, 12, 0)
        lat = 40.0
        assert solar_irradiance_wm2(summer_noon, lat) > solar_irradiance_wm2(winter_noon, lat)

    def test_cloud_cover_reduces_irradiance(self):
        noon = datetime(2026, 6, 21, 12, 0)
        clear = solar_irradiance_wm2(noon, cloud_cover=0.0)
        cloudy = solar_irradiance_wm2(noon, cloud_cover=1.0)
        assert cloudy < clear, "Overcast sky should reduce irradiance"
        assert cloudy >= 0.0, "Irradiance must not be negative"

    def test_full_cloud_cover_greatly_reduces_irradiance(self):
        noon = datetime(2026, 6, 21, 12, 0)
        clear = solar_irradiance_wm2(noon, cloud_cover=0.0)
        full_overcast = solar_irradiance_wm2(noon, cloud_cover=1.0)
        assert full_overcast < clear * 0.5, (
            "Full overcast should reduce irradiance by more than 50 %"
        )

    def test_near_equator_higher_than_high_latitude(self):
        noon = datetime(2026, 3, 20, 12, 0)  # equinox
        equator = solar_irradiance_wm2(noon, latitude_deg=0.0)
        arctic = solar_irradiance_wm2(noon, latitude_deg=70.0)
        assert equator > arctic, "Near equator at equinox should have higher irradiance"

    def test_symmetry_around_noon(self):
        """10:00 and 14:00 should produce the same irradiance."""
        am = solar_irradiance_wm2(datetime(2026, 6, 21, 10, 0), latitude_deg=40.0)
        pm = solar_irradiance_wm2(datetime(2026, 6, 21, 14, 0), latitude_deg=40.0)
        assert am == pytest.approx(pm, rel=1e-6)
