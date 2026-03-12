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


# ---------------------------------------------------------------------------
# Dual heat source — PhysicsRoomThermalModel
# ---------------------------------------------------------------------------

from intelliwarm.data import HeatSourceType, ZoneConfig


class TestPhysicsRoomThermalModelDualHeatSources:
    """Electric and furnace inputs are physically independent; the hybrid
    controller guarantees mutual exclusion, but the model accepts both."""

    def _electric_model(self) -> PhysicsRoomThermalModel:
        return PhysicsRoomThermalModel(
            room_name="office",
            thermal_capacitance_kj_k=200.0,
            conductance_ua_w_k=10.0,
            hvac_power_w=1000.0,
            furnace_power_w=0.0,
        )

    def _furnace_model(self) -> PhysicsRoomThermalModel:
        return PhysicsRoomThermalModel(
            room_name="bedroom",
            thermal_capacitance_kj_k=400.0,
            conductance_ua_w_k=10.0,
            hvac_power_w=0.0,
            furnace_power_w=7000.0,
        )

    def _dual_model(self) -> PhysicsRoomThermalModel:
        return PhysicsRoomThermalModel(
            room_name="living_room",
            thermal_capacitance_kj_k=400.0,
            conductance_ua_w_k=10.0,
            hvac_power_w=1500.0,
            furnace_power_w=7000.0,
        )

    # --- electric path ---

    def test_electric_only_heats_via_heating_power(self):
        model = self._electric_model()
        T_start = 15.0
        T_out = 5.0
        T1 = model.step(T_start, T_out, heating_power=1.0, furnace_heating_power=0.0)
        assert T1 > T_start

    def test_electric_only_ignores_furnace_fraction(self):
        """furnace_power_w=0 means furnace_heating_power has no effect."""
        model = self._electric_model()
        T_start = 15.0
        T_out = 5.0
        T_no_furnace = model.step(T_start, T_out, heating_power=0.5, furnace_heating_power=0.0)
        T_furnace_on = model.step(T_start, T_out, heating_power=0.5, furnace_heating_power=1.0)
        assert T_no_furnace == pytest.approx(T_furnace_on), (
            "furnace_power_w=0 should make furnace_heating_power a no-op"
        )

    # --- furnace path ---

    def test_furnace_heats_room_via_furnace_heating_power(self):
        model = self._furnace_model()
        T_start = 15.0
        T_out = 5.0
        T1 = model.step(T_start, T_out, heating_power=0.0, furnace_heating_power=1.0)
        assert T1 > T_start

    def test_furnace_only_ignores_electric_fraction(self):
        """electric_power_w=0 means heating_power has no effect."""
        model = self._furnace_model()
        T_start = 15.0
        T_out = 5.0
        T_no_elec = model.step(T_start, T_out, heating_power=0.0, furnace_heating_power=0.5)
        T_elec_on = model.step(T_start, T_out, heating_power=1.0, furnace_heating_power=0.5)
        assert T_no_elec == pytest.approx(T_elec_on), (
            "electric_power_w=0 should make heating_power a no-op"
        )

    def test_furnace_heats_more_than_electric_at_full_power(self):
        """Furnace (7 kW) should heat a room faster than electric (1.5 kW)."""
        common = dict(thermal_capacitance_kj_k=400.0, conductance_ua_w_k=10.0)
        elec = PhysicsRoomThermalModel(
            room_name="e", hvac_power_w=1500.0, furnace_power_w=0.0, **common)
        furn = PhysicsRoomThermalModel(
            room_name="f", hvac_power_w=0.0, furnace_power_w=7000.0, **common)
        T_start, T_out = 15.0, 5.0
        T_elec = elec.step(T_start, T_out, heating_power=1.0, furnace_heating_power=0.0)
        T_furn = furn.step(T_start, T_out, heating_power=0.0, furnace_heating_power=1.0)
        assert T_furn > T_elec, "Furnace should deliver more heat than electric"

    def test_combined_sources_add_linearly(self):
        """Both sources active should produce more heat than either alone."""
        model = self._dual_model()
        T_start, T_out = 15.0, 5.0
        T_elec_only = model.step(T_start, T_out, heating_power=1.0, furnace_heating_power=0.0)
        T_furn_only = model.step(T_start, T_out, heating_power=0.0, furnace_heating_power=1.0)
        T_both = model.step(T_start, T_out, heating_power=1.0, furnace_heating_power=1.0)
        assert T_both > T_elec_only
        assert T_both > T_furn_only
        # Linearity: the individual gains should sum to the combined gain
        delta_elec = T_elec_only - T_start
        delta_furn = T_furn_only - T_start
        # (small discrepancy from nonlinear UA term OK, use generous tolerance)
        assert T_both == pytest.approx(T_start + delta_elec + delta_furn, rel=0.05)

    def test_furnace_steady_state(self):
        """Furnace steady-state ΔT = furnace_power_w / UA."""
        furnace_w, ua = 7000.0, 10.0
        model = PhysicsRoomThermalModel(
            room_name="r",
            thermal_capacitance_kj_k=400.0,
            conductance_ua_w_k=ua,
            hvac_power_w=0.0,
            furnace_power_w=furnace_w,
        )
        T_out = 0.0
        T = T_out
        for _ in range(500):
            T = model.step(T, T_out, heating_power=0.0, furnace_heating_power=1.0,
                           dt_minutes=60)
        expected = furnace_w / ua
        assert abs(T - T_out - expected) < 1.0, (
            f"Furnace steady-state ΔT={T - T_out:.1f}°C, expected {expected:.1f}°C"
        )

    # --- properties ---

    def test_steady_state_delta_t_electric_only(self):
        model = self._electric_model()
        assert model.steady_state_delta_t == pytest.approx(100.0)  # 1000/10

    def test_steady_state_delta_t_furnace_only(self):
        model = self._furnace_model()
        assert model.steady_state_delta_t == pytest.approx(700.0)  # 7000/10

    def test_steady_state_delta_t_combined(self):
        model = self._dual_model()
        assert model.steady_state_delta_t == pytest.approx(850.0)  # (1500+7000)/10

    def test_hvac_power_w_property_returns_electric(self):
        """hvac_power_w backward-compat property must return electric_power_w."""
        model = self._dual_model()
        assert model.hvac_power_w == pytest.approx(model.electric_power_w)

    # --- simulate() ---

    def test_simulate_reads_furnace_from_forecast_inputs(self):
        model = self._furnace_model()
        T_start = 15.0
        inputs = [
            {"outdoor_temp": 5.0, "heating_power": 0.0, "furnace_heating_power": 1.0},
            {"outdoor_temp": 5.0, "heating_power": 0.0, "furnace_heating_power": 1.0},
        ]
        temps = model.simulate(T_start, inputs)
        assert len(temps) == 2
        assert temps[0] > T_start
        assert temps[1] > temps[0]  # continued heating


class TestFromRoomConfigWithZoneConfig:
    """from_room_config() correctly derives furnace_power_w from ZoneConfig."""

    def _residential_zone(self) -> ZoneConfig:
        return ZoneConfig(
            zone_id="Residential",
            has_furnace=True,
            furnace_btu_per_hour=60_000.0,
            furnace_efficiency=0.80,
        )

    def _work_zone(self) -> ZoneConfig:
        return ZoneConfig(
            zone_id="Work",
            has_furnace=False,
        )

    def test_furnace_zone_populates_furnace_power_w(self):
        rc = RoomConfig.from_legacy_config(
            "bedroom1",
            {"zone": "Residential", "target_temp": 21,
             "heater_power": 1500, "thermal_mass": 0.07,
             "heating_efficiency": 0.85},
        )
        model = PhysicsRoomThermalModel.from_room_config(
            rc, zone_config=self._residential_zone(), num_zone_rooms=2
        )
        # 60000 BTU/hr × 0.29307 W/(BTU/hr) × 0.80 / 2 rooms ≈ 7033 W
        expected = 60_000 * 0.29307 * 0.80 / 2
        assert model.furnace_power_w == pytest.approx(expected, rel=1e-3)
        assert model.electric_power_w == pytest.approx(1500.0)

    def test_no_furnace_zone_keeps_furnace_power_zero(self):
        rc = RoomConfig.from_legacy_config(
            "office",
            {"zone": "Work", "target_temp": 21,
             "heater_power": 1000, "thermal_mass": 0.05,
             "heating_efficiency": 0.85},
        )
        model = PhysicsRoomThermalModel.from_room_config(
            rc, zone_config=self._work_zone()
        )
        assert model.furnace_power_w == pytest.approx(0.0)

    def test_no_zone_config_keeps_furnace_power_zero(self):
        rc = RoomConfig.from_legacy_config(
            "bedroom1",
            {"zone": "Residential", "target_temp": 21,
             "heater_power": 1500, "thermal_mass": 0.07,
             "heating_efficiency": 0.85},
        )
        model = PhysicsRoomThermalModel.from_room_config(rc)
        assert model.furnace_power_w == pytest.approx(0.0)

    def test_furnace_room_heats_faster_than_electric(self):
        """A furnace-equipped room should warm more than an electric-only room per step."""
        rc = RoomConfig.from_legacy_config(
            "bedroom1",
            {"zone": "Residential", "target_temp": 21,
             "heater_power": 1500, "thermal_mass": 0.07,
             "heating_efficiency": 0.85},
        )
        elec_model = PhysicsRoomThermalModel.from_room_config(rc)
        furn_model = PhysicsRoomThermalModel.from_room_config(
            rc, zone_config=self._residential_zone(), num_zone_rooms=2
        )
        T_start, T_out = 15.0, 5.0
        T_elec = elec_model.step(T_start, T_out, heating_power=1.0, furnace_heating_power=0.0)
        T_furn = furn_model.step(T_start, T_out, heating_power=0.0, furnace_heating_power=1.0)
        assert T_furn > T_elec


class TestHouseSimulatorHeatSourceRouting:
    """HouseSimulator must dispatch the action to the right heat input."""

    def _make_simulator_with_sources(self) -> tuple:
        rc_elec = RoomConfig.from_legacy_config(
            "office",
            {"zone": "Work", "target_temp": 21, "heater_power": 1000,
             "thermal_mass": 0.05, "heating_efficiency": 0.85},
        )
        rc_furn = RoomConfig.from_legacy_config(
            "bedroom",
            {"zone": "Residential", "target_temp": 21, "heater_power": 1500,
             "thermal_mass": 0.07, "heating_efficiency": 0.85},
        )
        zone = ZoneConfig(zone_id="Residential", has_furnace=True,
                          furnace_btu_per_hour=60_000.0, furnace_efficiency=0.80)

        model_elec = PhysicsRoomThermalModel.from_room_config(rc_elec)
        model_furn = PhysicsRoomThermalModel.from_room_config(
            rc_furn, zone_config=zone, num_zone_rooms=1
        )

        sim = HouseSimulator(
            room_configs={"office": rc_elec, "bedroom": rc_furn},
            thermal_models={"office": model_elec, "bedroom": model_furn},
        )
        return sim, rc_elec, rc_furn

    def _state(self, heat_sources: dict) -> SimulationState:
        return SimulationState(
            timestamp=datetime(2026, 3, 12, 2, 0),  # night – no solar
            outdoor_temp=5.0,
            room_temperatures={"office": 15.0, "bedroom": 15.0},
            heating_actions={"office": HeatingAction.OFF, "bedroom": HeatingAction.OFF},
            occupancy={"office": 0.0, "bedroom": 0.0},
            heat_sources=heat_sources,
        )

    def test_electric_source_routed_to_electric_input(self):
        sim, _, _ = self._make_simulator_with_sources()
        state = self._state({"office": HeatSourceType.ELECTRIC})
        next_state = sim.step(state, {"office": HeatingAction.COMFORT})
        assert next_state.room_temperatures["office"] > 15.0

    def test_furnace_source_routes_to_furnace_input(self):
        sim, _, _ = self._make_simulator_with_sources()
        state = self._state({"bedroom": HeatSourceType.GAS_FURNACE})
        next_state = sim.step(state, {"bedroom": HeatingAction.COMFORT})
        assert next_state.room_temperatures["bedroom"] > 15.0

    def test_furnace_heats_more_than_electric_same_action(self):
        """COMFORT via furnace (7 kW) should heat more than COMFORT via electric (1 kW)."""
        sim, _, _ = self._make_simulator_with_sources()
        elec_state = self._state({"office": HeatSourceType.ELECTRIC,
                                  "bedroom": HeatSourceType.ELECTRIC})
        furn_state = self._state({"office": HeatSourceType.ELECTRIC,
                                  "bedroom": HeatSourceType.GAS_FURNACE})
        elec_next = sim.step(elec_state, {"office": HeatingAction.OFF,
                                          "bedroom": HeatingAction.COMFORT})
        furn_next = sim.step(furn_state, {"office": HeatingAction.OFF,
                                          "bedroom": HeatingAction.COMFORT})
        assert furn_next.room_temperatures["bedroom"] > elec_next.room_temperatures["bedroom"]

    def test_default_heat_source_is_electric(self):
        """Rooms not in heat_sources dict should default to electric routing."""
        sim, _, _ = self._make_simulator_with_sources()
        state = self._state({})  # empty heat_sources
        next_state = sim.step(state, {"office": HeatingAction.COMFORT})
        assert next_state.room_temperatures["office"] > 15.0


# ---------------------------------------------------------------------------
# sol_rad_tilt_wm2 — Duffie-Beckman tilted surface model
# ---------------------------------------------------------------------------

from intelliwarm.models import sol_rad_tilt_wm2


class TestSolRadTiltWm2:
    """Tilted surface irradiance follows Duffie-Beckman incidence angle physics."""

    # --- basic sanity ---

    def test_zero_at_night(self):
        midnight = datetime(2026, 1, 15, 0, 0)
        assert sol_rad_tilt_wm2(midnight, latitude_deg=40.0) == pytest.approx(0.0, abs=1.0)

    def test_positive_at_noon_summer(self):
        noon = datetime(2026, 6, 21, 12, 0)
        gti = sol_rad_tilt_wm2(noon, latitude_deg=40.0, surface_tilt_deg=90.0,
                                surface_azimuth_deg=0.0)
        assert gti > 100.0, "South-facing vertical window should receive solar at summer noon"

    def test_horizontal_surface_equals_ghi(self):
        """For a horizontal surface (tilt=0), GTI should equal GHI (albedo=0 eliminates reflected)."""
        noon = datetime(2026, 6, 21, 12, 0)
        ghi = solar_irradiance_wm2(noon, latitude_deg=40.0, cloud_cover=0.0)
        # Horizontal surface with zero albedo: no reflected component, isotropic diffuse=(1+1)/2=1
        # Direct = DNI*sin_elev, Diffuse = DHI*1, Total = GHI
        gti_horizontal = sol_rad_tilt_wm2(
            noon, latitude_deg=40.0, surface_tilt_deg=0.0,
            surface_azimuth_deg=0.0, cloud_cover=0.0, albedo=0.0
        )
        assert gti_horizontal == pytest.approx(ghi, rel=0.01), (
            "Horizontal surface irradiance should equal GHI (within 1 %)"
        )

    def test_returns_non_negative(self):
        """GTI must never be negative."""
        for month in [1, 6]:
            for hour in [0, 6, 12, 18]:
                ts = datetime(2026, month, 15, hour, 0)
                gti = sol_rad_tilt_wm2(ts, latitude_deg=40.0, surface_tilt_deg=90.0)
                assert gti >= 0.0

    # --- directional physics ---

    def test_south_facing_better_than_north_in_winter_noon(self):
        """At winter noon (sun in south), south-facing wall should get more than north-facing."""
        noon = datetime(2026, 1, 15, 12, 0)
        south = sol_rad_tilt_wm2(noon, latitude_deg=40.0, surface_tilt_deg=90.0,
                                  surface_azimuth_deg=0.0)
        north = sol_rad_tilt_wm2(noon, latitude_deg=40.0, surface_tilt_deg=90.0,
                                  surface_azimuth_deg=180.0)
        assert south > north, "South wall should receive more direct solar than north in winter"

    def test_east_facing_peaks_morning(self):
        """East-facing wall (azimuth=−90°) should receive more solar at 9:00 than 15:00."""
        ts_am = datetime(2026, 6, 21, 9, 0)
        ts_pm = datetime(2026, 6, 21, 15, 0)
        am = sol_rad_tilt_wm2(ts_am, latitude_deg=40.0, surface_tilt_deg=90.0,
                               surface_azimuth_deg=-90.0)
        pm = sol_rad_tilt_wm2(ts_pm, latitude_deg=40.0, surface_tilt_deg=90.0,
                               surface_azimuth_deg=-90.0)
        assert am > pm, "East-facing wall should get more solar in AM than PM"

    def test_west_facing_peaks_afternoon(self):
        """West-facing wall (azimuth=+90°) should receive more solar at 15:00 than 9:00."""
        ts_am = datetime(2026, 6, 21, 9, 0)
        ts_pm = datetime(2026, 6, 21, 15, 0)
        am = sol_rad_tilt_wm2(ts_am, latitude_deg=40.0, surface_tilt_deg=90.0,
                               surface_azimuth_deg=90.0)
        pm = sol_rad_tilt_wm2(ts_pm, latitude_deg=40.0, surface_tilt_deg=90.0,
                               surface_azimuth_deg=90.0)
        assert pm > am, "West-facing wall should get more solar in PM than AM"

    def test_vertical_south_winter_better_than_summer(self):
        """At 40°N south-facing vertical, winter direct beam > summer (sun lower = smaller incidence angle)."""
        winter_noon = datetime(2026, 1, 15, 12, 0)
        summer_noon = datetime(2026, 6, 21, 12, 0)
        winter = sol_rad_tilt_wm2(winter_noon, latitude_deg=40.0,
                                   surface_tilt_deg=90.0, surface_azimuth_deg=0.0)
        summer = sol_rad_tilt_wm2(summer_noon, latitude_deg=40.0,
                                   surface_tilt_deg=90.0, surface_azimuth_deg=0.0)
        # Winter has lower sun → smaller incidence angle on south-facing vertical → more direct
        assert winter > summer, (
            "South-facing vertical window should receive more direct beam in winter than summer "
            "(lower solar altitude in winter means sun hits more perpendicularly)"
        )

    # --- cloud cover ---

    def test_overcast_reduces_irradiance(self):
        noon = datetime(2026, 6, 21, 12, 0)
        clear = sol_rad_tilt_wm2(noon, latitude_deg=40.0, cloud_cover=0.0)
        overcast = sol_rad_tilt_wm2(noon, latitude_deg=40.0, cloud_cover=1.0)
        assert clear > overcast

    # --- albedo contribution ---

    def test_high_albedo_increases_reflected_on_tilted_surface(self):
        """Higher albedo should increase irradiance on a tilted surface (reflected term)."""
        noon = datetime(2026, 6, 21, 12, 0)
        low_albedo = sol_rad_tilt_wm2(noon, latitude_deg=40.0, surface_tilt_deg=60.0,
                                       surface_azimuth_deg=0.0, albedo=0.05)
        high_albedo = sol_rad_tilt_wm2(noon, latitude_deg=40.0, surface_tilt_deg=60.0,
                                        surface_azimuth_deg=0.0, albedo=0.60)
        assert high_albedo > low_albedo

    def test_horizontal_surface_has_no_reflected_contribution(self):
        """For horizontal surface, ground-reflected term = 0 (no visible ground from overhead)."""
        noon = datetime(2026, 6, 21, 12, 0)
        low = sol_rad_tilt_wm2(noon, latitude_deg=40.0, surface_tilt_deg=0.0, albedo=0.05)
        high = sol_rad_tilt_wm2(noon, latitude_deg=40.0, surface_tilt_deg=0.0, albedo=0.90)
        assert low == pytest.approx(high, rel=0.01), (
            "Horizontal surface has no reflected component regardless of albedo"
        )


# ---------------------------------------------------------------------------
# Infiltration — PhysicsRoomThermalModel
# ---------------------------------------------------------------------------


class TestInfiltration:
    """Infiltration conductance adds to total UA and accelerates heat exchange."""

    def _model(self, infiltration_ua: float = 0.0) -> PhysicsRoomThermalModel:
        return PhysicsRoomThermalModel(
            room_name="test",
            thermal_capacitance_kj_k=200.0,
            conductance_ua_w_k=10.0,
            hvac_power_w=0.0,
            infiltration_ua_w_k=infiltration_ua,
        )

    def test_effective_ua_is_sum_of_envelope_and_infiltration(self):
        model = self._model(infiltration_ua=5.0)
        assert model.effective_ua_w_k == pytest.approx(15.0)

    def test_zero_infiltration_effective_ua_equals_conductance(self):
        model = self._model(infiltration_ua=0.0)
        assert model.effective_ua_w_k == pytest.approx(10.0)

    def test_infiltration_increases_cooling_rate(self):
        """Room with infiltration should cool faster than without (same envelope UA)."""
        model_none = self._model(infiltration_ua=0.0)
        model_inf = self._model(infiltration_ua=20.0)  # heavy infiltration
        T_start, T_out = 25.0, 5.0
        T_none = model_none.step(T_start, T_out, heating_power=0.0)
        T_inf = model_inf.step(T_start, T_out, heating_power=0.0)
        assert T_inf < T_none, "More infiltration should cause faster cooling"

    def test_infiltration_steady_state_uses_effective_ua(self):
        """Steady-state ΔT should reflect total UA including infiltration."""
        hvac_w, envelope_ua, infiltration_ua = 1000.0, 10.0, 5.0
        model = PhysicsRoomThermalModel(
            room_name="r",
            thermal_capacitance_kj_k=400.0,
            conductance_ua_w_k=envelope_ua,
            hvac_power_w=hvac_w,
            infiltration_ua_w_k=infiltration_ua,
        )
        T_out = 0.0
        T = T_out
        for _ in range(500):
            T = model.step(T, T_out, heating_power=1.0)
        expected_delta = hvac_w / (envelope_ua + infiltration_ua)  # 1000/15 ≈ 66.7
        assert abs(T - T_out - expected_delta) < 1.0, (
            f"Steady-state ΔT={T - T_out:.1f}°C, expected ≈ {expected_delta:.1f}°C"
        )

    def test_steady_state_delta_t_property_uses_effective_ua(self):
        hvac_w, envelope_ua, inf_ua = 1000.0, 10.0, 5.0
        model = PhysicsRoomThermalModel(
            room_name="r",
            thermal_capacitance_kj_k=200.0,
            conductance_ua_w_k=envelope_ua,
            hvac_power_w=hvac_w,
            infiltration_ua_w_k=inf_ua,
        )
        assert model.steady_state_delta_t == pytest.approx(hvac_w / (envelope_ua + inf_ua))

    def test_time_constant_uses_effective_ua(self):
        """τ = C / UA_eff — should be smaller when infiltration is higher."""
        model_no_inf = self._model(infiltration_ua=0.0)
        model_with_inf = self._model(infiltration_ua=10.0)
        assert model_with_inf.time_constant_hours < model_no_inf.time_constant_hours

    def test_negative_infiltration_raises(self):
        with pytest.raises(ValueError, match="infiltration"):
            PhysicsRoomThermalModel(
                room_name="r",
                thermal_capacitance_kj_k=200.0,
                conductance_ua_w_k=10.0,
                hvac_power_w=0.0,
                infiltration_ua_w_k=-1.0,
            )

    def test_from_room_config_produces_positive_infiltration_ua(self):
        """from_room_config(infiltration_ach=0.5) should produce infiltration_ua_w_k > 0."""
        rc = RoomConfig.from_legacy_config(
            "bedroom1",
            {"zone": "Residential", "target_temp": 21,
             "heater_power": 1500, "thermal_mass": 0.07,
             "heating_efficiency": 0.85},
        )
        model = PhysicsRoomThermalModel.from_room_config(rc, infiltration_ach=0.5)
        assert model.infiltration_ua_w_k > 0.0

    def test_from_room_config_zero_infiltration_when_ach_zero(self):
        rc = RoomConfig.from_legacy_config(
            "bedroom1",
            {"zone": "Residential", "target_temp": 21,
             "heater_power": 1500, "thermal_mass": 0.07,
             "heating_efficiency": 0.85},
        )
        model = PhysicsRoomThermalModel.from_room_config(rc, infiltration_ach=0.0)
        assert model.infiltration_ua_w_k == pytest.approx(0.0, abs=1e-6)

    def test_from_room_config_surface_orientation_stored(self):
        """Solar tilt/azimuth passed to from_room_config should be stored on the model."""
        rc = RoomConfig.from_legacy_config(
            "bedroom1",
            {"zone": "Residential", "target_temp": 21,
             "heater_power": 1500, "thermal_mass": 0.07,
             "heating_efficiency": 0.85},
        )
        model = PhysicsRoomThermalModel.from_room_config(
            rc, solar_tilt_deg=45.0, solar_azimuth_deg=-90.0
        )
        assert model.solar_tilt_deg == pytest.approx(45.0)
        assert model.solar_azimuth_deg == pytest.approx(-90.0)


# ---------------------------------------------------------------------------
# HouseSimulator — per-room tilted solar
# ---------------------------------------------------------------------------


class TestSimulatorTiltedSolar:
    """Simulator routes tilted irradiance per room using each model's orientation."""

    def _state(self, timestamp: datetime) -> SimulationState:
        return SimulationState(
            timestamp=timestamp,
            outdoor_temp=5.0,
            room_temperatures={"south": 15.0, "north": 15.0},
            heating_actions={"south": HeatingAction.OFF, "north": HeatingAction.OFF},
            occupancy={"south": 0.0, "north": 0.0},
        )

    def _sim_with_orientations(self) -> HouseSimulator:
        rc = RoomConfig.from_legacy_config(
            "south",
            {"zone": "R", "target_temp": 21, "heater_power": 0,
             "thermal_mass": 0.07, "heating_efficiency": 0.85},
        )
        south_model = PhysicsRoomThermalModel(
            room_name="south",
            thermal_capacitance_kj_k=200.0,
            conductance_ua_w_k=5.0,
            hvac_power_w=0.0,
            solar_aperture_m2=2.0,
            solar_tilt_deg=90.0,
            solar_azimuth_deg=0.0,   # south-facing
        )
        north_model = PhysicsRoomThermalModel(
            room_name="north",
            thermal_capacitance_kj_k=200.0,
            conductance_ua_w_k=5.0,
            hvac_power_w=0.0,
            solar_aperture_m2=2.0,
            solar_tilt_deg=90.0,
            solar_azimuth_deg=180.0,  # north-facing
        )
        return HouseSimulator(
            room_configs={"south": rc, "north": rc},
            thermal_models={"south": south_model, "north": north_model},
            latitude_deg=40.0,
        )

    def test_south_room_warmer_than_north_at_winter_noon(self):
        """South-facing window gains more solar at winter noon than north-facing."""
        sim = self._sim_with_orientations()
        ts = datetime(2026, 1, 15, 12, 0)  # winter noon
        next_state = sim.step(self._state(ts), {"south": HeatingAction.OFF,
                                                "north": HeatingAction.OFF})
        assert next_state.room_temperatures["south"] >= next_state.room_temperatures["north"], (
            "South-facing room should be warmer (or equal) than north-facing at winter noon"
        )

    def test_simulator_albedo_parameter_accepted(self):
        """HouseSimulator should accept and store albedo parameter."""
        rc = RoomConfig.from_legacy_config(
            "r", {"zone": "X", "target_temp": 21, "heater_power": 0,
                  "thermal_mass": 0.07, "heating_efficiency": 0.85},
        )
        model = PhysicsRoomThermalModel(
            room_name="r",
            thermal_capacitance_kj_k=200.0,
            conductance_ua_w_k=5.0,
            hvac_power_w=0.0,
        )
        sim = HouseSimulator(
            room_configs={"r": rc},
            thermal_models={"r": model},
            albedo=0.60,
        )
        assert sim.albedo == pytest.approx(0.60)

    def test_legacy_model_falls_back_to_ghi(self):
        """A model without solar_tilt_deg/solar_azimuth_deg uses GHI fallback."""
        from intelliwarm.models.thermal_model import RoomThermalModel
        rc = RoomConfig.from_legacy_config(
            "r", {"zone": "X", "target_temp": 21, "heater_power": 0,
                  "thermal_mass": 0.07, "heating_efficiency": 0.85},
        )
        legacy_model = RoomThermalModel(room_name="r")
        sim = HouseSimulator(
            room_configs={"r": rc},
            thermal_models={"r": legacy_model},
        )
        ts = datetime(2026, 6, 21, 12, 0)
        state = SimulationState(
            timestamp=ts, outdoor_temp=15.0,
            room_temperatures={"r": 15.0},
            heating_actions={"r": HeatingAction.OFF},
            occupancy={"r": 0.0},
        )
        # Should not raise; GHI fallback works fine for legacy model
        next_state = sim.step(state, {"r": HeatingAction.OFF})
        assert "r" in next_state.room_temperatures

