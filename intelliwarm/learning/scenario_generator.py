"""
Deterministic scenario generation for IntelliWarm training environments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from intelliwarm.data import RoomConfig, ZoneConfig


@dataclass(frozen=True)
class TrainingScenario:
    """Deterministic training scenario spanning multiple rooms and zones."""

    name: str
    start_time: datetime
    horizon_steps: int
    step_minutes: int
    room_configs: Dict[str, RoomConfig]
    zone_configs: Dict[str, ZoneConfig]
    initial_temperatures: Dict[str, float]
    outdoor_temperatures: List[float]
    electricity_prices: List[float]
    gas_prices: List[float]
    description: str = ""


class SyntheticScenarioGenerator:
    """Build deterministic multi-room, multi-zone scenarios with varied schedules."""

    def build_scenario(
        self,
        name: str,
        start_time: datetime,
        room_configs: Dict[str, RoomConfig],
        zone_configs: Dict[str, ZoneConfig],
        initial_temperatures: Dict[str, float],
        outdoor_temperatures: List[float],
        electricity_prices: List[float],
        gas_prices: List[float],
        step_minutes: int = 60,
        description: str = "",
    ) -> TrainingScenario:
        horizon_steps = len(outdoor_temperatures)
        if len(electricity_prices) != horizon_steps or len(gas_prices) != horizon_steps:
            raise ValueError("Weather and price profiles must share the same horizon")

        return TrainingScenario(
            name=name,
            start_time=start_time,
            horizon_steps=horizon_steps,
            step_minutes=step_minutes,
            room_configs=room_configs,
            zone_configs=zone_configs,
            initial_temperatures=initial_temperatures,
            outdoor_temperatures=outdoor_temperatures,
            electricity_prices=electricity_prices,
            gas_prices=gas_prices,
            description=description,
        )

    def default_scenarios(self) -> List[TrainingScenario]:
        residential = ZoneConfig(
            zone_id="Residential",
            description="Bedrooms and family spaces",
            has_furnace=True,
            furnace_btu_per_hour=60000.0,
            furnace_efficiency=0.80,
        )
        work = ZoneConfig(
            zone_id="Work",
            description="Office and studio",
            has_furnace=False,
        )
        guest = ZoneConfig(
            zone_id="Guest",
            description="Occasionally used guest rooms",
            has_furnace=True,
            furnace_btu_per_hour=45000.0,
            furnace_efficiency=0.85,
        )

        return [
            self.build_scenario(
                name="winter-workday",
                start_time=datetime(2026, 1, 5, 6, 0),
                description="Cold weekday with a heated residential zone and an electric-only work zone.",
                room_configs={
                    "bedroom": RoomConfig.from_legacy_config(
                        "bedroom",
                        {
                            "zone": "Residential",
                            "target_temp": 21,
                            "heater_power": 1500,
                            "thermal_mass": 0.05,
                            "heating_efficiency": 1.4,
                            "occupancy_schedule": "6-8",
                        },
                    ),
                    "living_room": RoomConfig.from_legacy_config(
                        "living_room",
                        {
                            "zone": "Residential",
                            "target_temp": 21,
                            "heater_power": 1800,
                            "thermal_mass": 0.05,
                            "heating_efficiency": 1.6,
                            "occupancy_schedule": "17-22",
                        },
                    ),
                    "office": RoomConfig.from_legacy_config(
                        "office",
                        {
                            "zone": "Work",
                            "target_temp": 21,
                            "heater_power": 1200,
                            "thermal_mass": 0.06,
                            "heating_efficiency": 1.3,
                            "occupancy_schedule": "9-17",
                        },
                    ),
                },
                zone_configs={
                    "Residential": residential,
                    "Work": work,
                },
                initial_temperatures={
                    "bedroom": 18.5,
                    "living_room": 18.0,
                    "office": 17.5,
                },
                outdoor_temperatures=[-4.0, -4.0, -3.5, -3.0, -2.0, 0.0],
                electricity_prices=[0.20, 0.22, 0.26, 0.28, 0.18, 0.16],
                gas_prices=[1.15, 1.15, 1.15, 1.15, 1.15, 1.15],
            ),
            self.build_scenario(
                name="weekend-family",
                start_time=datetime(2026, 2, 7, 8, 0),
                description="Weekend occupancy spread across several residential rooms.",
                room_configs={
                    "main_bedroom": RoomConfig.from_legacy_config(
                        "main_bedroom",
                        {
                            "zone": "Residential",
                            "target_temp": 21,
                            "heater_power": 1500,
                            "thermal_mass": 0.05,
                            "heating_efficiency": 1.4,
                            "occupancy_schedule": "8-10",
                        },
                    ),
                    "kids_room": RoomConfig.from_legacy_config(
                        "kids_room",
                        {
                            "zone": "Residential",
                            "target_temp": 21,
                            "heater_power": 1400,
                            "thermal_mass": 0.05,
                            "heating_efficiency": 1.35,
                            "occupancy_schedule": "10-20",
                        },
                    ),
                    "guest_room": RoomConfig.from_legacy_config(
                        "guest_room",
                        {
                            "zone": "Guest",
                            "target_temp": 20,
                            "heater_power": 1200,
                            "thermal_mass": 0.05,
                            "heating_efficiency": 1.2,
                            "occupancy_schedule": "18-23",
                        },
                    ),
                },
                zone_configs={
                    "Residential": residential,
                    "Guest": guest,
                },
                initial_temperatures={
                    "main_bedroom": 19.0,
                    "kids_room": 18.0,
                    "guest_room": 17.0,
                },
                outdoor_temperatures=[2.0, 3.0, 4.0, 5.0, 4.0, 3.0],
                electricity_prices=[0.14, 0.14, 0.16, 0.16, 0.15, 0.14],
                gas_prices=[1.40, 1.40, 1.40, 1.40, 1.40, 1.40],
            ),
            self.build_scenario(
                name="mixed-use-peak",
                start_time=datetime(2026, 3, 11, 15, 0),
                description="Shoulder-season scenario with mixed commercial and residential schedules.",
                room_configs={
                    "studio": RoomConfig.from_legacy_config(
                        "studio",
                        {
                            "zone": "Work",
                            "target_temp": 20,
                            "heater_power": 1000,
                            "thermal_mass": 0.06,
                            "heating_efficiency": 1.1,
                            "occupancy_schedule": "15-22",
                        },
                    ),
                    "den": RoomConfig.from_legacy_config(
                        "den",
                        {
                            "zone": "Residential",
                            "target_temp": 21,
                            "heater_power": 1500,
                            "thermal_mass": 0.05,
                            "heating_efficiency": 1.45,
                            "occupancy_schedule": "18-23",
                        },
                    ),
                    "guest_office": RoomConfig.from_legacy_config(
                        "guest_office",
                        {
                            "zone": "Guest",
                            "target_temp": 20,
                            "heater_power": 1100,
                            "thermal_mass": 0.05,
                            "heating_efficiency": 1.15,
                            "occupancy_schedule": "16-19",
                        },
                    ),
                },
                zone_configs={
                    "Residential": residential,
                    "Work": work,
                    "Guest": guest,
                },
                initial_temperatures={
                    "studio": 19.5,
                    "den": 18.8,
                    "guest_office": 18.2,
                },
                outdoor_temperatures=[8.0, 7.5, 7.0, 6.5, 6.0, 5.5],
                electricity_prices=[0.18, 0.24, 0.28, 0.30, 0.24, 0.18],
                gas_prices=[1.25, 1.25, 1.25, 1.25, 1.25, 1.25],
            ),
        ]

    def get_scenario(self, name: str, scenarios: Optional[List[TrainingScenario]] = None) -> TrainingScenario:
        library = scenarios or self.default_scenarios()
        for scenario in library:
            if scenario.name == name:
                return scenario
        raise KeyError(f"Unknown scenario: {name}")

    def random_scenario(
        self,
        room_configs: Dict[str, RoomConfig],
        zone_configs: Dict[str, ZoneConfig],
        *,
        seed: Optional[int] = None,
        horizon_hours: int = 24,
        step_minutes: int = 60,
        gas_price_per_therm: float = 1.20,
    ) -> TrainingScenario:
        """Generate a randomised but physically realistic 24-hour scenario.

        Seasonal temperature baselines follow a Toronto-like climate.  Electricity
        prices follow a time-of-use pattern with random base rate and peak
        multiplier so the OPT policy must learn a genuine cost trade-off.

        Args:
            room_configs:         Per-room config to embed in the scenario.
            zone_configs:         Per-zone config to embed in the scenario.
            seed:                 Optional RNG seed for reproducibility.
            horizon_hours:        Episode length in steps (each step = step_minutes).
            step_minutes:         Duration of each timestep [minutes].
            gas_price_per_therm:  Gas price applied to every hour [$].

        Returns:
            A fresh :class:`TrainingScenario` instance.
        """
        rng = np.random.default_rng(seed)

        # ── Season & date ─────────────────────────────────────────────────────
        month = int(rng.integers(1, 13))
        day = int(rng.integers(1, 29))
        # Start at midnight or early morning for a full daily cycle
        start_hour = int(rng.choice([0, 3, 6]))
        start_time = datetime(2026, month, day, start_hour, 0)

        # Monthly mean outdoor temperature (°C) — Toronto-like climate
        _MEAN_T = {1: -8, 2: -6, 3: 0, 4: 8, 5: 15, 6: 20,
                   7: 23, 8: 22, 9: 17, 10: 10, 11: 3, 12: -4}
        base_t = float(_MEAN_T[month])

        # Daily cycle: trough ~6 AM, peak ~2 PM; amplitude varies
        daily_amplitude = float(rng.uniform(4.0, 10.0))
        outdoor_temps = []
        for h in range(horizon_hours):
            hour_of_day = (start_hour + h) % 24
            cycle = daily_amplitude * math.sin(math.pi * (hour_of_day - 6) / 14)
            noise = float(rng.normal(0, 0.6))
            outdoor_temps.append(round(base_t + cycle + noise, 1))

        # ── Electricity prices — time-of-use with random magnitude ────────────
        base_price = float(rng.uniform(0.07, 0.14))        # off-peak base
        peak_mult = float(rng.uniform(1.8, 3.5))           # peak multiplier
        elec_prices = []
        for h in range(horizon_hours):
            hour_of_day = (start_hour + h) % 24
            if 7 <= hour_of_day <= 9 or 17 <= hour_of_day <= 20:
                price = base_price * peak_mult * float(rng.uniform(0.90, 1.10))
            elif hour_of_day <= 5 or hour_of_day >= 23:
                price = base_price * float(rng.uniform(0.75, 0.95))
            else:
                price = base_price * float(rng.uniform(1.1, 1.7))
            elec_prices.append(round(price, 3))

        # ── Initial temperatures: each room slightly below its comfort minimum ─
        initial_temps = {
            room_id: round(float(rc.target_min_temp) - float(rng.uniform(1.0, 5.0)), 1)
            for room_id, rc in room_configs.items()
        }

        month_abbr = ["jan", "feb", "mar", "apr", "may", "jun",
                      "jul", "aug", "sep", "oct", "nov", "dec"][month - 1]
        scenario_id = int(rng.integers(1000, 9999)) if seed is None else seed
        name = f"random_{month_abbr}{day:02d}_{scenario_id}"

        return self.build_scenario(
            name=name,
            start_time=start_time,
            room_configs=room_configs,
            zone_configs=zone_configs,
            initial_temperatures=initial_temps,
            outdoor_temperatures=outdoor_temps,
            electricity_prices=elec_prices,
            gas_prices=[gas_price_per_therm] * horizon_hours,
            step_minutes=step_minutes,
            description=(
                f"Randomised {month_abbr} scenario: "
                f"base_T={base_t:.0f}°C, base_price=${base_price:.3f}/kWh, "
                f"peak_mult={peak_mult:.1f}x"
            ),
        )
