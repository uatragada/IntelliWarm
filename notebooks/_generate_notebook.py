"""Generate opt_heating_policy_training.ipynb."""
import json
import os

cells = []
_counter = [0]


def _id():
    _counter[0] += 1
    return f"cell{_counter[0]:04d}"


def md(src: str):
    cells.append({
        "cell_type": "markdown",
        "id": _id(),
        "metadata": {},
        "source": src,
    })


def code(src: str):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": _id(),
        "metadata": {},
        "outputs": [],
        "source": src,
    })


# ─────────────────────────────────────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────────────────────────────────────
md("""\
# 🏠 IntelliWarm — OPT Heating Policy Training
## Multi-Zone, Multi-Room Family House

This notebook trains an **OPT (optimal, most cost-effective) heating policy** for a
realistic family house with two gas-furnace zones and one electric-only zone using
**Proximal Policy Optimisation (PPO)** from Stable-Baselines3.

### What "OPT" means here
The trained policy minimises **total energy cost** (electricity + gas) while
**penalising rooms that are below their comfort temperature when occupancy starts**.
The agent must learn to:
- Preheat rooms *before* the family wakes up, arrives home, or returns from school.
- Choose the gas furnace when it is cheaper than running all room heaters independently.
- Use cheap overnight electricity for preheat when time-of-use pricing is favourable.

### House layout
| Room           | Zone        | Heat source | Occupancy |
|---------------|-------------|-------------|-----------|
| Living Room    | Main        | Gas furnace | Evenings + weekends |
| Kitchen        | Main        | Gas furnace | Mornings, lunch, dinner prep |
| Master Bedroom | Sleeping    | Gas furnace | Early mornings + nights |
| Kids Bedroom   | Sleeping    | Gas furnace | Nights + weekends |
| Home Office    | Office      | Electric    | Weekday 09:00–17:00 |

### Zones
| Zone     | Furnace     | Efficiency |
|---------|-------------|-----------|
| Main     | 80,000 BTU/hr | 80 % AFUE |
| Sleeping | 60,000 BTU/hr | 85 % AFUE |
| Office   | — (electric) | — |
""")

# ─────────────────────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────────────────────
md("## 📦 1. Setup & Dependencies")

code("""\
import subprocess, sys

def _pip(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

for _pkg, _import in [
    ("gymnasium",             "gymnasium"),
    ("stable-baselines3",     "stable_baselines3"),
    ("matplotlib",            "matplotlib"),
]:
    try:
        __import__(_import)
        print(f"✅ {_pkg} already installed")
    except ImportError:
        print(f"   Installing {_pkg} …")
        _pip(_pkg)
        print(f"✅ {_pkg} installed")

print("\\n🎉 Dependencies ready!")
""")

code("""\
import os, sys, warnings
sys.path.insert(0, os.path.abspath(".."))
warnings.filterwarnings("ignore")

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from collections import defaultdict

# ── IntelliWarm ───────────────────────────────────────────────────────────────
from intelliwarm.data import (
    RoomConfig, ZoneConfig, HeatSourceType, HeatingAction, OccupancyWindow,
    SimulationState,
)
from intelliwarm.control import BaselineController, HybridController
from intelliwarm.models import PhysicsRoomThermalModel, HouseSimulator
from intelliwarm.prediction import OccupancyPredictor
from intelliwarm.learning.scenario_generator import TrainingScenario, SyntheticScenarioGenerator
from intelliwarm.learning.gym_env import IntelliWarmMultiRoomEnv

# ── Gymnasium + SB3 ───────────────────────────────────────────────────────────
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

print("✅ All imports OK")
print(f"   gymnasium {gym.__version__}")
import stable_baselines3 as sb3
print(f"   stable-baselines3 {sb3.__version__}")
""")

# ─────────────────────────────────────────────────────────────────────────────
# FAMILY HOUSE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## 🏠 2. Family House Configuration

We define **three zones** and **five rooms** that represent a typical North-American
family home.  Occupancy schedules reflect a family where parents work-from-home
one weekday, kids attend school on weekdays, and everyone is home on weekends.
""")

code("""\
# ── Zones ─────────────────────────────────────────────────────────────────────
ZONES: Dict[str, ZoneConfig] = {
    "Main": ZoneConfig(
        zone_id="Main",
        description="Living areas — living room and kitchen",
        has_furnace=True,
        furnace_btu_per_hour=20_000.0,  # INPUT BTU/hr (zone-level; matches ~3.5 kW electric capacity)
        furnace_efficiency=0.80,
    ),
    "Sleeping": ZoneConfig(
        zone_id="Sleeping",
        description="Bedrooms — master and kids",
        has_furnace=True,
        furnace_btu_per_hour=15_000.0,  # INPUT BTU/hr (zone-level; matches ~2.5 kW electric capacity)
        furnace_efficiency=0.85,
    ),
    "Office": ZoneConfig(
        zone_id="Office",
        description="Home office — electric only",
        has_furnace=False,
    ),
}

for name, z in ZONES.items():
    src = f"{z.furnace_btu_per_hour/1000:.0f}k BTU/hr @ {z.furnace_efficiency*100:.0f}% AFUE" if z.has_furnace else "electric only"
    print(f"  Zone '{name}': {src}")
""")

code("""\
_WD = list(range(5))   # weekdays  (Mon=0 … Fri=4)
_WE = [5, 6]           # weekends  (Sat, Sun)

# ── Rooms ─────────────────────────────────────────────────────────────────────
ROOMS: Dict[str, RoomConfig] = {

    # ── Main zone ─────────────────────────────────────────────────────────────
    "living_room": RoomConfig(
        room_id="living_room", display_name="Living Room", zone="Main",
        target_min_temp=19.5, target_max_temp=22.0,
        heater_capacity=2000.0, heat_loss_factor=0.003, heating_efficiency=0.80,
        occupancy_schedule=(
            [OccupancyWindow(d, 17, 22, 0.90) for d in _WD] +   # weekday evenings
            [OccupancyWindow(d,  8, 22, 0.95) for d in _WE]     # all weekend
        ),
        heat_source=HeatSourceType.GAS_FURNACE,
    ),

    "kitchen": RoomConfig(
        room_id="kitchen", display_name="Kitchen", zone="Main",
        target_min_temp=18.5, target_max_temp=22.0,
        heater_capacity=1500.0, heat_loss_factor=0.005, heating_efficiency=0.80,
        occupancy_schedule=(
            [OccupancyWindow(d,  6,  8, 0.90) for d in _WD] +   # breakfast
            [OccupancyWindow(d, 17, 19, 0.85) for d in _WD] +   # dinner prep
            [OccupancyWindow(d,  7, 19, 0.75) for d in _WE]     # weekend all-day
        ),
        heat_source=HeatSourceType.GAS_FURNACE,
    ),

    # ── Sleeping zone ─────────────────────────────────────────────────────────
    "master_bedroom": RoomConfig(
        room_id="master_bedroom", display_name="Master Bedroom", zone="Sleeping",
        target_min_temp=18.0, target_max_temp=21.0,
        heater_capacity=1500.0, heat_loss_factor=0.004, heating_efficiency=0.85,
        occupancy_schedule=(
            [OccupancyWindow(d,  5,  8, 0.85) for d in _WD] +   # early weekday AM
            [OccupancyWindow(d, 21, 24, 0.95) for d in range(7)]  # every night
        ),
        heat_source=HeatSourceType.GAS_FURNACE,
    ),

    "kids_bedroom": RoomConfig(
        room_id="kids_bedroom", display_name="Kids Bedroom", zone="Sleeping",
        target_min_temp=19.0, target_max_temp=22.0,
        heater_capacity=1000.0, heat_loss_factor=0.005, heating_efficiency=0.85,
        occupancy_schedule=(
            [OccupancyWindow(d,  6,  8, 0.85) for d in _WD] +   # before school
            [OccupancyWindow(d, 19, 24, 0.95) for d in range(7)] +  # bedtime
            [OccupancyWindow(d,  8, 19, 0.70) for d in _WE]      # weekend days
        ),
        heat_source=HeatSourceType.GAS_FURNACE,
    ),

    # ── Office zone ───────────────────────────────────────────────────────────
    "home_office": RoomConfig(
        room_id="home_office", display_name="Home Office", zone="Office",
        target_min_temp=20.0, target_max_temp=23.0,
        heater_capacity=1200.0, heat_loss_factor=0.006, heating_efficiency=0.90,
        occupancy_schedule=[OccupancyWindow(d, 9, 17, 0.90) for d in _WD],
        heat_source=HeatSourceType.ELECTRIC,
    ),
}

print(f"Defined {len(ROOMS)} rooms across {len(ZONES)} zones:")
for room_id, rc in ROOMS.items():
    sched_count = len(rc.occupancy_schedule)
    print(f"  {rc.display_name:18s} → zone '{rc.zone}', "
          f"{rc.heater_capacity:.0f}W, {sched_count} occupancy windows")
""")

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING SCENARIOS
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## 🌡️ 3. Training Scenarios (24-hour horizon each)

Three 24-hour scenarios challenge the policy in different cost/comfort trade-off
situations:

| Scenario | Weather | Electricity | Gas-vs-electric trade-off |
|---------|---------|-------------|--------------------------|
| **winter_workday** | −14 °C to 2 °C | ToU peak 28 ¢/kWh | Furnace usually cheaper |
| **winter_weekend** | −10 °C to 4 °C | Flat ~14 ¢/kWh | Furnace vs electric depends on rooms needed |
| **spring_workday** | 3 °C to 15 °C | ToU 22 ¢/kWh peak | Electric sometimes competitive |

All rooms start cold (15–17 °C) so the policy must decide *when* and *how much*
to preheat before occupancy windows open.
""")

code("""\
gen = SyntheticScenarioGenerator()
STEP_MINUTES = 5
STEPS_PER_HOUR = 60 // STEP_MINUTES

def _expand_hourly_profile(values: Sequence[float]) -> List[float]:
    return [float(v) for v in values for _ in range(STEPS_PER_HOUR)]

# ── Winter Workday ────────────────────────────────────────────────────────────
# Monday Jan 5: kids go to school, parent WFH; ToU peak pricing 7-10am and 5-9pm
winter_workday = gen.build_scenario(
    name="winter_workday",
    start_time=datetime(2026, 1, 5, 0, 0),   # Monday
    room_configs=ROOMS,
    zone_configs=ZONES,
    initial_temperatures={r: 15.0 for r in ROOMS},
    outdoor_temperatures=_expand_hourly_profile([
        -12,-13,-13,-14,-14,-13,  # 00-05  pre-dawn coldest
         -11, -9, -7, -5, -3, -1,  # 06-11  warming
           0,  1,  2,  1,  0, -1,  # 12-17  mild afternoon
          -3, -5, -7, -9,-10,-11,  # 18-23  cooling fast
    ]),
    electricity_prices=_expand_hourly_profile([
        0.07, 0.07, 0.07, 0.07, 0.07, 0.07,  # 00-05  cheap overnight
        0.12, 0.22, 0.28, 0.26, 0.22, 0.16,  # 06-11  morning peak
        0.14, 0.14, 0.14, 0.14, 0.16, 0.20,  # 12-17  mid-day
        0.26, 0.28, 0.28, 0.24, 0.16, 0.10,  # 18-23  evening peak
    ]),
    gas_prices=[1.20] * (24 * STEPS_PER_HOUR),
    step_minutes=STEP_MINUTES,
    description="Cold January weekday: sharp morning/evening pricing peaks",
)

# ── Winter Weekend ────────────────────────────────────────────────────────────
# Saturday Jan 10: family home all day; flatter pricing; more rooms need heat
winter_weekend = gen.build_scenario(
    name="winter_weekend",
    start_time=datetime(2026, 1, 10, 0, 0),   # Saturday
    room_configs=ROOMS,
    zone_configs=ZONES,
    initial_temperatures={r: 16.0 for r in ROOMS},
    outdoor_temperatures=_expand_hourly_profile([
         -8, -9, -9,-10, -9, -8,   # 00-05
         -6, -4, -2,  0,  2,  3,   # 06-11
          4,  4,  3,  2,  1,  0,   # 12-17
         -1, -2, -3, -4, -5, -6,   # 18-23
    ]),
    electricity_prices=_expand_hourly_profile([
        0.08, 0.08, 0.08, 0.08, 0.08, 0.08,
        0.10, 0.12, 0.14, 0.14, 0.14, 0.14,
        0.14, 0.14, 0.14, 0.14, 0.14, 0.16,
        0.18, 0.18, 0.16, 0.14, 0.12, 0.10,
    ]),
    gas_prices=[1.20] * (24 * STEPS_PER_HOUR),
    step_minutes=STEP_MINUTES,
    description="Cold January Saturday: family home all day, flat electricity",
)

# ── Spring Workday ────────────────────────────────────────────────────────────
# Monday Mar 16: milder, heating less critical, electric sometimes competitive
spring_workday = gen.build_scenario(
    name="spring_workday",
    start_time=datetime(2026, 3, 16, 0, 0),   # Monday
    room_configs=ROOMS,
    zone_configs=ZONES,
    initial_temperatures={r: 17.5 for r in ROOMS},
    outdoor_temperatures=_expand_hourly_profile([
         4, 3, 3, 3, 4, 5,   # 00-05
         7, 9,11,12,13,14,   # 06-11
        15,15,14,13,11, 9,   # 12-17
         8, 7, 6, 6, 5, 5,   # 18-23
    ]),
    electricity_prices=_expand_hourly_profile([
        0.08, 0.08, 0.08, 0.08, 0.08, 0.08,
        0.10, 0.16, 0.20, 0.18, 0.14, 0.12,
        0.12, 0.12, 0.12, 0.12, 0.14, 0.18,
        0.22, 0.22, 0.18, 0.14, 0.10, 0.08,
    ]),
    gas_prices=[1.20] * (24 * STEPS_PER_HOUR),
    step_minutes=STEP_MINUTES,
    description="Mild March weekday: less heating needed, electric sometimes cheaper",
)

SCENARIOS = [winter_workday, winter_weekend, spring_workday]
print(
    f"Created {len(SCENARIOS)} training scenarios "
    f"(each {24 * STEPS_PER_HOUR} steps × {STEP_MINUTES} min = 24 h):"
)
for s in SCENARIOS:
    T_range = (min(s.outdoor_temperatures), max(s.outdoor_temperatures))
    E_range = (min(s.electricity_prices), max(s.electricity_prices))
    print(f"  {s.name:20s}  T_out=[{T_range[0]:+.0f}…{T_range[1]:+.0f}]°C  "
          f"elec=[{E_range[0]:.2f}…{E_range[1]:.2f}]$/kWh")
""")

code("""\
# ── Visualise scenario profiles ───────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 6), sharex=True)
hours = np.arange(SCENARIOS[0].horizon_steps) * (SCENARIOS[0].step_minutes / 60.0)

COLORS = ["#2196F3", "#FF5722", "#4CAF50"]

for col, scenario in enumerate(SCENARIOS):
    ax_temp  = axes[0, col]
    ax_price = axes[1, col]

    ax_temp.plot(hours, scenario.outdoor_temperatures, color=COLORS[col], lw=2)
    ax_temp.axhline(0, color="gray", lw=0.8, ls="--")
    ax_temp.fill_between(hours, scenario.outdoor_temperatures,
                         alpha=0.15, color=COLORS[col])
    ax_temp.set_title(scenario.name.replace("_", " ").title(), fontsize=11, fontweight="bold")
    ax_temp.set_ylabel("Outdoor temp (°C)" if col == 0 else "")
    ax_temp.set_ylim(-20, 20)
    ax_temp.grid(True, alpha=0.3)

    ax_price.plot(hours, [p * 100 for p in scenario.electricity_prices],
                  color=COLORS[col], lw=2, label="Electricity (¢/kWh)")
    ax_price.axhline(scenario.gas_prices[0] * 100 / 100 * 1.2, color="orange",
                     lw=1.5, ls=":", label="Gas equiv. ref")
    ax_price.fill_between(hours, [p * 100 for p in scenario.electricity_prices],
                          alpha=0.15, color=COLORS[col])
    ax_price.set_ylabel("Electricity (¢/kWh)" if col == 0 else "")
    ax_price.set_xlabel("Hour of day")
    ax_price.set_ylim(0, 35)
    ax_price.set_xticks(np.arange(0, 25, 4))
    ax_price.grid(True, alpha=0.3)
    if col == 2:
        ax_price.legend(fontsize=8, loc="upper right")

axes[0, 0].text(-0.25, 0.5, "Temperature", rotation=90, va="center",
                transform=axes[0, 0].transAxes, fontsize=10, color="gray")
axes[1, 0].text(-0.25, 0.5, "Price", rotation=90, va="center",
                transform=axes[1, 0].transAxes, fontsize=10, color="gray")

plt.suptitle("Training Scenario Profiles — 24-hour horizon at 5-minute resolution", fontsize=13, y=1.02)
plt.tight_layout()
plt.show()
""")

# ─────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## 🔧 4. Environment Setup

### PhysicsMultiRoomEnv
Subclass of `IntelliWarmMultiRoomEnv` that overrides `_build_simulator()` to use
`PhysicsRoomThermalModel` instead of the legacy first-order model.  The shared
multi-room observation includes each room's **full 24-hour future occupancy
forecast** (288 five-minute steps) in addition to current state, so PPO sees
the same schedule-style signal that the rule-based controllers use. This gives each room:
- Lumped thermal capacitance **C** [kJ/K] derived from heater design load.
- Envelope + **infiltration** conductance **UA** [W/K] (ASHRAE 62.2, 0.5 ACH default).
- Per-room **furnace power share** from `ZoneConfig` (BTU/hr × AFUE ÷ rooms).
- **Solar gain** on a south-facing vertical window (dm4bem incidence angle model).

### PreheatRewardWrapper
Adds an extra penalty on top of the base reward when a room is **approaching or in
an occupied period but is below the target minimum temperature**.

```
extra_penalty = preheat_boost × occupancy_prob × max(target_min − T_room, 0)
```
applied whenever `occupancy_prob > preheat_threshold` (default 0.35).

The wrapper also reports the added penalty back through `info`, so notebook
reward, comfort-violation, and cost diagnostics stay aligned.
""")

code("""\
class PhysicsMultiRoomEnv(IntelliWarmMultiRoomEnv):
    \"\"\"Multi-room environment backed by PhysicsRoomThermalModel (dm4bem physics).\"\"\"

    def __init__(
        self,
        *args,
        latitude_deg: float = 43.7,    # Toronto latitude (north-temperate)
        cloud_cover: float = 0.30,
        infiltration_ach: float = 0.50,  # ASHRAE 62.2 residential default
        albedo: float = 0.20,            # grass/asphalt ground
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._latitude_deg = latitude_deg
        self._cloud_cover = cloud_cover
        self._infiltration_ach = infiltration_ach
        self._albedo = albedo

    def _build_simulator(self, scenario: TrainingScenario) -> HouseSimulator:
        thermal_models = {}
        for room_name, room_config in scenario.room_configs.items():
            zone_id = room_config.zone
            zone_config = scenario.zone_configs.get(zone_id)
            num_zone_rooms = sum(
                1 for rc in scenario.room_configs.values() if rc.zone == zone_id
            )
            model = PhysicsRoomThermalModel.from_room_config(
                room_config,
                zone_config=zone_config,
                num_zone_rooms=num_zone_rooms,
                infiltration_ach=self._infiltration_ach,
            )
            thermal_models[room_name] = model

        occupancy_predictors = {
            room_name: OccupancyPredictor(room_name, room_config.occupancy_schedule)
            for room_name, room_config in scenario.room_configs.items()
        }
        return HouseSimulator(
            room_configs=scenario.room_configs,
            thermal_models=thermal_models,
            occupancy_predictors=occupancy_predictors,
            latitude_deg=self._latitude_deg,
            cloud_cover=self._cloud_cover,
            albedo=self._albedo,
        )

print("✅ PhysicsMultiRoomEnv defined")
""")

code("""\
class PreheatRewardWrapper(gym.Wrapper):
    \"\"\"
    Amplifies comfort penalties during pre-occupancy and occupied periods to
    give the policy a stronger gradient for learning to preheat ahead of arrival.

    At each step for every active room:
        extra_penalty += preheat_boost * effective_occ * max(target_min - T, 0)
    applied whenever occupancy_prob > preheat_threshold (default 0.35).

    The base energy and switching costs from IntelliWarmMultiRoomEnv are unchanged.
    \"\"\"

    def __init__(self, env: gym.Env, preheat_boost: float = 30.0,
                 preheat_threshold: float = 0.05):
        super().__init__(env)
        self.preheat_boost = preheat_boost
        self.preheat_threshold = preheat_threshold

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        max_rooms = self.env.max_rooms
        forecast_horizon = self.env.occupancy_forecast_horizon_steps
        room_feature_block = max_rooms * 6
        extra = 0.0
        for i in range(max_rooms):
            base = i * 6
            if float(obs[base + 5]) < 0.5:   # validity flag
                continue
            temp       = float(obs[base + 0])
            target_min = float(obs[base + 1])
            occ_now    = float(obs[base + 3])
            forecast_base = room_feature_block + (i * forecast_horizon)
            next_1h_occ = float(obs[forecast_base + 0]) if forecast_horizon >= 1 else 0.0
            next_2h_occ = float(obs[forecast_base + 1]) if forecast_horizon >= 2 else 0.0
            effective_occ = max(occ_now, next_1h_occ, next_2h_occ)
            if effective_occ > self.preheat_threshold and temp < target_min:
                extra += self.preheat_boost * effective_occ * (target_min - temp)

        info = dict(info)
        info["preheat_penalty"] = extra
        base_violation = float(info.get("comfort_violation", 0.0))
        comfort_weight = max(float(getattr(self.env, "comfort_penalty_weight", 1.0)), 1e-6)
        info["reported_comfort_violation"] = base_violation + (extra / comfort_weight)
        info["wrapped_reward"] = reward - extra

        return obs, reward - extra, terminated, truncated, info

print("✅ PreheatRewardWrapper defined")
""")

code("""\
# ── Create environment and explore spaces ─────────────────────────────────────
_ENV_KWARGS = dict(
    comfort_penalty_weight=25.0,   # strong comfort incentive
    energy_weight=1.0,
    switching_weight=0.10,         # light switching penalty
    invalid_source_penalty=2.0,    # penalise requesting furnace on electric zone
)

_probe = PhysicsMultiRoomEnv(scenarios=SCENARIOS, **_ENV_KWARGS)
_probe = PreheatRewardWrapper(_probe)
obs0, info0 = _probe.reset(seed=0)

print("═" * 60)
print("Environment spaces")
print("═" * 60)
print(f"  Observation: Box{_probe.observation_space.shape} float32")
print(f"    = {_probe.env.max_rooms} rooms × 6 state features")
print(f"    + {_probe.env.max_rooms} rooms × {_probe.env.occupancy_forecast_horizon_steps} occupancy-forecast features")
print(f"    + {_probe.env.max_zones} zones × 3 features")
print(f"    + 7 global features  (T_out, elec, gas, hour_sin, hour_cos, next_1h_occ, next_2h_occ)")
print(f"  Action:      Box{_probe.action_space.shape} float32")
print(f"    = {_probe.env.max_zones} zone source signals  [0..1, >=0.5 ⇒ furnace]")
print(f"    + {_probe.env.max_rooms} room heat demands     [0..1 continuous]")
print()
print(f"Initial observation (first episode, winter_workday):")
print(f"  Scenario  : {info0['scenario_name']}")
print(f"  Rooms     : {info0['room_names']}")
print(f"  Zones     : {info0['zone_names']}")
print(f"  Obs range : [{obs0.min():.2f}, {obs0.max():.2f}]")

_probe.close()
""")

# ─────────────────────────────────────────────────────────────────────────────
# BASELINES
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## 📊 5. Baseline Policy Benchmarks

We evaluate five hand-crafted baselines before training to establish cost and
comfort-violation reference points.

| Policy | Zone source | Room demand | Expected behaviour |
|--------|------------|-------------|-------------------|
| **Always OFF** | electric | 0 % | Cheapest energy, worst comfort |
| **ECO electric** | electric | 35 % constant | Low energy, modest comfort |
| **COMFORT electric** | electric | thermostat demand | Electric-only continuous thermostat |
| **Furnace COMFORT** | furnace (where available) | thermostat demand | Furnace zones run with shared continuous demand |
| **Smart ToU** | adaptive hybrid | thermostat demand | Continuous cost-aware heuristic |
""")

code("""\
def _make_eval_env(scenario: TrainingScenario):
    \"\"\"Create a fresh wrapped env for a single scenario.\"\"\"
    env = PhysicsMultiRoomEnv(scenarios=[scenario], **_ENV_KWARGS)
    env = PreheatRewardWrapper(env)
    return env


def rollout(policy_fn, env, seed: int = 42):
    \"\"\"Run one episode, return (total_reward, total_cost, total_violation, log).\"\"\"
    obs, info = env.reset(seed=seed)
    total_reward = total_cost = total_violation = 0.0
    log = []  # per-step dicts for plotting
    step_idx = 0
    while True:
        action = policy_fn(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward    += reward
        total_cost      += info.get("total_cost", 0.0)
        total_violation += info.get("reported_comfort_violation", info.get("comfort_violation", 0.0))
        log.append({
            "step": step_idx,
            "reward": reward,
            "total_cost": info.get("total_cost", 0.0),
            "comfort_violation": info.get("reported_comfort_violation", info.get("comfort_violation", 0.0)),
            "preheat_penalty": info.get("preheat_penalty", 0.0),
            "electric_cost": info.get("electric_cost", 0.0),
            "gas_cost": info.get("gas_cost", 0.0),
            "zone_heat_sources": dict(info.get("zone_heat_sources", {})),
        })
        step_idx += 1
        if terminated or truncated:
            break
    return total_reward, total_cost, total_violation, log


def eval_policy_all_scenarios(name: str, policy_fn):
    \"\"\"Evaluate policy once on each fixed scenario.\"\"\"
    results = {"name": name}
    for scenario in SCENARIOS:
        env = _make_eval_env(scenario)
        r, c, v, _ = rollout(policy_fn, env, seed=42)
        env.close()
        results[scenario.name] = {
            "reward": r,
            "cost": c,
            "violation": v,
        }
    return results

print("✅ Evaluation helpers ready")
""")

code("""\
# ── Define five baseline policies ─────────────────────────────────────────────
# Policy signature: (obs: np.ndarray, info: dict) -> Sequence[float]
# Action layout: [zone_source_signal..., room_heat_demand...]

BASELINE_CONTROLLERS = {
    room_name: BaselineController(
        room_config=room_config,
        min_temperature=18.0,
        max_temperature=24.0,
        preheat_lookahead_steps=2,
    )
    for room_name, room_config in ROOMS.items()
}

HYBRID_CONTROLLERS = {
    zone_name: HybridController(
        zone_config=zone_config,
        room_configs={
            room_name: room_config
            for room_name, room_config in ROOMS.items()
            if room_config.zone == zone_name
        },
        min_temperature=18.0,
        max_temperature=24.0,
        preheat_lookahead_steps=2,
    )
    for zone_name, zone_config in ZONES.items()
}

def _room_vector(room_demands: Dict[str, float], info: dict, fill: float = 0.0):
    values = [float(room_demands.get(room_name, fill)) for room_name in info.get("room_names", [])]
    values.extend([fill] * (info["max_rooms"] - len(values)))
    return values

def _active_zone_names(info: dict):
    return list(info.get("zone_names", []))

def _zone_vector(zone_signals: Dict[str, float], info: dict):
    values = [float(zone_signals.get(zone_name, 0.0)) for zone_name in _active_zone_names(info)]
    values.extend([0.0] * (info["max_zones"] - len(values)))
    return values

def _obs_context(obs, info):
    max_rooms = info["max_rooms"]
    horizon = info["occupancy_forecast_horizon_steps"]
    room_feature_block = max_rooms * 6
    room_context = {}
    for idx, room_name in enumerate(info.get("room_names", [])):
        base = idx * 6
        forecast_base = room_feature_block + (idx * horizon)
        occ_now = float(obs[base + 3])
        future_occ = [float(obs[forecast_base + j]) for j in range(horizon)]
        room_context[room_name] = {
            "temp": float(obs[base + 0]),
            "target_min": float(obs[base + 1]),
            "target_max": float(obs[base + 2]),
            "forecast": [occ_now] + future_occ,
            "current_occ": occ_now,
            "last_action": float(obs[base + 4]),
            "target": 0.5 * (float(obs[base + 1]) + float(obs[base + 2])),
        }

    outside_temp = float(obs[-7])
    electricity_price = float(obs[-6])
    gas_price = float(obs[-5])
    return room_context, outside_temp, electricity_price, gas_price

def _baseline_room_demands(obs, info):
    room_context, outside_temp, electricity_price, _ = _obs_context(obs, info)
    demands = {}
    for room_name, ctx in room_context.items():
        decision = BASELINE_CONTROLLERS[room_name].compute_decision(
            current_temp=ctx["temp"],
            occupancy_forecast=ctx["forecast"],
            energy_prices=[electricity_price],
            current_action=ctx["last_action"],
            outside_temp=outside_temp,
            target_temp=ctx["target"],
        )
        demands[room_name] = float(decision.action)
    return demands

def policy_always_off(obs, info):
    return [0.0] * (info["max_zones"] + info["max_rooms"])

def policy_eco_electric(obs, info):
    return [0.0] * info["max_zones"] + [0.35] * info["max_rooms"]

def policy_comfort_electric(obs, info):
    room_demands = _baseline_room_demands(obs, info)
    return [0.0] * info["max_zones"] + _room_vector(room_demands, info)

def policy_furnace_comfort(obs, info):
    \"\"\"Use furnace for furnace-equipped zones with continuous thermostat demand.\"\"\"
    room_demands = _baseline_room_demands(obs, info)
    zone_src = {
        zone_name: 1.0 if info.get("zone_has_furnace", {}).get(zone_name, False) else 0.0
        for zone_name in _active_zone_names(info)
    }
    return _zone_vector(zone_src, info) + _room_vector(room_demands, info)

def policy_smart_tou(obs, info):
    \"\"\"Continuous heuristic hybrid controller using the shared zone cost logic.\"\"\"
    room_context, outside_temp, electricity_price, gas_price = _obs_context(obs, info)
    zone_signals = {}
    room_demands = {}
    for zone_name in _active_zone_names(info):
        zone_rooms = {
            room_name: ctx
            for room_name, ctx in room_context.items()
            if ROOMS[room_name].zone == zone_name
        }
        decision = HYBRID_CONTROLLERS[zone_name].decide(
            room_temperatures={room_name: ctx["temp"] for room_name, ctx in zone_rooms.items()},
            occupancy_forecasts={room_name: ctx["forecast"] for room_name, ctx in zone_rooms.items()},
            electricity_price=electricity_price,
            gas_price=gas_price,
            outside_temp=outside_temp,
            current_actions={room_name: ctx["last_action"] for room_name, ctx in zone_rooms.items()},
            target_temps={room_name: ctx["target"] for room_name, ctx in zone_rooms.items()},
        )
        zone_signals[zone_name] = 1.0 if decision.heat_source == HeatSourceType.GAS_FURNACE else 0.0
        room_demands.update(decision.per_room_actions)

    return _zone_vector(zone_signals, info) + _room_vector(room_demands, info)

BASELINE_POLICIES = {
    "always_off":        policy_always_off,
    "eco_electric":      policy_eco_electric,
    "comfort_electric":  policy_comfort_electric,
    "furnace_comfort":   policy_furnace_comfort,
    "smart_tou":         policy_smart_tou,
}

print(f"Defined {len(BASELINE_POLICIES)} baseline policies")
""")

code("""\
# ── Evaluate all baselines ────────────────────────────────────────────────────
print("Evaluating baselines (1 deterministic rollout × 3 scenarios each) …")
baseline_results = {}
for pol_name, pol_fn in BASELINE_POLICIES.items():
    baseline_results[pol_name] = eval_policy_all_scenarios(pol_name, pol_fn)
    r_avg = np.mean([baseline_results[pol_name][s.name]["reward"] for s in SCENARIOS])
    c_avg = np.mean([baseline_results[pol_name][s.name]["cost"] for s in SCENARIOS])
    v_avg = np.mean([baseline_results[pol_name][s.name]["violation"] for s in SCENARIOS])
    print(f"  {pol_name:22s}  reward={r_avg:8.2f}  cost=${c_avg:.4f}  violation={v_avg:.3f}")
""")

code("""\
# ── Plot baseline comparison ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
SCENARIO_COLORS = {"winter_workday": "#1565C0", "winter_weekend": "#AD1457", "spring_workday": "#2E7D32"}
pol_names = list(BASELINE_POLICIES.keys())
x = np.arange(len(pol_names))
width = 0.25

for col, metric in enumerate(["reward", "cost", "violation"]):
    ax = axes[col]
    for s_idx, scenario in enumerate(SCENARIOS):
        vals = [baseline_results[p][scenario.name][metric] for p in pol_names]
        offset = (s_idx - 1) * width
        bars = ax.bar(x + offset, vals, width, label=scenario.name.replace("_", " "),
                      color=SCENARIO_COLORS[scenario.name], alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("_", "\\n") for p in pol_names], fontsize=9)
    titles = {"reward": "Episode Reward (↑ better)", "cost": "Total Energy Cost $ (↓ better)",
              "violation": "Comfort Violation (↓ better)"}
    ax.set_title(titles[metric], fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    if col == 0:
        ax.legend(fontsize=9, loc="lower right")

plt.suptitle("Baseline Policy Comparison — deterministic rollouts over 3 scenarios", fontsize=12)
plt.tight_layout()
plt.show()

# Print summary table
print("\\n─── Baseline summary (averaged across scenarios) ───")
print(f"{'Policy':22s}  {'Reward':>10}  {'Cost $':>10}  {'Violation':>10}")
print("─" * 60)
for p in pol_names:
    r = np.mean([baseline_results[p][s.name]["reward"] for s in SCENARIOS])
    c = np.mean([baseline_results[p][s.name]["cost"]   for s in SCENARIOS])
    v = np.mean([baseline_results[p][s.name]["violation"] for s in SCENARIOS])
    print(f"{p:22s}  {r:10.2f}  {c:10.4f}  {v:10.3f}")
""")

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## 🧠 6. Training the OPT Policy with PPO

We use **Proximal Policy Optimisation** (Schulman et al., 2017) from
Stable-Baselines3 with 4 parallel environment workers.

### Key PPO hyperparameters

| Param | Value | Rationale |
|-------|-------|-----------|
| `n_steps` | 1 152 | Exactly four 24-hour episodes per environment rollout |
| `batch_size` | 256 | Mini-batches from the collected rollout |
| `n_epochs` | 10 | Re-use rollout data for 10 gradient updates |
| `gamma` | 0.999 | Long planning horizon with 5-minute control intervals |
| `ent_coef` | 0.02 | Encourages action diversity early; decays with entropy |
| `learning_rate` | 3e-4 | Standard Adam LR for PPO |

### Training environment
- `PhysicsMultiRoomEnv` with `PreheatRewardWrapper(preheat_boost=30, preheat_threshold=0.05)`.
- Scenarios cycle round-robin across the 4 parallel workers.
- Each episode lasts exactly **288 steps** (one simulated day at 5-minute resolution).
- PPO outputs **continuous** room heat demands in `[0, 1]` plus continuous zone-source
  signals where `>= 0.5` requests the furnace.

### What OPT should learn
1. **Gas furnace during cold peaks** — furnace is cheaper than running all room heaters.
2. **Preheat bedrooms before 05:00/06:00** — family wakes up at 05:00–06:00; warm rooms = no penalty.
3. **Cheap overnight electricity** — run office heater at ECO during cheap hours (≤ 8 ¢/kWh).
4. **Reduce heating mid-day when unoccupied** — kitchen/living room empty 09:00–17:00 on weekdays.
""")

code("""\
N_ENVS = 4
TOTAL_TIMESTEPS = 100_000

def _make_train_env(seed: int = 0):
    \"\"\"Factory for one training environment instance (cycles all scenarios).\"\"\"
    def _init():
        env = PhysicsMultiRoomEnv(scenarios=SCENARIOS, **_ENV_KWARGS)
        env = PreheatRewardWrapper(env, preheat_boost=30.0, preheat_threshold=0.05)
        env.reset(seed=seed)
        return env
    return _init

vec_env = DummyVecEnv([_make_train_env(seed=i) for i in range(N_ENVS)])

print(f"VecEnv: {N_ENVS} parallel environments")
print(f"  obs_space : {vec_env.observation_space.shape}")
print(f"  act_space : Box{vec_env.action_space.shape}")
""")

code("""\
# ── Logging callback ─────────────────────────────────────────────────────────
class TrainingLogger(BaseCallback):
    \"\"\"Records per-episode reward, cost, and comfort violation during training.\"\"\"

    def __init__(self):
        super().__init__(verbose=0)
        self.ep_rewards: List[float] = []
        self.ep_costs:   List[float] = []
        self.ep_violations: List[float] = []
        self._ep_reward   = [0.0] * N_ENVS
        self._ep_cost     = [0.0] * N_ENVS
        self._ep_violation = [0.0] * N_ENVS

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", [0.0] * N_ENVS)
        dones   = self.locals.get("dones", [False]   * N_ENVS)
        infos   = self.locals.get("infos", [{}]       * N_ENVS)

        for i in range(N_ENVS):
            self._ep_reward[i]    += float(rewards[i])
            self._ep_cost[i]      += float(infos[i].get("total_cost", 0.0))
            self._ep_violation[i] += float(infos[i].get("reported_comfort_violation", infos[i].get("comfort_violation", 0.0)))
            if dones[i]:
                self.ep_rewards.append(self._ep_reward[i])
                self.ep_costs.append(self._ep_cost[i])
                self.ep_violations.append(self._ep_violation[i])
                self._ep_reward[i] = self._ep_cost[i] = self._ep_violation[i] = 0.0
        return True

logger = TrainingLogger()

# ── PPO model ────────────────────────────────────────────────────────────────
import torch

if torch.cuda.is_available():
    device = "cuda"
    device_label = f"cuda ({torch.cuda.get_device_name(0)})"
else:
    device = "cpu"
    device_label = "cpu"
    print("Warning: CUDA not available, falling back to CPU training.")

model = PPO(
    "MlpPolicy",
    vec_env,
    verbose=0,
    n_steps=1152,
    batch_size=256,
    n_epochs=10,
    learning_rate=3e-4,
    gamma=0.999,
    gae_lambda=0.95,
    ent_coef=0.02,
    clip_range=0.20,
    policy_kwargs=dict(net_arch=[128, 128]),   # two hidden layers
    seed=42,
    device=device,
)

print(f"Using PPO device: {device_label}")
print(f"PPO model ready — {sum(p.numel() for p in model.policy.parameters()):,} parameters")
print(f"Training for {TOTAL_TIMESTEPS:,} timesteps across {N_ENVS} envs …")
""")

code("""\
# ── Train! ────────────────────────────────────────────────────────────────────
import time
t0 = time.time()
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=logger, progress_bar=True)
elapsed = time.time() - t0
print(f"\\n✅ Training complete in {elapsed:.1f}s")
print(f"   Episodes logged: {len(logger.ep_rewards)}")
if logger.ep_rewards:
    last_k = 200
    r_last = np.mean(logger.ep_rewards[-last_k:])
    c_last = np.mean(logger.ep_costs[-last_k:])
    v_last = np.mean(logger.ep_violations[-last_k:])
    print(f"   Last {last_k} episodes — reward={r_last:.2f}  cost=${c_last:.4f}  violation={v_last:.3f}")
""")

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING CURVES
# ─────────────────────────────────────────────────────────────────────────────
md("## 📈 7. Training Curves")

code("""\
def _smooth(data, window: int = 50):
    \"\"\"Centred rolling mean, padded with edge values.\"\"\"
    if len(data) < window:
        return np.array(data)
    return np.convolve(data, np.ones(window) / window, mode="same")

fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
ep_idx = np.arange(len(logger.ep_rewards))

METRICS = [
    (logger.ep_rewards,    "Episode Reward (↑ better)",          "#1565C0"),
    (logger.ep_costs,      "Episode Energy Cost $ (↓ better)",   "#AD1457"),
    (logger.ep_violations, "Reported Comfort Violation (↓ better)", "#2E7D32"),
]

for ax, (data, title, color) in zip(axes, METRICS):
    ax.plot(ep_idx, data, alpha=0.20, color=color, lw=0.5)
    smooth = _smooth(data, window=min(100, max(10, len(data) // 20)))
    ax.plot(ep_idx, smooth, color=color, lw=2, label="Smoothed (100-ep window)")
    ax.set_ylabel(title, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper right" if "Reward" in title else "upper left")

axes[-1].set_xlabel("Training episode", fontsize=11)
plt.suptitle(f"PPO Training Progress — {TOTAL_TIMESTEPS:,} timesteps, {N_ENVS} parallel envs",
             fontsize=12)
plt.tight_layout()
plt.show()
""")

# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## 🏆 8. Policy Comparison: Baselines vs Trained OPT

We evaluate the trained PPO policy on all three scenarios (deterministic greedy,
`deterministic=True`) and compare against the five baselines.
""")

code("""\
# ── Trained policy wrapper ────────────────────────────────────────────────────
_last_info: dict = {}   # updated each step

def policy_trained_opt(obs: np.ndarray, info: dict) -> list:
    \"\"\"Deterministic greedy action from the trained PPO policy.\"\"\"
    action, _ = model.predict(obs.reshape(1, -1), deterministic=True)
    return action[0].tolist()

# ── Evaluate ──────────────────────────────────────────────────────────────────
print("Evaluating trained OPT policy …")
trained_results = eval_policy_all_scenarios("trained_opt", policy_trained_opt)

all_results = dict(baseline_results)
all_results["trained_opt"] = trained_results

# Print comparison table
print("\\n─── Final policy comparison (mean over scenarios) ───")
print(f"{'Policy':22s}  {'Reward':>10}  {'Cost $':>10}  {'Violation':>10}")
print("─" * 60)
all_names = list(BASELINE_POLICIES.keys()) + ["trained_opt"]
for p in all_names:
    r = np.mean([all_results[p][s.name]["reward"]    for s in SCENARIOS])
    c = np.mean([all_results[p][s.name]["cost"]      for s in SCENARIOS])
    v = np.mean([all_results[p][s.name]["violation"] for s in SCENARIOS])
    marker = " ◄ OPT" if p == "trained_opt" else ""
    print(f"{p:22s}  {r:10.2f}  {c:10.4f}  {v:10.3f}{marker}")
""")

code("""\
# ── Comparison plots ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
all_pol_names = all_names
x = np.arange(len(all_pol_names))
pal = plt.cm.tab10(np.linspace(0, 1, len(SCENARIOS)))

for col, metric in enumerate(["reward", "cost", "violation"]):
    ax = axes[col]
    for s_idx, scenario in enumerate(SCENARIOS):
        vals = [all_results[p][scenario.name][metric] for p in all_pol_names]
        offset = (s_idx - 1) * 0.22
        bars = ax.bar(x + offset, vals, 0.22,
                      label=scenario.name.replace("_", " "),
                      color=list(SCENARIO_COLORS.values())[s_idx], alpha=0.85)

    # Highlight trained OPT column
    opt_idx = all_pol_names.index("trained_opt")
    ax.axvspan(opt_idx - 0.4, opt_idx + 0.4, alpha=0.08, color="gold")
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("_", "\\n") for p in all_pol_names], fontsize=8.5)
    titles = {"reward": "Reward (↑)", "cost": "Cost $ (↓)", "violation": "Comfort Violation (↓)"}
    ax.set_title(titles[metric], fontweight="bold", fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    if col == 2:
        ax.legend(fontsize=9)

plt.suptitle("All Policies: Baselines vs Trained OPT  (gold = trained)", fontsize=12)
plt.tight_layout()
plt.show()
""")

# ─────────────────────────────────────────────────────────────────────────────
# BEHAVIOURAL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## 🔍 9. Behavioural Analysis: What Did the Policy Learn?

We roll out the trained policy on the **winter_workday** scenario (hardest case) and
visualise:
1. **Room temperatures** vs target bands over 24 hours.
2. **Continuous heating demand** per room.
3. **Zone heat-source choices** (furnace vs electric, across the day).
4. **Per-step cost breakdown** (electric vs gas).
""")

code("""\
# ── Single-episode rollout for analysis ───────────────────────────────────────
ROOM_ORDER = sorted(ROOMS.keys())   # alphabetical for consistent plots

def full_rollout(policy_fn, scenario_name: str, seed: int = 0):
    \"\"\"Run one deterministic episode; return per-step records.\"\"\"
    target = next(s for s in SCENARIOS if s.name == scenario_name)
    env = PhysicsMultiRoomEnv(scenarios=[target], **_ENV_KWARGS)
    env = PreheatRewardWrapper(env)

    obs, info = env.reset(seed=seed, options={"scenario_name": scenario_name})
    room_names_sorted = sorted(info["room_names"])
    zone_names_sorted = sorted(info["zone_names"])

    temps   = {r: [target.initial_temperatures[r]] for r in room_names_sorted}
    actions = {r: [] for r in room_names_sorted}
    sources = {z: [] for z in zone_names_sorted}
    step_costs_elec = []
    step_costs_gas  = []
    step_violations = []
    elapsed_hours = np.arange(target.horizon_steps) * (target.step_minutes / 60.0)

    while True:
        act = policy_fn(obs, info)
        obs, reward, terminated, truncated, info = env.step(act)

        # Room temperatures (from obs: first max_rooms * 6 elements)
        for i, r in enumerate(room_names_sorted):
            temps[r].append(float(obs[i * 6 + 0]))
            actions[r].append(float(obs[i * 6 + 4]))  # last_action_power

        # Zone sources
        zone_src_info = info.get("zone_heat_sources", {})
        for z in zone_names_sorted:
            sources[z].append(1 if zone_src_info.get(z, "electric") == "gas_furnace" else 0)

        step_costs_elec.append(info.get("electric_cost", 0.0))
        step_costs_gas.append(info.get("gas_cost", 0.0))
        step_violations.append(info.get("reported_comfort_violation", info.get("comfort_violation", 0.0)))

        if terminated or truncated:
            break

    env.close()
    return {
        "room_names": room_names_sorted,
        "zone_names": zone_names_sorted,
        "temps": temps,          # dict room→list[float] (t0 + horizon_steps values)
        "actions": actions,      # dict room→list[float] (horizon_steps values)
        "sources": sources,      # dict zone→list[int]   (horizon_steps values)
        "elapsed_hours": elapsed_hours,
        "elec_cost": step_costs_elec,
        "gas_cost":  step_costs_gas,
        "violations": step_violations,
    }

rollout_opt  = full_rollout(policy_trained_opt,  "winter_workday")
rollout_base = full_rollout(policy_comfort_electric, "winter_workday")
print("✅ Rollouts complete")
""")

code("""\
# ── Figure 1: Room temperatures ───────────────────────────────────────────────
def plot_temperatures(data: dict, title: str, scenario=winter_workday, ax_array=None):
    room_names = data["room_names"]
    n = len(room_names)
    if ax_array is None:
        fig, axes = plt.subplots(n, 1, figsize=(13, 2.8 * n), sharex=True)
        if n == 1:
            axes = [axes]
    else:
        axes = ax_array
        fig = None

    step_hours = scenario.step_minutes / 60.0
    hours_temp = np.arange(scenario.horizon_steps + 1) * step_hours
    hours_act  = np.arange(scenario.horizon_steps) * step_hours

    action_cmap = plt.cm.Blues

    for ax, room in zip(axes, room_names):
        temps_r = data["temps"][room]
        acts_r  = data["actions"][room]
        rc      = ROOMS[room]

        # Shade occupancy periods
        occ_predictor = OccupancyPredictor(room, rc.occupancy_schedule)
        for h in range(scenario.horizon_steps):
            ts = scenario.start_time + timedelta(minutes=scenario.step_minutes * h)
            occ = occ_predictor.predict(ts)
            if occ > 0.3:
                x0 = h * step_hours
                ax.axvspan(x0, x0 + step_hours, alpha=0.08 + 0.10 * occ, color="#FF9800", lw=0)

        # Comfort band
        ax.axhspan(rc.target_min_temp, rc.target_max_temp, alpha=0.12, color="green", label="Comfort band")
        ax.axhline(rc.target_min_temp, color="green", lw=0.8, ls="--")
        ax.axhline(rc.target_max_temp, color="green", lw=0.8, ls="--")

        # Temperature trace
        ax.plot(hours_temp, temps_r, lw=2, color="#1565C0", label="Room temp")

        # Continuous action power bars
        for h, act in enumerate(acts_r):
            col = action_cmap(0.20 + 0.75 * act)
            x0 = h * step_hours
            ax.axvspan(x0, x0 + step_hours, ymin=0, ymax=0.05, color=col, alpha=0.95)

        ax.set_ylabel(f"{rc.display_name}\\n(°C)", fontsize=9)
        ax.set_ylim(min(temps_r) - 1.5, max(temps_r) + 2.5)
        ax.grid(True, alpha=0.25)

    axes[0].set_title(title, fontsize=11, fontweight="bold")
    axes[-1].set_xlabel("Hour of day")
    axes[-1].set_xticks(np.arange(0, 25, 2))

    legend_patches = [
        mpatches.Patch(color=action_cmap(0.20), label="0% demand"),
        mpatches.Patch(color=action_cmap(0.45), label="~35% demand"),
        mpatches.Patch(color=action_cmap(0.72), label="~70% demand"),
        mpatches.Patch(color=action_cmap(0.95), label="100% demand"),
        mpatches.Patch(color="#FF9800", alpha=0.5, label="Occupancy window"),
    ]
    axes[-1].legend(handles=legend_patches, ncol=5, fontsize=8, loc="lower right")

    if fig is not None:
        plt.tight_layout()
        plt.show()

plot_temperatures(rollout_opt,  "Trained OPT — room temperatures (winter_workday)")
plot_temperatures(rollout_base, "COMFORT-electric baseline — room temperatures (winter_workday)")
""")

code("""\
# ── Figure 2: Zone heat-source choices & cost breakdown ──────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 7))

step_hours = winter_workday.step_minutes / 60.0
hours = np.arange(winter_workday.horizon_steps) * step_hours

# ── Top left: OPT zone sources ────────────────────────────────────────────────
ax = axes[0, 0]
zone_colors = {"Main": "#1565C0", "Sleeping": "#AD1457", "Office": "#2E7D32"}
for z in rollout_opt["zone_names"]:
    src = rollout_opt["sources"][z]
    ax.step(hours, src, where="post", label=f"{z} ({'furnace' if max(src) else 'electric'})",
            color=zone_colors.get(z, "gray"), lw=2)
ax.set_yticks([0, 1])
ax.set_yticklabels(["Electric", "Furnace"])
ax.set_title("OPT — Zone heat-source choice", fontweight="bold")
ax.set_xlabel("Hour of day"); ax.set_xticks(np.arange(0, 25, 4)); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# ── Top right: Baseline zone sources ─────────────────────────────────────────
ax = axes[0, 1]
for z in rollout_base["zone_names"]:
    src = rollout_base["sources"][z]
    ax.step(hours, src, where="post", label=z, color=zone_colors.get(z, "gray"), lw=2)
ax.set_yticks([0, 1])
ax.set_yticklabels(["Electric", "Furnace"])
ax.set_title("COMFORT-electric — Zone heat-source choice", fontweight="bold")
ax.set_xlabel("Hour of day"); ax.set_xticks(np.arange(0, 25, 4)); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# ── Bottom left: Cost breakdown OPT ──────────────────────────────────────────
ax = axes[1, 0]
ax.bar(hours, rollout_opt["elec_cost"], width=step_hours, color="#42A5F5", alpha=0.85, label="Electric")
ax.bar(hours, rollout_opt["gas_cost"],  bottom=rollout_opt["elec_cost"],
       width=step_hours, color="#FF7043", alpha=0.85, label="Gas")
ax2 = ax.twinx()
ax2.plot(hours, winter_workday.electricity_prices, color="black", lw=1.5, ls=":", label="Elec price")
ax2.set_ylabel("Elec price ($/kWh)", fontsize=9)
ax.set_title("OPT — 5-minute cost breakdown", fontweight="bold")
ax.set_xlabel("Hour of day"); ax.set_ylabel("Cost ($)"); ax.set_xticks(np.arange(0, 25, 4)); ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# ── Bottom right: Comfort violations ─────────────────────────────────────────
ax = axes[1, 1]
x = np.arange(len(hours))
w = 0.35
ax.bar(x - w/2, rollout_opt["violations"],  w, label="OPT",  color="#1565C0", alpha=0.8)
ax.bar(x + w/2, rollout_base["violations"], w, label="COMFORT-electric", color="#FF5722", alpha=0.8)
tick_idx = np.arange(0, len(hours), 12)
ax.set_xticks(tick_idx); ax.set_xticklabels([f"{hours[i]:.0f}" for i in tick_idx])
ax.set_title("Comfort violations — OPT vs baseline", fontweight="bold")
ax.set_xlabel("Hour of day"); ax.set_ylabel("Violation (°C · occupancy)")
ax.legend(fontsize=9); ax.grid(True, axis="y", alpha=0.3)

plt.suptitle("Behavioural Analysis — Winter Workday", fontsize=13)
plt.tight_layout()
plt.show()

opt_total_cost = sum(rollout_opt["elec_cost"]) + sum(rollout_opt["gas_cost"])
base_total_cost = sum(rollout_base["elec_cost"]) + sum(rollout_base["gas_cost"])
opt_total_viol  = sum(rollout_opt["violations"])
base_total_viol = sum(rollout_base["violations"])
savings = (base_total_cost - opt_total_cost) / max(base_total_cost, 1e-9) * 100
print(f"\\n┌─────────────────────────────────────────────────────────┐")
print(f"│  Winter workday — OPT vs COMFORT-electric baseline       │")
print(f"│  Energy cost  OPT: ${opt_total_cost:.4f}   Baseline: ${base_total_cost:.4f}  │")
print(f"│  Savings: {savings:+.1f}%                                       │")
print(f"│  Comfort viol OPT: {opt_total_viol:.3f}   Baseline: {base_total_viol:.3f}        │")
print(f"└─────────────────────────────────────────────────────────┘")
""")

# ─────────────────────────────────────────────────────────────────────────────
# SAVE MODEL
# ─────────────────────────────────────────────────────────────────────────────
md("## 💾 10. Save & Reload Model")

code("""\
import os
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(".")), "output", "opt_policy_ppo")
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
model.save(MODEL_PATH)
print(f"✅ Model saved to {MODEL_PATH}.zip")

# Reload and verify
loaded_model = PPO.load(MODEL_PATH)
obs_test, info_test = _make_train_env(seed=99)().reset()
a, _ = loaded_model.predict(obs_test, deterministic=True)
print(f"✅ Reload OK — sample action: {a.tolist()}")
""")

# ─────────────────────────────────────────────────────────────────────────────
# CONCLUSIONS
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## 🏁 11. Summary & Next Steps

### What was trained
A **PPO policy** for a 3-zone, 5-room family house that jointly decides:
- **Zone heat source** (gas furnace vs electric, per zone, every 5 minutes)
- **Room heating demand** (continuous 0–100 %, per room, every 5 minutes)

### Reward design
| Component | Weight | Effect |
|-----------|--------|--------|
| Energy cost (electric + gas) | 1.0 | Agent minimises energy spend |
| Comfort violation (occ × ΔT) | 25.0 | Penalises cold rooms during occupancy |
| Pre-occupancy penalty (wrapper) | 30.0 / °C | Extra signal to preheat *before* arrival |
| Switching penalty | 0.1 | Smooth action transitions |

### Key learned behaviours (typical)
- Turns on Sleeping-zone furnace around **03:00–05:00** before the family wakes up.
- Switches Main-zone furnace off mid-day when kitchen/living room are empty.
- Uses cheap overnight electricity (≤ 8 ¢/kWh) for the home office preheat.
- On spring days, often prefers electric over furnace when outdoor temps are mild.

### Next steps
1. **Longer training** — increase `TOTAL_TIMESTEPS` only if the learning curves still improve beyond 100k.
2. **VecNormalize** — wrap the VecEnv with `VecNormalize` for observation/reward normalisation.
3. **Live price integration** — replace synthetic price profiles with real time-of-use tariffs.
4. **Hardware deployment** — feed the trained policy into `IntelliWarmRuntime` as a controller type.
5. **Curriculum learning** — start with easy (spring) scenarios, progressively introduce cold winter days.
""")

# ─────────────────────────────────────────────────────────────────────────────
# WRITE THE NOTEBOOK
# ─────────────────────────────────────────────────────────────────────────────
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0",
        },
    },
    "cells": cells,
}

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "opt_heating_policy_training.ipynb")
with open(out_path, "w", encoding="utf-8") as fh:
    json.dump(notebook, fh, indent=1, ensure_ascii=False)

print(f"✅ Notebook written: {out_path}")
print(f"   Cells: {len(cells)}")
