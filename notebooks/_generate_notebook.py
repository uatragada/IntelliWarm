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
import json
import os, sys, warnings
from pathlib import Path
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
    RoomConfig, ZoneConfig, HeatSourceType, OccupancyWindow,
    SimulationState,
)
from intelliwarm.control import (
    BaselineController, HybridController, RoomHeatingIntent, ZoneSourceMode,
    room_intent_feature_value, room_intent_index,
    zone_source_mode_feature_value, zone_source_mode_index,
)
from intelliwarm.models import PhysicsRoomThermalModel, HouseSimulator
from intelliwarm.prediction import OccupancyPredictor
from intelliwarm.learning.scenario_generator import TrainingScenario, SyntheticScenarioGenerator
from intelliwarm.learning.gym_env import IntelliWarmMultiRoomEnv

# ── Gymnasium + SB3 ───────────────────────────────────────────────────────────
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

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
`PhysicsRoomThermalModel` instead of the legacy first-order model. The base
multi-room environment can expose the full scenario occupancy horizon; this
notebook caps the forecast to a 2-hour, 24-step window so PPO still sees
upcoming schedule changes without a huge observation vector. During PPO
training/evaluation we also append `smart_tou`'s recommended zone-source modes
and room intents as teacher features. This gives each room:
- Lumped thermal capacitance **C** [kJ/K] derived from heater design load.
- Envelope + **infiltration** conductance **UA** [W/K] (ASHRAE 62.2, 0.5 ACH default).
- Per-room **furnace power share** from `ZoneConfig` (BTU/hr × AFUE ÷ rooms).
- **Solar gain** on a south-facing vertical window (dm4bem incidence angle model).

### Comfort warmup ramp
During the first `comfort_warmup_steps` steps (default 24 = 2 hours), the comfort
penalty is linearly scaled from 0 to full strength. This avoids penalizing the
policy for unavoidable temperature deficits when rooms start cold.

### Observation compression
`max_forecast_steps=24` caps the occupancy forecast to a 2-hour lookahead at
5-minute resolution, reducing the observation from ~1500 dims to ~174 dims.
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
        include_smart_tou_features: bool = False,
        smart_tou_feature_fn=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._latitude_deg = latitude_deg
        self._cloud_cover = cloud_cover
        self._infiltration_ach = infiltration_ach
        self._albedo = albedo
        self._include_smart_tou_features = include_smart_tou_features
        self._smart_tou_feature_fn = smart_tou_feature_fn
        self._smart_tou_feature_dim = self.max_zones + self.max_rooms
        if self._include_smart_tou_features:
            base_dim = self.observation_space.shape[0]
            self.observation_space = gym.spaces.Box(
                low=-1000.0,
                high=1000.0,
                shape=(base_dim + self._smart_tou_feature_dim,),
                dtype=np.float32,
            )

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

    def _augment_observation(self, obs: np.ndarray, info: dict):
        info = dict(info)
        info["base_observation_dim"] = int(obs.shape[0])
        info["smart_tou_feature_dim"] = self._smart_tou_feature_dim if self._include_smart_tou_features else 0
        if not self._include_smart_tou_features:
            return obs, info

        if self._smart_tou_feature_fn is None:
            raise RuntimeError("smart_tou teacher features requested without a feature function.")

        smart_tou_features = np.asarray(self._smart_tou_feature_fn(obs, info), dtype=np.float32)
        info["smart_tou_zone_mode_offset"] = int(obs.shape[0])
        info["smart_tou_room_intent_offset"] = int(obs.shape[0] + self.max_zones)
        return np.concatenate([obs.astype(np.float32), smart_tou_features]), info

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        obs, info = super().reset(seed=seed, options=options)
        return self._augment_observation(obs, info)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        obs, info = self._augment_observation(obs, info)
        return obs, reward, terminated, truncated, info

print("✅ PhysicsMultiRoomEnv defined")
""")

code("""\
# ── Create environment and explore spaces ─────────────────────────────────────
_ENV_KWARGS = dict(
    comfort_penalty_weight=5.0,
    energy_weight=1.0,
    switching_weight=0.05,
    invalid_source_penalty=2.0,
    max_forecast_steps=24,
    comfort_warmup_steps=24,
)

_probe = PhysicsMultiRoomEnv(scenarios=SCENARIOS, **_ENV_KWARGS)
obs0, info0 = _probe.reset(seed=0)

print("═" * 60)
print("Environment spaces")
print("═" * 60)
print(f"  Observation: Box{_probe.observation_space.shape} float32")
print(f"    = {_probe.max_rooms} rooms × 6 state features")
print(f"    + {_probe.max_rooms} rooms × {_probe.occupancy_forecast_horizon_steps} occupancy-forecast features")
print(f"    + {_probe.max_zones} zones × 3 features")
print(f"    + 7 global features  (T_out, elec, gas, hour_sin, hour_cos, next_1h_occ_max, next_2h_occ_max)")
print(f"    + Optional PPO teacher features when enabled: {_probe.max_zones} smart_tou zone-mode hints + {_probe.max_rooms} smart_tou room-intent hints")
print(f"  Action:      MultiDiscrete{tuple(_probe.action_space.nvec.tolist())}")
print(f"    = {_probe.max_zones} zone source modes  [AUTO, ELECTRIC, GAS_FURNACE]")
print(f"    + {_probe.max_rooms} room intents       [OFF, PROTECT, MAINTAIN, PREHEAT, RECOVER]")
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

| Policy | Zone mode | Room intent | Expected behaviour |
|--------|-----------|-------------|-------------------|
| **Always OFF** | auto | OFF | Cheapest energy, worst comfort |
| **ECO electric** | electric | PROTECT | Low energy, modest comfort |
| **COMFORT electric** | electric | baseline inferred intents | Electric-only intent dispatch |
| **Furnace COMFORT** | furnace (where available) | baseline inferred intents | Furnace zones run with shared deterministic demand |
| **Smart ToU** | adaptive hybrid | baseline inferred intents | Intent-aware cost heuristic |
""")

code("""\
def _make_eval_env(scenario: TrainingScenario, include_smart_tou_features: bool = False):
    \"\"\"Create a fresh wrapped env for a single scenario.\"\"\"
    env = PhysicsMultiRoomEnv(
        scenarios=[scenario],
        include_smart_tou_features=include_smart_tou_features,
        smart_tou_feature_fn=_smart_tou_feature_vector if include_smart_tou_features else None,
        **_ENV_KWARGS,
    )
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
        total_violation += info.get("comfort_violation", 0.0)
        log.append({
            "step": step_idx,
            "reward": reward,
            "total_cost": info.get("total_cost", 0.0),
            "comfort_violation": info.get("comfort_violation", 0.0),
            "electric_cost": info.get("electric_cost", 0.0),
            "gas_cost": info.get("gas_cost", 0.0),
            "zone_heat_sources": dict(info.get("zone_heat_sources", {})),
        })
        step_idx += 1
        if terminated or truncated:
            break
    return total_reward, total_cost, total_violation, log


def eval_policy_all_scenarios(name: str, policy_fn, include_smart_tou_features: bool = False):
    \"\"\"Evaluate policy once on each fixed scenario.\"\"\"
    results = {"name": name}
    for scenario in SCENARIOS:
        env = _make_eval_env(scenario, include_smart_tou_features=include_smart_tou_features)
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
    base_obs = obs[: info.get("base_observation_dim", len(obs))]
    max_rooms = info["max_rooms"]
    horizon = info["occupancy_forecast_horizon_steps"]
    room_feature_block = max_rooms * 6
    room_context = {}
    for idx, room_name in enumerate(info.get("room_names", [])):
        base = idx * 6
        forecast_base = room_feature_block + (idx * horizon)
        occ_now = float(base_obs[base + 3])
        future_occ = [float(base_obs[forecast_base + j]) for j in range(horizon)]
        room_context[room_name] = {
            "temp": float(base_obs[base + 0]),
            "target_min": float(base_obs[base + 1]),
            "target_max": float(base_obs[base + 2]),
            "forecast": [occ_now] + future_occ,
            "current_occ": occ_now,
            "last_action": float(base_obs[base + 4]),
            "target": 0.5 * (float(base_obs[base + 1]) + float(base_obs[base + 2])),
        }

    outside_temp = float(base_obs[-7])
    electricity_price = float(base_obs[-6])
    gas_price = float(base_obs[-5])
    return room_context, outside_temp, electricity_price, gas_price

def _baseline_room_intents(obs, info):
    room_context, outside_temp, electricity_price, _ = _obs_context(obs, info)
    intents = {}
    for room_name, ctx in room_context.items():
        decision = BASELINE_CONTROLLERS[room_name].compute_decision(
            current_temp=ctx["temp"],
            occupancy_forecast=ctx["forecast"],
            energy_prices=[electricity_price],
            current_action=ctx["last_action"],
            outside_temp=outside_temp,
            target_temp=ctx["target"],
        )
        intents[room_name] = decision.metadata.get("room_intent", RoomHeatingIntent.OFF.value)
    return intents

def policy_always_off(obs, info):
    return (
        [zone_source_mode_index(ZoneSourceMode.AUTO)] * info["max_zones"]
        + [room_intent_index(RoomHeatingIntent.OFF)] * info["max_rooms"]
    )

def policy_eco_electric(obs, info):
    return (
        [zone_source_mode_index(ZoneSourceMode.ELECTRIC)] * info["max_zones"]
        + [room_intent_index(RoomHeatingIntent.PROTECT)] * info["max_rooms"]
    )

def policy_comfort_electric(obs, info):
    room_intents = _baseline_room_intents(obs, info)
    return (
        [zone_source_mode_index(ZoneSourceMode.ELECTRIC)] * info["max_zones"]
        + [room_intent_index(room_intents.get(room_name, RoomHeatingIntent.OFF.value)) for room_name in info["room_names"]]
        + [room_intent_index(RoomHeatingIntent.OFF)] * (info["max_rooms"] - len(info["room_names"]))
    )

def policy_furnace_comfort(obs, info):
    \"\"\"Use furnace mode for furnace-equipped zones with baseline room intents.\"\"\"
    room_intents = _baseline_room_intents(obs, info)
    zone_src = {
        zone_name: ZoneSourceMode.GAS_FURNACE if info.get("zone_has_furnace", {}).get(zone_name, False) else ZoneSourceMode.ELECTRIC
        for zone_name in _active_zone_names(info)
    }
    return (
        [zone_source_mode_index(zone_src.get(zone_name, ZoneSourceMode.AUTO)) for zone_name in info["zone_names"]]
        + [zone_source_mode_index(ZoneSourceMode.AUTO)] * (info["max_zones"] - len(info["zone_names"]))
        + [room_intent_index(room_intents.get(room_name, RoomHeatingIntent.OFF.value)) for room_name in info["room_names"]]
        + [room_intent_index(RoomHeatingIntent.OFF)] * (info["max_rooms"] - len(info["room_names"]))
    )

def _smart_tou_decision(obs, info):
    \"\"\"Intent-aware heuristic hybrid controller using the shared zone cost logic.\"\"\"
    room_context, outside_temp, electricity_price, gas_price = _obs_context(obs, info)
    room_intents = {}
    for room_name, ctx in room_context.items():
        decision = BASELINE_CONTROLLERS[room_name].compute_decision(
            current_temp=ctx["temp"],
            occupancy_forecast=ctx["forecast"],
            energy_prices=[electricity_price],
            current_action=ctx["last_action"],
            outside_temp=outside_temp,
            target_temp=ctx["target"],
        )
        room_intents[room_name] = decision.metadata.get("room_intent", RoomHeatingIntent.OFF.value)

    zone_modes = {}
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
            room_intents={room_name: room_intents[room_name] for room_name in zone_rooms},
            zone_source_preference=ZoneSourceMode.AUTO,
        )
        zone_modes[zone_name] = (
            ZoneSourceMode.GAS_FURNACE
            if decision.heat_source == HeatSourceType.GAS_FURNACE
            else ZoneSourceMode.ELECTRIC
        )

    return zone_modes, room_intents

def _smart_tou_feature_vector(obs, info):
    zone_modes, room_intents = _smart_tou_decision(obs, info)
    return _zone_vector(
        {
            zone_name: zone_source_mode_feature_value(mode)
            for zone_name, mode in zone_modes.items()
        },
        info,
    ) + _room_vector(
        {
            room_name: room_intent_feature_value(intent)
            for room_name, intent in room_intents.items()
        },
        info,
    )

def policy_smart_tou(obs, info):
    zone_modes, room_intents = _smart_tou_decision(obs, info)
    return (
        [zone_source_mode_index(zone_modes.get(zone_name, ZoneSourceMode.AUTO)) for zone_name in info["zone_names"]]
        + [zone_source_mode_index(ZoneSourceMode.AUTO)] * (info["max_zones"] - len(info["zone_names"]))
        + [room_intent_index(room_intents.get(room_name, RoomHeatingIntent.OFF.value)) for room_name in info["room_names"]]
        + [room_intent_index(RoomHeatingIntent.OFF)] * (info["max_rooms"] - len(info["room_names"]))
    )

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
Stable-Baselines3, but run training through `scripts/train_opt_heating_policy.py`
instead of directly inside Jupyter. This avoids the Windows notebook
`SubprocVecEnv` worker crashes while still saving the trained model and
per-episode metrics back into `output/` for the rest of this notebook.

### Key PPO hyperparameters

| Param | Value | Rationale |
|-------|-------|-----------|
| `n_steps` | 288 | One 24-hour episode per environment rollout |
| `batch_size` | 128 | Mini-batches from the collected rollout |
| `n_epochs` | 10 | Re-use rollout data for 10 gradient updates |
| `gamma` | 0.999 | Long planning horizon with 5-minute control intervals |
| `ent_coef` | 0.02 | Encourages action diversity early; decays with entropy |
| `learning_rate` | 3e-4 | Standard Adam LR for PPO |

### Training environment
- `PhysicsMultiRoomEnv` with `max_forecast_steps=24` (2h occupancy lookahead), `comfort_warmup_steps=24`, and `preoccupancy_penalty_weight=30.0`.
- Workers start on staggered scenarios and then keep cycling round-robin across the scenario library.
- Each episode lasts exactly **288 steps** (one simulated day at 5-minute resolution).
- PPO observations append `smart_tou`'s recommended zone source mode and room intent as teacher features.
- PPO outputs **discrete** high-level actions:
  one source mode per zone (`AUTO`, `ELECTRIC`, `GAS_FURNACE`) and one thermal
  intent per room (`OFF`, `PROTECT`, `MAINTAIN`, `PREHEAT`, `RECOVER`).
- Shared deterministic control logic converts those intents into actual normalized
  room demand and applies the final gas-vs-electric decision.

### What OPT should learn
1. **Gas furnace during cold peaks** — furnace is cheaper than running all room heaters.
2. **Preheat bedrooms before 05:00/06:00** — family wakes up at 05:00–06:00; warm rooms = no penalty.
3. **Cheap overnight electricity** — favor `PROTECT` / `MAINTAIN` in the office during cheap hours (≤ 8 ¢/kWh).
4. **Reduce heating mid-day when unoccupied** — kitchen/living room empty 09:00–17:00 on weekdays.
""")

code("""\
N_ENVS = 4
TOTAL_TIMESTEPS = 100_000
REPO_ROOT = Path(os.path.abspath(".."))
OUTPUT_DIR = REPO_ROOT / "output"
TRAINING_SCRIPT = REPO_ROOT / "scripts" / "train_opt_heating_policy.py"
MODEL_PATH = OUTPUT_DIR / "opt_policy_ppo"
MODEL_ZIP_PATH = MODEL_PATH.with_suffix(".zip")
TRAINING_SUMMARY_PATH = OUTPUT_DIR / "opt_policy_training_summary.json"

print(f"Training script : {TRAINING_SCRIPT}")
print(f"Model artifact  : {MODEL_ZIP_PATH}")
print(f"Metrics artifact: {TRAINING_SUMMARY_PATH}")
""")

code("""\
def load_training_artifacts():
    if not MODEL_ZIP_PATH.exists():
        raise FileNotFoundError(f"Missing trained model: {MODEL_ZIP_PATH}")
    if not TRAINING_SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Missing training summary: {TRAINING_SUMMARY_PATH}")

    with TRAINING_SUMMARY_PATH.open("r", encoding="utf-8") as fh:
        training_summary = json.load(fh)

    normalize_path = Path(
        training_summary.get(
            "normalize_path",
            str(OUTPUT_DIR / "opt_policy_vec_normalize.pkl"),
        )
    )
    if not normalize_path.exists():
        raise FileNotFoundError(f"Missing VecNormalize stats: {normalize_path}")

    model = PPO.load(str(MODEL_PATH), device="cpu")
    return training_summary, model, normalize_path

print("Ready to launch external PPO training and load saved artifacts, including VecNormalize stats.")
""")

code("""\
# ── Train externally, then load artifacts back into the notebook ─────────────
FORCE_RETRAIN = False

if FORCE_RETRAIN or not MODEL_ZIP_PATH.exists() or not TRAINING_SUMMARY_PATH.exists():
    cmd = [
        sys.executable,
        str(TRAINING_SCRIPT),
        "--timesteps", str(TOTAL_TIMESTEPS),
        "--n-envs", str(N_ENVS),
        "--output-dir", str(OUTPUT_DIR),
        "--no-progress-bar",
        "--progress-interval", "5000",
    ]
    print("Launching external trainer:")
    print("  " + " ".join(cmd))
    process = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"External trainer failed with exit code {return_code}")
else:
    print("Using existing training artifacts. Set FORCE_RETRAIN = True to rerun PPO.")

training_summary, model, NORMALIZE_PATH = load_training_artifacts()
print(f"\\n✅ Training artifacts loaded")
print(f"   VecEnv: {training_summary['vec_env_type']}")
print(f"   Device: {training_summary['device']}")
print(f"   Parameters: {training_summary['parameter_count']:,}")
print(f"   Elapsed: {training_summary['elapsed_seconds']:.1f}s")
print(f"   Episodes logged: {len(training_summary['episode_rewards'])}")
if training_summary["episode_rewards"]:
    last_k = 200
    r_last = np.mean(training_summary["episode_rewards"][-last_k:])
    c_last = np.mean(training_summary["episode_costs"][-last_k:])
    v_last = np.mean(training_summary["episode_violations"][-last_k:])
    print(f"   Last {last_k} episodes — raw_reward={r_last:.2f}  cost=${c_last:.4f}  violation={v_last:.3f}")
if training_summary.get("episode_rewards_normalized"):
    rn_last = np.mean(training_summary["episode_rewards_normalized"][-last_k:])
    print(f"   Last {last_k} episodes — normalized_reward={rn_last:.2f}")
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
ep_rewards = training_summary["episode_rewards"]
ep_costs = training_summary["episode_costs"]
ep_violations = training_summary["episode_violations"]
ep_idx = np.arange(len(ep_rewards))

METRICS = [
    (ep_rewards,    "Episode Reward (raw env return, ↑ better)", "#1565C0"),
    (ep_costs,      "Episode Energy Cost $ (↓ better)",   "#AD1457"),
    (ep_violations, "Reported Comfort Violation (↓ better)", "#2E7D32"),
]

for ax, (data, title, color) in zip(axes, METRICS):
    ax.plot(ep_idx, data, alpha=0.20, color=color, lw=0.5)
    smooth = _smooth(data, window=min(100, max(10, len(data) // 20)))
    ax.plot(ep_idx, smooth, color=color, lw=2, label="Smoothed (100-ep window)")
    ax.set_ylabel(title, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper right" if "Reward" in title else "upper left")

axes[-1].set_xlabel("Training episode", fontsize=11)
plt.suptitle(f"PPO Training Progress — {training_summary['total_timesteps']:,} timesteps, {training_summary['n_envs']} parallel envs",
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
# ── VecNormalize-aware trained-policy evaluation ──────────────────────────────
def _make_trained_eval_vec_env(scenario: TrainingScenario):
    vec_env = DummyVecEnv([
        lambda: _make_eval_env(scenario, include_smart_tou_features=True)
    ])
    vec_env = VecNormalize.load(str(NORMALIZE_PATH), vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    return vec_env


def _evaluate_trained_opt_scenario(scenario: TrainingScenario):
    vec_env = _make_trained_eval_vec_env(scenario)
    obs = vec_env.reset()
    total_reward = total_cost = total_violation = 0.0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = vec_env.step(action)
        info = infos[0]
        total_reward += float(rewards[0])
        total_cost += float(info.get("total_cost", 0.0))
        total_violation += float(info.get("comfort_violation", 0.0))
        if bool(dones[0]):
            break
    vec_env.close()
    return total_reward, total_cost, total_violation


def eval_trained_opt_all_scenarios(name: str = "trained_opt"):
    results = {"name": name}
    for scenario in SCENARIOS:
        r, c, v = _evaluate_trained_opt_scenario(scenario)
        results[scenario.name] = {
            "reward": r,
            "cost": c,
            "violation": v,
        }
    return results


# ── Evaluate ──────────────────────────────────────────────────────────────────
print("Evaluating trained OPT policy with saved VecNormalize stats …")
trained_results = eval_trained_opt_all_scenarios()

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

def full_rollout(policy_fn, scenario_name: str, seed: int = 0, include_smart_tou_features: bool = False):
    \"\"\"Run one deterministic episode; return per-step records.\"\"\"
    target = next(s for s in SCENARIOS if s.name == scenario_name)
    env = PhysicsMultiRoomEnv(
        scenarios=[target],
        include_smart_tou_features=include_smart_tou_features,
        smart_tou_feature_fn=_smart_tou_feature_vector if include_smart_tou_features else None,
        **_ENV_KWARGS,
    )

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
        step_violations.append(info.get("comfort_violation", 0.0))

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

def full_rollout_trained_opt(scenario_name: str, seed: int = 0):
    \"\"\"Run one deterministic episode for the PPO policy using saved VecNormalize stats.\"\"\"
    target = next(s for s in SCENARIOS if s.name == scenario_name)
    vec_env = _make_trained_eval_vec_env(target)
    raw_env = vec_env.venv.envs[0]
    obs = vec_env.reset()

    room_names_sorted = sorted(target.room_configs.keys())
    zone_names_sorted = sorted(target.zone_configs.keys())
    temps   = {r: [target.initial_temperatures[r]] for r in room_names_sorted}
    actions = {r: [] for r in room_names_sorted}
    sources = {z: [] for z in zone_names_sorted}
    step_costs_elec = []
    step_costs_gas  = []
    step_violations = []
    elapsed_hours = np.arange(target.horizon_steps) * (target.step_minutes / 60.0)

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = vec_env.step(action)
        info = infos[0]

        for room_name in room_names_sorted:
            temps[room_name].append(float(raw_env._current_state.room_temperatures[room_name]))
            actions[room_name].append(float(raw_env._last_effective_actions[room_name]))

        zone_src_info = info.get("zone_heat_sources", {})
        for zone_name in zone_names_sorted:
            sources[zone_name].append(
                1 if zone_src_info.get(zone_name, "electric") == "gas_furnace" else 0
            )

        step_costs_elec.append(info.get("electric_cost", 0.0))
        step_costs_gas.append(info.get("gas_cost", 0.0))
        step_violations.append(info.get("comfort_violation", 0.0))

        if bool(dones[0]):
            break

    vec_env.close()
    return {
        "room_names": room_names_sorted,
        "zone_names": zone_names_sorted,
        "temps": temps,
        "actions": actions,
        "sources": sources,
        "elapsed_hours": elapsed_hours,
        "elec_cost": step_costs_elec,
        "gas_cost":  step_costs_gas,
        "violations": step_violations,
    }

rollout_opt  = full_rollout_trained_opt("winter_workday")
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
print(f"Model artifact already saved by the training script: {MODEL_ZIP_PATH}")
loaded_model = PPO.load(str(MODEL_PATH), device="cpu")
loaded_vec_env = _make_trained_eval_vec_env(winter_workday)
obs_test = loaded_vec_env.reset()
a, _ = loaded_model.predict(obs_test, deterministic=True)
loaded_vec_env.close()
print(f"✅ Reload OK with VecNormalize — sample action: {a[0].tolist()}")
""")

# ─────────────────────────────────────────────────────────────────────────────
# CONCLUSIONS
# ─────────────────────────────────────────────────────────────────────────────
md("""\
## 🏁 11. Summary & Next Steps

### What was trained
A **PPO policy** for a 3-zone, 5-room family house that jointly decides:
- **Zone source mode** (`AUTO`, `ELECTRIC`, or `GAS_FURNACE`, per zone, every 5 minutes)
- **Room thermal intent** (`OFF`, `PROTECT`, `MAINTAIN`, `PREHEAT`, or `RECOVER`, per room, every 5 minutes)

Those high-level actions are then resolved by shared deterministic control logic into
the actual normalized room heat command and the final gas-vs-electric actuation.

### Reward design
| Component | Weight | Effect |
|-----------|--------|--------|
| Energy cost (electric + gas) | 1.0 | Agent minimises energy spend |
| Comfort violation (occ × ΔT) | 5.0 | Penalises cold rooms during occupancy |
| Pre-occupancy penalty (wrapper) | 30.0 / °C | Extra signal to preheat *before* arrival |
| Switching penalty | 0.05 | Smooth action transitions |

### Target behaviours to check during evaluation
- Turns on Sleeping-zone furnace around **03:00–05:00** before the family wakes up.
- Switches Main-zone furnace off mid-day when kitchen/living room are empty.
- Uses cheap overnight electricity (≤ 8 ¢/kWh) for the home office preheat.
- On spring days, often prefers electric over furnace when outdoor temps are mild.

### Next steps
1. **Longer training** — increase `TOTAL_TIMESTEPS` only if the learning curves still improve beyond 100k.
2. **Checkpoint selection** — compare multiple saved checkpoints on the deterministic scenario suite instead of trusting the latest one by default.
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

print(f"Notebook written: {out_path}")
print(f"   Cells: {len(cells)}")
