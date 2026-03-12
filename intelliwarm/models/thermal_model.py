"""
Thermal model module.

Two implementations are provided:

``RoomThermalModel`` – legacy first-order linear model (kept for backward
compatibility and for rapidly testing new control logic).

``PhysicsRoomThermalModel`` – lumped thermal capacitance model using real
physical parameters (capacitance C in kJ/K, conductance UA in W/K, HVAC
power, solar aperture, and occupant heat gain).  Uses sub-step Euler
integration so hourly simulation steps remain accurate.
"""

import logging
import math
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Solar irradiance utility
# ---------------------------------------------------------------------------

def solar_irradiance_wm2(
    timestamp: datetime,
    latitude_deg: float = 40.0,
    cloud_cover: float = 0.0,
) -> float:
    """
    Estimate global horizontal irradiance (W/m²) using a simplified
    astronomical clear-sky model with cloud correction.

    Accuracy: within ~10% of measured data for typical mid-latitude sites on
    clear days.  Suitable for HVAC simulation where sub-percent accuracy is
    not required but daily and seasonal patterns must be correct.

    Typical peak values:
      * Winter (Jan, 40°N):  ~350–450 W/m²
      * Summer (Jul, 40°N):  ~800–950 W/m²

    Args:
        timestamp:     Local solar time (assumed close to civil time).
        latitude_deg:  Site latitude in degrees north.
        cloud_cover:   Fraction of sky covered by cloud (0 = clear, 1 = overcast).

    Returns:
        Global horizontal irradiance in W/m² (≥ 0).
    """
    doy = timestamp.timetuple().tm_yday

    # Solar declination (radians)
    decl_rad = math.radians(
        23.45 * math.sin(math.radians(360.0 * (doy - 81) / 365.0))
    )
    lat_rad = math.radians(latitude_deg)

    # Hour angle (15° per hour from solar noon; approximate local solar time)
    solar_hour = timestamp.hour + timestamp.minute / 60.0
    hour_angle_rad = math.radians(15.0 * (solar_hour - 12.0))

    # Solar elevation
    sin_elev = (
        math.sin(lat_rad) * math.sin(decl_rad)
        + math.cos(lat_rad) * math.cos(decl_rad) * math.cos(hour_angle_rad)
    )
    if sin_elev <= 0.0:
        return 0.0

    elev_deg = math.degrees(math.asin(min(1.0, sin_elev)))

    # Kasten air-mass formula
    am = 1.0 / (sin_elev + 0.50572 * (elev_deg + 6.07995) ** -1.6364)

    # Clear-sky transmittance (Bird & Hulstrom simplified, broadband)
    transmittance = 0.7 ** (am ** 0.678)

    # Eccentricity correction (Earth–Sun distance)
    e0 = 1.0 + 0.033 * math.cos(2.0 * math.pi * doy / 365.0)

    ghi_clear = 1361.0 * e0 * transmittance * sin_elev

    # Kasten–Czeplak cloud-cover correction
    cf = max(0.0, min(1.0, float(cloud_cover)))
    cloud_factor = 1.0 - 0.75 * (cf ** 3.4)

    return max(0.0, ghi_clear * cloud_factor)


def sol_rad_tilt_wm2(
    timestamp: datetime,
    latitude_deg: float = 40.0,
    surface_tilt_deg: float = 90.0,
    surface_azimuth_deg: float = 0.0,
    cloud_cover: float = 0.0,
    albedo: float = 0.20,
) -> float:
    """
    Estimate solar irradiance on a tilted surface [W/m²].

    Combines the built-in clear-sky GHI model with the **Duffie-Beckman**
    incidence-angle model (*Solar Engineering of Thermal Processes*, 5th ed.,
    eq. 1.6.2) to decompose irradiance into:

    * **Direct (beam)** — ``DNI × cos θ`` where θ is the angle between the
      surface normal and the sun direction.
    * **Diffuse (sky)** — isotropic sky model:
      ``DHI × (1 + cos β) / 2``.
    * **Reflected (ground)** — ``GHI × albedo × (1 − cos β) / 2``.

    The beam/diffuse split uses a simple cloud-cover-based diffuse fraction
    (15 % on clear days, 100 % on fully overcast days), consistent with the
    Erbs correlation at the extremes.

    Surface orientation convention (Duffie 2020 / dm4bem):

    ============== ===============================================================
    Parameter       Description
    ============== ===============================================================
    tilt = 0°       Horizontal (skylight faces up)
    tilt = 90°      Vertical (wall or window)
    azimuth = 0°    South-facing
    azimuth = +90°  West-facing
    azimuth = −90°  East-facing
    azimuth = 180°  North-facing
    ============== ===============================================================

    Args:
        timestamp:            Local solar time.
        latitude_deg:         Site latitude [°N].
        surface_tilt_deg:     Surface tilt from horizontal [°].  0=horizontal,
                              90=vertical.
        surface_azimuth_deg:  Surface azimuth [°].  0=south, +90=west, −90=east.
        cloud_cover:          Fractional sky cloud cover [0–1].
        albedo:               Ground reflectance [0–1].  Typical values:
                              0.20 = grass/asphalt, 0.60 = fresh snow.

    Returns:
        Total solar irradiance on the tilted surface [W/m²] (≥ 0).

    References:
        J. A. Duffie, W. A. Beckman, N. Blair (2020) *Solar Engineering of
        Thermal Processes*, 5th ed., Wiley, eq. 1.6.2.
        Réglementation Thermique 2005 (Th-CE 2005), §11.2.1.

        C. Ghiaus (2021) `dm4bem
        <https://github.com/cghiaus/dm4bem>`_, ``sol_rad_tilt_surf()``.
    """
    # Step 1: GHI (cloud-corrected global horizontal irradiance)
    ghi = solar_irradiance_wm2(timestamp, latitude_deg, cloud_cover)
    if ghi <= 0.0:
        return 0.0

    # Step 2: Solar geometry
    doy = timestamp.timetuple().tm_yday
    d = math.radians(23.45 * math.sin(math.radians(360.0 * (doy - 81) / 365.0)))
    L = math.radians(latitude_deg)
    solar_hour = timestamp.hour + timestamp.minute / 60.0
    h = math.radians(15.0 * (solar_hour - 12.0))  # hour angle; positive = afternoon

    B = math.radians(surface_tilt_deg)
    Z = math.radians(surface_azimuth_deg)  # 0=south, +90=west, -90=east

    # Solar elevation (needed to project DNI onto horizontal)
    sin_elev = math.sin(L) * math.sin(d) + math.cos(L) * math.cos(d) * math.cos(h)
    if sin_elev <= 1e-6:
        return 0.0  # sun below horizon

    # Step 3: Incidence angle on tilted surface — Duffie-Beckman eq. 1.6.2
    cos_theta = (
        math.sin(d) * math.sin(L) * math.cos(B)
        - math.sin(d) * math.cos(L) * math.sin(B) * math.cos(Z)
        + math.cos(d) * math.cos(L) * math.cos(B) * math.cos(h)
        + math.cos(d) * math.sin(L) * math.sin(B) * math.cos(Z) * math.cos(h)
        + math.cos(d) * math.sin(B) * math.sin(Z) * math.sin(h)
    )
    cos_theta = max(0.0, cos_theta)  # clamp: surface in shadow if < 0

    # Step 4: Beam/diffuse split (cloud-cover-based diffuse fraction)
    # Clear day (cf=0): 15 % diffuse.  Overcast (cf=1): 100 % diffuse.
    cf = max(0.0, min(1.0, float(cloud_cover)))
    diffuse_frac = 0.15 + 0.85 * cf
    dhi = ghi * diffuse_frac                    # diffuse horizontal irradiance
    dni = (ghi * (1.0 - diffuse_frac)) / sin_elev  # direct normal irradiance

    # Step 5: Three components on tilted surface (Th-CE 2005 §11.2.1 / Duffie)
    i_direct = dni * cos_theta                         # beam component
    i_diffuse = dhi * (1.0 + math.cos(B)) / 2.0       # isotropic sky diffuse
    i_reflected = ghi * albedo * (1.0 - math.cos(B)) / 2.0  # ground-reflected

    return max(0.0, i_direct + i_diffuse + i_reflected)




class RoomThermalModel:
    """
    Models room thermal dynamics (legacy first-order linear model).

    Equation:
        T(t+1) = T(t) + α*H(t) - β*(T(t) - T_outside)

    Where:
        T(t)  = room temperature
        H(t)  = heating action (0-1)
        α     = heating efficiency coefficient  (°C/hr at full power)
        β     = heat loss coefficient (fraction of temp-difference lost per hour)

    This model is retained for fast unit tests and backward compatibility.
    For physics-accurate simulation use ``PhysicsRoomThermalModel``.
    """

    def __init__(self, room_name: str, alpha: float = 0.1, beta: float = 0.05):
        self.room_name = room_name
        self.alpha = alpha
        self.beta = beta
        self.logger = logging.getLogger("IntelliWarm.ThermalModel")

    def step(
        self,
        current_temp: float,
        outside_temp: float,
        heating_power: float,
        dt_minutes: int = 60,
        *,
        solar_irradiance_w_m2: float = 0.0,  # accepted but unused in legacy model
        occupancy: float = 0.0,              # accepted but unused in legacy model
        furnace_heating_power: float = 0.0,  # accepted but unused in legacy model
    ) -> float:
        """Advance the thermal model by one timestep."""
        dt_scale = dt_minutes / 60.0
        temperature_delta = (
            (self.alpha * heating_power) - (self.beta * (current_temp - outside_temp))
        ) * dt_scale
        return current_temp + temperature_delta

    def simulate(
        self,
        initial_temp: float,
        forecast_inputs: Iterable[Dict[str, float]],
        dt_minutes: int = 60,
    ) -> List[float]:
        """Simulate temperature evolution for a sequence of forecast inputs."""
        temperatures: List[float] = []
        current_temp = initial_temp

        for forecast_step in forecast_inputs:
            current_temp = self.step(
                current_temp=current_temp,
                outside_temp=float(forecast_step["outdoor_temp"]),
                heating_power=float(forecast_step["heating_power"]),
                dt_minutes=dt_minutes,
            )
            temperatures.append(current_temp)

        return temperatures
    
    def predict_temperature(
        self,
        current_temp: float,
        outside_temp: float,
        heating_actions: List[float],
        hours: int = 24
    ) -> List[float]:
        """
        Predict future temperatures
        
        Args:
            current_temp: Current room temperature (°C)
            outside_temp: Outside temperature (°C)
            heating_actions: List of heating actions [0-1] for each hour
            hours: Number of hours to predict
            
        Returns:
            List of predicted temperatures
        """
        forecast_inputs = [
            {"outdoor_temp": outside_temp, "heating_power": heating_actions[h]}
            for h in range(min(hours, len(heating_actions)))
        ]
        return self.simulate(current_temp, forecast_inputs, dt_minutes=60)
    
    def estimate_parameters(self, historical_data: List[Tuple[float, float, float]]):
        """
        Estimate α and β from historical data using least squares
        
        Args:
            historical_data: List of (T_current, H_action, outside_temp, T_next) tuples
        """
        if len(historical_data) < 10:
            self.logger.warning("Insufficient data for parameter estimation")
            return
        
        try:
            # Extract data
            T_current = np.array([d[0] for d in historical_data])
            H_action = np.array([d[1] for d in historical_data])
            T_outside = np.array([d[2] for d in historical_data])
            T_next = np.array([d[3] for d in historical_data])
            
            # Construct the system: T(t+1) - T(t) = α*H - β*(T - T_out)
            # Rearrange: dT = α*H - β*T + β*T_out
            dT = T_next - T_current
            
            # Design matrix: [H, -(T - T_outside)]
            A = np.column_stack([H_action, -(T_current - T_outside)])
            
            # Solve least squares: A * [α, β] = dT
            solution = np.linalg.lstsq(A, dT, rcond=None)[0]
            
            new_alpha = max(0, solution[0])
            new_beta = max(0, solution[1])
            
            self.alpha = new_alpha
            self.beta = new_beta
            
            self.logger.info(f"Parameters updated: α={self.alpha:.4f}, β={self.beta:.4f}")
        
        except Exception as e:
            self.logger.error(f"Parameter estimation failed: {e}")
    
    def get_parameters(self) -> Tuple[float, float]:
        """Return current model parameters"""
        return self.alpha, self.beta


# ---------------------------------------------------------------------------
# Physics-accurate lumped thermal capacitance model
# ---------------------------------------------------------------------------

class PhysicsRoomThermalModel:
    """
    Lumped thermal capacitance model for a single room.

    Governing equation (all quantities in SI units)::

        C [J/K] * dT/dt [K/s] = UA [W/K] * (T_out - T)
                                + Q_electric [W]
                                + Q_furnace [W]
                                + Q_solar [W]
                                + Q_occ [W]

    where::

        Q_electric = electric_power_w  * electric_heating_power_fraction
        Q_furnace  = furnace_power_w   * furnace_heating_power_fraction
        Q_solar    = solar_aperture_m2 * solar_irradiance_w_m2
        Q_occ      = occupant_gain_w   * occupancy_fraction

    Both heat sources are accepted simultaneously, but the hybrid controller
    guarantees that only one is active at a time for a given zone — this
    model simply sums whatever is provided.

    Integration uses sub-step forward Euler (``N_SUBSTEPS`` per call) to
    maintain accuracy over 60-minute simulation timesteps without the
    overhead of a stiff solver.

    Typical residential parameter ranges
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ======================== ============= ===================================
    Parameter                Typical range Notes
    ======================== ============= ===================================
    C (kJ/K)                 200 – 800     Air, walls, furnishings
    UA (W/K)                 5  – 40       Higher → leakier envelope
    electric_power_w (W)     800 – 3000    Rated electric space-heater output
    furnace_power_w (W)      3000 – 15000  Per-room share of zone furnace
    solar_aperture_m2        0.3 – 3.0     South-facing glazing equivalent
    occupant_gain_w          70 – 120      ASHRAE seated/standing metabolic
    ======================== ============= ===================================
    """

    N_SUBSTEPS: int = 6  # sub-steps per timestep (10 min at dt=60 min)

    def __init__(
        self,
        room_name: str,
        thermal_capacitance_kj_k: float,
        conductance_ua_w_k: float,
        hvac_power_w: float,
        solar_aperture_m2: float = 0.0,
        occupant_gain_w: float = 80.0,
        furnace_power_w: float = 0.0,
        infiltration_ua_w_k: float = 0.0,
        solar_tilt_deg: float = 90.0,
        solar_azimuth_deg: float = 0.0,
    ):
        """
        Args:
            room_name:                Identifier used for logging.
            thermal_capacitance_kj_k: Effective thermal mass C [kJ/K].
            conductance_ua_w_k:       Opaque envelope conductance UA [W/K].
                                      Represents conduction + surface convection
                                      through walls, ceiling, floor.
            hvac_power_w:             Rated electric space-heater output [W].
                                      (Accepted as ``hvac_power_w`` for
                                      backward compatibility; stored as
                                      ``electric_power_w`` internally.)
            solar_aperture_m2:        Effective glazing area [m²].
            occupant_gain_w:          Metabolic heat per occupant [W].
            furnace_power_w:          Per-room share of zone furnace output [W].
            infiltration_ua_w_k:      Infiltration thermal conductance [W/K].
                                      ``G_inf = ṁ_air × c_p_air`` where
                                      ``ṁ_air = ACH × V × ρ_air / 3600``.
                                      Adds to ``conductance_ua_w_k`` in the
                                      heat-loss term.  See ``effective_ua_w_k``.
            solar_tilt_deg:           Tilt of the solar aperture (window) from
                                      horizontal [°].  0=skylight, 90=vertical
                                      wall/window.  Used by
                                      :class:`HouseSimulator` to compute the
                                      correct irradiance for this room.
            solar_azimuth_deg:        Azimuth of the solar aperture [°] using
                                      the Duffie convention: 0=south, +90=west,
                                      −90=east, 180=north.
        """
        if thermal_capacitance_kj_k <= 0:
            raise ValueError("thermal_capacitance_kj_k must be positive")
        if conductance_ua_w_k < 0:
            raise ValueError("conductance_ua_w_k must be non-negative")
        if hvac_power_w < 0:
            raise ValueError("hvac_power_w must be non-negative")
        if furnace_power_w < 0:
            raise ValueError("furnace_power_w must be non-negative")
        if infiltration_ua_w_k < 0:
            raise ValueError("infiltration_ua_w_k must be non-negative")

        self.room_name = room_name
        self.thermal_capacitance_kj_k = thermal_capacitance_kj_k
        self.conductance_ua_w_k = conductance_ua_w_k
        self.electric_power_w = hvac_power_w  # electric space-heater rated output
        self.furnace_power_w = furnace_power_w  # per-room furnace contribution
        self.solar_aperture_m2 = solar_aperture_m2
        self.occupant_gain_w = occupant_gain_w
        self.infiltration_ua_w_k = infiltration_ua_w_k
        self.solar_tilt_deg = solar_tilt_deg
        self.solar_azimuth_deg = solar_azimuth_deg
        self.logger = logging.getLogger("IntelliWarm.PhysicsThermalModel")

    @property
    def hvac_power_w(self) -> float:
        """Backward-compatible alias — returns the electric space-heater power [W]."""
        return self.electric_power_w

    @property
    def effective_ua_w_k(self) -> float:
        """
        Total envelope conductance including air infiltration [W/K].

        This is the UA value used in the heat-balance equation::

            Q_loss = effective_ua_w_k × (T_room − T_outdoor)

        Separating the opaque-envelope and infiltration contributions allows
        independent calibration from blower-door tests (infiltration) and
        U-value audits (envelope).
        """
        return self.conductance_ua_w_k + self.infiltration_ua_w_k

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_room_config(
        cls,
        room_config: object,
        zone_config: Optional[object] = None,
        num_zone_rooms: int = 1,
        infiltration_ach: float = 0.5,
        solar_tilt_deg: float = 90.0,
        solar_azimuth_deg: float = 0.0,
    ) -> "PhysicsRoomThermalModel":
        """
        Derive physics parameters from a :class:`~intelliwarm.data.RoomConfig`
        (and optionally a :class:`~intelliwarm.data.ZoneConfig`) using heuristic
        estimates that reproduce realistic HVAC dynamics.

        Derivation
        ----------
        1. **Room volume** — ``heater_capacity / 45 W·m⁻³``
           (45 W/m³ is a typical residential design heat-load at −10 °C
           outside, 21 °C inside).
        2. **C** — ``volume × ρ_air × c_p_air × 10``
           (×10 multiplier for walls, floor, ceiling, and furnishings;
           consistent with ASHRAE 90.1).
        3. **UA (envelope)** — ``volume^(2/3) × 1.5 W/(K·m^(2/3))``
           (calibrated against RSI-3.5 walls and double-pane glazing with
           indoor h_i = 7.7 W/(m²·K), outdoor h_o = 25 W/(m²·K) per
           Dal Zotto et al. 2014 / dm4bem Annex 1).
        4. **Infiltration UA** — ``ṁ_air × c_p_air``
           where ``ṁ_air = ACH × V × ρ_air / 3600 [kg/s]``
           (ASHRAE 62.2 residential default: ACH = 0.5 h⁻¹;
           typical range 0.3–1.5 h⁻¹).
        5. **electric_power_w** — ``heater_capacity`` (rated output).
        6. **furnace_power_w** — from ``zone_config`` when the zone has a
           furnace::

               BTU/hr × 0.29307 W/(BTU/hr) × AFUE / num_zone_rooms

        7. **Solar aperture** — ``max(0.3, volume^(2/3) × 0.08)`` m².
        8. **solar_tilt_deg / solar_azimuth_deg** — passed through unchanged;
           default is a vertical south-facing window (90°, 0°).

        Args:
            room_config:         A :class:`~intelliwarm.data.RoomConfig`.
            zone_config:         Optional :class:`~intelliwarm.data.ZoneConfig`.
            num_zone_rooms:      Rooms sharing the zone furnace.
            infiltration_ach:    Air changes per hour [h⁻¹] for infiltration
                                 conductance calculation.
            solar_tilt_deg:      Window tilt [°].  0=horizontal, 90=vertical.
            solar_azimuth_deg:   Window azimuth [°].  0=south, +90=west.
        """
        heater_capacity = float(getattr(room_config, "heater_capacity", 1500.0))
        room_id = str(getattr(room_config, "room_id", "room"))

        volume_m3 = heater_capacity / 45.0
        c_kj_k = volume_m3 * 1.2 * 1.005 * 10.0
        ua_w_k = (volume_m3 ** (2.0 / 3.0)) * 1.5
        solar_ap_m2 = max(0.3, (volume_m3 ** (2.0 / 3.0)) * 0.08)

        # Infiltration conductance (dm4bem Annex 1 / ASHRAE 62.2)
        m_dot_kg_s = volume_m3 * 1.2 * infiltration_ach / 3_600.0  # kg/s
        infiltration_ua = m_dot_kg_s * 1_005.0  # W/K  (c_p_air ≈ 1005 J/(kg·K))

        # Per-room furnace output
        furnace_power_w = 0.0
        if zone_config is not None and getattr(zone_config, "has_furnace", False):
            btu_per_hour = float(getattr(zone_config, "furnace_btu_per_hour", 60_000.0))
            efficiency = float(getattr(zone_config, "furnace_efficiency", 0.80))
            furnace_total_w = btu_per_hour * 0.29307 * efficiency
            furnace_power_w = furnace_total_w / max(1, num_zone_rooms)

        return cls(
            room_name=room_id,
            thermal_capacitance_kj_k=c_kj_k,
            conductance_ua_w_k=ua_w_k,
            hvac_power_w=heater_capacity,
            solar_aperture_m2=solar_ap_m2,
            occupant_gain_w=80.0,
            furnace_power_w=furnace_power_w,
            infiltration_ua_w_k=infiltration_ua,
            solar_tilt_deg=solar_tilt_deg,
            solar_azimuth_deg=solar_azimuth_deg,
        )

    # ------------------------------------------------------------------
    # Core integration
    # ------------------------------------------------------------------

    def step(
        self,
        current_temp: float,
        outside_temp: float,
        heating_power: float,
        dt_minutes: int = 60,
        *,
        solar_irradiance_w_m2: float = 0.0,
        occupancy: float = 0.0,
        furnace_heating_power: float = 0.0,
    ) -> float:
        """
        Advance the room temperature by one timestep.

        Args:
            current_temp:           Room temperature at start of step [°C].
            outside_temp:           Outdoor dry-bulb temperature [°C].
            heating_power:          Normalised **electric** heater fraction [0–1].
            dt_minutes:             Timestep duration [min].
            solar_irradiance_w_m2:  Irradiance on the solar aperture surface
                                      [W/m²].  When called from
                                      :class:`HouseSimulator`, this value has
                                      already been corrected for surface tilt
                                      and azimuth via :func:`sol_rad_tilt_wm2`.
            occupancy:              Occupancy fraction [0–1].
            furnace_heating_power:  Normalised **gas furnace** fraction [0–1].
                                    The hybrid controller guarantees this and
                                    ``heating_power`` are not both > 0 for
                                    the same zone simultaneously.

        Returns:
            Room temperature at end of step [°C].
        """
        C_j_k = self.thermal_capacitance_kj_k * 1_000.0  # kJ/K → J/K
        dt_sec = dt_minutes * 60.0
        dt_sub = dt_sec / self.N_SUBSTEPS

        Q_electric = self.electric_power_w * max(0.0, min(1.0, heating_power))
        Q_furnace = self.furnace_power_w * max(0.0, min(1.0, furnace_heating_power))
        Q_solar = self.solar_aperture_m2 * max(0.0, solar_irradiance_w_m2)
        Q_occ = self.occupant_gain_w * max(0.0, min(1.0, occupancy))

        T = current_temp
        for _ in range(self.N_SUBSTEPS):
            Q_loss = self.effective_ua_w_k * (T - outside_temp)
            dT = (Q_electric + Q_furnace + Q_solar + Q_occ - Q_loss) / C_j_k * dt_sub
            T += dT

        return T

    def simulate(
        self,
        initial_temp: float,
        forecast_inputs: Iterable[Dict[str, float]],
        dt_minutes: int = 60,
    ) -> List[float]:
        """Simulate temperature evolution for a sequence of forecast inputs.

        Each entry in *forecast_inputs* may include:
        ``outdoor_temp``, ``heating_power``, ``furnace_heating_power``,
        ``solar_irradiance_w_m2``, ``occupancy``.
        """
        temperatures: List[float] = []
        current_temp = initial_temp

        for step_data in forecast_inputs:
            current_temp = self.step(
                current_temp=current_temp,
                outside_temp=float(step_data.get("outdoor_temp", 0.0)),
                heating_power=float(step_data.get("heating_power", 0.0)),
                dt_minutes=dt_minutes,
                solar_irradiance_w_m2=float(step_data.get("solar_irradiance_w_m2", 0.0)),
                occupancy=float(step_data.get("occupancy", 0.0)),
                furnace_heating_power=float(step_data.get("furnace_heating_power", 0.0)),
            )
            temperatures.append(current_temp)

        return temperatures

    @property
    def time_constant_hours(self) -> float:
        """RC time constant τ = C / UA_eff [hours], including infiltration."""
        if self.effective_ua_w_k == 0:
            return float("inf")
        return (self.thermal_capacitance_kj_k * 1_000.0) / (self.effective_ua_w_k * 3_600.0)

    @property
    def steady_state_delta_t(self) -> Optional[float]:
        """
        Maximum temperature rise above outdoor when **both** heat sources run
        at full output simultaneously [°C].  Returns ``None`` if effective UA
        is zero.

        Uses :attr:`effective_ua_w_k` (envelope + infiltration), so a room
        with more infiltration has a smaller maximum ΔT for the same heater
        capacity.

        In practice the hybrid controller activates only one source at a time;
        the individual contributions are::

            ΔT_electric = electric_power_w / effective_ua_w_k
            ΔT_furnace  = furnace_power_w  / effective_ua_w_k
        """
        if self.effective_ua_w_k == 0:
            return None
        return (self.electric_power_w + self.furnace_power_w) / self.effective_ua_w_k
