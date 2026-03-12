"""
Forecast bundle service for aligned occupancy, outdoor temperature, and energy prices.
"""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import datetime, timedelta
from typing import List, Optional, Protocol, Sequence

from intelliwarm.data import ForecastBundle, ForecastStep


class OutdoorForecastProvider(Protocol):
    """Contract for future outdoor temperature forecast providers."""

    def get_temperature_forecast(
        self,
        start_time: datetime,
        horizon_steps: int,
        step_minutes: int,
    ) -> List[float]:
        """Return an aligned outdoor temperature forecast."""


class DeterministicOutdoorForecast:
    """Offline-safe outdoor forecast using a deterministic diurnal profile."""

    def __init__(self, base_temp: float = 5.0, swing: float = 4.0):
        self.base_temp = float(base_temp)
        self.swing = float(swing)

    def get_temperature_forecast(
        self,
        start_time: datetime,
        horizon_steps: int,
        step_minutes: int,
    ) -> List[float]:
        forecast: List[float] = []
        for step_index in range(horizon_steps):
            timestamp = start_time + timedelta(minutes=step_minutes * step_index)
            hour = timestamp.hour + (timestamp.minute / 60.0)
            angle = ((hour - 15.0) / 24.0) * 2.0 * math.pi
            forecast.append(round(self.base_temp + (self.swing * math.cos(angle)), 3))
        return forecast


class ForecastBundleService:
    """Build a shared aligned forecast bundle for controllers and runtime consumers."""

    def __init__(
        self,
        energy_service,
        outdoor_provider: Optional[OutdoorForecastProvider] = None,
        step_minutes: int = 60,
    ):
        self.energy_service = energy_service
        self.outdoor_provider = outdoor_provider or DeterministicOutdoorForecast()
        self.step_minutes = int(step_minutes)

    def build_bundle(
        self,
        room_name: str,
        occupancy_predictor,
        horizon_steps: int,
        start_time: Optional[datetime] = None,
    ) -> ForecastBundle:
        """Create an aligned forecast bundle from occupancy, weather, and prices."""
        resolved_start = (start_time or datetime.now()).replace(second=0, microsecond=0)
        occupancy_probs = occupancy_predictor.predict_horizon(
            resolved_start,
            horizon_steps,
            self.step_minutes,
        )
        outdoor_temps = self.outdoor_provider.get_temperature_forecast(
            resolved_start,
            horizon_steps,
            self.step_minutes,
        )
        price_forecast = self.energy_service.get_price_forecast(
            horizon_steps,
            start_time=resolved_start,
        )

        steps = [
            ForecastStep(
                timestamp=resolved_start + timedelta(minutes=self.step_minutes * index),
                occupancy_probability=float(occupancy_probs[index]),
                outdoor_temp=float(outdoor_temps[index]),
                electricity_price=float(price_forecast[index]["electricity"]),
                gas_price=float(price_forecast[index]["gas"]),
            )
            for index in range(horizon_steps)
        ]

        return ForecastBundle(
            room_id=room_name,
            start_time=resolved_start,
            step_minutes=self.step_minutes,
            steps=steps,
            source="deterministic",
        )

    def override_bundle(
        self,
        bundle: ForecastBundle,
        occupancy_probabilities: Optional[Sequence[float]] = None,
        outdoor_temperatures: Optional[Sequence[float]] = None,
        electricity_prices: Optional[Sequence[float]] = None,
        gas_prices: Optional[Sequence[float]] = None,
    ) -> ForecastBundle:
        """Return a bundle copy with aligned overrides applied."""
        steps = list(bundle.steps)
        overrides = {
            "occupancy_probability": occupancy_probabilities,
            "outdoor_temp": outdoor_temperatures,
            "electricity_price": electricity_prices,
            "gas_price": gas_prices,
        }
        for values in overrides.values():
            if values is not None and len(values) != len(steps):
                raise ValueError("Forecast overrides must match bundle horizon length")

        updated_steps = []
        for index, step in enumerate(steps):
            updated_steps.append(
                replace(
                    step,
                    occupancy_probability=(
                        float(occupancy_probabilities[index])
                        if occupancy_probabilities is not None
                        else step.occupancy_probability
                    ),
                    outdoor_temp=(
                        float(outdoor_temperatures[index])
                        if outdoor_temperatures is not None
                        else step.outdoor_temp
                    ),
                    electricity_price=(
                        float(electricity_prices[index])
                        if electricity_prices is not None
                        else step.electricity_price
                    ),
                    gas_price=(
                        float(gas_prices[index])
                        if gas_prices is not None
                        else step.gas_price
                    ),
                )
            )

        return replace(bundle, steps=updated_steps)
