"""
Energy pricing module.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional


class PriceProvider(ABC):
    """Provider contract for current and forecasted utility prices."""

    @abstractmethod
    def get_current_prices(self, electricity_price: float, gas_price: float) -> Dict[str, float]:
        """Return the latest electricity and gas prices."""

    @abstractmethod
    def get_price_forecast(
        self,
        hours: int,
        start_time: Optional[datetime],
        electricity_price: float,
        gas_price: float,
    ) -> List[Dict]:
        """Return the forecasted utility prices."""


class StaticPriceProvider(PriceProvider):
    """Always returns the provided base prices."""

    def get_current_prices(self, electricity_price: float, gas_price: float) -> Dict[str, float]:
        return {
            "electricity": electricity_price,
            "gas": gas_price,
        }

    def get_price_forecast(
        self,
        hours: int,
        start_time: Optional[datetime],
        electricity_price: float,
        gas_price: float,
    ) -> List[Dict]:
        reference_time = start_time or datetime.now()
        return [
            {
                "hour": (reference_time.hour + offset) % 24,
                "electricity": electricity_price,
                "gas": gas_price,
            }
            for offset in range(hours)
        ]


class TimeOfUsePriceProvider(PriceProvider):
    """Offline-safe time-of-use pricing fallback."""

    def __init__(
        self,
        peak_multiplier: float = 1.3,
        off_peak_multiplier: float = 0.8,
        shoulder_multiplier: float = 1.0,
    ):
        self.peak_multiplier = peak_multiplier
        self.off_peak_multiplier = off_peak_multiplier
        self.shoulder_multiplier = shoulder_multiplier

    def get_current_prices(self, electricity_price: float, gas_price: float) -> Dict[str, float]:
        return {
            "electricity": electricity_price,
            "gas": gas_price,
        }

    def _electricity_multiplier(self, hour: int) -> float:
        if (6 <= hour < 9) or (16 <= hour < 21):
            return self.peak_multiplier
        if hour >= 23 or hour < 6:
            return self.off_peak_multiplier
        return self.shoulder_multiplier

    def get_price_forecast(
        self,
        hours: int,
        start_time: Optional[datetime],
        electricity_price: float,
        gas_price: float,
    ) -> List[Dict]:
        reference_time = start_time or datetime.now()
        forecast: List[Dict] = []
        for offset in range(hours):
            hour = (reference_time.hour + offset) % 24
            forecast.append(
                {
                    "hour": hour,
                    "electricity": electricity_price * self._electricity_multiplier(hour),
                    "gas": gas_price,
                }
            )
        return forecast


class CallbackPriceProvider(PriceProvider):
    """Adapter for future live/provider integrations without changing service callers."""

    def __init__(self, current_reader, forecast_reader):
        self.current_reader = current_reader
        self.forecast_reader = forecast_reader

    def get_current_prices(self, electricity_price: float, gas_price: float) -> Dict[str, float]:
        prices = dict(self.current_reader())
        return {
            "electricity": float(prices.get("electricity", electricity_price)),
            "gas": float(prices.get("gas", gas_price)),
        }

    def get_price_forecast(
        self,
        hours: int,
        start_time: Optional[datetime],
        electricity_price: float,
        gas_price: float,
    ) -> List[Dict]:
        forecast = []
        for item in self.forecast_reader(hours, start_time):
            forecast.append(
                {
                    "hour": int(item["hour"]),
                    "electricity": float(item.get("electricity", electricity_price)),
                    "gas": float(item.get("gas", gas_price)),
                }
            )
        return forecast


class EnergyPriceService:
    """Manages electricity and gas prices."""

    def __init__(
        self,
        electricity_price: float = 0.12,
        gas_price: float = 5.0,
        provider: Optional[PriceProvider] = None,
    ):
        self.electricity_price = electricity_price
        self.gas_price = gas_price
        self.provider = provider or TimeOfUsePriceProvider()
        self.logger = logging.getLogger("IntelliWarm.Pricing")
        self.price_history: List[Dict] = []

    def _current_snapshot(self) -> Dict[str, float]:
        snapshot = self.provider.get_current_prices(self.electricity_price, self.gas_price)
        return {
            "electricity": float(snapshot.get("electricity", self.electricity_price)),
            "gas": float(snapshot.get("gas", self.gas_price)),
        }

    def get_current_electricity_price(self) -> float:
        """Get current electricity price."""
        return self._current_snapshot()["electricity"]

    def get_current_gas_price(self) -> float:
        """Get current gas price."""
        return self._current_snapshot()["gas"]

    def set_provider(self, provider: PriceProvider):
        """Swap pricing providers without changing service callers."""
        self.provider = provider
        self.logger.info("Energy price provider updated: %s", provider.__class__.__name__)

    def set_electricity_price(self, price: float):
        """Update base electricity price."""
        self.electricity_price = float(price)
        self._record_price()
        self.logger.info("Electricity price updated: $%s/kWh", price)

    def set_gas_price(self, price: float):
        """Update base gas price."""
        self.gas_price = float(price)
        self._record_price()
        self.logger.info("Gas price updated: $%s/therm", price)

    def _record_price(self):
        snapshot = self._current_snapshot()
        self.price_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "electricity": snapshot["electricity"],
                "gas": snapshot["gas"],
            }
        )

    def get_price_forecast(self, hours: int = 24, start_time: datetime = None) -> List[Dict]:
        """Get price forecast for the next N hours."""
        return self.provider.get_price_forecast(
            hours=hours,
            start_time=start_time,
            electricity_price=self.electricity_price,
            gas_price=self.gas_price,
        )

    def calculate_energy_cost(self, kwh: float, use_gas: bool = False) -> float:
        """Calculate energy cost."""
        if use_gas:
            return kwh * self.get_current_gas_price()
        return kwh * self.get_current_electricity_price()
