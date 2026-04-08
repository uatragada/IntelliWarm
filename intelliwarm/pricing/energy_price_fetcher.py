"""
Energy pricing module.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable, Dict, List, Mapping, Optional
from urllib.request import Request, urlopen


class PriceProvider(ABC):
    """Provider contract for current and forecasted utility prices."""

    @abstractmethod
    def get_current_prices(self, electricity_price: float, gas_price: float) -> Dict[str, Any]:
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

    source_name = "static"

    def get_current_prices(self, electricity_price: float, gas_price: float) -> Dict[str, Any]:
        return {
            "electricity": electricity_price,
            "gas": gas_price,
            "source": self.source_name,
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
                "source": self.source_name,
            }
            for offset in range(hours)
        ]


class TimeOfUsePriceProvider(PriceProvider):
    """Offline-safe time-of-use pricing fallback."""

    source_name = "time_of_use"

    def __init__(
        self,
        peak_multiplier: float = 1.3,
        off_peak_multiplier: float = 0.8,
        shoulder_multiplier: float = 1.0,
    ):
        self.peak_multiplier = peak_multiplier
        self.off_peak_multiplier = off_peak_multiplier
        self.shoulder_multiplier = shoulder_multiplier

    def get_current_prices(self, electricity_price: float, gas_price: float) -> Dict[str, Any]:
        return {
            "electricity": electricity_price,
            "gas": gas_price,
            "source": self.source_name,
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
                    "source": self.source_name,
                }
            )
        return forecast


def _json_path_get(payload: Any, path: str, default: Any = None) -> Any:
    """Return a nested value from a dot-separated JSON path."""
    if path is None or str(path).strip() == "":
        return payload

    current = payload
    for part in str(path).split("."):
        if isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError) as exc:
                if default is not None:
                    return default
                raise KeyError(path) from exc
            continue

        if isinstance(current, Mapping) and part in current:
            current = current[part]
            continue

        if default is not None:
            return default
        raise KeyError(path)
    return current


class HttpJsonPriceProvider(PriceProvider):
    """Live price provider backed by configurable HTTP JSON endpoints.

    This adapter keeps vendor-specific API shapes out of runtime code. Configure
    the endpoint URLs and JSON paths for whatever utility, aggregator, or local
    gateway supplies production electricity and gas prices.
    """

    source_name = "live_http_json"

    def __init__(
        self,
        current_url: str,
        forecast_url: str = "",
        headers: Optional[Mapping[str, str]] = None,
        api_key: str = "",
        api_key_header: str = "",
        api_key_prefix: str = "",
        current_electricity_path: str = "electricity",
        current_gas_path: str = "gas",
        forecast_items_path: str = "prices",
        forecast_hour_path: str = "hour",
        forecast_electricity_path: str = "electricity",
        forecast_gas_path: str = "gas",
        request_timeout_seconds: float = 5.0,
        http_get_json: Optional[Callable[[str, Dict[str, str], float], Any]] = None,
    ):
        if not current_url:
            raise ValueError("HttpJsonPriceProvider requires current_url")

        self.current_url = str(current_url)
        self.forecast_url = str(forecast_url or "")
        self.headers = dict(headers or {})
        self.api_key = str(api_key or "")
        self.api_key_header = str(api_key_header or "")
        self.api_key_prefix = str(api_key_prefix or "")
        self.current_electricity_path = str(current_electricity_path or "electricity")
        self.current_gas_path = str(current_gas_path or "gas")
        self.forecast_items_path = str(forecast_items_path or "")
        self.forecast_hour_path = str(forecast_hour_path or "")
        self.forecast_electricity_path = str(forecast_electricity_path or "electricity")
        self.forecast_gas_path = str(forecast_gas_path or "gas")
        self.request_timeout_seconds = float(request_timeout_seconds)
        self.http_get_json = http_get_json or self._default_get_json

    def _request_headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/json"}
        headers.update(self.headers)
        if self.api_key and self.api_key_header:
            headers[self.api_key_header] = f"{self.api_key_prefix}{self.api_key}"
        return headers

    @staticmethod
    def _default_get_json(url: str, headers: Dict[str, str], timeout: float) -> Any:
        request = Request(url, headers=headers)
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
        return json.loads(body)

    @staticmethod
    def _format_url(url: str, hours: int, start_time: Optional[datetime]) -> str:
        reference_time = start_time or datetime.now()
        try:
            return url.format(
                hours=hours,
                start_time=reference_time.isoformat(),
                start_hour=reference_time.hour,
            )
        except KeyError:
            return url

    def _fetch_json(self, url: str) -> Any:
        return self.http_get_json(url, self._request_headers(), self.request_timeout_seconds)

    def get_current_prices(self, electricity_price: float, gas_price: float) -> Dict[str, Any]:
        payload = self._fetch_json(self.current_url)
        return {
            "electricity": float(_json_path_get(payload, self.current_electricity_path)),
            "gas": float(_json_path_get(payload, self.current_gas_path)),
            "source": self.source_name,
        }

    def get_price_forecast(
        self,
        hours: int,
        start_time: Optional[datetime],
        electricity_price: float,
        gas_price: float,
    ) -> List[Dict]:
        reference_time = start_time or datetime.now()
        if not self.forecast_url:
            current = self.get_current_prices(electricity_price, gas_price)
            return [
                {
                    "hour": (reference_time.hour + offset) % 24,
                    "electricity": current["electricity"],
                    "gas": current["gas"],
                    "source": current["source"],
                }
                for offset in range(hours)
            ]

        url = self._format_url(self.forecast_url, hours=hours, start_time=start_time)
        payload = self._fetch_json(url)
        items = _json_path_get(payload, self.forecast_items_path, default=payload)
        if isinstance(items, Mapping) and "prices" in items:
            items = items["prices"]
        if not isinstance(items, list) or not items:
            raise ValueError("live price forecast response did not contain any price items")

        forecast: List[Dict] = []
        for offset in range(hours):
            item = items[min(offset, len(items) - 1)]
            fallback_hour = (reference_time.hour + offset) % 24
            hour_value = (
                _json_path_get(item, self.forecast_hour_path, default=fallback_hour)
                if self.forecast_hour_path
                else fallback_hour
            )
            forecast.append(
                {
                    "hour": int(hour_value),
                    "electricity": float(_json_path_get(item, self.forecast_electricity_path)),
                    "gas": float(_json_path_get(item, self.forecast_gas_path)),
                    "source": self.source_name,
                }
            )
        return forecast


class CallbackPriceProvider(PriceProvider):
    """Adapter for future live/provider integrations without changing service callers."""

    source_name = "callback"

    def __init__(self, current_reader, forecast_reader):
        self.current_reader = current_reader
        self.forecast_reader = forecast_reader

    def get_current_prices(self, electricity_price: float, gas_price: float) -> Dict[str, Any]:
        prices = dict(self.current_reader())
        return {
            "electricity": float(prices.get("electricity", electricity_price)),
            "gas": float(prices.get("gas", gas_price)),
            "source": str(prices.get("source", self.source_name)),
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
                    "source": str(item.get("source", self.source_name)),
                }
            )
        return forecast


def build_price_provider_from_config(
    energy_config: Mapping[str, Any],
    http_get_json: Optional[Callable[[str, Dict[str, str], float], Any]] = None,
) -> PriceProvider:
    """Build the configured price provider for application bootstrap."""
    provider_name = str(energy_config.get("provider") or "time_of_use").strip().lower().replace("-", "_")
    if provider_name in {"time_of_use", "tou", ""}:
        return TimeOfUsePriceProvider(
            peak_multiplier=float(energy_config.get("peak_multiplier", 1.3)),
            off_peak_multiplier=float(energy_config.get("off_peak_multiplier", 0.8)),
            shoulder_multiplier=float(energy_config.get("shoulder_multiplier", 1.0)),
        )

    if provider_name == "static":
        return StaticPriceProvider()

    if provider_name in {"http_json", "live_http_json", "http"}:
        return HttpJsonPriceProvider(
            current_url=str(energy_config.get("current_prices_url", "")),
            forecast_url=str(energy_config.get("forecast_prices_url", "")),
            api_key=str(energy_config.get("api_key", "")),
            api_key_header=str(energy_config.get("api_key_header", "")),
            api_key_prefix=str(energy_config.get("api_key_prefix", "")),
            current_electricity_path=str(energy_config.get("current_electricity_path", "electricity")),
            current_gas_path=str(energy_config.get("current_gas_path", "gas")),
            forecast_items_path=str(energy_config.get("forecast_items_path", "prices")),
            forecast_hour_path=str(energy_config.get("forecast_hour_path", "hour")),
            forecast_electricity_path=str(energy_config.get("forecast_electricity_path", "electricity")),
            forecast_gas_path=str(energy_config.get("forecast_gas_path", "gas")),
            request_timeout_seconds=float(energy_config.get("request_timeout_seconds", 5.0)),
            http_get_json=http_get_json,
        )

    raise ValueError(f"Unsupported energy price provider: {provider_name}")


class EnergyPriceService:
    """Manages electricity and gas prices."""

    def __init__(
        self,
        electricity_price: float = 0.12,
        gas_price: float = 5.0,
        provider: Optional[PriceProvider] = None,
        fallback_provider: Optional[PriceProvider] = None,
    ):
        self.electricity_price = electricity_price
        self.gas_price = gas_price
        self.provider = provider or TimeOfUsePriceProvider()
        self.fallback_provider = fallback_provider or TimeOfUsePriceProvider()
        self.logger = logging.getLogger("IntelliWarm.Pricing")
        self.price_history: List[Dict] = []
        self.last_price_source = self._provider_source(self.provider)
        self.last_price_error: Optional[str] = None

    @staticmethod
    def _provider_source(provider: PriceProvider) -> str:
        return str(getattr(provider, "source_name", provider.__class__.__name__))

    def _normalize_snapshot(
        self,
        snapshot: Dict[str, Any],
        source: str,
    ) -> Dict[str, Any]:
        electricity = float(snapshot.get("electricity", self.electricity_price))
        gas = float(snapshot.get("gas", self.gas_price))
        if electricity < 0 or gas < 0:
            raise ValueError("energy prices must be non-negative")
        return {
            "electricity": electricity,
            "gas": gas,
            "source": str(snapshot.get("source", source)),
        }

    def _snapshot_from_provider(self, provider: PriceProvider) -> Dict[str, Any]:
        source = self._provider_source(provider)
        snapshot = provider.get_current_prices(self.electricity_price, self.gas_price)
        return self._normalize_snapshot(snapshot, source)

    def _current_snapshot(self) -> Dict[str, Any]:
        try:
            snapshot = self._snapshot_from_provider(self.provider)
            self.last_price_source = snapshot["source"]
            self.last_price_error = None
            return snapshot
        except Exception as exc:
            primary_source = self._provider_source(self.provider)
            self.logger.warning("Energy price provider failed (%s): %s", primary_source, exc)
            fallback = self._snapshot_from_provider(self.fallback_provider)
            fallback["source"] = f"{fallback['source']}_fallback"
            self.last_price_source = fallback["source"]
            self.last_price_error = str(exc)
            return fallback

    def get_current_electricity_price(self) -> float:
        """Get current electricity price."""
        return self._current_snapshot()["electricity"]

    def get_current_gas_price(self) -> float:
        """Get current gas price."""
        return self._current_snapshot()["gas"]

    def set_provider(self, provider: PriceProvider):
        """Swap pricing providers without changing service callers."""
        self.provider = provider
        self.last_price_source = self._provider_source(provider)
        self.last_price_error = None
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
        try:
            forecast = self._forecast_from_provider(self.provider, hours, start_time)
            self.last_price_source = forecast[0]["source"] if forecast else self._provider_source(self.provider)
            self.last_price_error = None
            return forecast
        except Exception as exc:
            primary_source = self._provider_source(self.provider)
            self.logger.warning("Energy price forecast provider failed (%s): %s", primary_source, exc)
            forecast = self._forecast_from_provider(self.fallback_provider, hours, start_time)
            for point in forecast:
                point["source"] = f"{point['source']}_fallback"
            self.last_price_source = forecast[0]["source"] if forecast else self._provider_source(self.fallback_provider)
            self.last_price_error = str(exc)
            return forecast

    def _forecast_from_provider(
        self,
        provider: PriceProvider,
        hours: int,
        start_time: Optional[datetime],
    ) -> List[Dict]:
        source = self._provider_source(provider)
        forecast = provider.get_price_forecast(
            hours=hours,
            start_time=start_time,
            electricity_price=self.electricity_price,
            gas_price=self.gas_price,
        )
        if len(forecast) != hours:
            raise ValueError(f"price forecast length {len(forecast)} did not match requested horizon {hours}")

        normalized = []
        reference_time = start_time or datetime.now()
        for offset, point in enumerate(forecast):
            electricity = float(point.get("electricity", self.electricity_price))
            gas = float(point.get("gas", self.gas_price))
            if electricity < 0 or gas < 0:
                raise ValueError("energy prices must be non-negative")
            normalized.append(
                {
                    "hour": int(point.get("hour", (reference_time.hour + offset) % 24)),
                    "electricity": electricity,
                    "gas": gas,
                    "source": str(point.get("source", source)),
                }
            )
        return normalized

    def calculate_energy_cost(self, kwh: float, use_gas: bool = False) -> float:
        """Calculate energy cost."""
        if use_gas:
            return kwh * self.get_current_gas_price()
        return kwh * self.get_current_electricity_price()
