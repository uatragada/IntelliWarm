"""
Tests for typed config loading and environment overrides.
"""

import pytest

from intelliwarm.core import SystemConfig
from intelliwarm.core.config import ConfigValidationError


def test_system_config_resolves_env_placeholders_and_overrides(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
system:
  debug: true
  logging_level: INFO
  poll_interval: 15
  optimization_horizon: 12
comfort:
  min_temperature: 18
  max_temperature: 24
  default_target: 21
optimization:
  comfort_weight: 1.0
  switching_weight: 0.5
  energy_weight: 1.0
energy:
  electricity_price: 0.12
  gas_price: 5.0
database:
  path: "${TEST_INTELLIWARM_DB}"
devices:
  enable_control: false
rooms:
  office:
    zone: Work
    room_size: 125
    target_temp: 22
    heater_power: 1200
    occupancy_schedule: "8-17"
zones:
  Work:
    description: Workspace
""".strip(),
        encoding="utf-8",
    )

    monkeypatch.setenv("TEST_INTELLIWARM_DB", "env.db")
    monkeypatch.setenv("INTELLIWARM_DEBUG", "false")
    monkeypatch.setenv("INTELLIWARM_ELECTRICITY_PRICE", "0.22")

    config = SystemConfig(str(config_path))

    assert config.debug is False
    assert config.electricity_price == 0.22
    assert config.database_path == "env.db"
    assert config.get_room_config("office")["target_temp"] == 22
    assert config.zones["Work"]["description"] == "Workspace"


def test_build_room_config_uses_typed_defaults(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
comfort:
  default_target: 20
rooms:
  den:
    zone: Residential
    room_size: 110
""".strip(),
        encoding="utf-8",
    )

    config = SystemConfig(str(config_path))
    room_config = config.build_room_config(room_name="den")

    assert room_config["zone"] == "Residential"
    assert room_config["room_size"] == 110.0
    assert room_config["target_temp"] == 20.0
    assert room_config["heating_efficiency"] == 0.85


def test_build_room_config_preserves_structured_schedule_and_weather_settings(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
weather_api:
  enabled: true
  provider: openweathermap
  api_key: "${WEATHER_API_KEY}"
  update_interval: 1800
rooms:
  office:
    zone: Work
    occupancy_schedule:
      - day_of_week: 2
        start_hour: 8
        end_hour: 17
        probability: 0.9
""".strip(),
        encoding="utf-8",
    )

    config = SystemConfig(str(config_path))
    room_config = config.build_room_config(room_name="office")

    assert isinstance(room_config["occupancy_schedule"], list)
    assert room_config["occupancy_schedule"][0]["probability"] == 0.9
    assert config.weather_api_config["enabled"] is True
    assert config.weather_api_config["provider"] == "openweathermap"


def test_invalid_config_raises_validation_error(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
comfort:
  min_temperature: 24
  max_temperature: 18
devices:
  enable_control: true
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigValidationError):
        SystemConfig(str(config_path))
