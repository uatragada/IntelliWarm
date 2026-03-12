"""
Tests for Flask app bootstrap and config-backed startup.
"""

from intelliwarm.control import HardwareDeviceBackend
from intelliwarm.sensors import HardwareSensorBackend
from intelliwarm.services import create_app, create_runtime_bootstrap


def _write_app_config(config_path):
    config_path.write_text(
        """
system:
  debug: false
  poll_interval: 10
  optimization_horizon: 6
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
  path: "bootstrap.db"
rooms:
  office:
    zone: Work
    room_size: 100
    target_temp: 21
    heater_power: 1000
    occupancy_schedule: "8-17"
zones:
  Work:
    description: Workspace
""".strip(),
        encoding="utf-8",
    )


def test_runtime_bootstrap_loads_configured_rooms(tmp_path):
    config_path = tmp_path / "config.yaml"
    _write_app_config(config_path)

    bootstrap = create_runtime_bootstrap(
        config_path=str(config_path),
        database_path=str(tmp_path / "runtime.db"),
    )

    assert bootstrap.runtime.room_names == ["office"]
    assert bootstrap.runtime.zones == [{"name": "Work", "description": "Workspace"}]
    assert bootstrap.database.db_path.endswith("runtime.db")
    assert isinstance(bootstrap.sensor_manager.backend, HardwareSensorBackend)
    assert isinstance(bootstrap.device_controller.backend, HardwareDeviceBackend)


def test_create_app_registers_runtime_and_routes(tmp_path):
    config_path = tmp_path / "config.yaml"
    _write_app_config(config_path)

    app = create_app(
        config_path=str(config_path),
        database_path=str(tmp_path / "app.db"),
    )

    assert "intelliwarm_bootstrap" in app.extensions
    assert "dashboard" in app.view_functions
    assert "demo_timeline" in app.view_functions

    with app.test_client() as client:
        response = client.get("/api/rooms")

    assert response.status_code == 200
    payload = response.get_json()
    assert len(payload) == 1
    assert payload[0]["room"] == "office"
    assert payload[0]["zone"] == "Work"


def test_optimization_endpoint_accepts_baseline_controller(tmp_path):
    config_path = tmp_path / "config.yaml"
    _write_app_config(config_path)

    app = create_app(
        config_path=str(config_path),
        database_path=str(tmp_path / "app.db"),
    )

    with app.test_client() as client:
        response = client.get("/api/optimization/office?controller=baseline")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["controller"] == "baseline"
    assert payload["next_action_label"] in {"OFF", "ECO", "COMFORT", "PREHEAT"}


def test_demo_timeline_route_stays_registered_after_modularization(tmp_path):
    config_path = tmp_path / "config.yaml"
    _write_app_config(config_path)

    app = create_app(
        config_path=str(config_path),
        database_path=str(tmp_path / "app.db"),
    )

    with app.test_client() as client:
        response = client.get("/demo_timeline")

    assert response.status_code == 200
