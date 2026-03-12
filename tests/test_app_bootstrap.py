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
    assert payload[0]["configured_heat_source"] == "electric"


def test_optimization_endpoint_routes_baseline_requests_through_hybrid_controller(tmp_path):
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
    assert payload["controller"] == "hybrid"
    assert payload["next_action_label"] in {"OFF", "ECO", "COMFORT", "PREHEAT"}
    assert "hybrid_decision" in payload


def test_add_room_form_supports_target_temperature_and_heat_source(tmp_path):
    config_path = tmp_path / "config.yaml"
    _write_app_config(config_path)

    app = create_app(
        config_path=str(config_path),
        database_path=str(tmp_path / "app.db"),
    )

    with app.test_client() as client:
        response = client.post(
            "/add_room",
            data={
                "roomName": "Guest Room",
                "roomSize": "180",
                "zone": "Work",
                "targetTemp": "22",
                "heatSource": "gas_furnace",
            },
            follow_redirects=True,
        )
        rooms_response = client.get("/api/rooms")

    assert response.status_code == 200
    payload = rooms_response.get_json()
    guest_room = next(room for room in payload if room["room"] == "Guest Room")
    assert guest_room["target_temp"] == 22.0
    assert guest_room["configured_heat_source"] == "gas_furnace"


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
