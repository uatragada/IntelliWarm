"""
Flask application bootstrap helpers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

from flask import Flask, current_app, jsonify, redirect, render_template, request, url_for

from intelliwarm.control import DeviceController
from intelliwarm.core import SystemConfig, SystemScheduler
from intelliwarm.pricing import EnergyPriceService
from intelliwarm.sensors import SensorManager
from intelliwarm.storage import Database

from .forecast_bundle import ForecastBundleService
from .runtime import IntelliWarmRuntime


@dataclass
class RuntimeBootstrap:
    project_root: Path
    config: SystemConfig
    database: Database
    sensor_manager: SensorManager
    device_controller: DeviceController
    energy_service: EnergyPriceService
    forecast_service: ForecastBundleService
    scheduler: SystemScheduler
    runtime: IntelliWarmRuntime
    logger: logging.Logger


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_database_path(project_root: Path, configured_path: Optional[str]) -> Path:
    candidate = Path(configured_path or "intelliwarm.db")
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return candidate


def _default_display_temp_f(temp_c: float) -> float:
    return round((temp_c * 9 / 5) + 32, 1)


def create_runtime_bootstrap(
    config_path: Optional[str] = None,
    database_path: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> RuntimeBootstrap:
    project_root = _project_root()
    resolved_config_path = Path(config_path) if config_path else project_root / "configs" / "config.yaml"
    config = SystemConfig(str(resolved_config_path))

    logging_level = getattr(logging, str(config.logging_level).upper(), logging.INFO)
    logging.basicConfig(level=logging_level)
    app_logger = logger or logging.getLogger("IntelliWarm.App")

    resolved_database_path = _resolve_database_path(project_root, database_path or config.database_path)
    database = Database(str(resolved_database_path))
    sensor_manager = SensorManager()
    device_controller = DeviceController()
    energy_service = EnergyPriceService(config.electricity_price, config.gas_price)
    forecast_service = ForecastBundleService(energy_service=energy_service)
    scheduler = SystemScheduler()
    runtime = IntelliWarmRuntime(
        config=config,
        database=database,
        sensor_manager=sensor_manager,
        device_controller=device_controller,
        energy_service=energy_service,
        forecast_service=forecast_service,
        logger=app_logger,
    )
    runtime.bootstrap_from_config()

    return RuntimeBootstrap(
        project_root=project_root,
        config=config,
        database=database,
        sensor_manager=sensor_manager,
        device_controller=device_controller,
        energy_service=energy_service,
        forecast_service=forecast_service,
        scheduler=scheduler,
        runtime=runtime,
        logger=app_logger,
    )


def _bootstrap(app: Flask) -> RuntimeBootstrap:
    return app.extensions["intelliwarm_bootstrap"]


def _add_room_from_form(runtime: IntelliWarmRuntime, form: Mapping[str, str]) -> bool:
    name = (form.get("roomName") or "").strip()
    if not name:
        return False

    zone = (form.get("zone") or "Unassigned").strip()
    room_size_raw = (form.get("roomSize") or "").strip()
    room_config = runtime.config.build_room_config(zone=zone)
    room_size = float(room_size_raw) if room_size_raw else float(room_config.get("room_size", 150.0))
    initial_sensor_temp = float(room_config.get("target_temp", runtime.config.default_target_temp))

    runtime.add_room(
        name=name,
        room_size=room_size,
        zone=zone,
        room_config=room_config,
        initial_sensor_temp=initial_sensor_temp,
        initial_occupancy=bool(room_config.get("initial_occupancy", False)),
        display_temp_f=room_config.get("display_temp_f") or _default_display_temp_f(initial_sensor_temp),
        humidity=float(room_config.get("humidity", 45.0)),
        heating_source=str(room_config.get("heating_source", "Off")),
    )
    return True


def _apply_home_configuration(runtime: IntelliWarmRuntime, form: Mapping[str, str]):
    zone_name = (form.get("zoneName") or "").strip()
    zone_desc = (form.get("zoneDescription") or "").strip()
    if zone_name:
        runtime.add_zone(zone_name, zone_desc)

    elec_price = (form.get("electricityPrice") or "").strip()
    gas_price = (form.get("gasPrice") or "").strip()

    if elec_price:
        runtime.update_utility_rates(electricity_price=float(elec_price))

    if gas_price:
        runtime.update_utility_rates(gas_price=float(gas_price))


def register_routes(app: Flask):
    @app.route("/")
    def dashboard():
        return render_template("dashboard.html", rooms=_bootstrap(current_app).runtime.rooms)

    @app.route("/api/rooms", methods=["GET"])
    def api_get_rooms():
        return jsonify(_bootstrap(current_app).runtime.get_rooms_api_data())

    @app.route("/api/optimization/<room_name>", methods=["GET"])
    def api_get_optimization(room_name: str):
        controller_type = request.args.get("controller", default="mpc", type=str)
        if controller_type not in {"mpc", "baseline"}:
            return jsonify({"error": "Unsupported controller type"}), 400

        plan = _bootstrap(current_app).runtime.optimize_heating_plan(
            room_name,
            controller_type=controller_type,
        )
        return jsonify(plan) if plan else (jsonify({"error": "Optimization failed"}), 500)

    @app.route("/add_room", methods=["GET", "POST"])
    def add_room():
        bootstrap = _bootstrap(current_app)
        if request.method == "POST":
            if _add_room_from_form(bootstrap.runtime, request.form):
                return redirect(url_for("add_room"))

        return render_template("add_room.html", zones=[zone["name"] for zone in bootstrap.runtime.zones])

    @app.route("/config_home", methods=["GET", "POST"])
    def config_home():
        bootstrap = _bootstrap(current_app)
        if request.method == "POST":
            _apply_home_configuration(bootstrap.runtime, request.form)
            return redirect(url_for("config_home"))

        return render_template(
            "config_home.html",
            zones=bootstrap.runtime.zones,
            rates=bootstrap.runtime.utility_rates,
        )

    @app.route("/demo")
    def demo():
        if not _bootstrap(current_app).runtime.load_demo_dataset():
            return redirect(url_for("dashboard"))
        return redirect(url_for("demo_timeline"))

    @app.route("/demo_timeline")
    def demo_timeline():
        runtime = _bootstrap(current_app).runtime
        if not runtime.demo_loaded:
            runtime.load_demo_dataset()
        return render_template("demo_timeline.html")

    @app.route("/api/demo/timeline/meta", methods=["GET"])
    def api_demo_timeline_meta():
        meta = _bootstrap(current_app).runtime.get_demo_timeline_meta()
        if meta is None:
            return jsonify({"error": "Demo dataset not available"}), 500
        return jsonify(meta)

    @app.route("/api/demo/timeline/point", methods=["GET"])
    def api_demo_timeline_point():
        index = request.args.get("index", default=0, type=int)
        point = _bootstrap(current_app).runtime.get_demo_timeline_point(index)
        if point is None:
            return jsonify({"error": "Demo dataset not available"}), 500
        return jsonify(point)


def create_app(
    config_path: Optional[str] = None,
    database_path: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> Flask:
    bootstrap = create_runtime_bootstrap(
        config_path=config_path,
        database_path=database_path,
        logger=logger,
    )
    app = Flask(
        __name__,
        template_folder=str(bootstrap.project_root / "templates"),
    )
    app.extensions["intelliwarm_bootstrap"] = bootstrap
    register_routes(app)
    return app


def start_runtime_scheduler(app: Flask):
    bootstrap = _bootstrap(app)
    bootstrap.scheduler.add_task(
        "optimization",
        bootstrap.runtime.run_optimization_cycle,
        bootstrap.config.poll_interval,
    )
    bootstrap.scheduler.start()
    bootstrap.logger.info("Started optimization scheduler")
