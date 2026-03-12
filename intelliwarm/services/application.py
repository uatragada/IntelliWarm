"""
Flask application bootstrap helpers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from flask import Flask

from intelliwarm.control import DeviceController
from intelliwarm.core import SystemConfig, SystemScheduler
from intelliwarm.pricing import EnergyPriceService
from intelliwarm.routes import register_route_modules
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
    default_device_id = config.thermostat_id or config.smart_plug_id
    sensor_manager = SensorManager.with_hardware_fallback()
    device_controller = DeviceController.with_hardware_fallback(
        enable_hardware=config.enable_device_control,
        default_device_id=default_device_id,
    )
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
    register_route_modules(app)
    return app


def start_runtime_scheduler(app: Flask):
    bootstrap = app.extensions["intelliwarm_bootstrap"]
    bootstrap.scheduler.add_task(
        "optimization",
        bootstrap.runtime.run_optimization_cycle,
        bootstrap.config.poll_interval,
    )
    bootstrap.scheduler.start()
    bootstrap.logger.info("Started optimization scheduler")
