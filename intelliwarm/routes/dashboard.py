"""
Dashboard and room management route registration.
"""

from __future__ import annotations

from flask import Flask, jsonify, redirect, render_template, request, url_for

from .shared import add_room_from_form, apply_home_configuration, current_bootstrap


def register_dashboard_routes(app: Flask):
    """Register dashboard-facing routes on the Flask app."""

    @app.route("/")
    def dashboard():
        dashboard_data = current_bootstrap().runtime.get_dashboard_data()
        return render_template(
            "dashboard.html",
            dashboard=dashboard_data,
            rooms=dashboard_data["rooms"],
            zones=dashboard_data["zones"],
            runtime=dashboard_data["runtime"],
        )

    @app.route("/api/rooms", methods=["GET"])
    def api_get_rooms():
        return jsonify(current_bootstrap().runtime.get_rooms_api_data())

    @app.route("/api/optimization/<room_name>", methods=["GET"])
    def api_get_optimization(room_name: str):
        controller_type = request.args.get("controller", default="hybrid", type=str)
        if controller_type not in {"mpc", "baseline", "hybrid"}:
            return jsonify({"error": "Unsupported controller type"}), 400

        plan = current_bootstrap().runtime.optimize_heating_plan(
            room_name,
            controller_type=controller_type,
        )
        return jsonify(plan) if plan else (jsonify({"error": "Optimization failed"}), 500)

    @app.route("/add_room", methods=["GET", "POST"])
    def add_room():
        bootstrap = current_bootstrap()
        if request.method == "POST":
            if add_room_from_form(bootstrap.runtime, request.form):
                return redirect(url_for("add_room"))

        return render_template("add_room.html", zones=bootstrap.runtime.get_zone_status_data())

    @app.route("/config_home", methods=["GET", "POST"])
    def config_home():
        bootstrap = current_bootstrap()
        if request.method == "POST":
            apply_home_configuration(bootstrap.runtime, request.form)
            return redirect(url_for("config_home"))

        return render_template(
            "config_home.html",
            zones=bootstrap.runtime.get_zone_status_data(),
            rates=bootstrap.runtime.utility_rates,
        )
