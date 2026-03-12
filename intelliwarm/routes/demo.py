"""
Demo dataset and timeline route registration.
"""

from __future__ import annotations

from flask import Flask, jsonify, redirect, render_template, request, url_for

from .shared import current_bootstrap


def register_demo_routes(app: Flask):
    """Register demo and timeline routes on the Flask app."""

    @app.route("/demo")
    def demo():
        if not current_bootstrap().runtime.load_demo_dataset():
            return redirect(url_for("dashboard"))
        return redirect(url_for("demo_timeline"))

    @app.route("/demo_timeline")
    def demo_timeline():
        runtime = current_bootstrap().runtime
        if not runtime.demo_loaded:
            runtime.load_demo_dataset()
        return render_template("demo_timeline.html")

    @app.route("/api/demo/timeline/meta", methods=["GET"])
    def api_demo_timeline_meta():
        meta = current_bootstrap().runtime.get_demo_timeline_meta()
        if meta is None:
            return jsonify({"error": "Demo dataset not available"}), 500
        return jsonify(meta)

    @app.route("/api/demo/timeline/point", methods=["GET"])
    def api_demo_timeline_point():
        index = request.args.get("index", default=0, type=int)
        point = current_bootstrap().runtime.get_demo_timeline_point(index)
        if point is None:
            return jsonify({"error": "Demo dataset not available"}), 500
        return jsonify(point)
