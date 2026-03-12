"""
IntelliWarm Flask application entrypoint.
"""

from intelliwarm.services import create_app, start_runtime_scheduler


app = create_app()


if __name__ == "__main__":
    bootstrap = app.extensions["intelliwarm_bootstrap"]
    start_runtime_scheduler(app)
    app.run(debug=bootstrap.config.debug, port=5000)
