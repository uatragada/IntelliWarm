"""
Flask route module registration.
"""

from .dashboard import register_dashboard_routes
from .demo import register_demo_routes


def register_route_modules(app):
    """Register all route modules on the Flask application."""
    register_dashboard_routes(app)
    register_demo_routes(app)


__all__ = ["register_route_modules"]
