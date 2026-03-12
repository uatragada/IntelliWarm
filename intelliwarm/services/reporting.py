"""
Reporting services layered on the existing SQLite storage.
"""

from __future__ import annotations


class ReportService:
    """Build room and portfolio reports from persisted runtime data."""

    def __init__(self, database):
        self.database = database

    def build_room_report(self, room_name: str, limit: int = 10):
        room = self.database.get_room(room_name)
        if room is None:
            return None

        summary = self.database.get_room_summary(room_name) or {}
        return {
            "room": room_name,
            "room_details": room,
            "summary": summary,
            "recent_optimizations": self.database.get_recent_optimizations(room_name, limit=limit),
            "temperature_history": self.database.get_temperature_history(room_name, limit=limit),
        }

    def build_portfolio_report(self, limit_per_room: int = 5):
        room_reports = []
        total_runs = 0
        total_predicted_cost = 0.0
        for room in self.database.get_all_rooms():
            report = self.build_room_report(room["name"], limit=limit_per_room)
            if report is None:
                continue

            total_runs += int(report["summary"].get("optimization_runs", 0))
            total_predicted_cost += float(report["summary"].get("total_predicted_cost", 0.0))
            room_reports.append(report)

        return {
            "room_count": len(room_reports),
            "optimization_runs": total_runs,
            "total_predicted_cost": round(total_predicted_cost, 4),
            "rooms": room_reports,
        }
