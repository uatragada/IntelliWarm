"""
Database Layer
SQLite implementation with ORM models
"""

import logging
import sqlite3
from typing import Any, Dict, List, Optional


class Database:
    """SQLite database handler for IntelliWarm"""
    
    def __init__(self, db_path: str = "intelliwarm.db"):
        self.db_path = db_path
        self.logger = logging.getLogger("IntelliWarm.Database")
        self._init_schema()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_column(self, cursor, table_name: str, column_name: str, definition: str):
        cursor.execute(f"PRAGMA table_info({table_name})")
        existing_columns = {row[1] for row in cursor.fetchall()}
        if column_name not in existing_columns:
            cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}")

    def _init_schema(self):
        """Create database tables if they don't exist"""
        conn = self._connect()
        cursor = conn.cursor()

        # Rooms table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rooms (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                zone TEXT,
                room_size REAL,
                target_temp REAL,
                heater_power REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Temperature logs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS temperature_logs (
                id INTEGER PRIMARY KEY,
                room_id INTEGER,
                temperature REAL,
                outside_temperature REAL,
                humidity REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (room_id) REFERENCES rooms(id)
            )
        """)
        
        # Energy prices table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS energy_prices (
                id INTEGER PRIMARY KEY,
                electricity_price REAL,
                gas_price REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Optimization runs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS optimization_runs (
                id INTEGER PRIMARY KEY,
                room_id INTEGER,
                heating_action REAL,
                predicted_cost REAL,
                controller_type TEXT DEFAULT 'mpc',
                action_label TEXT DEFAULT '',
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (room_id) REFERENCES rooms(id)
            )
        """)
        self._ensure_column(cursor, "optimization_runs", "controller_type", "TEXT DEFAULT 'mpc'")
        self._ensure_column(cursor, "optimization_runs", "action_label", "TEXT DEFAULT ''")
        
        # Model parameters table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_parameters (
                id INTEGER PRIMARY KEY,
                room_id INTEGER,
                alpha REAL,
                beta REAL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (room_id) REFERENCES rooms(id)
            )
        """)
        
        # Occupancy logs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS occupancy_logs (
                id INTEGER PRIMARY KEY,
                room_id INTEGER,
                occupancy_probability REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (room_id) REFERENCES rooms(id)
            )
        """)
        
        conn.commit()
        conn.close()
        self.logger.info("Database initialized: %s", self.db_path)

    def add_temperature_log(self, room_name: str, temp: float, humidity: float = None, outside_temp: float = None):
        """Log temperature reading"""
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute("SELECT id FROM rooms WHERE name = ?", (room_name,))
        result = cursor.fetchone()

        if result:
            room_id = result[0]
            cursor.execute("""
                INSERT INTO temperature_logs (room_id, temperature, humidity, outside_temperature)
                VALUES (?, ?, ?, ?)
            """, (room_id, temp, humidity, outside_temp))
            conn.commit()

        conn.close()

    def add_room(self, name: str, zone: str, room_size: float, target_temp: float, heater_power: float):
        """Add room to database"""
        conn = self._connect()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT INTO rooms (name, zone, room_size, target_temp, heater_power)
                VALUES (?, ?, ?, ?, ?)
            """, (name, zone, room_size, target_temp, heater_power))
            conn.commit()
            self.logger.info("Room added: %s", name)
        except sqlite3.IntegrityError:
            self.logger.warning("Room already exists: %s", name)

        conn.close()

    def get_room(self, room_name: str) -> Dict:
        """Get room details"""
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM rooms WHERE name = ?", (room_name,))
        result = cursor.fetchone()
        conn.close()

        return dict(result) if result else None

    def get_all_rooms(self) -> List[Dict]:
        """Get all rooms"""
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM rooms")
        results = cursor.fetchall()
        conn.close()

        return [dict(row) for row in results]

    def record_optimization(
        self,
        room_name: str,
        heating_action: float,
        predicted_cost: float,
        controller_type: str = "mpc",
        action_label: str = "",
    ):
        """Record optimization decision"""
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute("SELECT id FROM rooms WHERE name = ?", (room_name,))
        result = cursor.fetchone()

        if result:
            room_id = result[0]
            cursor.execute("""
                INSERT INTO optimization_runs (
                    room_id,
                    heating_action,
                    predicted_cost,
                    controller_type,
                    action_label
                )
                VALUES (?, ?, ?, ?, ?)
            """, (room_id, heating_action, predicted_cost, controller_type, action_label))
            conn.commit()

        conn.close()

    def get_temperature_history(self, room_name: str, limit: int = 100) -> List[Dict]:
        """Get recent temperature readings"""
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT tl.* FROM temperature_logs tl
            JOIN rooms r ON tl.room_id = r.id
            WHERE r.name = ?
            ORDER BY tl.timestamp DESC
            LIMIT ?
        """, (room_name, limit))

        results = cursor.fetchall()
        conn.close()

        return [dict(row) for row in results]

    def save_model_parameters(self, room_name: str, alpha: float, beta: float):
        """Save thermal model parameters"""
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute("SELECT id FROM rooms WHERE name = ?", (room_name,))
        result = cursor.fetchone()

        if result:
            room_id = result[0]
            cursor.execute("""
                DELETE FROM model_parameters WHERE room_id = ?
            """, (room_id,))

            cursor.execute("""
                INSERT INTO model_parameters (room_id, alpha, beta)
                VALUES (?, ?, ?)
            """, (room_id, alpha, beta))
            conn.commit()

        conn.close()

    def get_model_parameters(self, room_name: str) -> Dict:
        """Get thermal model parameters"""
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT mp.alpha, mp.beta FROM model_parameters mp
            JOIN rooms r ON mp.room_id = r.id
            WHERE r.name = ?
            ORDER BY mp.updated_at DESC
            LIMIT 1
        """, (room_name,))

        result = cursor.fetchone()
        conn.close()

        return dict(result) if result else {"alpha": 0.1, "beta": 0.05}

    def get_recent_optimizations(self, room_name: Optional[str] = None, limit: int = 20) -> List[Dict]:
        """Return recent optimization runs with room and controller metadata."""
        conn = self._connect()
        cursor = conn.cursor()

        if room_name:
            cursor.execute(
                """
                SELECT
                    r.name AS room_name,
                    r.zone,
                    o.heating_action,
                    o.predicted_cost,
                    o.controller_type,
                    o.action_label,
                    o.timestamp
                FROM optimization_runs o
                JOIN rooms r ON o.room_id = r.id
                WHERE r.name = ?
                ORDER BY o.timestamp DESC, o.id DESC
                LIMIT ?
                """,
                (room_name, limit),
            )
        else:
            cursor.execute(
                """
                SELECT
                    r.name AS room_name,
                    r.zone,
                    o.heating_action,
                    o.predicted_cost,
                    o.controller_type,
                    o.action_label,
                    o.timestamp
                FROM optimization_runs o
                JOIN rooms r ON o.room_id = r.id
                ORDER BY o.timestamp DESC, o.id DESC
                LIMIT ?
                """,
                (limit,),
            )

        results = cursor.fetchall()
        conn.close()
        return [dict(row) for row in results]

    def get_room_summary(self, room_name: str) -> Optional[Dict]:
        """Return aggregated reporting metrics for a room."""
        conn = self._connect()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                r.name AS room_name,
                r.zone,
                COUNT(o.id) AS optimization_runs,
                COALESCE(AVG(o.predicted_cost), 0.0) AS avg_predicted_cost,
                COALESCE(SUM(o.predicted_cost), 0.0) AS total_predicted_cost,
                (
                    SELECT controller_type
                    FROM optimization_runs o2
                    WHERE o2.room_id = r.id
                    ORDER BY o2.timestamp DESC, o2.id DESC
                    LIMIT 1
                ) AS last_controller_type,
                (
                    SELECT action_label
                    FROM optimization_runs o3
                    WHERE o3.room_id = r.id
                    ORDER BY o3.timestamp DESC, o3.id DESC
                    LIMIT 1
                ) AS last_action_label,
                (
                    SELECT timestamp
                    FROM optimization_runs o4
                    WHERE o4.room_id = r.id
                    ORDER BY o4.timestamp DESC, o4.id DESC
                    LIMIT 1
                ) AS last_optimization_at
            FROM rooms r
            LEFT JOIN optimization_runs o ON o.room_id = r.id
            WHERE r.name = ?
            GROUP BY r.id, r.name, r.zone
            """,
            (room_name,),
        )

        result = cursor.fetchone()
        conn.close()
        return dict(result) if result else None
