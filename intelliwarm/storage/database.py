"""
Database Layer
SQLite implementation with ORM models
"""

import sqlite3
import json
import logging
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path


class Database:
    """SQLite database handler for IntelliWarm"""
    
    def __init__(self, db_path: str = "intelliwarm.db"):
        self.db_path = db_path
        self.logger = logging.getLogger("IntelliWarm.Database")
        self._init_schema()
    
    def _init_schema(self):
        """Create database tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
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
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (room_id) REFERENCES rooms(id)
            )
        """)
        
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
        self.logger.info(f"Database initialized: {self.db_path}")
    
    def add_temperature_log(self, room_name: str, temp: float, humidity: float = None, outside_temp: float = None):
        """Log temperature reading"""
        conn = sqlite3.connect(self.db_path)
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
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO rooms (name, zone, room_size, target_temp, heater_power)
                VALUES (?, ?, ?, ?, ?)
            """, (name, zone, room_size, target_temp, heater_power))
            conn.commit()
            self.logger.info(f"Room added: {name}")
        except sqlite3.IntegrityError:
            self.logger.warning(f"Room already exists: {name}")
        
        conn.close()
    
    def get_room(self, room_name: str) -> Dict:
        """Get room details"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM rooms WHERE name = ?", (room_name,))
        result = cursor.fetchone()
        conn.close()
        
        return dict(result) if result else None
    
    def get_all_rooms(self) -> List[Dict]:
        """Get all rooms"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM rooms")
        results = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in results]
    
    def record_optimization(self, room_name: str, heating_action: float, predicted_cost: float):
        """Record optimization decision"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT id FROM rooms WHERE name = ?", (room_name,))
        result = cursor.fetchone()
        
        if result:
            room_id = result[0]
            cursor.execute("""
                INSERT INTO optimization_runs (room_id, heating_action, predicted_cost)
                VALUES (?, ?, ?)
            """, (room_id, heating_action, predicted_cost))
            conn.commit()
        
        conn.close()
    
    def get_temperature_history(self, room_name: str, limit: int = 100) -> List[Dict]:
        """Get recent temperature readings"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
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
        conn = sqlite3.connect(self.db_path)
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
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
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
