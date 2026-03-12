# IntelliWarm — Intelligent HVAC Optimization Platform

**Version 1.0** | **Status: In Development**

IntelliWarm is an intelligent heating optimization system designed to minimize energy costs while maintaining occupant comfort. It combines predictive thermal modeling, occupancy forecasting, and Model Predictive Control (MPC) to dynamically optimize heating schedules.

---

## 🎯 System Overview

IntelliWarm acts as a **decision engine** between sensors and HVAC systems, providing:

- **Real-time Temperature Monitoring**: Aggregates sensor data from multiple rooms
- **Predictive Thermal Modeling**: Learns room thermal dynamics and predicts future temperatures
- **Occupancy Forecasting**: Predicts when rooms are occupied to optimize comfort
- **Cost-Based Optimization**: Uses MPC to minimize energy costs while respecting comfort constraints
- **Device Control**: Communicates with smart thermostats, plugs, and relays
- **Historical Analytics**: Logs all sensor data, optimization decisions, and energy costs

**Primary Goal**: Minimize energy cost while maintaining comfort constraints.

---

## 📦 Architecture

The system is structured into modular services:

```
IntelliWarm/
├── intelliwarm/
│   ├── sensors/          → Temperature & occupancy sensors
│   ├── models/           → Thermal dynamics model
│   ├── prediction/       → Occupancy prediction engine
│   ├── pricing/          → Energy price management
│   ├── optimizer/        → MPC optimization controller
│   ├── control/          → Device control interface
│   ├── storage/          → SQLite database layer
│   ├── learning/         → Model retraining pipeline
│   └── core/             → Config & scheduler
├── configs/              → Configuration files (YAML)
├── app.py                → Flask web application
└── templates/            → HTML templates
```

### Key Components

| Module | Purpose | Key Classes |
|--------|---------|-------------|
| **Sensors** | Collects temperature & occupancy data | `SensorManager`, `TemperatureSensor`, `OccupancySensor` |
| **Models** | Thermal dynamics prediction | `RoomThermalModel` |
| **Prediction** | Occupancy forecasting | `OccupancyPredictor` |
| **Pricing** | Energy price management | `EnergyPriceService` |
| **Optimizer** | MPC cost minimization | `MPCController`, `CostFunction` |
| **Control** | Device command execution | `DeviceController`, `SimulatedHeater` |
| **Storage** | Data persistence | `Database` (SQLite) |

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/uatragada/IntelliWarm.git
cd IntelliWarm

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure System

Edit `configs/config.yaml` to define:
- Room configurations (target temps, heater power)
- Comfort constraints
- Energy prices
- Optimization parameters

Example:
```yaml
rooms:
  bedroom1:
    zone: "Residential"
    room_size: 150  # sq ft
    target_temp: 21
    heater_power: 1500  # watts
    occupancy_schedule: "9-18"  # 9am-6pm
```

### 3. Run Application

```bash
# Start Flask application
python app.py

# Visit: http://localhost:5000
```

### 4. Load Demo Data

Click "Load Demo" to populate with sample rooms from CSV data.

---

## 🏗️ Phase 1 — Core MVP (Current Phase)

- [x] Sensor data ingestion module
- [x] Thermal model with parameter learning
- [x] MPC cost minimization optimization
- [x] Simulation environment (no real device control)
- [x] Database storage layer
- [x] Flask dashboard scaffolding

### Running the System

```python
from intelliwarm.core import SystemConfig
from intelliwarm.sensors import SensorManager
from intelliwarm.models import RoomThermalModel
from intelliwarm.optimizer import MPCController

# Load config
config = SystemConfig("configs/config.yaml")

# Initialize sensors
sensors = SensorManager()
sensors.register_temperature_sensor("bedroom", 20.0)

# Create thermal model
model = RoomThermalModel("bedroom", alpha=0.1, beta=0.05)

# Create optimizer
optimizer = MPCController(config, model)

# Compute optimal plan
plan = optimizer.compute_optimal_plan(
    room_name="bedroom",
    current_temp=20.0,
    outside_temp=5.0,
    target_temp=21.0,
    energy_prices=[0.12] * 24,
    occupancy_probs=[0.1] * 9 + [0.8] * 9 + [0.1] * 6,
    current_action=0.0
)

print(f"Optimal heating actions: {plan['optimal_actions']}")
print(f"Predicted temperatures: {plan['predicted_temperatures']}")
print(f"Total cost: ${plan['total_cost']:.2f}")
```

---

## 🔄 Thermal Model

The system models room heating dynamics:

```
T(t+1) = T(t) + α*H(t) - β*(T(t) - T_outside)
```

Where:
- **T(t)** = room temperature at time t
- **H(t)** = heating power (0-1, normalized)
- **α** = heating efficiency coefficient
- **β** = heat loss coefficient

Parameters are automatically learned from historical sensor data using least squares estimation.

---

## 📊 Cost Optimization

The MPC optimizer minimizes:

```
Cost = Energy_Cost + λ₁*Discomfort_Penalty + λ₂*Switching_Penalty
```

Where:
- **Energy_Cost** = heating duration × energy price
- **Discomfort_Penalty** = penalty for temperature deviations outside comfort zone
- **Switching_Penalty** = penalty for frequent on/off cycles

---

## 💾 Database

SQLite database stores:
- Room configurations
- Temperature logs
- Occupancy predictions
- Energy prices
- Optimization decisions
- Model parameters

Access via the `Database` class:

```python
from intelliwarm.storage import Database

db = Database("intelliwarm.db")

# Log temperature
db.add_temperature_log("bedroom", temp=20.5, humidity=45, outside_temp=5.0)

# Get history
history = db.get_temperature_history("bedroom", limit=100)

# Save thermal model parameters
db.save_model_parameters("bedroom", alpha=0.12, beta=0.048)
```

---

## 🌐 API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Dashboard |
| `/add_room` | GET/POST | Add new room |
| `/config_home` | GET/POST | Configure zones & prices |
| `/demo` | GET | Load demo data |
| `/api/rooms` | GET | Get all rooms (JSON) |
| `/api/optimization/<room>` | GET | Get optimization plan |

---

## Phase 2 — Smart System (Planned)

- [ ] Integration with real weather APIs
- [ ] Occupancy learning from historical data
- [ ] Multi-room optimization (shared resources)
- [ ] Bayesian occupancy prediction model
- [ ] Advanced dashboard with heatmaps

---

## Phase 3 — Production (Planned)

- [ ] Real device control (smart plugs, thermostats)
- [ ] Cloud deployment (Azure, AWS)
- [ ] Multi-building scaling
- [ ] RL-based adaptive control
- [ ] Mobile app

---

## 🧪 Testing

Run unit tests:

```bash
pytest tests/
```

Key test files:
- `tests/test_thermal_model.py`
- `tests/test_optimizer.py`
- `tests/test_database.py`

---

## 📚 Documentation

- **[SRS Document](docs/SRS.md)** — Full Software Requirements Specification
- **[Architecture Guide](docs/ARCHITECTURE.md)** — System design details
- **[API Reference](docs/API.md)** — Python module documentation

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| Backend | Python 3.9+, Flask |
| ML/Optimization | NumPy, SciPy, scikit-learn |
| Database | SQLite (MVP) / PostgreSQL (production) |
| Configuration | YAML |
| Frontend | HTML/CSS/JavaScript (Plotly) |

---

## 🤝 Contributing

Contributions welcome! Areas for enhancement:

1. **Real device integration** — Smart plugs, thermostats, relays
2. **Weather API integration** — OpenWeatherMap, NOAA
3. **ML forecasting** — ARIMA, LSTM for occupancy
4. **Reinforcement learning** — Q-learning for adaptive control
5. **Multi-zone optimization** — Coupling between rooms
6. **Cloud deployment** — Docker, Kubernetes

---

## 📄 License

MIT License — See LICENSE file

---

## 👤 Author

**Uday Atragada** — [GitHub](https://github.com/uatragada)

---

## 🔗 References

- Research: HVAC Optimization via MPC ([GitHub Link](https://github.com))
- Energy Efficiency: Building automation best practices
- Control Theory: Model Predictive Control fundamentals

---

## ❓ FAQ

**Q: Does IntelliWarm control real HVAC systems?**  
A: Currently, it runs in simulation mode. Phase 3 will add real device control via smart plugs/thermostats.

**Q: What's the optimization speed?**  
A: Target < 2 seconds per room per cycle (configurable in `config.yaml`).

**Q: Can I use this in production?**  
A: Phase 1 is for MVP/research. Production deployment requires Phase 3 completion.

**Q: How is occupancy predicted?**  
A: Currently rule-based via schedules. Phase 2 will add ML-based prediction.

---

**Last Updated**: March 2026
