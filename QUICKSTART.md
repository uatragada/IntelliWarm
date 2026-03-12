# Quick Start Guide

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Initialize Configuration

The system uses `configs/config.yaml` for all settings. Key sections:

```yaml
# Comfort levels
comfort:
  min_temperature: 18
  max_temperature: 24
  default_target: 21

# Rooms you want to optimize
rooms:
  bedroom:
    target_temp: 21
    occupancy_schedule: "9-18"  # 9am to 6pm

# Energy prices
energy:
  electricity_price: 0.12  # $/kWh
  gas_price: 5.0           # $/therm
```

## 3. Run Examples

Test the system without the web UI:

```bash
python examples.py
```

This demonstrates:
- Basic optimization
- Multi-room setups
- Thermal model learning
- Cost analysis

## 4. Start Web Dashboard

```bash
python app.py
```

Then open: **http://localhost:5000**

### Dashboard Features

- Add rooms
- Configure zones and energy prices
- Load demo data from CSV
- View room status
- Monitor optimization plans

## 5. Run Tests

```bash
pip install pytest
pytest tests/
```

## 6. Explore the Code

### Key Modules

| Module | File | Purpose |
|--------|------|---------|
| Thermal Model | `intelliwarm/models/thermal_model.py` | Physics-based temperature prediction |
| Optimizer | `intelliwarm/optimizer/mpc_controller.py` | Cost minimization engine |
| Sensors | `intelliwarm/sensors/sensor_manager.py` | Data collection interface |
| Database | `intelliwarm/storage/database.py` | Data persistence |

### Example: Manual Optimization

```python
from intelliwarm.core import SystemConfig
from intelliwarm.models import RoomThermalModel
from intelliwarm.optimizer import MPCController, CostFunction

# Load config
config = SystemConfig("configs/config.yaml")

# Create model
model = RoomThermalModel("bedroom", alpha=0.1, beta=0.05)

# Create optimizer
optimizer = MPCController(config, model)

# Compute optimal plan
plan = optimizer.compute_optimal_plan(
    room_name="bedroom",
    current_temp=18.0,
    outside_temp=5.0,
    target_temp=21.0,
    energy_prices=[0.12] * 24,
    occupancy_probs=[0.8]*12 + [0.1]*12,
    current_action=0.0
)

print(f"Next action: {plan['next_action']:.0%}")  # 0-100%
print(f"Cost: ${plan['total_cost']:.2f}")
```

## 7. Architecture Overview

```
Sensors (hardware)
    ↓
SensorManager (read data)
    ↓
RoomThermalModel (predict temperature)
    ↓
OccupancyPredictor (forecast occupancy)
    ↓
EnergyPriceService (get price forecast)
    ↓
MPCController (optimize)
    ↓
DeviceController (send commands)
    ↓
Database (log everything)
```

## Next Steps

1. **Add real rooms** - Configure actual room properties in `config.yaml`
2. **Connect sensors** - Implement real sensor readers
3. **Integrate devices** - Add smart plug/thermostat control
4. **Tune parameters** - Adjust comfort weights and thermal coefficients
5. **Analyze results** - Review saved optimization decisions

## Troubleshooting

### ImportError: No module named 'intelliwarm'

```bash
# Make sure you're in the project root
cd /path/to/IntelliWarm
python app.py
```

### Database errors

```bash
# Delete old database to reset
rm intelliwarm.db
python app.py  # Will create a fresh database
```

### Missing dependencies

```bash
pip install -r requirements.txt --upgrade
```

## More Documentation

- **Full SRS**: See README.md (System Overview section)
- **API Reference**: See docstrings in Python modules
- **Examples**: Run `python examples.py`
- **Tests**: Browse `tests/test_modules.py`

---

**Questions?** Open an issue on GitHub or review the code comments.
