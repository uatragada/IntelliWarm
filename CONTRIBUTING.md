# Contributing to IntelliWarm

We welcome contributions! Here's how to help.

## Development Setup

1. **Fork the repository**
2. **Create a branch**: `git checkout -b feature/your-feature`
3. **Install dev dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install pytest pylint black
   ```

## Code Standards

### Style Guide
- Use **PEP 8** formatting
- Run linter: `pylint intelliwarm/`
- Format code: `black intelliwarm/`
- Aim for > 80% test coverage

### Docstrings
Every class and function should have a docstring:

```python
def compute_optimal_plan(self, room_name: str, current_temp: float) -> Dict:
    """
    Compute optimal heating schedule.
    
    Args:
        room_name: Name of the room
        current_temp: Current temperature in Celsius
        
    Returns:
        Dict with 'optimal_actions' and 'total_cost'
    """
```

### Type Hints
Use type hints for all function signatures:

```python
def set_heater(self, room_name: str, power_level: float) -> None:
    pass
```

## Testing

Add tests for new features:

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_modules.py::TestOptimizer::test_mpc_optimization

# Generate coverage report
pytest tests/ --cov=intelliwarm
```

## Commit Messages

Use clear, descriptive commit messages:

```
[Feature] Add weather API integration
[Bugfix] Fix thermal model parameter learning
[Docs] Update README with new API endpoints
[Test] Add tests for occupancy predictor
```

## Areas for Contribution

### High Priority
1. **Real device integration** → Smart plugs, thermostats
2. **Weather API** → Live temperature forecasts
3. **ML occupancy prediction** → Learn from patterns
4. **Multi-zone optimization** → Shared heating resources

### Medium Priority
5. **Dashboard improvements** → Charts, heatmaps
6. **Database migration** → PostgreSQL support
7. **Cloud deployment** → Docker, Kubernetes
8. **Mobile app** → React Native frontend

### Nice to Have
9. Reinforcement learning control
10. Cost/energy analytics reports
11. User authentication
12. Multi-property support

## Submitting a Pull Request

1. **Keep it focused** - One feature per PR
2. **Write tests** - New code needs tests
3. **Update docs** - Edit README if changing behavior
4. **Run tests locally**:
   ```bash
   pytest tests/ -v
   pylint intelliwarm/
   black intelliwarm/
   ```
5. **Push and create PR** - Include description of changes

## Project Structure

```
intelliwarm/
├── core/          # Config & scheduler
├── sensors/       # Temperature/occupancy
├── models/        # Thermal model
├── prediction/    # Occupancy forecasting
├── pricing/       # Energy prices
├── optimizer/     # MPC controller
├── control/       # Device commands
├── storage/       # Database
└── learning/      # Model retraining

tests/            # Unit tests
configs/          # Configuration files
templates/        # HTML templates
```

## Development Workflow

### Adding a new module:

1. Create directory: `intelliwarm/mymodule/`
2. Add `__init__.py` with exports
3. Implement core logic in feature files
4. Add unit tests in `tests/test_mymodule.py`
5. Update imports in `intelliwarm/__init__.py`
6. Document in README.md

### Example: Adding energy rate tariffs

```python
# intelliwarm/pricing/tariff_model.py
class TimeOfUseTariff:
    """Time-of-use electricity pricing"""
    
    def get_rate(self, hour: int) -> float:
        """Get electricity rate for hour of day"""
        if 6 <= hour < 9 or 17 <= hour < 21:
            return 0.18  # Peak
        elif 23 <= hour or hour < 6:
            return 0.08  # Off-peak
        else:
            return 0.12  # Mid-peak
```

## Questions?

- **Issues**: Open a GitHub issue
- **Discussions**: Use GitHub discussions
- **Email**: uday@example.com

## Code of Conduct

- Be respectful and inclusive
- Assume good intent
- Help others learn
- Report issues constructively

---

**Thank you for helping make IntelliWarm better!** 🌡️
