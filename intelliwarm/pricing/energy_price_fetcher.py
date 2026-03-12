"""
Energy Pricing Module
Retrieves and manages energy prices
"""

import logging
from typing import Dict, List
from datetime import datetime


class EnergyPriceService:
    """Manages electricity and gas prices"""
    
    def __init__(self, electricity_price: float = 0.12, gas_price: float = 5.0):
        """
        Initialize energy price service
        
        Args:
            electricity_price: $/kWh
            gas_price: $/therm
        """
        self.electricity_price = electricity_price
        self.gas_price = gas_price
        self.logger = logging.getLogger("IntelliWarm.Pricing")
        self.price_history: List[Dict] = []
    
    def get_current_electricity_price(self) -> float:
        """Get current electricity price"""
        return self.electricity_price
    
    def get_current_gas_price(self) -> float:
        """Get current gas price"""
        return self.gas_price
    
    def set_electricity_price(self, price: float):
        """Update electricity price"""
        self.electricity_price = price
        self._record_price()
        self.logger.info(f"Electricity price updated: ${price}/kWh")
    
    def set_gas_price(self, price: float):
        """Update gas price"""
        self.gas_price = price
        self._record_price()
        self.logger.info(f"Gas price updated: ${price}/therm")
    
    def _record_price(self):
        """Record price change"""
        self.price_history.append({
            "timestamp": datetime.now().isoformat(),
            "electricity": self.electricity_price,
            "gas": self.gas_price
        })
    
    def get_price_forecast(self, hours: int = 24, start_time: datetime = None) -> List[Dict]:
        """
        Get price forecast for next N hours
        
        Args:
            hours: Number of hours to forecast
            
        Returns:
            List of forecasted prices
        """
        # TODO: Integrate with real API (OpenWeather, utility API, etc.)
        # For now, return constant prices
        forecast = []
        current_price = {
            "electricity": self.electricity_price,
            "gas": self.gas_price
        }
        
        reference_time = start_time or datetime.now()

        for h in range(hours):
            # Simple pattern: prices may vary by time of day (TOU)
            hour = (reference_time.hour + h) % 24
            
            # Peak hours: 6am-9am and 4pm-9pm
            if (6 <= hour < 9) or (16 <= hour < 21):
                multiplier = 1.3
            # Off-peak: 11pm-6am
            elif (23 <= hour) or (hour < 6):
                multiplier = 0.8
            else:
                multiplier = 1.0
            
            forecast.append({
                "hour": hour,
                "electricity": current_price["electricity"] * multiplier,
                "gas": current_price["gas"]  # Gas typically doesn't have TOU pricing
            })
        
        return forecast
    
    def calculate_energy_cost(self, kwh: float, use_gas: bool = False) -> float:
        """
        Calculate energy cost
        
        Args:
            kwh: Energy in kWh (or therms if gas)
            use_gas: If True, treat input as therms and use gas price
            
        Returns:
            Cost in dollars
        """
        if use_gas:
            return kwh * self.gas_price
        else:
            return kwh * self.electricity_price
