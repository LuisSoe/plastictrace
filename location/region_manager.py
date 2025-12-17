"""Region selection and persistence."""

import json
from pathlib import Path
from typing import Dict, Optional


class RegionManager:
    """Manages region selection and persistence."""
    
    def __init__(self, config_file: str = "data/region_config.json"):
        """
        Initialize region manager.
        
        Args:
            config_file: Path to config file for persistence
        """
        self.config_file = Path(config_file)
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Default region (Indonesia, Jakarta)
        self.default_region = {
            "country": "ID",
            "province": "DKI Jakarta",
            "city": "Jakarta"
        }
    
    def get_current_region(self) -> Dict[str, str]:
        """Get current selected region."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                    return data.get("region", self.default_region)
            except:
                pass
        
        return self.default_region.copy()
    
    def set_region(self, country: str, province: str, city: str):
        """
        Set current region.
        
        Args:
            country: Country code
            province: Province name
            city: City name
        """
        region = {
            "country": country,
            "province": province,
            "city": city
        }
        
        data = {"region": region}
        
        with open(self.config_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_indonesia_provinces(self) -> Dict[str, List[str]]:
        """Get Indonesia provinces and cities (simplified)."""
        return {
            "DKI Jakarta": ["Jakarta", "Jakarta Pusat", "Jakarta Selatan", "Jakarta Utara", "Jakarta Barat", "Jakarta Timur"],
            "Jawa Barat": ["Bandung", "Bekasi", "Bogor", "Depok", "Cimahi"],
            "Banten": ["Tangerang", "Serang", "Cilegon"],
            "Jawa Tengah": ["Semarang", "Surakarta", "Magelang"],
            "Jawa Timur": ["Surabaya", "Malang", "Sidoarjo"],
            "Bali": ["Denpasar", "Badung", "Gianyar"]
        }

