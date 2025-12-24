"""
Domain models for detections and locations.
"""
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple


@dataclass
class Detection:
    """Detection result from ML model."""
    label: str
    confidence: float
    probs: List[float]
    raw_label: str
    raw_conf: float
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)
    
    @property
    def is_unknown(self) -> bool:
        """Check if detection is unknown."""
        return self.label == "Unknown"


@dataclass
class Location:
    """Location record for drop-off centers."""
    id: str
    name: str
    lat: float
    lon: float
    address: str
    hours: Optional[str] = None
    phone: Optional[str] = None
    types: List[str] = None  # e.g., ["PET", "HDPE", "PP", "PS"]
    source: str = "seed"
    updated_at: Optional[str] = None
    distance_km: Optional[float] = None  # Calculated distance from user
    
    def __post_init__(self):
        if self.types is None:
            self.types = []
    
    def accepts_type(self, plastic_type: str) -> bool:
        """Check if location accepts the plastic type."""
        if plastic_type in self.types:
            return True
        # Check for GENERAL tag
        if "GENERAL" in self.types or "MIXED" in self.types:
            return True
        return False

