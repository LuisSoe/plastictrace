"""Schema for drop-off locations."""

from dataclasses import dataclass, asdict
from typing import List, Optional, Dict


@dataclass
class DropOffLocation:
    """Drop-off location record."""
    id: str
    name: str
    lat: float
    lng: float
    address: str
    hours: Optional[str] = None
    contact: Optional[str] = None
    source: str = "seed"  # "seed", "community", "partner"
    accepted_types: List[str] = None  # e.g., ["PET","HDPE","PP"]
    conditions_required: Optional[Dict[str, bool]] = None  # {clean: true, label_removed: false}
    notes: Optional[str] = None
    
    def __post_init__(self):
        if self.accepted_types is None:
            self.accepted_types = []
        if self.conditions_required is None:
            self.conditions_required = {}
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'DropOffLocation':
        """Create from dictionary."""
        return cls(**data)
    
    def accepts_type(self, plastic_type: str, eligible_tags: List[str]) -> bool:
        """
        Check if location accepts the plastic type.
        
        Args:
            plastic_type: Plastic type to check
            eligible_tags: Eligible tags from recommendation
            
        Returns:
            True if location accepts the type
        """
        # Check direct match
        if plastic_type in self.accepted_types:
            return True
        
        # Check tag matches
        for tag in eligible_tags:
            if tag in self.accepted_types:
                return True
        
        # Check for GENERAL tag
        if "GENERAL" in self.accepted_types or "MIXED" in self.accepted_types:
            return True
        
        return False
    
    def meets_conditions(self, conditions) -> tuple[bool, Optional[str]]:
        """
        Check if location's condition requirements are met.
        
        Args:
            conditions: ConditionsData object
            
        Returns:
            Tuple of (meets_requirements, reason_if_not)
        """
        if not self.conditions_required:
            return True, None
        
        # Check clean requirement
        if self.conditions_required.get("clean") is True:
            if conditions.clean is False:
                return False, "Requires cleaning"
        
        # Check label_removed requirement
        if self.conditions_required.get("label_removed") is True:
            if conditions.label_present is True:
                return False, "Requires label removal"
        
        return True, None

