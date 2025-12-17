"""Rules engine for recycling recommendations based on region and conditions."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from feedback.schema import ConditionsData


@dataclass
class Recommendation:
    """Disposal recommendation."""
    recyclable: Union[bool, str]  # True, False, or "depends"
    instructions: List[str]
    warnings: List[str]
    dropoff_required: bool
    eligible_dropoff_tags: List[str]


@dataclass
class Ruleset:
    """Ruleset for a region."""
    region: Dict[str, str]  # {country, province, city}
    rules: Dict[str, Dict]  # Per plastic type rules


class RulesEngine:
    """Pluggable rules engine for recycling recommendations."""
    
    def __init__(self, rules_dir: str = "rules"):
        """
        Initialize rules engine.
        
        Args:
            rules_dir: Directory containing rules JSON files
        """
        self.rules_dir = Path(rules_dir)
        self.rules_dir.mkdir(exist_ok=True)
        self._rulesets: Dict[str, Ruleset] = {}
        self._default_ruleset: Optional[Ruleset] = None
        
        # Load default ruleset
        self._load_default_ruleset()
    
    def _load_default_ruleset(self):
        """Load default ruleset."""
        default_path = self.rules_dir / "default.json"
        if default_path.exists():
            with open(default_path, 'r') as f:
                data = json.load(f)
                self._default_ruleset = Ruleset(
                    region=data.get("region", {}),
                    rules=data.get("rules", {})
                )
        else:
            # Create default ruleset
            self._default_ruleset = self._create_default_ruleset()
            self._save_ruleset(self._default_ruleset, default_path)
    
    def _create_default_ruleset(self) -> Ruleset:
        """Create default ruleset with base rules."""
        return Ruleset(
            region={"country": "ID", "province": "Default", "city": "Default"},
            rules={
                "PET": {
                    "recyclable": True,
                    "base_instructions": [
                        "Rinse thoroughly with water",
                        "Remove cap and label if possible",
                        "Flatten to save space"
                    ],
                    "condition_overrides": {
                        "dirty": {
                            "warnings": ["Item must be cleaned before recycling"],
                            "dropoff_required": True
                        }
                    },
                    "eligible_dropoff_tags": ["PET", "BOTTLES", "GENERAL"]
                },
                "HDPE": {
                    "recyclable": True,
                    "base_instructions": [
                        "Rinse if used for food",
                        "Remove labels if possible",
                        "Check local guidelines"
                    ],
                    "condition_overrides": {
                        "dirty": {
                            "warnings": ["Clean before recycling"],
                            "dropoff_required": True
                        }
                    },
                    "eligible_dropoff_tags": ["HDPE", "HARD_PLASTIC", "GENERAL"]
                },
                "PP": {
                    "recyclable": True,
                    "base_instructions": [
                        "Rinse if used for food",
                        "Check label for specific instructions",
                        "Avoid if oily or contaminated"
                    ],
                    "condition_overrides": {
                        "dirty": {
                            "warnings": ["Must be clean for recycling"],
                            "dropoff_required": True
                        },
                        "mixed": {
                            "warnings": ["Mixed materials may not be accepted"],
                            "dropoff_required": True
                        }
                    },
                    "eligible_dropoff_tags": ["PP", "CONTAINERS", "GENERAL"]
                },
                "PS": {
                    "recyclable": "depends",
                    "base_instructions": [
                        "Check if local facility accepts PS",
                        "Do not burn or incinerate",
                        "Consider alternative disposal if not recyclable"
                    ],
                    "warnings": [
                        "Many centers do not accept PS foam",
                        "Styrofoam is difficult to recycle"
                    ],
                    "eligible_dropoff_tags": ["PS", "SPECIALTY"]
                },
                "OTHER": {
                    "recyclable": "depends",
                    "base_instructions": [
                        "Check resin code (number in triangle)",
                        "Consult local recycling guidelines",
                        "Do not contaminate recycling stream"
                    ],
                    "warnings": [
                        "Verification required",
                        "May not be accepted at all centers"
                    ],
                    "eligible_dropoff_tags": ["MIXED", "GENERAL"]
                },
                "UNKNOWN": {
                    "recyclable": "depends",
                    "base_instructions": [
                        "Verify resin code manually",
                        "Check for recycling symbol",
                        "Consult local guidelines",
                        "Do not contaminate recycling stream"
                    ],
                    "warnings": [
                        "Manual verification required",
                        "Uncertainty in classification"
                    ],
                    "eligible_dropoff_tags": ["MIXED", "GENERAL", "VERIFICATION"]
                }
            }
        )
    
    def _save_ruleset(self, ruleset: Ruleset, path: Path):
        """Save ruleset to file."""
        data = {
            "region": ruleset.region,
            "rules": ruleset.rules
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_ruleset(self, region: Dict[str, str]) -> Ruleset:
        """
        Load ruleset for a region.
        
        Args:
            region: Dict with country, province, city
            
        Returns:
            Ruleset for region (falls back to default)
        """
        # Try to load region-specific ruleset
        region_key = f"{region.get('country', 'ID')}-{region.get('province', '').lower().replace(' ', '-')}-{region.get('city', '').lower().replace(' ', '-')}"
        region_path = self.rules_dir / f"{region_key}.json"
        
        if region_path.exists():
            with open(region_path, 'r') as f:
                data = json.load(f)
                return Ruleset(
                    region=data.get("region", region),
                    rules=data.get("rules", {})
                )
        
        # Fall back to default
        return self._default_ruleset
    
    def recommend(self, plastic_type: str, conditions: Optional[ConditionsData],
                  region: Dict[str, str]) -> Recommendation:
        """
        Get disposal recommendation.
        
        Args:
            plastic_type: Detected plastic type
            conditions: Condition flags
            region: Region dict with country, province, city
            
        Returns:
            Recommendation object
        """
        ruleset = self.load_ruleset(region)
        
        # Get rule for plastic type (fallback to OTHER if not found)
        rule = ruleset.rules.get(plastic_type, ruleset.rules.get("OTHER", {}))
        
        # Start with base values
        recyclable = rule.get("recyclable", "depends")
        instructions = rule.get("base_instructions", []).copy()
        warnings = rule.get("warnings", []).copy()
        dropoff_required = rule.get("dropoff_required", False)
        eligible_dropoff_tags = rule.get("eligible_dropoff_tags", ["GENERAL"]).copy()
        
        # Apply condition overrides
        if conditions and "condition_overrides" in rule:
            overrides = rule["condition_overrides"]
            
            # Check dirty condition
            if conditions.clean is False and "dirty" in overrides:
                override = overrides["dirty"]
                warnings.extend(override.get("warnings", []))
                if override.get("dropoff_required"):
                    dropoff_required = True
            
            # Check mixed condition
            if conditions.mixed is True and "mixed" in overrides:
                override = overrides["mixed"]
                warnings.extend(override.get("warnings", []))
                if override.get("dropoff_required"):
                    dropoff_required = True
            
            # Check label_present condition
            if conditions.label_present is True and "label_present" in overrides:
                override = overrides["label_present"]
                if "instructions" in override:
                    instructions.extend(override["instructions"])
        
        return Recommendation(
            recyclable=recyclable,
            instructions=instructions,
            warnings=warnings,
            dropoff_required=dropoff_required,
            eligible_dropoff_tags=eligible_dropoff_tags
        )

