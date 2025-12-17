"""Location filtering and ranking."""

import math
from typing import List, Tuple, Optional
from dataclasses import dataclass

from location.dropoff_schema import DropOffLocation
from location.rules_engine import Recommendation
from feedback.schema import ConditionsData


@dataclass
class RankedLocation:
    """Ranked location with reason."""
    location: DropOffLocation
    distance: float  # in km (or None if no user location)
    rank_score: float  # 0-1, higher is better
    reason: str  # e.g., "Accepts PET", "Requires cleaning"
    excluded: bool = False  # True if should be excluded


class LocationFilterRanker:
    """Filters and ranks drop-off locations."""
    
    def __init__(self):
        """Initialize filter/ranker."""
        pass
    
    def filter_and_rank(self,
                       locations: List[DropOffLocation],
                       plastic_type: str,
                       recommendation: Recommendation,
                       conditions: Optional[ConditionsData],
                       user_lat: Optional[float] = None,
                       user_lng: Optional[float] = None) -> List[RankedLocation]:
        """
        Filter and rank locations.
        
        Args:
            locations: List of all locations
            plastic_type: Detected plastic type
            recommendation: Recommendation from rules engine
            conditions: Condition flags
            user_lat: User latitude (optional)
            user_lng: User longitude (optional)
            
        Returns:
            List of ranked locations (sorted by rank_score, highest first)
        """
        ranked = []
        
        for location in locations:
            # Check if location accepts the type
            accepts = location.accepts_type(plastic_type, recommendation.eligible_dropoff_tags)
            
            if not accepts:
                # Exclude if doesn't accept type
                ranked.append(RankedLocation(
                    location=location,
                    distance=self._calculate_distance(user_lat, user_lng, location.lat, location.lng) if user_lat and user_lng else None,
                    rank_score=0.0,
                    reason=f"Does not accept {plastic_type}",
                    excluded=True
                ))
                continue
            
            # Check condition requirements
            meets_conditions, condition_reason = location.meets_conditions(conditions) if conditions else (True, None)
            
            # Calculate distance
            distance = self._calculate_distance(user_lat, user_lng, location.lat, location.lng) if user_lat and user_lng else None
            
            # Calculate rank score
            rank_score = self._calculate_rank_score(
                accepts=True,
                meets_conditions=meets_conditions,
                distance=distance,
                source=location.source
            )
            
            # Determine reason
            if meets_conditions:
                reason = f"Accepts {plastic_type}"
                if distance is not None:
                    reason += f" ({distance:.1f} km away)"
            else:
                reason = condition_reason or "Condition requirements not met"
                # Downgrade but don't exclude
                rank_score *= 0.5
            
            ranked.append(RankedLocation(
                location=location,
                distance=distance,
                rank_score=rank_score,
                reason=reason,
                excluded=False
            ))
        
        # Sort by rank_score (highest first), then by distance if available
        ranked.sort(key=lambda x: (
            -x.rank_score,  # Negative for descending
            x.distance if x.distance is not None else float('inf')
        ))
        
        return ranked
    
    def _calculate_distance(self, lat1: Optional[float], lng1: Optional[float],
                           lat2: float, lng2: float) -> Optional[float]:
        """
        Calculate distance between two points (Haversine formula).
        
        Returns distance in kilometers.
        """
        if lat1 is None or lng1 is None:
            return None
        
        # Earth radius in km
        R = 6371.0
        
        # Convert to radians
        lat1_rad = math.radians(lat1)
        lng1_rad = math.radians(lng1)
        lat2_rad = math.radians(lat2)
        lng2_rad = math.radians(lng2)
        
        # Differences
        dlat = lat2_rad - lat1_rad
        dlng = lng2_rad - lng1_rad
        
        # Haversine formula
        a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlng / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        distance = R * c
        return distance
    
    def _calculate_rank_score(self, accepts: bool, meets_conditions: bool,
                              distance: Optional[float], source: str) -> float:
        """
        Calculate ranking score for a location.
        
        Args:
            accepts: Whether location accepts the type
            meets_conditions: Whether condition requirements are met
            distance: Distance in km (None if unknown)
            source: Location source (seed, community, partner)
            
        Returns:
            Score from 0.0 to 1.0
        """
        if not accepts:
            return 0.0
        
        score = 1.0
        
        # Distance factor (closer is better)
        if distance is not None:
            # Normalize distance (assume max 50km is reasonable)
            distance_factor = max(0.0, 1.0 - (distance / 50.0))
            score *= (0.3 + 0.7 * distance_factor)  # Distance is 70% of score
        else:
            # No distance info, use base score
            score *= 0.5
        
        # Source factor (prefer verified sources)
        if source == "partner":
            score *= 1.1  # Slight boost
        elif source == "community":
            score *= 0.9  # Slight penalty
        
        # Condition factor
        if not meets_conditions:
            score *= 0.5  # Downgrade but don't exclude
        
        return min(1.0, max(0.0, score))

