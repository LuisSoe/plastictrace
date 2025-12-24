"""
Geographic utilities for distance calculation and filtering.
"""
import math
from typing import List, Tuple
from domain.models import Location


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate distance between two points using Haversine formula.
    
    Args:
        lat1, lon1: First point (latitude, longitude)
        lat2, lon2: Second point (latitude, longitude)
    
    Returns:
        Distance in kilometers
    """
    R = 6371  # Earth radius in km
    
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c


def filter_locations(
    locations: List[Location],
    user_lat: float,
    user_lon: float,
    radius_km: float,
    types_selected: List[str]
) -> List[Location]:
    """
    Filter locations by radius and accepted types.
    
    Args:
        locations: List of locations
        user_lat, user_lon: User's location
        radius_km: Search radius in kilometers
        types_selected: List of plastic types to filter by
    
    Returns:
        Filtered and sorted list of locations (by distance)
    """
    filtered = []
    
    for loc in locations:
        # Check if location accepts any of the selected types
        if types_selected:
            accepts = any(loc.accepts_type(t) for t in types_selected)
            if not accepts:
                continue
        
        # Calculate distance
        distance = haversine_distance(user_lat, user_lon, loc.lat, loc.lon)
        
        # Filter by radius
        if distance <= radius_km:
            loc.distance_km = distance
            filtered.append(loc)
    
    # Sort by distance
    filtered.sort(key=lambda x: x.distance_km or float('inf'))
    
    return filtered

