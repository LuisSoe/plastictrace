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
    types_selected: List[str],
    max_results: int = None
) -> List[Location]:
    """
    Filter locations by radius and accepted types, return nearest locations.
    
    Args:
        locations: List of locations
        user_lat, user_lon: User's location
        radius_km: Search radius in kilometers (use None or very large value for no radius limit)
        types_selected: List of plastic types to filter by
        max_results: Maximum number of results to return (None for all)
    
    Returns:
        Filtered and sorted list of locations (by distance), limited to max_results
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
        loc.distance_km = distance
        
        # Filter by radius (if radius_km is None or very large, don't filter)
        if radius_km is None or radius_km >= 10000 or distance <= radius_km:
            filtered.append(loc)
    
    # Sort by distance
    filtered.sort(key=lambda x: x.distance_km or float('inf'))
    
    # Limit to max_results nearest locations
    if max_results is not None and max_results > 0:
        filtered = filtered[:max_results]
    
    return filtered

