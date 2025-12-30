"""
One-time script to geocode all locations from Excel and save to JSON cache.
Run this once to create the cache file, then the app will use the cache instead of Excel.

Usage:
    python location/geocode_all.py
"""

import json
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from location.excel_loader import load_locations_from_sipsn
from domain.models import Location


def geocode_and_save_all(output_file: str = "data/locations_geocoded.json"):
    """
    Geocode all locations from Excel and save to JSON file.
    
    Args:
        output_file: Path to output JSON file
    """
    print("Loading locations from Excel file...")
    print("This will geocode locations (takes ~1 second per location)...")
    print("=" * 60)
    
    # Load and geocode all locations from alamat (address) field
    # Note: This will take ~1 second per location due to rate limiting
    # For 200 locations, expect ~3-4 minutes
    print("Geocoding all locations from 'Alamat' (address) column...")
    print("Each location takes ~1 second to geocode.")
    print("This will create accurate latitude/longitude for each location.\n")
    
    locations = load_locations_from_sipsn(
        "data_sipsn.xlsx", 
        enable_geocoding=True,  # Enable geocoding to get coordinates from alamat
        max_locations=200  # Process up to 200 locations
    )
    
    print(f"\n{'=' * 60}")
    print(f"Geocoded {len(locations)} locations")
    
    # Convert to JSON-serializable format
    locations_data = []
    for loc in locations:
        locations_data.append({
            'id': loc.id,
            'name': loc.name,
            'lat': loc.lat,
            'lon': loc.lon,
            'address': loc.address,
            'hours': loc.hours,
            'phone': loc.phone,
            'types': loc.types,
            'source': loc.source
        })
    
    # Save to JSON file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(locations_data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved to: {output_path.absolute()}")
    print(f"\nYou can now use this cache file instead of the Excel file!")
    
    # Show statistics
    unique_coords = len(set((loc['lat'], loc['lon']) for loc in locations_data))
    default_coords = sum(1 for loc in locations_data if loc['lat'] == -6.2297 and loc['lon'] == 106.7997)
    geocoded = len(locations_data) - default_coords
    
    print(f"\nStatistics:")
    print(f"  Total locations: {len(locations_data)}")
    print(f"  Successfully geocoded: {geocoded}")
    print(f"  Using default coordinates: {default_coords}")
    print(f"  Unique coordinate pairs: {unique_coords}")


if __name__ == "__main__":
    geocode_and_save_all()

