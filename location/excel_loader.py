"""Load drop-off locations from Excel file (data_sipsn.xlsx) or cached JSON file."""

import os
import time
import json
import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple
from domain.models import Location

# Cache for geocoding results to avoid re-querying the same addresses
_geocode_cache: dict[str, Optional[Tuple[float, float]]] = {}


def geocode_address(address: str, max_retries: int = 3) -> Optional[Tuple[float, float]]:
    """
    Geocode an address to get latitude and longitude using Nominatim (OpenStreetMap).
    
    Args:
        address: Address string to geocode
        max_retries: Maximum number of retry attempts
        
    Returns:
        Tuple of (lat, lon) if successful, None otherwise
    """
    # Check cache first
    if address in _geocode_cache:
        return _geocode_cache[address]
    
    try:
        import urllib.request
        import urllib.parse
        import json
        
        # Rate limiting: wait 1 second between requests (Nominatim requires max 1 req/sec)
        time.sleep(1.0)
        
        # Clean and format address for better geocoding results
        # Remove extra spaces and normalize
        search_query = " ".join(address.split())
        
        # Add "Indonesia" to address if not present (helps with Indonesian addresses)
        if "Indonesia" not in search_query and "indonesia" not in search_query.lower():
            search_query = f"{search_query}, Indonesia"
        
        # Try multiple query formats with fallback strategy:
        # 1. Full address
        # 2. Kecamatan + Kabupaten (if available)
        # 3. Just Kabupaten
        # 4. Just first part of address
        queries_to_try = [search_query]  # Start with full address
        
        # Extract Kecamatan and Kabupaten from address
        parts = [p.strip() for p in address.split(',')]
        kecamatan = None
        kabupaten = None
        
        for part in parts:
            part_lower = part.lower()
            if 'kecamatan' in part_lower or 'kec.' in part_lower:
                kecamatan = part.replace('Kecamatan', '').replace('Kec.', '').strip()
            elif 'kabupaten' in part_lower or 'kab.' in part_lower or 'kota' in part_lower:
                kabupaten = part.replace('Kabupaten', '').replace('Kab.', '').replace('Kota', '').strip()
        
        # Build fallback queries
        if kecamatan and kabupaten:
            queries_to_try.append(f"{kecamatan}, {kabupaten}, Indonesia")
        if kabupaten:
            queries_to_try.append(f"{kabupaten}, Indonesia")
        if len(parts) > 0:
            queries_to_try.append(f"{parts[0]}, Indonesia")
        
        # Try each query format
        for query in queries_to_try:
            try:
                # Build geocoding URL
                base_url = "https://nominatim.openstreetmap.org/search"
                params = {
                    'q': query,
                    'format': 'json',
                    'limit': 1,
                    'countrycodes': 'id',  # Limit to Indonesia
                    'addressdetails': 1
                }
                url = f"{base_url}?{urllib.parse.urlencode(params)}"
                
                # Make request with proper user agent (required by Nominatim)
                req = urllib.request.Request(url, headers={'User-Agent': 'PlastiTrace/1.0'})
                
                with urllib.request.urlopen(req, timeout=10) as response:
                    data = json.loads(response.read().decode())
                    
                    if data and len(data) > 0:
                        result = data[0]
                        lat = float(result['lat'])
                        lon = float(result['lon'])
                        coords = (lat, lon)
                        _geocode_cache[address] = coords
                        return coords
            except Exception as e:
                # Try next query format if this one fails
                continue
        
        # All query formats failed
        _geocode_cache[address] = None
        return None
    except urllib.error.HTTPError as e:
        print(f"HTTP error geocoding '{address}': {e.code} - {e.reason}")
        if e.code == 429:
            print("  Rate limit exceeded, waiting longer...")
            time.sleep(2.0)  # Wait longer if rate limited
        _geocode_cache[address] = None
        return None
    except Exception as e:
        print(f"Geocoding error for '{address}': {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        _geocode_cache[address] = None
        return None


def load_locations_from_cache(cache_file: str = "data/locations_geocoded.json") -> List[Location]:
    """
    Load locations from cached JSON file (faster, no geocoding needed).
    
    Args:
        cache_file: Path to cached JSON file
        
    Returns:
        List of Location objects
    """
    cache_path = Path(cache_file)
    if not cache_path.exists():
        print(f"Cache file not found: {cache_file}")
        print("Run 'python -m location.geocode_all' to create the cache file first.")
        return []
    
    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            locations_data = json.load(f)
        
        locations = []
        for data in locations_data:
            location = Location(
                id=data['id'],
                name=data['name'],
                lat=data['lat'],
                lon=data['lon'],
                address=data['address'],
                hours=data.get('hours'),
                phone=data.get('phone'),
                types=data.get('types', ['GENERAL']),
                source=data.get('source', 'sipsn')
            )
            locations.append(location)
        
        print(f"Loaded {len(locations)} locations from cache file: {cache_file}")
        return locations
    except Exception as e:
        print(f"Error loading locations from cache: {e}")
        import traceback
        traceback.print_exc()
        return []


def load_locations_from_sipsn(file_path: str = "data_sipsn.xlsx", enable_geocoding: bool = False, max_locations: int = 200) -> List[Location]:
    """
    Load location data from SIPSN Excel file.
    
    Args:
        file_path: Path to the Excel file
        
    Returns:
        List of Location objects
    """
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found. Using empty locations.")
        return []
    
    locations = []
    
    try:
        # Use openpyxl engine specifically for .xlsx files
        df_raw = pd.read_excel(file_path, header=None, engine='openpyxl')
        
        # Find header row (containing 'Nama' or similar)
        header_row = -1
        for i, row in df_raw.iterrows():
            row_values = [str(val).lower() if pd.notna(val) else '' for val in row.values]
            if any('nama' in val or 'name' in val or 'fasilitas' in val for val in row_values):
                header_row = i
                break
        
        if header_row == -1:
            print("Warning: Could not find header row in Excel. Using empty locations.")
            return []
        
        # Set columns and cleanup
        df = df_raw.iloc[header_row+1:].copy()
        df.columns = df_raw.iloc[header_row]
        
        # Normalize column names (case-insensitive)
        column_map = {}
        for col in df.columns:
            col_lower = str(col).lower()
            if 'nama' in col_lower or 'name' in col_lower:
                column_map[col] = 'name'
            elif 'alamat' in col_lower or 'address' in col_lower:
                column_map[col] = 'address'
            elif 'lat' in col_lower or 'latitude' in col_lower:
                column_map[col] = 'lat'
            elif 'lng' in col_lower or 'lon' in col_lower or 'longitude' in col_lower:
                column_map[col] = 'lon'
            elif 'kecamatan' in col_lower:
                column_map[col] = 'kecamatan'
            elif 'kabupaten' in col_lower or 'kota' in col_lower:
                column_map[col] = 'kabupaten'
            elif 'jam' in col_lower or 'hours' in col_lower or 'waktu' in col_lower:
                column_map[col] = 'hours'
            elif 'kontak' in col_lower or 'contact' in col_lower or 'phone' in col_lower or 'telp' in col_lower:
                column_map[col] = 'contact'
        
        df = df.rename(columns=column_map)
        
        # Drop rows without name
        if 'name' in df.columns:
            df = df.dropna(subset=['name'])
        else:
            print("Warning: No 'name' column found. Using empty locations.")
            return []
        
        # Process each row (limit for performance, especially with geocoding)
        processed_count = 0
        geocoded_count = 0
        # Always geocode from alamat, but limit to avoid very long wait times
        max_geocoding = 200 if enable_geocoding else 0  # Process all locations if geocoding enabled
        
        for idx, row in df.iterrows():
            if processed_count >= max_locations:
                print(f"Reached limit of {max_locations} locations. Stopping.")
                break
            try:
                name = str(row['name']).strip()
                if not name or name == 'nan':
                    continue
                
                # Build address string from Alamat + Kecamatan + Kabupaten/Kota
                # This full address will be used for geocoding to get accurate coordinates
                addr_parts = []
                if 'address' in df.columns and pd.notna(row.get('address')):
                    addr_parts.append(str(row['address']))
                if 'kecamatan' in df.columns and pd.notna(row.get('kecamatan')):
                    addr_parts.append(str(row['kecamatan']))
                if 'kabupaten' in df.columns and pd.notna(row.get('kabupaten')):
                    addr_parts.append(str(row['kabupaten']))
                address = ", ".join(addr_parts) if addr_parts else "Jakarta"
                
                # Store the full address for geocoding (will be used to get lat/lon)
                full_address_for_geocoding = address
                
                # Get coordinates - try multiple column name variations
                lat = None
                lon = None
                
                # Try to find latitude
                lat_cols = [col for col in df.columns if 'lat' in str(col).lower()]
                for col in lat_cols:
                    if pd.notna(row.get(col)):
                        try:
                            lat_val = row[col]
                            # Handle string coordinates
                            if isinstance(lat_val, str):
                                lat_val = lat_val.replace(',', '.').strip()
                            lat = float(lat_val)
                            if -90 <= lat <= 90:  # Valid latitude range
                                break
                        except (ValueError, TypeError):
                            continue
                
                # Try to find longitude
                lon_cols = [col for col in df.columns if any(
                    keyword in str(col).lower() 
                    for keyword in ['lon', 'lng', 'longitude', 'long']
                )]
                for col in lon_cols:
                    if pd.notna(row.get(col)):
                        try:
                            lon_val = row[col]
                            # Handle string coordinates
                            if isinstance(lon_val, str):
                                lon_val = lon_val.replace(',', '.').strip()
                            lon = float(lon_val)
                            if -180 <= lon <= 180:  # Valid longitude range
                                break
                        except (ValueError, TypeError):
                            continue
                
                # If no coordinates found in Excel, ALWAYS geocode from Alamat (address) field
                # The alamat field contains the address which we geocode to get lat/lon
                if lat is None or lon is None:
                    if full_address_for_geocoding and full_address_for_geocoding != "Jakarta":
                        # Geocode from the full address (Alamat + Kecamatan + Kabupaten)
                        if enable_geocoding and geocoded_count < max_geocoding:
                            print(f"Geocoding {geocoded_count + 1}/{max_geocoding}: {name}...")
                            print(f"  Address: {full_address_for_geocoding}")
                            coords = geocode_address(full_address_for_geocoding)
                            if coords:
                                lat, lon = coords
                                geocoded_count += 1
                                print(f"  ✓ Geocoded to {lat}, {lon}")
                            else:
                                # Fallback: use default Jakarta coordinates if geocoding fails
                                print(f"  ✗ Geocoding failed, using default coordinates")
                                lat = -6.2297  # Jakarta Selatan default
                                lon = 106.7997
                        elif not enable_geocoding:
                            # Geocoding disabled, use default coordinates
                            lat = -6.2297
                            lon = 106.7997
                        else:
                            # Reached geocoding limit, use default coordinates
                            lat = -6.2297
                            lon = 106.7997
                    else:
                        # No address available, use default Jakarta coordinates
                        lat = -6.2297
                        lon = 106.7997
                
                # Get hours
                hours = None
                if 'hours' in df.columns and pd.notna(row.get('hours')):
                    hours = str(row['hours'])
                
                # Get contact
                contact = None
                if 'contact' in df.columns and pd.notna(row.get('contact')):
                    contact = str(row['contact'])
                
                # Generate ID
                location_id = f"sipsn_{idx}"
                
                # Extract accepted types from Excel if available
                types = ["GENERAL"]  # Default to accepting all types
                
                # Check for type-related columns
                type_columns = [col for col in df.columns if any(
                    keyword in str(col).lower() 
                    for keyword in ['type', 'jenis', 'kategori', 'accept', 'terima']
                )]
                
                if type_columns:
                    for type_col in type_columns:
                        if pd.notna(row.get(type_col)):
                            type_value = str(row[type_col]).strip().upper()
                            # Map common type names
                            type_mapping = {
                                'PET': 'PET',
                                'HDPE': 'HDPE',
                                'PP': 'PP',
                                'PS': 'PS',
                                'PVC': 'PVC',
                                'LDPE': 'LDPE',
                                'UMUM': 'GENERAL',
                                'CAMPURAN': 'MIXED',
                                'BOTOL': 'BOTTLES',
                                'KEMASAN': 'CONTAINERS'
                            }
                            # Try to find matching type
                            for key, value in type_mapping.items():
                                if key in type_value:
                                    if value not in types:
                                        types.append(value)
                                    break
                
                # If no specific types found, keep GENERAL
                if len(types) == 0:
                    types = ["GENERAL"]
                
                # Create location
                location = Location(
                    id=location_id,
                    name=name,
                    lat=lat,
                    lon=lon,
                    address=address,
                    hours=hours,
                    phone=contact,
                    types=types,
                    source="sipsn"
                )
                
                locations.append(location)
                processed_count += 1
                
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                continue
        
        print(f"Loaded {len(locations)} locations from {file_path}")
        return locations
        
    except Exception as e:
        print(f"Error loading locations from Excel: {e}")
        import traceback
        traceback.print_exc()
        return []

