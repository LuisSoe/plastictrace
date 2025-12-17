"""Drop-off location store with seed data and local cache."""

import json
import sqlite3
from pathlib import Path
from typing import List, Optional
import threading

from location.dropoff_schema import DropOffLocation


class DropOffStore:
    """Store for drop-off locations with seed data and local cache."""
    
    def __init__(self, data_dir: str = "data", seed_file: str = "data/dropoff_seed.json"):
        """
        Initialize drop-off store.
        
        Args:
            data_dir: Data directory
            seed_file: Path to seed JSON file
        """
        self.data_dir = Path(data_dir)
        self.seed_file = Path(seed_file)
        self.db_path = self.data_dir / "dropoff_locations.db"
        
        # Create directories
        self.data_dir.mkdir(exist_ok=True)
        
        # Thread lock
        self._lock = threading.Lock()
        
        # Initialize database
        self._init_db()
        
        # Load seed data if database is empty
        if self.count_locations() == 0:
            self._load_seed_data()
    
    def _init_db(self):
        """Initialize database schema."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dropoff_locations (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    lat REAL NOT NULL,
                    lng REAL NOT NULL,
                    address TEXT NOT NULL,
                    hours TEXT,
                    contact TEXT,
                    source TEXT NOT NULL DEFAULT 'seed',
                    accepted_types_json TEXT NOT NULL,
                    conditions_required_json TEXT,
                    notes TEXT
                )
            """)
            
            # Create indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_source ON dropoff_locations(source)
            """)
            
            conn.commit()
            conn.close()
    
    def _load_seed_data(self):
        """Load seed data from JSON file."""
        if not self.seed_file.exists():
            # Create default seed data
            self._create_default_seed_data()
        
        with open(self.seed_file, 'r') as f:
            data = json.load(f)
            locations = [DropOffLocation.from_dict(loc) for loc in data.get("locations", [])]
            
            for location in locations:
                self.add_location(location)
    
    def _create_default_seed_data(self):
        """Create default seed data file."""
        # Example locations for Indonesia (Jakarta area)
        default_locations = {
            "locations": [
                {
                    "id": "seed_001",
                    "name": "Jakarta Recycling Center",
                    "lat": -6.2088,
                    "lng": 106.8456,
                    "address": "Jl. Sudirman, Jakarta Pusat",
                    "hours": "Mon-Sat 8:00-17:00",
                    "contact": "+62-21-12345678",
                    "source": "seed",
                    "accepted_types": ["PET", "HDPE", "PP", "GENERAL"],
                    "conditions_required": {"clean": True},
                    "notes": "Accepts most plastic types when clean"
                },
                {
                    "id": "seed_002",
                    "name": "Bekasi Drop-off Point",
                    "lat": -6.2383,
                    "lng": 106.9756,
                    "address": "Jl. Ahmad Yani, Bekasi",
                    "hours": "Mon-Fri 9:00-16:00",
                    "contact": None,
                    "source": "seed",
                    "accepted_types": ["PET", "HDPE", "BOTTLES"],
                    "conditions_required": None,
                    "notes": "Specializes in bottles"
                },
                {
                    "id": "seed_003",
                    "name": "Depok Community Collection",
                    "lat": -6.4025,
                    "lng": 106.7942,
                    "address": "Jl. Margonda Raya, Depok",
                    "hours": "Sat-Sun 10:00-14:00",
                    "contact": None,
                    "source": "seed",
                    "accepted_types": ["GENERAL", "MIXED"],
                    "conditions_required": None,
                    "notes": "Community-run collection point"
                },
                {
                    "id": "seed_004",
                    "name": "Tangerang Plastic Hub",
                    "lat": -6.1781,
                    "lng": 106.6300,
                    "address": "Jl. Daan Mogot, Tangerang",
                    "hours": "Daily 7:00-19:00",
                    "contact": "+62-21-87654321",
                    "source": "seed",
                    "accepted_types": ["PP", "HDPE", "CONTAINERS"],
                    "conditions_required": {"clean": True},
                    "notes": "Focus on food containers"
                },
                {
                    "id": "seed_005",
                    "name": "South Jakarta Collection",
                    "lat": -6.2297,
                    "lng": 106.7997,
                    "address": "Jl. Fatmawati, Jakarta Selatan",
                    "hours": "Mon-Sat 8:00-18:00",
                    "contact": None,
                    "source": "seed",
                    "accepted_types": ["PET", "GENERAL"],
                    "conditions_required": None,
                    "notes": "General plastic collection"
                }
            ]
        }
        
        self.seed_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.seed_file, 'w') as f:
            json.dump(default_locations, f, indent=2)
    
    def add_location(self, location: DropOffLocation) -> bool:
        """
        Add a location to the store.
        
        Args:
            location: DropOffLocation to add
            
        Returns:
            True if successful
        """
        try:
            with self._lock:
                conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO dropoff_locations (
                        id, name, lat, lng, address, hours, contact, source,
                        accepted_types_json, conditions_required_json, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    location.id,
                    location.name,
                    location.lat,
                    location.lng,
                    location.address,
                    location.hours,
                    location.contact,
                    location.source,
                    json.dumps(location.accepted_types),
                    json.dumps(location.conditions_required) if location.conditions_required else None,
                    location.notes
                ))
                
                conn.commit()
                conn.close()
                return True
        except Exception as e:
            print(f"Error adding location: {e}")
            return False
    
    def get_all_locations(self) -> List[DropOffLocation]:
        """Get all locations."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM dropoff_locations")
            rows = cursor.fetchall()
            conn.close()
            
            return [self._row_to_location(dict(row)) for row in rows]
    
    def get_location(self, location_id: str) -> Optional[DropOffLocation]:
        """Get location by ID."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM dropoff_locations WHERE id = ?", (location_id,))
            row = cursor.fetchone()
            conn.close()
            
            if row is None:
                return None
            
            return self._row_to_location(dict(row))
    
    def count_locations(self) -> int:
        """Get total location count."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM dropoff_locations")
            count = cursor.fetchone()[0]
            conn.close()
            
            return count
    
    def _row_to_location(self, row: dict) -> DropOffLocation:
        """Convert database row to DropOffLocation."""
        return DropOffLocation(
            id=row["id"],
            name=row["name"],
            lat=row["lat"],
            lng=row["lng"],
            address=row["address"],
            hours=row.get("hours"),
            contact=row.get("contact"),
            source=row.get("source", "seed"),
            accepted_types=json.loads(row["accepted_types_json"]),
            conditions_required=json.loads(row["conditions_required_json"]) if row.get("conditions_required_json") else None,
            notes=row.get("notes")
        )

