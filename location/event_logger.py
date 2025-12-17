"""Event logging for map interactions linked to ScanRecord."""

import sqlite3
from pathlib import Path
from typing import Optional
from datetime import datetime
import threading


class MapEventLogger:
    """Logs map interaction events linked to scan records."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize event logger.
        
        Args:
            data_dir: Data directory
        """
        self.data_dir = Path(data_dir)
        self.db_path = self.data_dir / "map_events.db"
        
        # Create directory
        self.data_dir.mkdir(exist_ok=True)
        
        # Thread lock
        self._lock = threading.Lock()
        
        # Initialize database
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS map_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scan_record_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    location_id TEXT,
                    timestamp TEXT NOT NULL,
                    metadata_json TEXT
                )
            """)
            
            # Create index
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_scan_record ON map_events(scan_record_id)
            """)
            
            conn.commit()
            conn.close()
    
    def log_event(self, scan_record_id: str, event_type: str,
                  location_id: Optional[str] = None,
                  metadata: Optional[dict] = None):
        """
        Log a map event.
        
        Args:
            scan_record_id: ID of the scan record
            event_type: Type of event (opened_map, selected_location, opened_external_maps)
            location_id: ID of location (if applicable)
            metadata: Additional metadata
        """
        try:
            with self._lock:
                conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
                cursor = conn.cursor()
                
                import json
                metadata_json = json.dumps(metadata) if metadata else None
                timestamp = datetime.utcnow().isoformat() + "Z"
                
                cursor.execute("""
                    INSERT INTO map_events (
                        scan_record_id, event_type, location_id, timestamp, metadata_json
                    ) VALUES (?, ?, ?, ?, ?)
                """, (scan_record_id, event_type, location_id, timestamp, metadata_json))
                
                conn.commit()
                conn.close()
        except Exception as e:
            print(f"Error logging event: {e}")

