"""Local dataset storage (SQLite) with append-only writes."""

import sqlite3
import json
import os
from typing import List, Optional, Dict, Any
from pathlib import Path
import threading

from feedback.schema import ScanRecord, FrameQualityData, ConditionsData, ImageRefData, DeviceData


class DatasetStore:
    """SQLite-based dataset store with append-only writes."""
    
    def __init__(self, data_dir: str = "data", db_name: str = "records.db"):
        """
        Initialize dataset store.
        
        Args:
            data_dir: Base directory for data storage
            db_name: SQLite database filename
        """
        self.data_dir = Path(data_dir)
        self.db_path = self.data_dir / db_name
        
        # Create directories
        self.data_dir.mkdir(exist_ok=True)
        (self.data_dir / "images" / "original").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "images" / "snapshot").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "images" / "roi").mkdir(parents=True, exist_ok=True)
        
        # Thread lock for database access
        self._lock = threading.Lock()
        
        # Initialize database
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS scan_records (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    schema_version TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    app_version TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    pred_label TEXT NOT NULL,
                    pred_confidence REAL NOT NULL,
                    stability REAL NOT NULL,
                    vote_ratio REAL NOT NULL,
                    margin REAL NOT NULL,
                    entropy REAL NOT NULL,
                    user_label TEXT,
                    is_confirmed INTEGER NOT NULL DEFAULT 0,
                    frame_quality_json TEXT NOT NULL,
                    conditions_json TEXT NOT NULL,
                    image_ref_json TEXT NOT NULL,
                    device_json TEXT NOT NULL,
                    priority_score REAL NOT NULL DEFAULT 0.0,
                    high_value INTEGER NOT NULL DEFAULT 0
                )
            """)
            
            # Create indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON scan_records(timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_pred_label ON scan_records(pred_label)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_high_value ON scan_records(high_value)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_label ON scan_records(user_label)
            """)
            
            conn.commit()
            conn.close()
    
    def save_record(self, record: ScanRecord) -> bool:
        """
        Save a scan record (append-only, non-blocking).
        
        Args:
            record: ScanRecord to save
            
        Returns:
            True if successful
        """
        try:
            with self._lock:
                conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO scan_records (
                        id, timestamp, schema_version, created_at,
                        app_version, model_version,
                        pred_label, pred_confidence, stability, vote_ratio, margin, entropy,
                        user_label, is_confirmed,
                        frame_quality_json, conditions_json, image_ref_json, device_json,
                        priority_score, high_value
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.id,
                    record.timestamp,
                    record.schema_version,
                    record.created_at,
                    record.app_version,
                    record.model_version,
                    record.pred_label,
                    record.pred_confidence,
                    record.stability,
                    record.vote_ratio,
                    record.margin,
                    record.entropy,
                    record.user_label,
                    1 if record.is_confirmed else 0,
                    json.dumps(record.frame_quality.to_dict()),
                    json.dumps(record.conditions.to_dict()),
                    json.dumps(record.image_ref.to_dict()),
                    json.dumps(record.device.to_dict()),
                    record.priority_score,
                    1 if record.high_value else 0
                ))
                
                conn.commit()
                conn.close()
                return True
        except Exception as e:
            print(f"Error saving record: {e}")
            return False
    
    def get_record(self, record_id: str) -> Optional[ScanRecord]:
        """Get a record by ID."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM scan_records WHERE id = ?", (record_id,))
            row = cursor.fetchone()
            conn.close()
            
            if row is None:
                return None
            
            return self._row_to_record(dict(row))
    
    def get_all_records(self, limit: Optional[int] = None, offset: int = 0) -> List[ScanRecord]:
        """Get all records, optionally limited."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = "SELECT * FROM scan_records ORDER BY timestamp DESC"
            if limit:
                query += f" LIMIT {limit} OFFSET {offset}"
            
            cursor.execute(query)
            rows = cursor.fetchall()
            conn.close()
            
            return [self._row_to_record(dict(row)) for row in rows]
    
    def get_high_value_records(self) -> List[ScanRecord]:
        """Get all high-value records."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM scan_records WHERE high_value = 1 ORDER BY timestamp DESC")
            rows = cursor.fetchall()
            conn.close()
            
            return [self._row_to_record(dict(row)) for row in rows]
    
    def get_corrected_records(self) -> List[ScanRecord]:
        """Get all user-corrected records."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM scan_records 
                WHERE user_label IS NOT NULL AND user_label != pred_label
                ORDER BY timestamp DESC
            """)
            rows = cursor.fetchall()
            conn.close()
            
            return [self._row_to_record(dict(row)) for row in rows]
    
    def get_by_label(self, label: str) -> List[ScanRecord]:
        """Get records by predicted label."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM scan_records WHERE pred_label = ? ORDER BY timestamp DESC", (label,))
            rows = cursor.fetchall()
            conn.close()
            
            return [self._row_to_record(dict(row)) for row in rows]
    
    def count_records(self) -> int:
        """Get total record count."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM scan_records")
            count = cursor.fetchone()[0]
            conn.close()
            
            return count
    
    def _row_to_record(self, row: Dict[str, Any]) -> ScanRecord:
        """Convert database row to ScanRecord."""
        # Parse JSON fields
        frame_quality = FrameQualityData(**json.loads(row["frame_quality_json"]))
        conditions = ConditionsData(**json.loads(row["conditions_json"]))
        image_ref = ImageRefData(**json.loads(row["image_ref_json"]))
        device = DeviceData(**json.loads(row["device_json"]))
        
        record = ScanRecord(
            id=row["id"],
            timestamp=row["timestamp"],
            schema_version=row.get("schema_version", "1.0.0"),
            created_at=row.get("created_at", ""),
            app_version=row["app_version"],
            model_version=row["model_version"],
            pred_label=row["pred_label"],
            pred_confidence=row["pred_confidence"],
            stability=row["stability"],
            vote_ratio=row["vote_ratio"],
            margin=row["margin"],
            entropy=row["entropy"],
            user_label=row.get("user_label"),
            is_confirmed=bool(row["is_confirmed"]),
            frame_quality=frame_quality,
            conditions=conditions,
            image_ref=image_ref,
            device=device,
            priority_score=row.get("priority_score", 0.0),
            high_value=bool(row.get("high_value", 0))
        )
        
        return record

