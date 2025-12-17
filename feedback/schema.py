"""Data schema for scan records with versioning."""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
from datetime import datetime
import uuid
import json


SCHEMA_VERSION = "1.0.0"


@dataclass
class FrameQualityData:
    """Frame quality metrics."""
    blur_score: float
    brightness: float
    is_blurry: bool
    is_too_dark: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConditionsData:
    """Condition flags for scan."""
    clean: Optional[bool] = None
    label_present: Optional[bool] = None
    crushed: Optional[bool] = None
    mixed: Optional[bool] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class ImageRefData:
    """Image file references."""
    original_path: str
    snapshot_path: str
    roi_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DeviceData:
    """Device/platform information."""
    platform: str
    camera_resolution: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ScanRecord:
    """Structured scan record with all metadata."""
    # Required fields (no defaults) - must come first
    # Identifiers
    id: str
    timestamp: str  # ISO8601
    
    # App/Model versions
    app_version: str
    model_version: str
    
    # Prediction data
    pred_label: str
    pred_confidence: float
    stability: float
    vote_ratio: float
    margin: float
    entropy: float
    
    # Frame quality
    frame_quality: FrameQualityData
    
    # Image references
    image_ref: ImageRefData
    
    # Device info
    device: DeviceData
    
    # Optional/default fields (must come after required fields)
    # Schema metadata
    schema_version: str = SCHEMA_VERSION
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    
    # User feedback
    user_label: Optional[str] = None
    is_confirmed: bool = False
    
    # Conditions
    conditions: ConditionsData = field(default_factory=ConditionsData)
    
    # Active learning
    priority_score: float = 0.0
    high_value: bool = False
    
    @classmethod
    def create(cls, pred_label: str, pred_confidence: float, stability: float,
               vote_ratio: float, margin: float, entropy: float,
               frame_quality: FrameQualityData,
               image_ref: ImageRefData,
               app_version: str = "1.0.0",
               model_version: str = "1.0.0",
               device: Optional[DeviceData] = None) -> 'ScanRecord':
        """Create a new scan record."""
        import platform
        
        record_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat() + "Z"
        
        if device is None:
            device = DeviceData(
                platform=platform.system(),
                camera_resolution="unknown"
            )
        
        return cls(
            id=record_id,
            timestamp=timestamp,
            app_version=app_version,
            model_version=model_version,
            pred_label=pred_label,
            pred_confidence=pred_confidence,
            stability=stability,
            vote_ratio=vote_ratio,
            margin=margin,
            entropy=entropy,
            frame_quality=frame_quality,
            image_ref=image_ref,
            device=device
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "app_version": self.app_version,
            "model_version": self.model_version,
            "pred_label": self.pred_label,
            "pred_confidence": float(self.pred_confidence),
            "stability": float(self.stability),
            "vote_ratio": float(self.vote_ratio),
            "margin": float(self.margin),
            "entropy": float(self.entropy),
            "user_label": self.user_label,
            "is_confirmed": self.is_confirmed,
            "frame_quality": self.frame_quality.to_dict(),
            "conditions": self.conditions.to_dict(),
            "image_ref": self.image_ref.to_dict(),
            "device": self.device.to_dict(),
            "priority_score": float(self.priority_score),
            "high_value": self.high_value
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScanRecord':
        """Create from dictionary."""
        # Handle nested objects
        frame_quality = FrameQualityData(**data.get("frame_quality", {}))
        conditions = ConditionsData(**data.get("conditions", {}))
        image_ref = ImageRefData(**data.get("image_ref", {}))
        device = DeviceData(**data.get("device", {}))
        
        return cls(
            id=data["id"],
            timestamp=data["timestamp"],
            schema_version=data.get("schema_version", SCHEMA_VERSION),
            created_at=data.get("created_at", ""),
            app_version=data["app_version"],
            model_version=data["model_version"],
            pred_label=data["pred_label"],
            pred_confidence=float(data["pred_confidence"]),
            stability=float(data["stability"]),
            vote_ratio=float(data["vote_ratio"]),
            margin=float(data["margin"]),
            entropy=float(data["entropy"]),
            user_label=data.get("user_label"),
            is_confirmed=data.get("is_confirmed", False),
            frame_quality=frame_quality,
            conditions=conditions,
            image_ref=image_ref,
            device=device,
            priority_score=float(data.get("priority_score", 0.0)),
            high_value=data.get("high_value", False)
        )

