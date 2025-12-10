"""
Pydantic models for API request/response validation.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class DomainRequest(BaseModel):
    """Request model for single domain prediction."""
    domain: str = Field(..., min_length=1, max_length=253, description="Domain name to analyze")

    class Config:
        json_schema_extra = {
            "example": {
                "domain": "google.com"
            }
        }


class BatchDomainRequest(BaseModel):
    """Request model for batch domain prediction."""
    domains: List[str] = Field(..., min_length=1, max_length=1000, description="List of domains to analyze")

    class Config:
        json_schema_extra = {
            "example": {
                "domains": ["google.com", "xk23jf9sd.net", "facebook.com"]
            }
        }


class PredictionResponse(BaseModel):
    """Response model for domain prediction."""
    domain: str
    is_dga: bool
    confidence: float = Field(..., ge=0, le=1)
    dga_probability: float = Field(..., ge=0, le=1)
    legit_probability: float = Field(..., ge=0, le=1)
    model_used: str
    timestamp: datetime


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    predictions: List[PredictionResponse]
    total: int
    dga_count: int
    legit_count: int


class FeatureResponse(BaseModel):
    """Response model with detailed features."""
    domain: str
    is_dga: bool
    confidence: float
    dga_probability: float
    legit_probability: float
    features: Dict[str, Any]
    model_used: str
    timestamp: datetime


class DetectionLog(BaseModel):
    """Model for logged detections."""
    id: int
    domain: str
    is_dga: bool
    confidence: float
    source: Optional[str] = None
    user_agent: Optional[str] = None
    timestamp: datetime


class StatsResponse(BaseModel):
    """Response model for detection statistics."""
    total_scans: int
    dga_detected: int
    legit_detected: int
    detection_rate: float
    top_dga_domains: List[Dict[str, Any]]
    recent_detections: List[DetectionLog]
    hourly_stats: Dict[str, int]


class ModelInfoResponse(BaseModel):
    """Response model for model information."""
    model_name: str
    model_type: str
    is_loaded: bool
    metrics: Optional[Dict[str, Any]] = None
    feature_importance: Optional[Dict[str, float]] = None


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    models_loaded: Dict[str, bool]
    version: str
    uptime_seconds: float
