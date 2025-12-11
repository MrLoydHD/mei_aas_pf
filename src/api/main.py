"""
FastAPI backend for DGA Detection System.
Provides REST endpoints for domain classification and statistics.
"""

import os
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.api.models import (
    DomainRequest, BatchDomainRequest,
    PredictionResponse, BatchPredictionResponse, FeatureResponse,
    StatsResponse, ModelInfoResponse, HealthResponse, DetectionLog
)
from src.api.database import init_db, get_db, DetectionRepository
from src.ml.random_forest_model import RandomForestDGADetector
from src.ml.lstm_model import LSTMDGADetector

# Global state
startup_time = time.time()
rf_model: Optional[RandomForestDGADetector] = None
lstm_model: Optional[LSTMDGADetector] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database and load models on startup."""
    global rf_model, lstm_model, startup_time

    startup_time = time.time()

    print("Initializing database...")
    init_db()

    print("Loading models...")

    # Load Random Forest model
    rf_model_path = os.getenv("RF_MODEL_PATH", "models/random_forest.joblib")
    if os.path.exists(rf_model_path):
        try:
            rf_model = RandomForestDGADetector()
            rf_model.load(rf_model_path)
            print(f"Random Forest model loaded from {rf_model_path}")
        except Exception as e:
            print(f"Failed to load Random Forest model: {e}")

    # Load LSTM model
    lstm_model_path = os.getenv("LSTM_MODEL_PATH", "models/lstm")
    if os.path.exists(lstm_model_path):
        try:
            lstm_model = LSTMDGADetector()
            lstm_model.load(lstm_model_path)
            print(f"LSTM model loaded from {lstm_model_path}")
        except Exception as e:
            print(f"Failed to load LSTM model: {e}")

    if not rf_model and not lstm_model:
        print("WARNING: No models loaded. Please train models first.")

    yield  # App runs here

    # Cleanup on shutdown (if needed)
    print("Shutting down...")


# Initialize app with lifespan
app = FastAPI(
    title="DGA Detection API",
    description="API for detecting Domain Generation Algorithm (DGA) generated domains",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware for browser extension and frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_active_model(model_type: str = "auto"):
    """Get the active model based on preference."""
    if model_type == "rf" or model_type == "random_forest":
        if not rf_model or not rf_model.is_trained:
            raise HTTPException(status_code=503, detail="Random Forest model not available")
        return rf_model, "random_forest"
    elif model_type == "lstm" or model_type == "deep_learning":
        if not lstm_model or not lstm_model.is_trained:
            raise HTTPException(status_code=503, detail="LSTM model not available")
        return lstm_model, "lstm"
    else:
        # Auto mode - prefer LSTM if available, fallback to RF
        if lstm_model and lstm_model.is_trained:
            return lstm_model, "lstm"
        elif rf_model and rf_model.is_trained:
            return rf_model, "random_forest"
        else:
            raise HTTPException(status_code=503, detail="No models available")


def extract_domain_from_url(url_or_domain: str) -> str:
    """Extract domain from URL or return domain as-is."""
    import re

    # Remove protocol
    domain = re.sub(r'^https?://', '', url_or_domain)

    # Remove path, query, fragment
    domain = domain.split('/')[0].split('?')[0].split('#')[0]

    # Remove port
    domain = domain.split(':')[0]

    # Remove 'www.' prefix
    if domain.startswith('www.'):
        domain = domain[4:]

    # Extract main domain (remove TLD for analysis)
    parts = domain.split('.')
    if len(parts) >= 2:
        # Return second-level domain for analysis
        return parts[-2]

    return domain


# ============== API Endpoints ==============

@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "DGA Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        models_loaded={
            "random_forest": rf_model is not None and rf_model.is_trained,
            "lstm": lstm_model is not None and lstm_model.is_trained
        },
        version="1.0.0",
        uptime_seconds=time.time() - startup_time
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_domain(
    request: DomainRequest,
    req: Request,
    model_type: str = Query("auto", description="Model to use: auto, rf, lstm"),
    log: bool = Query(True, description="Log this detection"),
    db: Session = Depends(get_db)
):
    """
    Predict if a domain is DGA-generated.

    - **domain**: Domain name or URL to analyze
    - **model_type**: Which model to use (auto, rf, lstm)
    - **log**: Whether to log this detection
    """
    model, model_name = get_active_model(model_type)

    # Extract domain from URL if needed
    domain = extract_domain_from_url(request.domain)

    # Make prediction
    result = model.predict(domain)

    # Log detection
    if log:
        repo = DetectionRepository(db)
        repo.create(
            domain=request.domain,  # Log original input
            is_dga=result['is_dga'],
            confidence=result['confidence'],
            dga_probability=result['dga_probability'],
            model_used=model_name,
            source="api",
            user_agent=req.headers.get("user-agent"),
            ip_address=req.client.host if req.client else None
        )

    return PredictionResponse(
        domain=request.domain,
        is_dga=result['is_dga'],
        confidence=result['confidence'],
        dga_probability=result['dga_probability'],
        legit_probability=result['legit_probability'],
        model_used=model_name,
        timestamp=datetime.utcnow()
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(
    request: BatchDomainRequest,
    req: Request,
    model_type: str = Query("auto", description="Model to use: auto, rf, lstm"),
    log: bool = Query(True, description="Log these detections"),
    db: Session = Depends(get_db)
):
    """
    Predict DGA status for multiple domains.

    - **domains**: List of domain names to analyze
    - **model_type**: Which model to use (auto, rf, lstm)
    - **log**: Whether to log these detections
    """
    model, model_name = get_active_model(model_type)

    # Extract domains
    domains = [extract_domain_from_url(d) for d in request.domains]

    # Make predictions
    results = model.predict_batch(domains)

    predictions = []
    dga_count = 0

    repo = DetectionRepository(db) if log else None

    for i, result in enumerate(results):
        if result['is_dga']:
            dga_count += 1

        # Log detection
        if log and repo:
            repo.create(
                domain=request.domains[i],
                is_dga=result['is_dga'],
                confidence=result['confidence'],
                dga_probability=result['dga_probability'],
                model_used=model_name,
                source="api_batch",
                user_agent=req.headers.get("user-agent"),
                ip_address=req.client.host if req.client else None
            )

        predictions.append(PredictionResponse(
            domain=request.domains[i],
            is_dga=result['is_dga'],
            confidence=result['confidence'],
            dga_probability=result['dga_probability'],
            legit_probability=result['legit_probability'],
            model_used=model_name,
            timestamp=datetime.utcnow()
        ))

    return BatchPredictionResponse(
        predictions=predictions,
        total=len(predictions),
        dga_count=dga_count,
        legit_count=len(predictions) - dga_count
    )


@app.post("/predict/detailed", response_model=FeatureResponse, tags=["Prediction"])
async def predict_with_features(
    request: DomainRequest,
    req: Request,
    db: Session = Depends(get_db)
):
    """
    Predict with detailed feature analysis (Random Forest only).

    Returns the extracted features along with the prediction.
    """
    if not rf_model or not rf_model.is_trained:
        raise HTTPException(status_code=503, detail="Random Forest model required for detailed analysis")

    domain = extract_domain_from_url(request.domain)
    result = rf_model.predict(domain)

    # Log detection
    repo = DetectionRepository(db)
    repo.create(
        domain=request.domain,
        is_dga=result['is_dga'],
        confidence=result['confidence'],
        dga_probability=result['dga_probability'],
        model_used="random_forest",
        source="api_detailed",
        user_agent=req.headers.get("user-agent"),
        ip_address=req.client.host if req.client else None
    )

    return FeatureResponse(
        domain=request.domain,
        is_dga=result['is_dga'],
        confidence=result['confidence'],
        dga_probability=result['dga_probability'],
        legit_probability=result['legit_probability'],
        features=result['features'],
        model_used="random_forest",
        timestamp=datetime.utcnow()
    )


@app.get("/stats", response_model=StatsResponse, tags=["Statistics"])
async def get_statistics(
    hours: int = Query(24, ge=1, le=168, description="Hours to look back"),
    db: Session = Depends(get_db)
):
    """
    Get detection statistics.

    - **hours**: Number of hours to look back for statistics
    """
    repo = DetectionRepository(db)

    total = repo.get_total_count()
    dga = repo.get_dga_count()
    legit = repo.get_legit_count()

    recent = repo.get_recent(limit=20)
    recent_logs = [
        DetectionLog(
            id=r.id,
            domain=r.domain,
            is_dga=r.is_dga,
            confidence=r.confidence,
            source=r.source,
            user_agent=r.user_agent,
            timestamp=r.timestamp
        )
        for r in recent
    ]

    return StatsResponse(
        total_scans=total,
        dga_detected=dga,
        legit_detected=legit,
        detection_rate=dga / total if total > 0 else 0,
        top_dga_domains=repo.get_top_dga_domains(limit=10),
        recent_detections=recent_logs,
        hourly_stats=repo.get_hourly_stats(hours=hours)
    )


@app.get("/models", tags=["Models"])
async def get_models_info():
    """Get information about loaded models."""
    models = []

    if rf_model:
        models.append(ModelInfoResponse(
            model_name="Random Forest DGA Detector",
            model_type="random_forest",
            is_loaded=rf_model.is_trained,
            metrics=rf_model.metrics if rf_model.is_trained else None,
            feature_importance=rf_model.get_feature_importance() if rf_model.is_trained else None
        ))

    if lstm_model:
        models.append(ModelInfoResponse(
            model_name="LSTM DGA Detector",
            model_type="lstm",
            is_loaded=lstm_model.is_trained,
            metrics=lstm_model.metrics if lstm_model.is_trained else None,
            feature_importance=None  # LSTM doesn't have explicit feature importance
        ))

    return {"models": models}


# Extension-specific endpoint
@app.post("/extension/check", tags=["Extension"])
async def extension_check(
    request: DomainRequest,
    req: Request,
    db: Session = Depends(get_db)
):
    """
    Lightweight endpoint optimized for browser extension.

    Returns minimal data for fast response.
    """
    model, model_name = get_active_model("auto")

    domain = extract_domain_from_url(request.domain)
    result = model.predict(domain)

    # Log with extension source
    repo = DetectionRepository(db)
    repo.create(
        domain=request.domain,
        is_dga=result['is_dga'],
        confidence=result['confidence'],
        dga_probability=result['dga_probability'],
        model_used=model_name,
        source="extension",
        user_agent=req.headers.get("user-agent"),
        ip_address=req.client.host if req.client else None
    )

    return {
        "is_dga": result['is_dga'],
        "confidence": result['confidence'],
        "risk_level": "high" if result['dga_probability'] > 0.8 else "medium" if result['dga_probability'] > 0.5 else "low"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
