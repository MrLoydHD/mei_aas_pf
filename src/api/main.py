"""
FastAPI backend for DGA Detection System.
Provides REST endpoints for domain classification and statistics.
"""

import os
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.api.models import (
    DomainRequest, BatchDomainRequest,
    PredictionResponse, BatchPredictionResponse, FeatureResponse,
    StatsResponse, ModelInfoResponse, HealthResponse, DetectionLog,
    FamilyPredictionResponse, FamilyInfo
)
from src.api.database import init_db, get_db, DetectionRepository, User
from src.api.auth import router as auth_router, get_current_user
from src.ml.random_forest_model import RandomForestDGADetector
from src.ml.lstm_model import LSTMDGADetector
from src.ml.xgboost_model import XGBoostDGADetector
from src.ml.gradient_boosting_model import GradientBoostingDGADetector
from src.ml.transformer_model import TransformerDGADetector
from src.ml.family_classifier import (
    DGAFamilyClassifier, LSTMFamilyClassifier,
    XGBoostFamilyClassifier, GradientBoostingFamilyClassifier,
    TransformerFamilyClassifier, DGA_FAMILY_INFO
)

# Optional: DistilBERT (requires transformers library)
try:
    from src.ml.distilbert_model import DistilBERTDGADetector
    from src.ml.family_classifier import DistilBERTFamilyClassifier
    DISTILBERT_AVAILABLE = True
except ImportError:
    DISTILBERT_AVAILABLE = False

# Global state
startup_time = time.time()
rf_model: Optional[RandomForestDGADetector] = None
lstm_model: Optional[LSTMDGADetector] = None
xgb_model: Optional[XGBoostDGADetector] = None
gb_model: Optional[GradientBoostingDGADetector] = None
transformer_model: Optional[TransformerDGADetector] = None
distilbert_model = None  # Optional
family_rf_model: Optional[DGAFamilyClassifier] = None
family_lstm_model: Optional[LSTMFamilyClassifier] = None
family_xgb_model: Optional[XGBoostFamilyClassifier] = None
family_gb_model: Optional[GradientBoostingFamilyClassifier] = None
family_transformer_model: Optional[TransformerFamilyClassifier] = None
family_distilbert_model = None  # Optional


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database and load models on startup."""
    global rf_model, lstm_model, xgb_model, gb_model, transformer_model, distilbert_model
    global family_rf_model, family_lstm_model, family_xgb_model, family_gb_model
    global family_transformer_model, family_distilbert_model, startup_time

    startup_time = time.time()

    print("Initializing database...")
    init_db()

    print("Loading models...")

    # ============== Binary DGA Detection Models ==============

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

    # Load XGBoost model
    xgb_model_path = os.getenv("XGB_MODEL_PATH", "models/xgboost.joblib")
    if os.path.exists(xgb_model_path):
        try:
            xgb_model = XGBoostDGADetector()
            xgb_model.load(xgb_model_path)
            print(f"XGBoost model loaded from {xgb_model_path}")
        except Exception as e:
            print(f"Failed to load XGBoost model: {e}")

    # Load Gradient Boosting model
    gb_model_path = os.getenv("GB_MODEL_PATH", "models/gradient_boosting.joblib")
    if os.path.exists(gb_model_path):
        try:
            gb_model = GradientBoostingDGADetector()
            gb_model.load(gb_model_path)
            print(f"Gradient Boosting model loaded from {gb_model_path}")
        except Exception as e:
            print(f"Failed to load Gradient Boosting model: {e}")

    # Load Transformer model
    transformer_model_path = os.getenv("TRANSFORMER_MODEL_PATH", "models/transformer")
    if os.path.exists(transformer_model_path):
        try:
            transformer_model = TransformerDGADetector()
            transformer_model.load(transformer_model_path)
            print(f"Transformer model loaded from {transformer_model_path}")
        except Exception as e:
            print(f"Failed to load Transformer model: {e}")

    # Load DistilBERT model (optional)
    if DISTILBERT_AVAILABLE:
        distilbert_model_path = os.getenv("DISTILBERT_MODEL_PATH", "models/distilbert")
        if os.path.exists(distilbert_model_path):
            try:
                distilbert_model = DistilBERTDGADetector()
                distilbert_model.load(distilbert_model_path)
                print(f"DistilBERT model loaded from {distilbert_model_path}")
            except Exception as e:
                print(f"Failed to load DistilBERT model: {e}")

    # ============== Family Classification Models ==============

    # Load Family Classifier RF model
    family_rf_path = os.getenv("FAMILY_RF_MODEL_PATH", "models/family_classifier_rf.joblib")
    if os.path.exists(family_rf_path):
        try:
            family_rf_model = DGAFamilyClassifier()
            family_rf_model.load(family_rf_path)
            print(f"Family RF classifier loaded from {family_rf_path}")
        except Exception as e:
            print(f"Failed to load Family RF classifier: {e}")

    # Load Family Classifier LSTM model
    family_lstm_path = os.getenv("FAMILY_LSTM_MODEL_PATH", "models/family_classifier_lstm")
    if os.path.exists(family_lstm_path):
        try:
            family_lstm_model = LSTMFamilyClassifier()
            family_lstm_model.load(family_lstm_path)
            print(f"Family LSTM classifier loaded from {family_lstm_path}")
        except Exception as e:
            print(f"Failed to load Family LSTM classifier: {e}")

    # Load Family Classifier XGBoost model
    family_xgb_path = os.getenv("FAMILY_XGB_MODEL_PATH", "models/family_classifier_xgb.joblib")
    if os.path.exists(family_xgb_path):
        try:
            family_xgb_model = XGBoostFamilyClassifier()
            family_xgb_model.load(family_xgb_path)
            print(f"Family XGBoost classifier loaded from {family_xgb_path}")
        except Exception as e:
            print(f"Failed to load Family XGBoost classifier: {e}")

    # Load Family Classifier Gradient Boosting model
    family_gb_path = os.getenv("FAMILY_GB_MODEL_PATH", "models/family_classifier_gb.joblib")
    if os.path.exists(family_gb_path):
        try:
            family_gb_model = GradientBoostingFamilyClassifier()
            family_gb_model.load(family_gb_path)
            print(f"Family GB classifier loaded from {family_gb_path}")
        except Exception as e:
            print(f"Failed to load Family GB classifier: {e}")

    # Load Family Classifier Transformer model
    family_transformer_path = os.getenv("FAMILY_TRANSFORMER_MODEL_PATH", "models/family_classifier_transformer")
    if os.path.exists(family_transformer_path):
        try:
            family_transformer_model = TransformerFamilyClassifier()
            family_transformer_model.load(family_transformer_path)
            print(f"Family Transformer classifier loaded from {family_transformer_path}")
        except Exception as e:
            print(f"Failed to load Family Transformer classifier: {e}")

    # Load Family Classifier DistilBERT model (optional)
    if DISTILBERT_AVAILABLE:
        family_distilbert_path = os.getenv("FAMILY_DISTILBERT_MODEL_PATH", "models/family_classifier_distilbert")
        if os.path.exists(family_distilbert_path):
            try:
                family_distilbert_model = DistilBERTFamilyClassifier()
                family_distilbert_model.load(family_distilbert_path)
                print(f"Family DistilBERT classifier loaded from {family_distilbert_path}")
            except Exception as e:
                print(f"Failed to load Family DistilBERT classifier: {e}")

    # ============== Summary ==============
    binary_models = sum([
        rf_model is not None,
        lstm_model is not None,
        xgb_model is not None,
        gb_model is not None,
        transformer_model is not None,
        distilbert_model is not None
    ])
    family_models = sum([
        family_rf_model is not None and family_rf_model.model is not None,
        family_lstm_model is not None and family_lstm_model.model is not None,
        family_xgb_model is not None and family_xgb_model.model is not None,
        family_gb_model is not None and family_gb_model.model is not None,
        family_transformer_model is not None and family_transformer_model.model is not None,
        family_distilbert_model is not None
    ])

    if binary_models == 0:
        print("WARNING: No DGA detection models loaded. Please train models first.")
    else:
        print(f"Loaded {binary_models} binary DGA detection model(s)")

    if family_models == 0:
        print("INFO: No family classifiers loaded. Family classification disabled.")
    else:
        print(f"Loaded {family_models} family classification model(s)")

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

# Include auth router
app.include_router(auth_router)


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
    elif model_type == "xgb" or model_type == "xgboost":
        if not xgb_model or not xgb_model.is_trained:
            raise HTTPException(status_code=503, detail="XGBoost model not available")
        return xgb_model, "xgboost"
    elif model_type == "gb" or model_type == "gradient_boosting":
        if not gb_model or not gb_model.is_trained:
            raise HTTPException(status_code=503, detail="Gradient Boosting model not available")
        return gb_model, "gradient_boosting"
    elif model_type == "transformer":
        if not transformer_model or not transformer_model.is_trained:
            raise HTTPException(status_code=503, detail="Transformer model not available")
        return transformer_model, "transformer"
    elif model_type == "distilbert":
        if not DISTILBERT_AVAILABLE or not distilbert_model or not distilbert_model.is_trained:
            raise HTTPException(status_code=503, detail="DistilBERT model not available")
        return distilbert_model, "distilbert"
    else:
        # Auto mode - prefer best available model (by typical accuracy ranking)
        # Priority: DistilBERT > Transformer > LSTM > XGBoost > GB > RF
        if DISTILBERT_AVAILABLE and distilbert_model and distilbert_model.is_trained:
            return distilbert_model, "distilbert"
        elif transformer_model and transformer_model.is_trained:
            return transformer_model, "transformer"
        elif lstm_model and lstm_model.is_trained:
            return lstm_model, "lstm"
        elif xgb_model and xgb_model.is_trained:
            return xgb_model, "xgboost"
        elif gb_model and gb_model.is_trained:
            return gb_model, "gradient_boosting"
        elif rf_model and rf_model.is_trained:
            return rf_model, "random_forest"
        else:
            raise HTTPException(status_code=503, detail="No models available")


# Known dynamic DNS / free hosting services where the subdomain is the interesting part
DDNS_PROVIDERS = {
    # No-IP services
    'ddns.net', 'no-ip.org', 'no-ip.biz', 'no-ip.info', 'no-ip.com',
    'hopto.org', 'zapto.org', 'sytes.net', 'redirectme.net', 'bounceme.net',
    'myftp.biz', 'myftp.org', 'myvnc.com', 'serveftp.com', 'servegame.com',
    'servehttp.com', 'servequake.com',
    # DynDNS
    'dyndns.org', 'dyndns.info', 'dyndns.tv', 'dyndns.biz',
    'dnsalias.com', 'dnsalias.net', 'dnsalias.org',
    # DuckDNS
    'duckdns.org',
    # Dynu
    'dynu.com', 'dynu.net',
    # FreeDNS / Afraid.org
    'afraid.org', 'freedns.afraid.org', 'mooo.com', 'chickenkiller.com',
    # Other popular DDNS
    'changeip.com', 'changeip.net', 'changeip.org',
    'dns.army', 'dns.navy', 'dns2go.com',
    'dnsdojo.com', 'dnsdojo.net', 'dnsdojo.org',
    'doesntexist.com', 'doesntexist.org',
    'doomdns.com', 'doomdns.org',
    'dvrdns.org', 'dynalias.com', 'dynalias.net', 'dynalias.org',
    'gotdns.com', 'gotdns.org',
    'selfip.com', 'selfip.net', 'selfip.org',
    'webhop.net', 'webhop.org', 'webhop.biz',
    'ydns.eu', 'yi.org',
    # Cloud/hosting providers often abused
    'cloudns.cc', 'cloudns.net',
    # Other
    '3utilities.com', 'blogsyte.com', 'brasilia.me', 'cable-modem.org',
    'ciscofreak.com', 'collegefan.org', 'couchpotatofries.org',
    'damnserver.com', 'ddns.info', 'ddns.mobi', 'ddns.name',
    'ddnsking.com', 'ditchyourip.com',
    'etowns.net', 'etowns.org',
    'game-host.org', 'game-server.cc',
    'getmyip.com', 'giize.com', 'gleeze.com',
    'homeftp.net', 'homeftp.org', 'homeip.net', 'homelinux.com',
    'homelinux.net', 'homelinux.org', 'homeunix.com', 'homeunix.net',
    'homeunix.org', 'iownyour.biz', 'iownyour.org',
    'is-a-chef.com', 'is-a-geek.com', 'is-a-geek.net', 'is-a-geek.org',
    'kicks-ass.net', 'kicks-ass.org', 'misconfused.org',
    'mypets.ws', 'myphotos.cc', 'neat-url.com',
    'office-on-the.net', 'podzone.net', 'podzone.org',
    'privatedns.org', 'privatizehealthinsurance.net',
    'scrapping.cc', 'selfip.biz', 'sellsyourhome.org',
    'servebbs.com', 'servebbs.net', 'servebbs.org',
    'servecounterstrike.com', 'serveftp.net', 'serveftp.org',
    'serveirc.com', 'serveminecraft.net', 'servepics.com',
    'shacknet.nu', 'trickip.net', 'trickip.org',
    'vicp.cc', 'vicp.net', 'vpndns.net',
    'wikaba.com', 'xicp.cn', 'xicp.net',
    'yombo.me', 'yourtrap.com', 'zaizheli.net',
}


def extract_domain_from_url(url_or_domain: str, strip_tld: bool = True) -> str:
    """Extract domain from URL or return domain as-is.

    Args:
        url_or_domain: URL or domain name
        strip_tld: If True, returns the relevant domain part for DGA analysis
                   (handles dynamic DNS services correctly by extracting subdomain)
                   If False, returns full domain with TLD
    """
    import re
    import tldextract

    # Remove protocol
    domain = re.sub(r'^https?://', '', url_or_domain)

    # Remove path, query, fragment
    domain = domain.split('/')[0].split('?')[0].split('#')[0]

    # Remove port
    domain = domain.split(':')[0]

    # Remove 'www.' prefix
    if domain.startswith('www.'):
        domain = domain[4:]

    if strip_tld:
        # Use tldextract for proper domain parsing
        extracted = tldextract.extract(domain)

        # Check if this is a dynamic DNS provider
        # The registered domain is domain.suffix (e.g., "ddns.net")
        registered_domain = f"{extracted.domain}.{extracted.suffix}".lower()

        if registered_domain in DDNS_PROVIDERS and extracted.subdomain:
            # For DDNS services, the subdomain is the DGA-generated part
            # e.g., "ixekrihagimau.ddns.net" -> "ixekrihagimau"
            # Handle multi-level subdomains by taking the leftmost part
            subdomain_parts = extracted.subdomain.split('.')
            # Filter out common prefixes
            for part in subdomain_parts:
                if part.lower() not in {'www', 'mail', 'ftp', 'smtp', 'pop', 'imap'}:
                    return part
            return subdomain_parts[0]

        # For regular domains, return the registered domain name (without TLD)
        # e.g., "google.com" -> "google"
        return extracted.domain

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
            # Binary detection models
            "random_forest": rf_model is not None and rf_model.is_trained,
            "lstm": lstm_model is not None and lstm_model.is_trained,
            "xgboost": xgb_model is not None and xgb_model.is_trained,
            "gradient_boosting": gb_model is not None and gb_model.is_trained,
            "transformer": transformer_model is not None and transformer_model.is_trained,
            "distilbert": DISTILBERT_AVAILABLE and distilbert_model is not None and distilbert_model.is_trained,
            # Family classification models
            "family_classifier_rf": family_rf_model is not None and family_rf_model.model is not None,
            "family_classifier_lstm": family_lstm_model is not None and family_lstm_model.model is not None,
            "family_classifier_xgb": family_xgb_model is not None and family_xgb_model.model is not None,
            "family_classifier_gb": family_gb_model is not None and family_gb_model.model is not None,
            "family_classifier_transformer": family_transformer_model is not None and family_transformer_model.model is not None,
            "family_classifier_distilbert": DISTILBERT_AVAILABLE and family_distilbert_model is not None
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
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user)
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
            source="dashboard",
            user_agent=req.headers.get("user-agent"),
            ip_address=req.client.host if req.client else None,
            user_id=current_user.id if current_user else None
        )

    return PredictionResponse(
        domain=request.domain,
        is_dga=result['is_dga'],
        confidence=result['confidence'],
        dga_probability=result['dga_probability'],
        legit_probability=result['legit_probability'],
        model_used=model_name,
        timestamp=datetime.now(timezone.utc)
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(
    request: BatchDomainRequest,
    req: Request,
    model_type: str = Query("auto", description="Model to use: auto, rf, lstm"),
    log: bool = Query(True, description="Log these detections"),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user)
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
                source="dashboard_batch",
                user_agent=req.headers.get("user-agent"),
                ip_address=req.client.host if req.client else None,
                user_id=current_user.id if current_user else None
            )

        predictions.append(PredictionResponse(
            domain=request.domains[i],
            is_dga=result['is_dga'],
            confidence=result['confidence'],
            dga_probability=result['dga_probability'],
            legit_probability=result['legit_probability'],
            model_used=model_name,
            timestamp=datetime.now(timezone.utc)
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
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user)
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
        source="dashboard_detailed",
        user_agent=req.headers.get("user-agent"),
        ip_address=req.client.host if req.client else None,
        user_id=current_user.id if current_user else None
    )

    return FeatureResponse(
        domain=request.domain,
        is_dga=result['is_dga'],
        confidence=result['confidence'],
        dga_probability=result['dga_probability'],
        legit_probability=result['legit_probability'],
        features=result['features'],
        model_used="random_forest",
        timestamp=datetime.now(timezone.utc)
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

    # ============== Binary Detection Models ==============

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

    if xgb_model:
        models.append(ModelInfoResponse(
            model_name="XGBoost DGA Detector",
            model_type="xgboost",
            is_loaded=xgb_model.is_trained,
            metrics=xgb_model.metrics if xgb_model.is_trained else None,
            feature_importance=xgb_model.get_feature_importance() if xgb_model.is_trained else None
        ))

    if gb_model:
        models.append(ModelInfoResponse(
            model_name="Gradient Boosting DGA Detector",
            model_type="gradient_boosting",
            is_loaded=gb_model.is_trained,
            metrics=gb_model.metrics if gb_model.is_trained else None,
            feature_importance=gb_model.get_feature_importance() if gb_model.is_trained else None
        ))

    if transformer_model:
        models.append(ModelInfoResponse(
            model_name="Transformer DGA Detector",
            model_type="transformer",
            is_loaded=transformer_model.is_trained,
            metrics=transformer_model.metrics if transformer_model.is_trained else None,
            feature_importance=None  # Transformer doesn't have explicit feature importance
        ))

    if DISTILBERT_AVAILABLE and distilbert_model:
        models.append(ModelInfoResponse(
            model_name="DistilBERT DGA Detector",
            model_type="distilbert",
            is_loaded=distilbert_model.is_trained,
            metrics=distilbert_model.metrics if distilbert_model.is_trained else None,
            feature_importance=None  # DistilBERT doesn't have explicit feature importance
        ))

    # ============== Family Classification Models ==============

    if family_rf_model and family_rf_model.model is not None:
        models.append(ModelInfoResponse(
            model_name="Family Classifier (Random Forest)",
            model_type="family_classifier_rf",
            is_loaded=True,
            metrics=family_rf_model.metrics,
            feature_importance=family_rf_model.get_feature_importance()
        ))

    if family_lstm_model and family_lstm_model.model is not None:
        models.append(ModelInfoResponse(
            model_name="Family Classifier (LSTM)",
            model_type="family_classifier_lstm",
            is_loaded=True,
            metrics=family_lstm_model.metrics,
            feature_importance=None  # LSTM doesn't have explicit feature importance
        ))

    if family_xgb_model and family_xgb_model.model is not None:
        models.append(ModelInfoResponse(
            model_name="Family Classifier (XGBoost)",
            model_type="family_classifier_xgb",
            is_loaded=True,
            metrics=family_xgb_model.metrics,
            feature_importance=family_xgb_model.get_feature_importance()
        ))

    if family_gb_model and family_gb_model.model is not None:
        models.append(ModelInfoResponse(
            model_name="Family Classifier (Gradient Boosting)",
            model_type="family_classifier_gb",
            is_loaded=True,
            metrics=family_gb_model.metrics,
            feature_importance=family_gb_model.get_feature_importance()
        ))

    if family_transformer_model and family_transformer_model.model is not None:
        models.append(ModelInfoResponse(
            model_name="Family Classifier (Transformer)",
            model_type="family_classifier_transformer",
            is_loaded=True,
            metrics=family_transformer_model.metrics,
            feature_importance=None  # Transformer doesn't have explicit feature importance
        ))

    if DISTILBERT_AVAILABLE and family_distilbert_model:
        models.append(ModelInfoResponse(
            model_name="Family Classifier (DistilBERT)",
            model_type="family_classifier_distilbert",
            is_loaded=True,
            metrics=getattr(family_distilbert_model, 'metrics', None),
            feature_importance=None  # DistilBERT doesn't have explicit feature importance
        ))

    return {"models": models}


# ============== Family Classification Endpoints ==============

def get_family_model(family_model_type: str = "auto"):
    """Get the active family classifier based on preference."""
    if family_model_type == "rf":
        if not family_rf_model or family_rf_model.model is None:
            return None, None
        return family_rf_model, "family_classifier_rf"
    elif family_model_type == "lstm":
        if not family_lstm_model or family_lstm_model.model is None:
            return None, None
        return family_lstm_model, "family_classifier_lstm"
    elif family_model_type == "xgb" or family_model_type == "xgboost":
        if not family_xgb_model or family_xgb_model.model is None:
            return None, None
        return family_xgb_model, "family_classifier_xgb"
    elif family_model_type == "gb" or family_model_type == "gradient_boosting":
        if not family_gb_model or family_gb_model.model is None:
            return None, None
        return family_gb_model, "family_classifier_gb"
    elif family_model_type == "transformer":
        if not family_transformer_model or family_transformer_model.model is None:
            return None, None
        return family_transformer_model, "family_classifier_transformer"
    elif family_model_type == "distilbert":
        if not DISTILBERT_AVAILABLE or not family_distilbert_model:
            return None, None
        return family_distilbert_model, "family_classifier_distilbert"
    else:
        # Auto mode - prefer best available by typical accuracy
        # Priority: DistilBERT > Transformer > LSTM > XGBoost > GB > RF
        if DISTILBERT_AVAILABLE and family_distilbert_model:
            return family_distilbert_model, "family_classifier_distilbert"
        elif family_transformer_model and family_transformer_model.model is not None:
            return family_transformer_model, "family_classifier_transformer"
        elif family_lstm_model and family_lstm_model.model is not None:
            return family_lstm_model, "family_classifier_lstm"
        elif family_xgb_model and family_xgb_model.model is not None:
            return family_xgb_model, "family_classifier_xgb"
        elif family_gb_model and family_gb_model.model is not None:
            return family_gb_model, "family_classifier_gb"
        elif family_rf_model and family_rf_model.model is not None:
            return family_rf_model, "family_classifier_rf"
        return None, None


# Default accuracy weights for ensemble voting (will be updated based on actual model metrics)
DEFAULT_MODEL_ACCURACIES = {
    'rf': 0.795,
    'lstm': 0.938,
    'xgb': 0.85,  # Expected - XGBoost typically slightly better than RF
    'gb': 0.84,   # Expected - GB similar to XGBoost
    'transformer': 0.92,  # Expected - Transformer typically high
    'distilbert': 0.95,   # Expected - Fine-tuned BERT models typically best
}


def ensemble_family_prediction(domain: str) -> Optional[Dict]:
    """
    Run all available family classifiers and combine results using weighted voting.

    Weights are based on model accuracy - higher accuracy models get more weight.
    Uses actual metrics if available, falls back to default estimates.

    Returns combined prediction with higher confidence.
    """
    # Check which models are available
    model_availability = {
        'rf': family_rf_model and family_rf_model.model is not None,
        'lstm': family_lstm_model and family_lstm_model.model is not None,
        'xgb': family_xgb_model and family_xgb_model.model is not None,
        'gb': family_gb_model and family_gb_model.model is not None,
        'transformer': family_transformer_model and family_transformer_model.model is not None,
        'distilbert': DISTILBERT_AVAILABLE and family_distilbert_model is not None
    }

    available_models = [k for k, v in model_availability.items() if v]

    if not available_models:
        return None

    # Get predictions from all available models
    predictions = {}
    model_refs = {
        'rf': family_rf_model,
        'lstm': family_lstm_model,
        'xgb': family_xgb_model,
        'gb': family_gb_model,
        'transformer': family_transformer_model,
        'distilbert': family_distilbert_model
    }

    for model_name in available_models:
        model = model_refs[model_name]
        try:
            predictions[model_name] = model.predict(domain)
        except Exception as e:
            print(f"{model_name.upper()} family prediction error: {e}")

    # Filter out failed predictions
    predictions = {k: v for k, v in predictions.items() if v is not None}

    if not predictions:
        return None

    # If only one model available/successful, return its result
    if len(predictions) == 1:
        model_name, result = list(predictions.items())[0]
        return {**result, 'ensemble': False, 'models_used': [model_name]}

    # Multiple models - calculate weights based on accuracy
    model_accuracies = {}
    for model_name in predictions:
        model = model_refs[model_name]
        # Try to get actual accuracy from model metrics
        if hasattr(model, 'metrics') and model.metrics and 'accuracy' in model.metrics:
            model_accuracies[model_name] = model.metrics['accuracy']
        else:
            model_accuracies[model_name] = DEFAULT_MODEL_ACCURACIES.get(model_name, 0.8)

    # Normalize weights
    total_accuracy = sum(model_accuracies.values())
    weights = {k: v / total_accuracy for k, v in model_accuracies.items()}

    # Weighted voting for family
    family_votes = {}
    for model_name, result in predictions.items():
        family = result['family']
        weight = weights[model_name]
        weighted_conf = result['confidence'] * weight

        if family not in family_votes:
            family_votes[family] = {
                'weighted_score': 0,
                'voters': [],
                'result': result  # Keep one result for metadata
            }
        family_votes[family]['weighted_score'] += weighted_conf
        family_votes[family]['voters'].append(model_name)

    # Find winner (family with highest weighted score)
    winner_family = max(family_votes.keys(), key=lambda f: family_votes[f]['weighted_score'])
    winner_data = family_votes[winner_family]
    winner_result = winner_data['result']

    # Calculate agreement
    all_predictions = [result['family'] for result in predictions.values()]
    agreement_count = all_predictions.count(winner_family)
    full_agreement = agreement_count == len(predictions)

    # Combined confidence based on agreement
    if full_agreement:
        # All models agree - high confidence
        combined_confidence = min(1.0, winner_data['weighted_score'] * 1.2)
    elif agreement_count > len(predictions) / 2:
        # Majority agreement
        combined_confidence = winner_data['weighted_score']
    else:
        # Close race - reduce confidence
        combined_confidence = winner_data['weighted_score'] * 0.85

    # Build alternatives from all models
    all_alternatives = {}
    for model_name, result in predictions.items():
        weight = weights[model_name]
        for alt in result.get('alternatives', []):
            if alt['family'] not in all_alternatives:
                all_alternatives[alt['family']] = 0
            all_alternatives[alt['family']] += alt['confidence'] * weight

        # Also add the model's primary prediction if it's not the winner
        if result['family'] != winner_family:
            if result['family'] not in all_alternatives:
                all_alternatives[result['family']] = 0
            all_alternatives[result['family']] += result['confidence'] * weight

    # Remove winner from alternatives and sort
    all_alternatives.pop(winner_family, None)
    sorted_alts = sorted(all_alternatives.items(), key=lambda x: x[1], reverse=True)[:3]
    alternatives = [{'family': f, 'confidence': c} for f, c in sorted_alts]

    # Build model predictions summary
    model_predictions = {m: p['family'] for m, p in predictions.items()}

    return {
        'family': winner_family,
        'confidence': combined_confidence,
        'description': winner_result['description'],
        'threat_level': winner_result['threat_level'],
        'first_seen': winner_result['first_seen'],
        'malware_type': winner_result['malware_type'],
        'alternatives': alternatives,
        'ensemble': True,
        'models_used': list(predictions.keys()),
        'agreement': full_agreement,
        'agreement_ratio': agreement_count / len(predictions),
        'model_predictions': model_predictions,
        'weights': weights
    }


@app.post("/predict/family", response_model=FamilyPredictionResponse, tags=["Family Classification"])
async def predict_with_family(
    request: DomainRequest,
    req: Request,
    model_type: str = Query("auto", description="DGA model to use: auto, rf, lstm, xgb, gb, transformer, distilbert"),
    family_model_type: str = Query("auto", description="Family model to use: auto (ensemble), rf, lstm, xgb, gb, transformer, distilbert"),
    log: bool = Query(True, description="Log this detection"),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user)
):
    """
    Predict if a domain is DGA-generated and classify its malware family.

    Uses a two-stage approach:
    1. Binary DGA detection using selected model
    2. If DGA detected, classifies the malware family

    Returns threat intelligence including family name, threat level, and malware type.
    """
    model, model_name = get_active_model(model_type)

    # Extract domain from URL - strip TLD for binary detection
    domain_for_detection = extract_domain_from_url(request.domain, strip_tld=True)
    # Keep full domain with TLD for family classification (model was trained with TLDs)
    domain_for_family = extract_domain_from_url(request.domain, strip_tld=False)

    # Stage 1: Binary DGA detection
    result = model.predict(domain_for_detection)

    family_info = None
    family_name = None
    family_confidence = None
    threat_level = None
    family_model_used = None

    # Stage 2: Family classification (only if DGA detected and model available)
    if result['is_dga']:
        family_result = None

        if family_model_type == "auto":
            # Use ensemble mode - run both models and combine results
            family_result = ensemble_family_prediction(domain_for_family)
            if family_result:
                family_model_used = "ensemble" if family_result.get('ensemble') else family_result.get('models_used', ['unknown'])[0]
        else:
            # Use specific model
            active_family_model, family_model_used = get_family_model(family_model_type)
            if active_family_model is not None:
                try:
                    family_result = active_family_model.predict(domain_for_family)
                except Exception as e:
                    print(f"Family classification error: {e}")

        if family_result:
            family_info = FamilyInfo(
                family=family_result['family'],
                confidence=family_result['confidence'],
                description=family_result['description'],
                threat_level=family_result['threat_level'],
                first_seen=family_result['first_seen'],
                malware_type=family_result['malware_type'],
                alternatives=family_result['alternatives']
            )
            family_name = family_result['family']
            family_confidence = family_result['confidence']
            threat_level = family_result['threat_level']

    # Log detection with family info
    if log:
        repo = DetectionRepository(db)
        repo.create(
            domain=request.domain,
            is_dga=result['is_dga'],
            confidence=result['confidence'],
            dga_probability=result['dga_probability'],
            model_used=model_name,
            source="dashboard_family",
            user_agent=req.headers.get("user-agent"),
            ip_address=req.client.host if req.client else None,
            user_id=current_user.id if current_user else None,
            family=family_name,
            family_confidence=family_confidence,
            threat_level=threat_level
        )

    return FamilyPredictionResponse(
        domain=request.domain,
        is_dga=result['is_dga'],
        dga_confidence=result['confidence'],
        family_info=family_info,
        model_used=model_name,
        family_model_used=family_model_used,
        timestamp=datetime.now(timezone.utc)
    )


@app.get("/families", tags=["Family Classification"])
async def get_families_info():
    """
    Get information about all known DGA families.

    Returns threat intelligence metadata for each DGA family including:
    - Description
    - Threat level (critical, high, medium, low)
    - First seen date
    - Malware type (ransomware, banking_trojan, botnet, etc.)
    """
    return {
        "families": DGA_FAMILY_INFO,
        "total_families": len(DGA_FAMILY_INFO),
        "models_loaded": {
            "family_rf": family_rf_model is not None and family_rf_model.model is not None,
            "family_lstm": family_lstm_model is not None and family_lstm_model.model is not None,
            "family_xgb": family_xgb_model is not None and family_xgb_model.model is not None,
            "family_gb": family_gb_model is not None and family_gb_model.model is not None,
            "family_transformer": family_transformer_model is not None and family_transformer_model.model is not None,
            "family_distilbert": DISTILBERT_AVAILABLE and family_distilbert_model is not None
        },
        # Legacy fields for backward compatibility
        "family_rf_loaded": family_rf_model is not None and family_rf_model.model is not None,
        "family_lstm_loaded": family_lstm_model is not None and family_lstm_model.model is not None
    }


@app.get("/stats/families", tags=["Family Classification"])
async def get_family_statistics(
    db: Session = Depends(get_db)
):
    """
    Get detection statistics grouped by DGA family.

    Returns count and average confidence for each detected family.
    """
    repo = DetectionRepository(db)

    return {
        "family_stats": repo.get_family_stats(),
        "threat_level_distribution": repo.get_threat_level_distribution()
    }


# Extension-specific endpoint
@app.post("/extension/check", tags=["Extension"])
async def extension_check(
    request: DomainRequest,
    req: Request,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user)
):
    """
    Lightweight endpoint optimized for browser extension.

    Returns minimal data for fast response.
    Optionally accepts authentication to link detections to user.
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
        ip_address=req.client.host if req.client else None,
        user_id=current_user.id if current_user else None
    )

    return {
        "is_dga": result['is_dga'],
        "confidence": result['confidence'],
        "risk_level": "high" if result['dga_probability'] > 0.8 else "medium" if result['dga_probability'] > 0.5 else "low"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
