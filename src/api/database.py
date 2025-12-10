"""
Database setup for logging detections.
Uses SQLite with SQLAlchemy for simplicity.
"""

import os
from datetime import datetime
from typing import Optional, List

from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# Database URL - use SQLite for simplicity
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/dga_detection.db")

# Create engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


class DetectionRecord(Base):
    """Database model for detection logs."""
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True)
    domain = Column(String(253), index=True, nullable=False)
    is_dga = Column(Boolean, nullable=False)
    confidence = Column(Float, nullable=False)
    dga_probability = Column(Float, nullable=False)
    model_used = Column(String(50), nullable=False)
    source = Column(String(100), nullable=True)  # e.g., "extension", "api", "dashboard"
    user_agent = Column(String(500), nullable=True)
    ip_address = Column(String(45), nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)


def init_db():
    """Initialize database tables."""
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    Base.metadata.create_all(bind=engine)


def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class DetectionRepository:
    """Repository for detection records."""

    def __init__(self, db: Session):
        self.db = db

    def create(
        self,
        domain: str,
        is_dga: bool,
        confidence: float,
        dga_probability: float,
        model_used: str,
        source: Optional[str] = None,
        user_agent: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> DetectionRecord:
        """Create a new detection record."""
        record = DetectionRecord(
            domain=domain,
            is_dga=is_dga,
            confidence=confidence,
            dga_probability=dga_probability,
            model_used=model_used,
            source=source,
            user_agent=user_agent,
            ip_address=ip_address
        )
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record

    def get_recent(self, limit: int = 100) -> List[DetectionRecord]:
        """Get recent detection records."""
        return self.db.query(DetectionRecord)\
            .order_by(DetectionRecord.timestamp.desc())\
            .limit(limit)\
            .all()

    def get_by_domain(self, domain: str, limit: int = 10) -> List[DetectionRecord]:
        """Get detection records for a specific domain."""
        return self.db.query(DetectionRecord)\
            .filter(DetectionRecord.domain == domain)\
            .order_by(DetectionRecord.timestamp.desc())\
            .limit(limit)\
            .all()

    def get_dga_count(self) -> int:
        """Get count of DGA detections."""
        return self.db.query(DetectionRecord)\
            .filter(DetectionRecord.is_dga == True)\
            .count()

    def get_legit_count(self) -> int:
        """Get count of legitimate detections."""
        return self.db.query(DetectionRecord)\
            .filter(DetectionRecord.is_dga == False)\
            .count()

    def get_total_count(self) -> int:
        """Get total count of detections."""
        return self.db.query(DetectionRecord).count()

    def get_top_dga_domains(self, limit: int = 10) -> List[dict]:
        """Get most frequently detected DGA domains."""
        from sqlalchemy import func

        results = self.db.query(
            DetectionRecord.domain,
            func.count(DetectionRecord.id).label('count'),
            func.avg(DetectionRecord.confidence).label('avg_confidence')
        ).filter(DetectionRecord.is_dga == True)\
         .group_by(DetectionRecord.domain)\
         .order_by(func.count(DetectionRecord.id).desc())\
         .limit(limit)\
         .all()

        return [
            {
                'domain': r.domain,
                'count': r.count,
                'avg_confidence': round(r.avg_confidence, 4)
            }
            for r in results
        ]

    def get_hourly_stats(self, hours: int = 24) -> dict:
        """Get hourly detection statistics."""
        from sqlalchemy import func
        from datetime import timedelta

        cutoff = datetime.utcnow() - timedelta(hours=hours)

        results = self.db.query(
            func.strftime('%Y-%m-%d %H:00', DetectionRecord.timestamp).label('hour'),
            func.count(DetectionRecord.id).label('total'),
            func.sum(DetectionRecord.is_dga.cast(Integer)).label('dga_count')
        ).filter(DetectionRecord.timestamp >= cutoff)\
         .group_by(func.strftime('%Y-%m-%d %H:00', DetectionRecord.timestamp))\
         .order_by(func.strftime('%Y-%m-%d %H:00', DetectionRecord.timestamp))\
         .all()

        return {
            r.hour: {
                'total': r.total,
                'dga': r.dga_count or 0,
                'legit': r.total - (r.dga_count or 0)
            }
            for r in results
        }
