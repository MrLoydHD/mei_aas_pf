"""
Database setup for logging detections.
Uses SQLite with SQLAlchemy for simplicity.
"""

import os
from datetime import datetime, timezone
from typing import Optional, List

from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship

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


class User(Base):
    """Database model for users."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    google_id = Column(String(255), unique=True, index=True, nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    name = Column(String(255), nullable=True)
    picture = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship to detections
    detections = relationship("DetectionRecord", back_populates="user")


class DetectionRecord(Base):
    """Database model for detection logs."""
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True)
    domain = Column(String(253), index=True, nullable=False)
    is_dga = Column(Boolean, nullable=False)
    confidence = Column(Float, nullable=False)
    dga_probability = Column(Float, nullable=False)
    model_used = Column(String(50), nullable=False)
    # Family classification fields
    family = Column(String(50), nullable=True, index=True)
    family_confidence = Column(Float, nullable=True)
    threat_level = Column(String(20), nullable=True)
    source = Column(String(100), nullable=True)  # e.g., "extension", "api", "dashboard"
    user_agent = Column(String(500), nullable=True)
    ip_address = Column(String(45), nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)

    # Relationship to user
    user = relationship("User", back_populates="detections")


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
        ip_address: Optional[str] = None,
        user_id: Optional[int] = None,
        family: Optional[str] = None,
        family_confidence: Optional[float] = None,
        threat_level: Optional[str] = None
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
            ip_address=ip_address,
            user_id=user_id,
            family=family,
            family_confidence=family_confidence,
            threat_level=threat_level
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

        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

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

    def get_user_stats(self, user_id: int) -> dict:
        """Get detection statistics for a specific user."""
        total = self.db.query(DetectionRecord).filter(DetectionRecord.user_id == user_id).count()
        dga = self.db.query(DetectionRecord).filter(
            DetectionRecord.user_id == user_id,
            DetectionRecord.is_dga == True
        ).count()

        return {
            'total_checked': total,
            'dga_detected': dga,
            'legit_detected': total - dga
        }

    def get_user_detections(self, user_id: int, limit: int = 100) -> List[DetectionRecord]:
        """Get detection records for a specific user."""
        return self.db.query(DetectionRecord)\
            .filter(DetectionRecord.user_id == user_id)\
            .order_by(DetectionRecord.timestamp.desc())\
            .limit(limit)\
            .all()

    def get_family_stats(self) -> List[dict]:
        """Get detection statistics by DGA family."""
        from sqlalchemy import func

        results = self.db.query(
            DetectionRecord.family,
            func.count(DetectionRecord.id).label('count'),
            func.avg(DetectionRecord.family_confidence).label('avg_confidence'),
            DetectionRecord.threat_level
        ).filter(
            DetectionRecord.is_dga == True,
            DetectionRecord.family.isnot(None)
        ).group_by(DetectionRecord.family, DetectionRecord.threat_level)\
         .order_by(func.count(DetectionRecord.id).desc())\
         .all()

        return [
            {
                'family': r.family,
                'count': r.count,
                'avg_confidence': round(r.avg_confidence, 4) if r.avg_confidence else 0,
                'threat_level': r.threat_level
            }
            for r in results
        ]

    def get_threat_level_distribution(self) -> dict:
        """Get distribution of detected DGAs by threat level."""
        from sqlalchemy import func

        results = self.db.query(
            DetectionRecord.threat_level,
            func.count(DetectionRecord.id).label('count')
        ).filter(
            DetectionRecord.is_dga == True,
            DetectionRecord.threat_level.isnot(None)
        ).group_by(DetectionRecord.threat_level)\
         .all()

        return {r.threat_level: r.count for r in results}


class UserRepository:
    """Repository for user records."""

    def __init__(self, db: Session):
        self.db = db

    def get_by_google_id(self, google_id: str) -> Optional[User]:
        """Get user by Google ID."""
        return self.db.query(User).filter(User.google_id == google_id).first()

    def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        return self.db.query(User).filter(User.email == email).first()

    def get_by_id(self, user_id: int) -> Optional[User]:
        """Get user by ID."""
        return self.db.query(User).filter(User.id == user_id).first()

    def create(
        self,
        google_id: str,
        email: str,
        name: Optional[str] = None,
        picture: Optional[str] = None
    ) -> User:
        """Create a new user."""
        user = User(
            google_id=google_id,
            email=email,
            name=name,
            picture=picture
        )
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user

    def update_last_login(self, user: User) -> User:
        """Update user's last login time."""
        user.last_login = datetime.now(timezone.utc)
        self.db.commit()
        self.db.refresh(user)
        return user

    def get_or_create(
        self,
        google_id: str,
        email: str,
        name: Optional[str] = None,
        picture: Optional[str] = None
    ) -> tuple[User, bool]:
        """Get existing user or create new one. Returns (user, created)."""
        user = self.get_by_google_id(google_id)
        if user:
            # Update profile info if changed (name, picture can change on Google)
            if name and user.name != name:
                user.name = name
            if picture and user.picture != picture:
                user.picture = picture
            self.update_last_login(user)
            return user, False

        user = self.create(google_id, email, name, picture)
        return user, True
