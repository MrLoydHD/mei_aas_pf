"""
Google OAuth authentication for DGA Detection System.
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
from jose import jwt, JWTError
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.api.database import get_db, UserRepository, DetectionRepository, User

# Configuration
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24 * 7  # 7 days

router = APIRouter(prefix="/auth", tags=["Authentication"])
security = HTTPBearer(auto_error=False)


# Request/Response models
class GoogleAuthRequest(BaseModel):
    credential: str  # Google ID token


class UserResponse(BaseModel):
    id: int
    email: str
    name: Optional[str]
    picture: Optional[str]

    class Config:
        from_attributes = True


class AuthResponse(BaseModel):
    user: UserResponse
    token: str
    created: bool  # True if new user was created


class UserStatsResponse(BaseModel):
    total_checked: int
    dga_detected: int
    legit_detected: int


class SyncStatsRequest(BaseModel):
    total_checked: int
    dga_detected: int
    legit_detected: int


def create_jwt_token(user_id: int, email: str) -> str:
    """Create a JWT token for the user."""
    payload = {
        "sub": str(user_id),
        "email": email,
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRATION_HOURS),
        "iat": datetime.now(timezone.utc)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_jwt_token(token: str) -> Optional[dict]:
    """Verify and decode a JWT token."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except JWTError:
        return None


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db)
) -> Optional[User]:
    """Get the current authenticated user from JWT token."""
    if not credentials:
        return None

    token = credentials.credentials
    payload = verify_jwt_token(token)
    if not payload:
        return None

    user_id = int(payload.get("sub", 0))
    if not user_id:
        return None

    user_repo = UserRepository(db)
    return user_repo.get_by_id(user_id)


async def require_user(
    user: Optional[User] = Depends(get_current_user)
) -> User:
    """Require an authenticated user."""
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


@router.post("/google", response_model=AuthResponse)
async def google_auth(
    request: GoogleAuthRequest,
    db: Session = Depends(get_db)
):
    """
    Authenticate with Google ID token.

    The frontend sends the Google ID token after user signs in with Google.
    We verify it and create/get the user, returning a JWT for subsequent requests.
    """
    try:
        # Get client ID from environment
        client_id = os.getenv("GOOGLE_CLIENT_ID", "")
        if not client_id:
            raise HTTPException(status_code=500, detail="GOOGLE_CLIENT_ID not configured")

        # Verify the Google ID token
        idinfo = id_token.verify_oauth2_token(
            request.credential,
            google_requests.Request(),
            client_id
        )

        # Extract user info from token
        google_id = idinfo.get("sub")
        email = idinfo.get("email")
        name = idinfo.get("name")
        picture = idinfo.get("picture")

        if not google_id or not email:
            raise HTTPException(status_code=400, detail="Invalid token: missing user info")

        # Get or create user
        user_repo = UserRepository(db)
        user, created = user_repo.get_or_create(
            google_id=google_id,
            email=email,
            name=name,
            picture=picture
        )

        # Create JWT token
        token = create_jwt_token(user.id, user.email)

        return AuthResponse(
            user=UserResponse(
                id=user.id,
                email=user.email,
                name=user.name,
                picture=user.picture
            ),
            token=token,
            created=created
        )

    except ValueError as e:
        raise HTTPException(status_code=401, detail=f"Invalid Google token: {str(e)}")


@router.get("/me", response_model=UserResponse)
async def get_me(user: User = Depends(require_user)):
    """Get current authenticated user info."""
    return UserResponse(
        id=user.id,
        email=user.email,
        name=user.name,
        picture=user.picture
    )


@router.get("/stats", response_model=UserStatsResponse)
async def get_user_stats(
    user: User = Depends(require_user),
    db: Session = Depends(get_db)
):
    """Get detection statistics for the authenticated user."""
    repo = DetectionRepository(db)
    stats = repo.get_user_stats(user.id)
    return UserStatsResponse(**stats)


@router.post("/sync", response_model=UserStatsResponse)
async def sync_stats(
    request: SyncStatsRequest,
    user: User = Depends(require_user),
    db: Session = Depends(get_db)
):
    """
    Sync extension stats to user account.

    This creates placeholder detection records to match the extension's stats.
    Used when user first signs in on extension to sync their existing data.
    """
    repo = DetectionRepository(db)
    current_stats = repo.get_user_stats(user.id)

    # Calculate how many records we need to add
    dga_to_add = max(0, request.dga_detected - current_stats['dga_detected'])
    legit_to_add = max(0, request.legit_detected - current_stats['legit_detected'])

    # Create placeholder records for synced data
    for _ in range(dga_to_add):
        repo.create(
            domain="synced-dga-domain",
            is_dga=True,
            confidence=0.9,
            dga_probability=0.9,
            model_used="extension_sync",
            source="extension_sync",
            user_id=user.id
        )

    for _ in range(legit_to_add):
        repo.create(
            domain="synced-legit-domain",
            is_dga=False,
            confidence=0.9,
            dga_probability=0.1,
            model_used="extension_sync",
            source="extension_sync",
            user_id=user.id
        )

    # Return updated stats
    updated_stats = repo.get_user_stats(user.id)
    return UserStatsResponse(**updated_stats)


@router.post("/verify")
async def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Verify if a JWT token is valid."""
    if not credentials:
        raise HTTPException(status_code=401, detail="No token provided")

    payload = verify_jwt_token(credentials.credentials)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    return {"valid": True, "user_id": payload.get("sub")}


class ExtensionAuthRequest(BaseModel):
    access_token: str  # Google OAuth access token from chrome.identity


@router.post("/extension", response_model=AuthResponse)
async def extension_auth(
    request: ExtensionAuthRequest,
    db: Session = Depends(get_db)
):
    """
    Authenticate with Google access token from Chrome extension.

    The extension uses chrome.identity.getAuthToken() which returns an access token.
    We validate it by calling Google's userinfo API.
    """
    import httpx

    try:
        # Validate access token by calling Google userinfo API
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://www.googleapis.com/oauth2/v3/userinfo",
                headers={"Authorization": f"Bearer {request.access_token}"}
            )

            if response.status_code != 200:
                raise HTTPException(status_code=401, detail="Invalid access token")

            userinfo = response.json()

        # Extract user info
        google_id = userinfo.get("sub")
        email = userinfo.get("email")
        name = userinfo.get("name")
        picture = userinfo.get("picture")

        if not google_id or not email:
            raise HTTPException(status_code=400, detail="Invalid token: missing user info")

        # Get or create user
        user_repo = UserRepository(db)
        user, created = user_repo.get_or_create(
            google_id=google_id,
            email=email,
            name=name,
            picture=picture
        )

        # Create JWT token
        token = create_jwt_token(user.id, user.email)

        return AuthResponse(
            user=UserResponse(
                id=user.id,
                email=user.email,
                name=user.name,
                picture=user.picture
            ),
            token=token,
            created=created
        )

    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Failed to validate token: {str(e)}")


class UserDetectionLog(BaseModel):
    """Model for user detection logs."""
    id: int
    domain: str
    is_dga: bool
    confidence: float
    source: Optional[str] = None
    timestamp: datetime

    class Config:
        from_attributes = True


class UserDetectionsResponse(BaseModel):
    """Response model for user detections."""
    detections: list[UserDetectionLog]
    total: int
    dga_count: int
    legit_count: int


@router.get("/detections", response_model=UserDetectionsResponse)
async def get_user_detections(
    user: User = Depends(require_user),
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Get detection history for the authenticated user."""
    repo = DetectionRepository(db)
    detections = repo.get_user_detections(user.id, limit=limit)
    stats = repo.get_user_stats(user.id)

    return UserDetectionsResponse(
        detections=[
            UserDetectionLog(
                id=d.id,
                domain=d.domain,
                is_dga=d.is_dga,
                confidence=d.confidence,
                source=d.source,
                timestamp=d.timestamp
            )
            for d in detections
        ],
        total=stats['total_checked'],
        dga_count=stats['dga_detected'],
        legit_count=stats['legit_detected']
    )
