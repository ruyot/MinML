"""
Authentication and rate limiting middleware.
"""

import time
from typing import Dict, Optional, Tuple
from collections import defaultdict, deque
import hashlib

from fastapi import HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.security.utils import get_authorization_scheme_param

from .config import Settings


class RateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.requests: Dict[str, deque] = defaultdict(deque)
        self.enabled = settings.rate_limit_enabled
        self.max_requests = settings.rate_limit_requests
        self.window_seconds = settings.rate_limit_window
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting."""
        # Use X-Forwarded-For if available (for reverse proxy setups)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        else:
            client_ip = request.client.host if request.client else "unknown"
        
        # Include API key in rate limiting if available
        api_key = request.headers.get(self.settings.api_key_header)
        if api_key:
            # Hash the API key for privacy
            api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]
            return f"{client_ip}:{api_key_hash}"
        
        return client_ip
    
    def is_allowed(self, request: Request) -> Tuple[bool, Dict[str, int]]:
        """Check if request is allowed and return rate limit info."""
        if not self.enabled:
            return True, {}
        
        client_id = self._get_client_id(request)
        now = time.time()
        window_start = now - self.window_seconds
        
        # Clean old requests
        while self.requests[client_id] and self.requests[client_id][0] < window_start:
            self.requests[client_id].popleft()
        
        current_requests = len(self.requests[client_id])
        
        # Check if limit exceeded
        if current_requests >= self.max_requests:
            rate_limit_info = {
                "requests": current_requests,
                "limit": self.max_requests,
                "window": self.window_seconds,
                "reset_time": int(self.requests[client_id][0] + self.window_seconds),
                "retry_after": int(self.requests[client_id][0] + self.window_seconds - now)
            }
            return False, rate_limit_info
        
        # Add current request
        self.requests[client_id].append(now)
        
        rate_limit_info = {
            "requests": current_requests + 1,
            "limit": self.max_requests,
            "window": self.window_seconds,
            "remaining": self.max_requests - current_requests - 1
        }
        
        return True, rate_limit_info
    
    def add_rate_limit_headers(self, response, rate_limit_info: Dict[str, int]):
        """Add rate limit headers to response."""
        if not rate_limit_info:
            return
        
        response.headers["X-RateLimit-Limit"] = str(rate_limit_info.get("limit", ""))
        response.headers["X-RateLimit-Remaining"] = str(rate_limit_info.get("remaining", ""))
        response.headers["X-RateLimit-Window"] = str(rate_limit_info.get("window", ""))
        
        if "reset_time" in rate_limit_info:
            response.headers["X-RateLimit-Reset"] = str(rate_limit_info["reset_time"])
        
        if "retry_after" in rate_limit_info:
            response.headers["Retry-After"] = str(rate_limit_info["retry_after"])


class APIKeyAuth:
    """API key authentication."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.enabled = bool(settings.valid_api_keys)
        self.valid_keys = set(settings.valid_api_keys)
        self.header_name = settings.api_key_header
    
    def authenticate(self, request: Request) -> Optional[str]:
        """Authenticate request and return API key if valid."""
        if not self.enabled:
            return None
        
        api_key = request.headers.get(self.header_name)
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API key required",
                headers={"WWW-Authenticate": f"ApiKey header={self.header_name}"}
            )
        
        if api_key not in self.valid_keys:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key"
            )
        
        return api_key


class OptionalBearerAuth(HTTPBearer):
    """Optional Bearer token authentication."""
    
    def __init__(self, settings: Settings):
        super().__init__(auto_error=False)
        self.settings = settings
    
    async def __call__(self, request: Request) -> Optional[HTTPAuthorizationCredentials]:
        """Extract bearer token if present."""
        authorization = request.headers.get("Authorization")
        if not authorization:
            return None
        
        scheme, credentials = get_authorization_scheme_param(authorization)
        if not (authorization and scheme and credentials):
            return None
        
        if scheme.lower() != "bearer":
            return None
        
        return HTTPAuthorizationCredentials(scheme=scheme, credentials=credentials)


async def verify_auth_and_rate_limit(
    request: Request,
    settings: Settings,
    rate_limiter: RateLimiter,
    api_auth: APIKeyAuth
) -> Tuple[Optional[str], Dict[str, int]]:
    """Verify authentication and rate limiting."""
    
    # Check rate limiting first
    allowed, rate_limit_info = rate_limiter.is_allowed(request)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={
                "Retry-After": str(rate_limit_info.get("retry_after", 60)),
                "X-RateLimit-Limit": str(rate_limit_info.get("limit", "")),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(rate_limit_info.get("reset_time", ""))
            }
        )
    
    # Check API key authentication
    api_key = None
    if api_auth.enabled:
        api_key = api_auth.authenticate(request)
    
    return api_key, rate_limit_info


def create_auth_components(settings: Settings) -> Tuple[RateLimiter, APIKeyAuth, OptionalBearerAuth]:
    """Create authentication and rate limiting components."""
    rate_limiter = RateLimiter(settings)
    api_auth = APIKeyAuth(settings)
    bearer_auth = OptionalBearerAuth(settings)
    
    return rate_limiter, api_auth, bearer_auth 