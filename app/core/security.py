"""
Internal API token authentication.

Laravel must send the header:
    X-API-Key: <shared secret>

This dependency rejects all requests without a valid key.
"""

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.core.config import Settings, get_settings

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    api_key: str | None = Security(_api_key_header),
    settings: Settings = Depends(get_settings),
) -> str:
    """
    Validate the X-API-Key header against the configured secret.

    Returns the key on success, raises 401 otherwise.
    """
    if api_key is None or api_key != settings.api_secret_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    return api_key
