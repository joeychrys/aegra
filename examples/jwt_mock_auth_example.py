"""Example JWT authentication handler for Aegra.

This demonstrates how to implement authentication and authorization using
LangGraph SDK's Auth system with @auth.authenticate and @auth.on handlers.

It shows two authentication modes:
- Mock tokens for local development (format: mock-jwt-<user_id>-<role>-<team_id>)
- Real JWT/JWKS validation for production (validates against a JWKS endpoint)

Set the JWKS_URL environment variable to enable real JWT validation.
When JWKS_URL is not set, the handler falls back to mock token parsing.

Configuration:
Add this to your aegra.json or langgraph.json:

{
  "graphs": {
    "agent": "./graphs/react_agent/graph.py:graph"
  },
  "auth": {
    "path": "./jwt_mock_auth_example.py:auth"
  }
}

Note: If no auth path is configured in aegra.json, Aegra automatically
falls back to anonymous (noop) authentication. You only need this file
when you want to enforce token-based access control.

This example includes:
- Authentication handler (@auth.authenticate) with mock and real JWT modes
- Authorization handlers (@auth.on) for fine-grained access control
- Custom user fields (role, team_id, subscription_tier)
- Metadata injection in create operations
- Filter application in search operations
"""

import os
from typing import Any

import jwt
import structlog
from jwt import PyJWKClient
from langgraph_sdk import Auth

logger = structlog.getLogger(__name__)

auth = Auth()

# JWT Configuration (only used when JWKS_URL is set)
JWKS_URL = os.getenv("JWKS_URL")
JWT_ISSUER = os.getenv("JWT_ISSUER", "http://localhost:3000")
JWT_AUDIENCE = os.getenv("JWT_AUDIENCE", "http://localhost:3000")

# Lazy-initialized JWKS client
_jwks_client: PyJWKClient | None = None


def _get_jwks_client() -> PyJWKClient:
    """Get or create the JWKS client with caching.

    Returns:
        PyJWKClient instance for fetching signing keys from the JWKS endpoint.
    """
    global _jwks_client
    if _jwks_client is None:
        _jwks_client = PyJWKClient(
            JWKS_URL,
            cache_keys=True,
            lifespan=3600,  # Cache keys for 1 hour
        )
    return _jwks_client


def _parse_mock_token(token: str) -> Auth.types.MinimalUserDict:
    """Parse a mock JWT token for local development.

    Token format: mock-jwt-<user_id>-<role>-<team_id>
    Example: mock-jwt-alice-admin-team123

    Args:
        token: The raw token string (without "Bearer " prefix).

    Returns:
        User data dict with identity and custom fields.

    Raises:
        Auth.exceptions.HTTPException: If token format is invalid.
    """
    if not token.startswith("mock-jwt-"):
        raise Auth.exceptions.HTTPException(status_code=401, detail="Invalid token format")

    parts = token.split("-")[2:]  # Skip "mock-jwt"
    if len(parts) < 2:
        raise Auth.exceptions.HTTPException(status_code=401, detail="Token missing required fields")

    user_id = parts[0]
    role = parts[1]
    team_id = parts[2] if len(parts) > 2 else "team_default"
    subscription_tier = "premium" if role in ("admin", "premium") else "free"

    return {
        "identity": user_id,
        "display_name": f"User {user_id}",
        "is_authenticated": True,
        "permissions": [role, f"{role}:read", f"{role}:write"],
        "role": role,
        "subscription_tier": subscription_tier,
        "team_id": team_id,
        "email": f"{user_id}@example.com",
    }


def _validate_jwt(token: str) -> Auth.types.MinimalUserDict:
    """Validate a real JWT token against the configured JWKS endpoint.

    Supports EdDSA (Ed25519), RS256, and ES256 algorithms.

    Args:
        token: The raw JWT string (without "Bearer " prefix).

    Returns:
        User data dict with identity and display_name.

    Raises:
        Auth.exceptions.HTTPException: If token validation fails.
    """
    try:
        jwks_client = _get_jwks_client()
        signing_key = jwks_client.get_signing_key_from_jwt(token)

        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["EdDSA", "RS256", "ES256"],
            issuer=JWT_ISSUER,
            audience=JWT_AUDIENCE,
            options={
                "require": ["exp", "iat", "sub", "iss", "aud"],
            },
        )

        user_id = payload.get("sub")
        if not user_id:
            raise Auth.exceptions.HTTPException(status_code=401, detail="Invalid token: missing user identity")

        return {
            "identity": user_id,
            "display_name": payload.get("name", "Unknown User"),
            "is_authenticated": True,
        }

    except jwt.ExpiredSignatureError:
        logger.warning("JWT token expired")
        raise Auth.exceptions.HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidIssuerError:
        logger.warning("JWT invalid issuer")
        raise Auth.exceptions.HTTPException(status_code=401, detail="Invalid token issuer")
    except jwt.InvalidAudienceError:
        logger.warning("JWT invalid audience")
        raise Auth.exceptions.HTTPException(status_code=401, detail="Invalid token audience")
    except jwt.DecodeError as e:
        logger.warning("JWT decode error", error=str(e))
        raise Auth.exceptions.HTTPException(status_code=401, detail="Invalid token format")
    except jwt.PyJWKClientError as e:
        logger.error("JWKS client error", error=str(e))
        raise Auth.exceptions.HTTPException(status_code=503, detail="Authentication service unavailable")


# --- Authentication handler ---


@auth.authenticate
async def authenticate(headers: dict[str, str]) -> Auth.types.MinimalUserDict:
    """Authenticate requests via Bearer token.

    When JWKS_URL is set, validates real JWT tokens against the JWKS endpoint.
    Otherwise, parses mock tokens for local development.

    Args:
        headers: Request headers dict.

    Returns:
        User data dict with identity, display_name, and custom fields.

    Raises:
        Auth.exceptions.HTTPException: If token is missing or invalid.
    """
    authorization = headers.get("authorization") or headers.get("Authorization")

    if isinstance(authorization, bytes):
        authorization = authorization.decode("utf-8")

    if not authorization or not authorization.startswith("Bearer "):
        raise Auth.exceptions.HTTPException(
            status_code=401,
            detail="Missing or invalid Authorization header",
        )

    token = authorization[7:]  # Strip "Bearer "

    if JWKS_URL:
        return _validate_jwt(token)
    return _parse_mock_token(token)


# --- Authorization handlers ---


@auth.on
async def authorize(ctx: Auth.types.AuthContext, value: dict[str, Any]) -> dict[str, Any]:
    """Global authorization handler — user-scoped access by default.

    Applies an owner filter so users only see resources they created.
    Injects the owner into metadata on create/update operations.

    Args:
        ctx: Auth context containing the authenticated user.
        value: The resource value being authorized (may include metadata).

    Returns:
        Filter dict to apply to database queries.
    """
    user_id = ctx.user.identity

    owner_filter = {"owner": user_id}

    if value.get("metadata") is None:
        value["metadata"] = {}
    value["metadata"].update(owner_filter)

    return owner_filter


@auth.on.threads.create
async def on_thread_create(ctx: Auth.types.AuthContext, value: dict[str, Any]) -> dict[str, Any]:
    """Inject team_id into thread metadata on creation.

    This ensures data isolation at the team level when combined
    with the team-based search filter.

    Args:
        ctx: Auth context containing the authenticated user.
        value: The thread creation payload.

    Returns:
        Owner filter dict.
    """
    if value.get("metadata") is None:
        value["metadata"] = {}

    user_id = ctx.user.identity
    value["metadata"]["owner"] = user_id

    team_id = getattr(ctx.user, "team_id", None)
    if team_id:
        value["metadata"]["team_id"] = team_id

    return {"owner": user_id}


@auth.on.threads.search
async def on_thread_search(ctx: Auth.types.AuthContext, value: dict[str, Any]) -> dict[str, Any]:
    """Filter thread searches by team_id (or fall back to user_id).

    Args:
        ctx: Auth context containing the authenticated user.
        value: The search payload (unused).

    Returns:
        Filter dict to scope the search query.
    """
    team_id = getattr(ctx.user, "team_id", None)
    if team_id:
        return {"metadata": {"team_id": team_id}}
    return {"owner": ctx.user.identity}


@auth.on.assistants.create
async def on_assistant_create(ctx: Auth.types.AuthContext, value: dict[str, Any]) -> dict[str, Any]:
    """Inject creator info and team_id into assistant metadata.

    Args:
        ctx: Auth context containing the authenticated user.
        value: The assistant creation payload.

    Returns:
        Owner filter dict.
    """
    if value.get("metadata") is None:
        value["metadata"] = {}

    user_id = ctx.user.identity
    value["metadata"]["created_by"] = user_id
    value["metadata"]["owner"] = user_id

    team_id = getattr(ctx.user, "team_id", None)
    if team_id:
        value["metadata"]["team_id"] = team_id

    return {"owner": user_id}


@auth.on.assistants.delete
async def on_assistant_delete(ctx: Auth.types.AuthContext, value: dict[str, Any]) -> bool:
    """Only admins can delete assistants.

    Args:
        ctx: Auth context containing the authenticated user.
        value: The delete payload (unused).

    Returns:
        True if allowed, False if denied.
    """
    role = getattr(ctx.user, "role", None)
    return role == "admin"
