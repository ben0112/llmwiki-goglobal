import asyncio
import hashlib
import logging

import jwt as pyjwt
from jwt import PyJWKClient
from mcp.server.auth.provider import AccessToken, TokenVerifier

from config import settings

logger = logging.getLogger(__name__)

API_KEY_PREFIX = "sv_"

_jwks_client: PyJWKClient | None = None


def _get_jwks_client() -> PyJWKClient:
    global _jwks_client
    if _jwks_client is None:
        jwks_url = f"{settings.SUPABASE_URL}/auth/v1/.well-known/jwks.json"
        _jwks_client = PyJWKClient(jwks_url)
    return _jwks_client


_EXPECTED_ISSUER = f"{settings.SUPABASE_URL.rstrip('/')}/auth/v1"


async def _verify_api_key(raw_key: str) -> AccessToken | None:
    """Verify an `sv_` API key (created via /v1/api-keys) by SHA-256 lookup.

    Static bearer credentials keep MCP clients working on self-hosted
    deployments whose auth server has no OAuth 2.1 support. Mirrors
    api/auth.py verify_api_key; last_used_at is updated in the same statement.
    """
    from db import service_queryrow

    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    try:
        row = await service_queryrow(
            "UPDATE api_keys SET last_used_at = now() "
            "WHERE key_hash = $1 AND revoked_at IS NULL "
            "RETURNING user_id",
            key_hash,
        )
    except Exception as e:
        logger.warning("MCP auth rejected: api-key lookup failed: %s", e)
        return None
    if row is None:
        logger.info("MCP auth rejected: unknown or revoked API key")
        return None
    user_id = str(row["user_id"])
    logger.info("MCP auth (api key): %s", user_id)
    return AccessToken(token=raw_key, client_id=user_id, scopes=[])


class SupabaseTokenVerifier(TokenVerifier):

    async def verify_token(self, token: str) -> AccessToken | None:
        if token.startswith(API_KEY_PREFIX):
            return await _verify_api_key(token)
        try:
            signing_key = await asyncio.to_thread(
                _get_jwks_client().get_signing_key_from_jwt, token
            )
            payload = pyjwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256", "ES256"],
                audience="authenticated",
                issuer=_EXPECTED_ISSUER,
                leeway=30,
                options={
                    "require": ["exp", "iat", "sub", "aud", "iss"],
                    "verify_exp": True,
                    "verify_iat": True,
                    "verify_nbf": True,
                },
            )
        except pyjwt.ExpiredSignatureError:
            logger.info("MCP auth rejected: token expired")
            return None
        except pyjwt.PyJWTError as e:
            logger.info("MCP auth rejected: %s: %s", type(e).__name__, e)
            return None
        except Exception as e:
            logger.warning("MCP auth rejected: JWKS fetch failed: %s", e)
            return None

        sub = payload.get("sub", "")
        if not sub:
            logger.warning("JWT has no sub claim")
            return None

        scopes = []
        scope_str = payload.get("scope", "")
        if isinstance(scope_str, str) and scope_str:
            scopes = scope_str.split()

        logger.info("MCP auth: %s", sub)
        return AccessToken(
            token=token,
            client_id=sub,
            scopes=scopes,
            extra={"claims": payload},
        )
