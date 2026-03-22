"""Security Service for HOPEFX Trading Bot.

Provides basic JWT token generation/validation and password hashing stubs
that can be wired into the FastAPI application layer.
"""

import hashlib
import hmac
import os
import time
from typing import Dict, Any, Optional


# ---------------------------------------------------------------------------
# Simple HMAC-SHA256 token helpers (no external JWT library required)
# ---------------------------------------------------------------------------

class SecurityService:
    """Handles token issuance and validation for API authentication.

    This is a lightweight stub using HMAC-SHA256 signed tokens.
    Replace with a full JWT library (e.g. ``python-jose``) for production.
    """

    def __init__(self, secret_key: Optional[str] = None) -> None:
        self._secret = (secret_key or os.environ.get("APP_SECRET_KEY", "changeme")).encode()

    # ------------------------------------------------------------------
    # Token helpers
    # ------------------------------------------------------------------

    def create_token(self, payload: Dict[str, Any], ttl_seconds: int = 3600) -> str:
        """Create a signed token encoding *payload*.

        Args:
            payload: Arbitrary key-value data to embed in the token.
            ttl_seconds: Token lifetime in seconds (default: 1 hour).

        Returns:
            A hex-encoded ``<timestamp>.<payload_hash>.<signature>`` string.
        """
        import json, base64
        payload["exp"] = int(time.time()) + ttl_seconds
        body = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
        sig = hmac.new(self._secret, body.encode(), hashlib.sha256).hexdigest()
        return f"{body}.{sig}"

    def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate *token* and return its payload if valid.

        Args:
            token: A token previously issued by :meth:`create_token`.

        Returns:
            The decoded payload dict, or ``None`` if the token is invalid
            or has expired.
        """
        import json, base64
        try:
            body, sig = token.rsplit(".", 1)
        except ValueError:
            return None

        expected = hmac.new(self._secret, body.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, sig):
            return None

        try:
            payload = json.loads(base64.urlsafe_b64decode(body + "==").decode())
        except Exception:
            return None

        if payload.get("exp", 0) < int(time.time()):
            return None  # Expired

        return payload

    # ------------------------------------------------------------------
    # Password helpers
    # ------------------------------------------------------------------

    @staticmethod
    def hash_password(password: str, salt: Optional[bytes] = None) -> str:
        """Hash *password* with PBKDF2-HMAC-SHA256.

        Args:
            password: Plain-text password to hash.
            salt: Optional salt bytes; a random salt is generated when omitted.

        Returns:
            A ``<hex_salt>:<hex_digest>`` string suitable for storage.
        """
        if salt is None:
            salt = os.urandom(16)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 260_000)
        return f"{salt.hex()}:{dk.hex()}"

    @staticmethod
    def verify_password(password: str, stored_hash: str) -> bool:
        """Verify *password* against a hash produced by :meth:`hash_password`.

        Args:
            password: Plain-text password to check.
            stored_hash: The stored ``<hex_salt>:<hex_digest>`` value.

        Returns:
            ``True`` if the password matches, ``False`` otherwise.
        """
        try:
            salt_hex, dk_hex = stored_hash.split(":")
        except ValueError:
            return False
        salt = bytes.fromhex(salt_hex)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 260_000)
        return hmac.compare_digest(dk.hex(), dk_hex)

