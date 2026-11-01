"""Receipt signing and secret redaction for the control plane (blueprint §10, §13).

The runtime "returns a signed or hashed receipt" and must "redact secrets"
before a receipt is appended to the event log or shown to the model (§10).

* :class:`Redactor` removes secrets by key name (``api_key``, ``password``...),
  by well-known credential shapes (``sk-...``, ``AKIA...``, bearer tokens,
  PEM private keys) and by exact registered secret values. Redaction happens
  *before* hashing, so a signed receipt can be shared without leaking.
* :class:`ReceiptSigner` computes ``receipt_hash = content_hash(body)`` and an
  HMAC-SHA256 signature over it with a control-plane key. Previous keys can be
  kept for verification during key rotation. Verification uses constant-time
  comparison and fails closed on any malformed input.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from typing import Any, Iterable, Mapping, Optional

from ..common import content_hash

__all__ = ["REDACTED", "Redactor", "ReceiptSigner"]

REDACTED = "[REDACTED]"

_SECRET_KEY_RE = re.compile(
    r"(?i)(secret|passw(or)?d|passphrase|api[_-]?key|access[_-]?key|auth[_-]?token|access[_-]?token|"
    r"refresh[_-]?token|session[_-]?token|private[_-]?key|credential|authorization|cookie|signing[_-]?key)"
)
_SECRET_VALUE_RES = (
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/\-]{16,}=*"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"),
)


class Redactor:
    """Recursively replaces secrets in JSON-like structures with ``[REDACTED]``."""

    def __init__(self, secret_values: Iterable[str] = ()) -> None:
        self._values: set[str] = set()
        for v in secret_values:
            self.register_secret(v)

    def register_secret(self, value: str | bytes) -> None:
        """Register an exact secret value (e.g. a tool credential) to scrub everywhere."""
        if isinstance(value, bytes):
            try:
                value = value.decode("utf-8")
            except UnicodeDecodeError:
                value = value.hex()
        if len(value) >= 6:  # very short values would cause absurd over-redaction
            self._values.add(value)

    def redact_text(self, text: str) -> str:
        for v in sorted(self._values, key=len, reverse=True):
            if v in text:
                text = text.replace(v, REDACTED)
        for rx in _SECRET_VALUE_RES:
            text = rx.sub(REDACTED, text)
        return text

    def redact(self, obj: Any) -> Any:
        """Return a redacted deep copy of ``obj`` (dict keys are kept, values scrubbed)."""
        if isinstance(obj, str):
            return self.redact_text(obj)
        if isinstance(obj, Mapping):
            out: dict[Any, Any] = {}
            for k, v in obj.items():
                if isinstance(k, str) and _SECRET_KEY_RE.search(k) and v is not None:
                    out[k] = REDACTED
                else:
                    out[k] = self.redact(v)
            return out
        if isinstance(obj, (list, tuple)):
            return [self.redact(v) for v in obj]
        return obj


class ReceiptSigner:
    """HMAC-SHA256 signer for receipt bodies (with optional rotated verification keys)."""

    SCHEME = "hmac-sha256"

    def __init__(self, secret: bytes, key_id: str = "cp-k1", previous_keys: Optional[Mapping[str, bytes]] = None) -> None:
        if not isinstance(secret, (bytes, bytearray)) or len(secret) < 16:
            raise ValueError("control-plane signing secret must be at least 16 bytes")
        self.key_id = key_id
        self._keys: dict[str, bytes] = {key_id: bytes(secret)}
        for kid, key in (previous_keys or {}).items():
            self._keys.setdefault(kid, bytes(key))

    def _mac(self, key_id: str, receipt_hash: str) -> Optional[str]:
        key = self._keys.get(key_id)
        if key is None:
            return None
        digest = hmac.new(key, receipt_hash.encode("utf-8"), hashlib.sha256).hexdigest()
        return f"{self.SCHEME}:{digest}"

    def sign(self, body: Mapping[str, Any]) -> tuple[str, str]:
        """Return ``(receipt_hash, signature)`` for a receipt body signed with the active key."""
        if body.get("key_id") != self.key_id:
            raise ValueError("receipt body must carry the active key_id before signing")
        receipt_hash = content_hash(dict(body))
        mac = self._mac(self.key_id, receipt_hash)
        assert mac is not None
        return receipt_hash, mac

    def verify(self, receipt: Mapping[str, Any]) -> bool:
        """True iff the hash matches the body and the HMAC matches the hash. Fails closed."""
        try:
            d = dict(receipt)
            claimed_hash = d.pop("receipt_hash")
            signature = d.pop("signature")
            key_id = d["key_id"]
            if not isinstance(claimed_hash, str) or not isinstance(signature, str):
                return False
            if content_hash(d) != claimed_hash:
                return False
            expected = self._mac(key_id, claimed_hash)
            return expected is not None and hmac.compare_digest(expected, signature)
        except (KeyError, TypeError, ValueError):
            return False

    @property
    def secrets(self) -> tuple[bytes, ...]:
        """Key material (for registration with a :class:`Redactor`; never logged)."""
        return tuple(self._keys.values())
