#!/usr/bin/env python3
"""Service-account (M2M) credential provider for MCP HTTP servers.

Exchanges a long-lived service-account password for a short-lived Bearer
access token, injects ``Authorization: Bearer <access_token>`` into MCP
requests, and renews automatically.  This is distinct from the
browser-based PKCE flow (``auth: oauth``) — no user interaction is
required.

Grant strategy is **explicit**, never inferred from which fields happen to
be present.  ``service_account.grant_type`` selects it and is required.

Supported strategies
--------------------
``authentik_app_password``
    Authentik's service-account extension.  Posts ``grant_type=
    client_credentials`` together with a resource-owner ``username`` /
    ``password`` pair.  Note this is *not* the RFC 6749 §4.4.2
    client-credentials request, which carries no username/password — it is
    a provider extension that happens to reuse the same wire grant name.
    Providers whose M2M flow is plain client authentication (Keycloak
    service accounts, Auth0 M2M) are **not** supported by this strategy;
    adding a standards-conforming ``client_credentials`` strategy is a
    separate, additive change.

Configuration in config.yaml::

    mcp_servers:
      toolhive:
        url: https://mcp.example/mcp
        auth: service_account
        service_account:
          grant_type: authentik_app_password         # required, explicit
          token_url: https://idp.example/application/o/toolhive/token/
          client_id: toolhive
          username: zug
          password_env: AUTHENTIK_ZUG_APP_PASSWORD   # env-var name, not value
          scope: "openid profile groups toolhive-audience"
          client_secret_env: OPTIONAL_CLIENT_SECRET  # optional

Secret values (password, client_secret) are **never** stored in
config.yaml. Only the environment-variable *names* appear there; the
values are read at runtime via ``agent.secret_scope.get_secret`` which
honours the active profile's isolated secret scope under multiplexing and
falls back to ``os.environ`` in single-profile mode.

Token caching
-------------
Tokens are cached at
``$HERMES_HOME/mcp-tokens/service-account/<server>-<digest>.json`` with
file permissions 0o600 and atomic write (O_EXCL temp-then-rename).

Three separate collisions are ruled out by that layout:

- **Across profiles** — the path is rooted at the owning profile's
  ``HERMES_HOME``, so two profiles using the same server name never share a
  token. The owning home is passed explicitly by ``MCPServerTask``; it is not
  re-derived from ambient state.
- **Against browser OAuth** — service-account tokens live in their own
  ``service-account/`` subdirectory. The previous ``<server>-sa.json``
  convention shared a directory with browser OAuth's ``<server>.json``, so a
  server named ``foo-sa`` aliased the service-account cache of ``foo``.
- **Across server names and identities** — ``<digest>`` is a SHA-256 prefix
  over the raw server name and the credential identity, which restores the
  distinctions the filename sanitizer erases (``a/b`` and ``a_b`` both
  sanitize to ``a_b``).

Cached tokens are additionally **bound** to the identity that minted them —
``grant_type``, ``token_url``, ``client_id``, ``username``, ``scope`` and the
credential env-var *names* (see :func:`sa_identity_fingerprint`). Changing any
of them invalidates the cache instead of continuing to present a token for the
previous identity. Only env-var names are hashed; no secret value is.

The access token is cached; the service-account password is never written
to disk. If the server returns a ``refresh_token``, it is cached and used
on subsequent renewals, falling back to a fresh service-account exchange
if the refresh fails. When a refresh response omits ``refresh_token``
(permitted by RFC 6749 §6, meaning "keep the one you have") the existing
refresh token is retained rather than dropped.

httpx compatibility
-------------------
``ServiceAccountAuth`` inherits from the ``Auth`` class exported by
whichever httpx distribution the installed MCP SDK uses (plain ``httpx``
for mcp < 2.0, ``httpx2`` for mcp >= 2.0).  The base class is resolved
once at module import time via :func:`_resolve_auth_base` and stored in
``_SA_AUTH_BASE``.  This makes the provider a valid ``isinstance(...,
httpx.Auth)`` object and therefore acceptable to ``AsyncClient(auth=...)``.

Security
--------
- **Transport.**  ``https://`` token endpoints are always accepted.  Plain
  ``http://`` is accepted **only** for loopback hosts (``localhost``,
  ``127.0.0.1``, ``::1``), where the request never reaches a network — the
  same carve-out RFC 8252 §8.3 makes for native-app loopback.  A plaintext
  non-loopback endpoint is refused, because the token request carries a
  long-lived service-account password (RFC 6749 §2.3.1).  This is enforced
  when the config is validated *and* again immediately before every token
  request, so a config that bypassed validation cannot slip past.  Loopback
  plaintext logs a warning on each exchange.
- **Redirects.**  Token-endpoint redirects are **not** followed, regardless
  of scheme.  A 307/308 is method-preserving, so an authorization server (or
  a compromised or misconfigured one) could otherwise redirect the POST —
  password and client secret included — to an origin the config never
  authorised.  The config proves exactly one secret sink; runtime does not
  widen it.  A 3xx from the token endpoint is surfaced as an error, and the
  ``Location`` is never logged.
- TLS verification is always on; no way to disable it from config.
- Passwords, access tokens, Authorization header values, and token
  responses are never logged.  Errors are redacted before surfacing.
- ``password_env`` accepts only a legal environment-variable name.
- The password is fetched once per token exchange and not held in memory
  beyond the HTTP request coroutine.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import secrets
import stat
import time
import threading as _threading
from urllib.parse import urlsplit
from pathlib import Path
from typing import Any, Optional

from agent.secret_scope import get_secret as _get_scoped_secret

logger = logging.getLogger(__name__)

# ── How many seconds before nominal expiry to proactively renew the token.
_PROACTIVE_RENEW_BUFFER_SECONDS = 60

# ── Env-var name validation — same rule as shell identifier.
_ENV_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# ── Grant strategies.  The config value is a Hermes-level discriminator, not
#    the OAuth wire ``grant_type`` — see GRANT_WIRE_TYPES below.
GRANT_AUTHENTIK_APP_PASSWORD = "authentik_app_password"

#: Config-level grant strategies this provider implements.  Adding a
#: standards-conforming ``client_credentials`` strategy is additive: extend
#: this set, GRANT_WIRE_TYPES, and _build_exchange_form.
SUPPORTED_GRANT_TYPES: frozenset[str] = frozenset({GRANT_AUTHENTIK_APP_PASSWORD})

#: Config strategy → the ``grant_type`` value actually sent on the wire.
#: Authentik's service-account extension reuses the ``client_credentials``
#: wire name while adding a resource-owner username/password pair, so the
#: two names deliberately differ here.
GRANT_WIRE_TYPES: dict[str, str] = {
    GRANT_AUTHENTIK_APP_PASSWORD: "client_credentials",
}

#: Fields required per grant strategy, on top of the common ones.
_GRANT_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    GRANT_AUTHENTIK_APP_PASSWORD: ("username", "password_env"),
}

#: Required regardless of strategy.
_COMMON_REQUIRED_FIELDS: tuple[str, ...] = ("token_url", "client_id")

#: Hosts for which a plaintext ``http://`` token endpoint is accepted.
#: Loopback never leaves the machine, so there is no network path on which the
#: credential could be observed — the same carve-out RFC 8252 §8.3 makes for
#: native-app loopback redirects. Every other host must be https://: the token
#: request carries a long-lived service-account password (RFC 6749 §2.3.1).
_PLAINTEXT_OK_HOSTS: frozenset[str] = frozenset(
    {"localhost", "127.0.0.1", "::1", "[::1]"}
)


def _token_url_scheme_error(name: str, token_url: str) -> Optional[str]:
    """Return an error string when *token_url* may not carry credentials.

    ``https://`` is always accepted. ``http://`` is accepted only for loopback
    hosts. Anything else — a non-loopback http:// host, or a non-HTTP scheme —
    is refused.
    """
    text = str(token_url)
    if text.startswith("https://"):
        return None
    if not text.startswith("http://"):
        return (
            f"MCP server '{name}': service_account.token_url must be an "
            "http(s):// URL"
        )
    try:
        host = (urlsplit(text).hostname or "").lower()
    except ValueError:
        host = ""
    if host in _PLAINTEXT_OK_HOSTS:
        return None
    return (
        f"MCP server '{name}': service_account.token_url must use https:// — "
        "the token request carries the service-account password, which must "
        "not cross a network in plaintext (RFC 6749 §2.3.1). Plain http:// is "
        "accepted only for loopback hosts (localhost, 127.0.0.1, ::1)."
    )


def _is_loopback_plaintext(token_url: str) -> bool:
    """True when *token_url* is an accepted plaintext loopback endpoint."""
    text = str(token_url)
    if not text.startswith("http://"):
        return False
    try:
        return (urlsplit(text).hostname or "").lower() in _PLAINTEXT_OK_HOSTS
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# httpx Auth base — resolved from the SDK's own httpx distribution
# ---------------------------------------------------------------------------


def _resolve_auth_base() -> type:
    """Return the ``Auth`` base class from the MCP SDK's httpx distribution.

    mcp >= 2.0 ships its transports against ``httpx2`` rather than the
    upstream ``httpx``.  ``AsyncClient(auth=...)`` uses ``isinstance(auth,
    Auth)`` from *its own* httpx module, so the provider must inherit from
    the same class.  We mirror what :func:`tools.mcp_tool.sdk_httpx` does
    (read the transport module's ``httpx2`` or ``httpx`` attribute) without
    importing all of ``mcp_tool`` to avoid a heavy circular-import at
    module load time.
    """
    try:
        from mcp.client import streamable_http as _transport

        _mod = getattr(_transport, "httpx2", None) or getattr(_transport, "httpx", None)
        if _mod is not None and hasattr(_mod, "Auth"):
            return _mod.Auth
    except ImportError:
        pass
    # Fallback: httpx2 then httpx
    try:
        import httpx2 as _h

        return _h.Auth
    except ImportError:
        import httpx as _h  # type: ignore[no-redef]

        return _h.Auth


_SA_AUTH_BASE: type = _resolve_auth_base()


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def validate_service_account_config(name: str, cfg: dict) -> list[str]:
    """Return a list of human-readable validation errors for a service_account block.

    ``name`` is the MCP server name (for error messages).  ``cfg`` is the
    value of the ``service_account:`` sub-key in the server config.
    """
    errors: list[str] = []
    if not isinstance(cfg, dict):
        return [f"MCP server '{name}': service_account must be a mapping"]

    # Grant strategy is explicit — never inferred from field presence.
    grant_type = cfg.get("grant_type")
    if not grant_type:
        errors.append(
            f"MCP server '{name}': service_account.grant_type is required. "
            f"Supported: {', '.join(sorted(SUPPORTED_GRANT_TYPES))}"
        )
    elif str(grant_type) not in SUPPORTED_GRANT_TYPES:
        errors.append(
            f"MCP server '{name}': service_account.grant_type "
            f"'{grant_type}' is not supported. "
            f"Supported: {', '.join(sorted(SUPPORTED_GRANT_TYPES))}"
        )

    required = _COMMON_REQUIRED_FIELDS + _GRANT_REQUIRED_FIELDS.get(
        str(grant_type), ()
    )
    for field in required:
        if not cfg.get(field):
            errors.append(f"MCP server '{name}': service_account.{field} is required")

    token_url = cfg.get("token_url", "")
    if token_url:
        scheme_error = _token_url_scheme_error(name, token_url)
        if scheme_error:
            errors.append(scheme_error)

    for env_field in ("password_env", "client_secret_env"):
        val = cfg.get(env_field)
        if val and not _ENV_VAR_NAME_RE.match(str(val)):
            errors.append(
                f"MCP server '{name}': service_account.{env_field} must be a "
                "valid environment-variable name (letters, digits, underscores)"
            )

    return errors


def _resolve_password(cfg: dict, server_name: str) -> str:
    """Fetch the service-account password from the active profile secret scope.

    Reads the env-var named in ``password_env`` via ``agent.secret_scope.get_secret``
    so the active profile's isolated scope is honoured under multiplexing. Falls
    back to ``os.environ`` in single-profile mode (when no secret scope is
    installed and multiplexing is inactive).

    Raises ``ValueError`` with a non-secret message if the secret is missing or
    empty. In multiplex mode with no scope installed, ``get_secret`` raises
    ``UnscopedSecretError`` (a ``RuntimeError`` subclass) before this function
    constructs its own error — that propagates as-is to the caller.
    """
    env_name = cfg.get("password_env", "")
    if not env_name:
        raise ValueError(
            f"MCP service-account '{server_name}': password_env is required"
        )
    if not _ENV_VAR_NAME_RE.match(str(env_name)):
        raise ValueError(
            f"MCP service-account '{server_name}': password_env "
            f"'{env_name}' is not a valid environment-variable name"
        )
    value = _get_scoped_secret(str(env_name)) or ""
    if not value:
        raise ValueError(
            f"MCP service-account '{server_name}': environment variable "
            f"'{env_name}' is not set or is empty. "
            f"Set it in the profile's $HERMES_HOME/.env before connecting."
        )
    return value


def _resolve_client_secret(cfg: dict) -> Optional[str]:
    """Return the optional client secret from the profile secret scope, or None."""
    env_name = cfg.get("client_secret_env", "")
    if not env_name:
        return None
    return _get_scoped_secret(str(env_name)) or None


# ---------------------------------------------------------------------------
# Token cache (disk)
# ---------------------------------------------------------------------------


def sa_identity_fingerprint(cfg: dict) -> str:
    """Return a stable fingerprint of the credential identity in *cfg*.

    A cached access token is only valid for the exact identity it was minted
    for. Every field below changes *who* the token represents or *what* it is
    good for, so a change to any of them must invalidate the cache rather than
    keep presenting a token for the previous identity:

    - ``token_url``  — which authorization server issued it
    - ``client_id``  — which OAuth client it belongs to
    - ``username``   — which service account it authenticates
    - ``scope``      — what it is authorized to do
    - ``password_env`` / ``client_secret_env`` — which credential minted it

    ``password_env`` is an env-var NAME, never a value; nothing secret is
    hashed here, and the digest is not a secret either.
    """
    material = "\x00".join(
        str(cfg.get(field, ""))
        for field in (
            "grant_type",
            "token_url",
            "client_id",
            "username",
            "scope",
            "password_env",
            "client_secret_env",
        )
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]


def _get_sa_token_dir(hermes_home: Optional[str | Path] = None) -> Path:
    """Return the service-account token directory for a profile.

    Deliberately a SUBDIRECTORY of the browser-OAuth token dir rather than a
    sibling filename convention. The old layout wrote
    ``mcp-tokens/<server>-sa.json`` alongside browser OAuth's
    ``mcp-tokens/<server>.json``, so a server literally named ``foo-sa``
    produced the same path as the service-account cache for server ``foo`` —
    one server's credential silently served as another's. Separate namespaces
    make that structurally impossible.
    """
    from tools.mcp_oauth import _get_token_dir

    return _get_token_dir(hermes_home) / "service-account"


def _get_sa_token_path(
    server_name: str,
    hermes_home: Optional[str | Path] = None,
    identity: Optional[str] = None,
) -> Path:
    """Return the path to the service-account token cache file.

    The filename is ``<sanitized-name>-<digest>.json``. ``_safe_filename`` is
    lossy — it maps every non-word character to ``_``, so ``tool/hive``,
    ``tool.hive`` and ``tool_hive`` all sanitize to ``tool_hive`` and would
    otherwise share one cache file. The digest is taken over the RAW server
    name (plus the credential identity when known), which restores the
    distinction the sanitizer destroys.

    Rooted at the owning profile's ``HERMES_HOME``, so two profiles using the
    same server name never share a token — see ``build_service_account_auth``.
    """
    from tools.mcp_oauth import _safe_filename

    digest_material = server_name if identity is None else f"{server_name}\x00{identity}"
    digest = hashlib.sha256(digest_material.encode("utf-8")).hexdigest()[:16]
    return _get_sa_token_dir(hermes_home) / f"{_safe_filename(server_name)}-{digest}.json"


def _read_token_cache(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _write_token_cache(path: Path, data: dict) -> None:
    """Atomically write token cache to *path* with mode 0o600."""
    from hermes_constants import secure_parent_dir

    path.parent.mkdir(parents=True, exist_ok=True)
    secure_parent_dir(path)
    tmp = path.with_suffix(f".tmp.{os.getpid()}.{secrets.token_hex(4)}")
    try:
        fd = os.open(
            str(tmp),
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            stat.S_IRUSR | stat.S_IWUSR,
        )
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except OSError:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _delete_token_cache(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


class _CachedToken:
    """In-memory view of a cached service-account token."""

    def __init__(
        self,
        access_token: str,
        expires_at: float,
        refresh_token: Optional[str] = None,
        identity: Optional[str] = None,
        issued_at: Optional[float] = None,
        lifetime: Optional[float] = None,
    ):
        self.access_token = access_token
        self.expires_at = expires_at
        self.refresh_token = refresh_token
        # Fingerprint of the credential identity this token was minted for
        # (see sa_identity_fingerprint). Checked on load; a mismatch means the
        # operator changed username/client_id/scope/token_url and the cached
        # token no longer represents what the config asks for.
        self.identity = identity
        self.issued_at = issued_at if issued_at is not None else time.time()
        # NOMINAL lifetime in seconds, i.e. the ``expires_in`` the
        # authorization server actually returned. ``None`` means unknown.
        #
        # It is deliberately NOT inferred from ``expires_at - now``: that
        # conflates "a 30-second token" with "a one-hour token observed 30
        # seconds before it expires", and the two need opposite handling —
        # the first should still be used, the second must be renewed.
        self.lifetime = lifetime

    def renew_buffer(self) -> float:
        """Seconds before expiry at which this token should be renewed.

        A flat 60s buffer is wrong for genuinely short-lived tokens: an
        authorization server issuing ``expires_in: 30`` would produce a token
        that is *never* valid, so ``_get_cached_token`` always misses,
        ``_acquire_token`` re-exchanges on every single MCP request, and the
        provider melts the token endpoint while never caching anything. When
        the nominal lifetime is known and short, scale the buffer down to half
        of it so the token is still used for the first half of its life.

        When the lifetime is unknown, keep the flat buffer — that is the
        historical behaviour and the safe default.
        """
        if self.lifetime is None:
            return float(_PROACTIVE_RENEW_BUFFER_SECONDS)
        if self.lifetime <= 0:
            return 0.0
        return min(float(_PROACTIVE_RENEW_BUFFER_SECONDS), self.lifetime / 2.0)

    def is_valid(self, buffer: Optional[float] = None) -> bool:
        if buffer is None:
            buffer = self.renew_buffer()
        return time.time() < self.expires_at - buffer

    @classmethod
    def from_dict(cls, data: dict) -> "_CachedToken | None":
        at = data.get("access_token")
        ea = data.get("expires_at")
        if not at or not ea:
            return None
        try:
            expires_at = float(ea)
            issued_at = data.get("issued_at")
            issued_at = float(issued_at) if issued_at is not None else None
            lifetime = data.get("lifetime")
            lifetime = float(lifetime) if lifetime is not None else None
            return cls(
                access_token=str(at),
                expires_at=expires_at,
                refresh_token=data.get("refresh_token") or None,
                identity=data.get("identity") or None,
                issued_at=issued_at,
                lifetime=lifetime,
            )
        except (TypeError, ValueError):
            return None

    def to_dict(self) -> dict:
        d: dict = {
            "access_token": self.access_token,
            "expires_at": self.expires_at,
            "issued_at": self.issued_at,
        }
        if self.lifetime is not None:
            d["lifetime"] = self.lifetime
        if self.refresh_token:
            d["refresh_token"] = self.refresh_token
        if self.identity:
            d["identity"] = self.identity
        return d


# ---------------------------------------------------------------------------
# HTTP token exchange
# ---------------------------------------------------------------------------


async def _post_token_request(
    http_client: Any,
    token_url: str,
    form: dict,
    server_name: str,
) -> dict:
    """POST form-encoded data to token_url and parse the JSON response.

    Never logs form values (which include the password).  Raises ``ValueError``
    with a redacted error message on any failure.

    The transport requirement is re-checked here rather than trusted from
    validation time: this is the last point before a credential-bearing body
    leaves the process, and the caller may have been handed a config that
    never passed through :func:`validate_service_account_config`. ``https://``
    is always allowed; ``http://`` only for loopback, which never reaches a
    network.

    The caller must supply a client with redirects disabled; a 3xx from the
    token endpoint therefore falls through to the redirect branch and is
    reported as an error rather than replaying the form at a new origin.
    """
    if _token_url_scheme_error(server_name, token_url) is not None:
        raise ValueError(
            f"MCP service-account '{server_name}': refusing to send "
            "credentials to a plaintext non-loopback token endpoint"
        )
    if _is_loopback_plaintext(token_url):
        logger.warning(
            "MCP service-account '%s': token endpoint is plaintext http:// on "
            "a loopback host. This is accepted for local development only — "
            "use https:// for any endpoint reachable over a network.",
            server_name,
        )

    try:
        resp = await http_client.post(
            token_url,
            data=form,
            headers={"Accept": "application/json"},
        )
    except Exception as exc:
        # Redact the URL in case query-string values snuck in somehow.
        raise ValueError(
            f"MCP service-account '{server_name}': token endpoint request failed"
        ) from exc

    if 300 <= resp.status_code < 400:
        # Redirects are deliberately not followed: replaying a
        # password-bearing POST at a Location the config never authorised is
        # credential egress to an unproven sink.  Never log the Location.
        raise ValueError(
            f"MCP service-account '{server_name}': token endpoint returned a "
            f"redirect (HTTP {resp.status_code}); redirects are not followed "
            "because the request carries credentials. Point token_url at the "
            "authorization server's final https:// token endpoint."
        )

    if not (200 <= resp.status_code < 300):
        # Never include the response body — it may echo back the error_description
        # which can include credential hints.
        raise ValueError(
            f"MCP service-account '{server_name}': token endpoint returned "
            f"HTTP {resp.status_code}"
        )

    try:
        body = resp.json()
    except Exception:
        raise ValueError(
            f"MCP service-account '{server_name}': token endpoint returned "
            "non-JSON response"
        )

    if not isinstance(body, dict) or "access_token" not in body:
        raise ValueError(
            f"MCP service-account '{server_name}': token response missing "
            "'access_token' field"
        )

    return body


def _parse_token_response(
    body: dict,
    server_name: str,
    *,
    now: Optional[float] = None,
) -> _CachedToken:
    """Parse a standard token response body into a ``_CachedToken``."""
    access_token = str(body.get("access_token", ""))
    if not access_token:
        raise ValueError(
            f"MCP service-account '{server_name}': empty access_token in response"
        )
    expires_in = body.get("expires_in")
    if expires_in is None:
        # Default to 1 hour when the server omits expires_in.
        expires_in = 3600
    try:
        expires_in = int(expires_in)
    except (TypeError, ValueError):
        expires_in = 3600
    t = now if now is not None else time.time()
    return _CachedToken(
        access_token=access_token,
        expires_at=t + expires_in,
        refresh_token=body.get("refresh_token") or None,
        issued_at=t,
        # The server's own expires_in is the only trustworthy source for the
        # nominal lifetime — see _CachedToken.renew_buffer.
        lifetime=float(expires_in),
    )


# ---------------------------------------------------------------------------
# Per-server refresh deduplication
# ---------------------------------------------------------------------------

# Keyed by (hermes_home_str, server_name) → asyncio.Lock.
_refresh_locks: dict[tuple[str, str], asyncio.Lock] = {}
_refresh_locks_mu = _threading.Lock()


def _get_refresh_lock(server_name: str, hermes_home: str) -> asyncio.Lock:
    key = (hermes_home, server_name)
    with _refresh_locks_mu:
        lock = _refresh_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            _refresh_locks[key] = lock
        return lock


def _clear_refresh_locks_for_tests() -> None:
    """Test-only: reset the global lock table."""
    with _refresh_locks_mu:
        _refresh_locks.clear()


# ---------------------------------------------------------------------------
# ServiceAccountAuth — inherits from the SDK's httpx.Auth
# ---------------------------------------------------------------------------


class ServiceAccountAuth(_SA_AUTH_BASE):  # type: ignore[valid-type,misc]
    """httpx.Auth subclass for service-account M2M token exchange.

    Inherits from the ``Auth`` class of whichever httpx distribution the
    installed MCP SDK uses (``httpx`` for mcp < 2.0, ``httpx2`` for mcp
    >= 2.0).  This satisfies ``AsyncClient(auth=...)``'s ``isinstance``
    check on both SDK generations.

    The provider:
    - Caches tokens to disk at ``$HERMES_HOME/mcp-tokens/<server>-sa.json``.
    - Proactively renews tokens before they expire (60s buffer).
    - On a 401, obtains a fresh token and retries the request once.
    - Deduplicates concurrent refresh attempts within the process.
    - Never logs passwords, tokens, or Authorization header values.
    """

    # Tell httpx we need to read both request and response bodies so the
    # base-class sync stub in auth_flow() is never called (we fully override
    # async_auth_flow).  Setting these to False is safe because our
    # async_auth_flow is a complete async generator that never delegates to
    # the sync auth_flow() path.
    requires_request_body = False
    requires_response_body = False

    def __init__(
        self,
        server_name: str,
        sa_config: dict,
        *,
        hermes_home: Optional[str | Path] = None,
    ):
        # httpx.Auth.__init__ takes no arguments, but call it for compat.
        super().__init__()
        self._server_name = server_name
        self._cfg = dict(sa_config)
        from hermes_constants import get_hermes_home

        self._hermes_home = str(
            Path(hermes_home).expanduser().resolve(strict=False)
            if hermes_home is not None
            else get_hermes_home()
        )
        self._identity = sa_identity_fingerprint(self._cfg)
        self._cache_path = _get_sa_token_path(
            server_name, self._hermes_home, self._identity
        )
        # In-memory token — avoids a disk read on every request.
        self._mem_token: Optional[_CachedToken] = None

    @property
    def _refresh_lock(self) -> asyncio.Lock:
        # Keyed by (home, server, identity): two profiles never share a lock,
        # and neither do two different credential identities that happen to
        # use the same server name.
        return _get_refresh_lock(
            f"{self._server_name}\x00{self._identity}", self._hermes_home
        )

    # -- Token resolution ----------------------------------------------------

    def _load_from_disk(self) -> Optional[_CachedToken]:
        data = _read_token_cache(self._cache_path)
        if data is None:
            return None
        token = _CachedToken.from_dict(data)
        if token is None:
            return None
        # Identity binding: a token minted for a different token_url /
        # client_id / username / scope must never be presented for this
        # config, even if it landed at this path (stale file, restored
        # backup, edited config). Absent identity = written by an older
        # build; treat as a mismatch and re-mint rather than trusting it.
        if token.identity != self._identity:
            logger.debug(
                "MCP service-account '%s': cached token identity does not "
                "match the current config; discarding and re-minting",
                self._server_name,
            )
            _delete_token_cache(self._cache_path)
            return None
        return token

    def _save_to_disk(self, token: _CachedToken) -> None:
        # Stamp the identity at the single write boundary rather than in each
        # exchange path, so no minting route can persist an unbound token.
        token.identity = self._identity
        try:
            _write_token_cache(self._cache_path, token.to_dict())
        except OSError as exc:
            logger.warning(
                "MCP service-account '%s': failed to write token cache: %s",
                self._server_name,
                exc,
            )

    def _get_cached_token(self) -> Optional[_CachedToken]:
        """Return a valid in-memory or disk-cached token, or None."""
        if self._mem_token is not None and self._mem_token.is_valid():
            return self._mem_token
        disk = self._load_from_disk()
        if disk is not None and disk.is_valid():
            self._mem_token = disk
            return disk
        return None

    def _build_exchange_form(
        self, password: str, client_secret: Optional[str]
    ) -> dict:
        """Build the token-request form for this server's grant strategy.

        Dispatch is on the explicit ``grant_type`` discriminator, so a new
        strategy adds a branch here rather than changing meaning for existing
        configs.
        """
        cfg = self._cfg
        grant = str(cfg.get("grant_type", ""))
        if grant not in SUPPORTED_GRANT_TYPES:
            raise ValueError(
                f"MCP service-account '{self._server_name}': unsupported "
                f"service_account.grant_type '{grant}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_GRANT_TYPES))}"
            )

        form: dict = {
            "grant_type": GRANT_WIRE_TYPES[grant],
            "client_id": cfg["client_id"],
        }
        if grant == GRANT_AUTHENTIK_APP_PASSWORD:
            # Authentik's service-account extension: client_credentials on the
            # wire plus a resource-owner username/password pair.
            form["username"] = cfg["username"]
            form["password"] = password
        if cfg.get("scope"):
            form["scope"] = cfg["scope"]
        if client_secret:
            form["client_secret"] = client_secret
        return form

    async def _exchange_service_account(self, http_client: Any) -> _CachedToken:
        """Exchange the service-account credential for an access token."""
        cfg = self._cfg
        password = _resolve_password(cfg, self._server_name)
        client_secret = _resolve_client_secret(cfg)

        form = self._build_exchange_form(password, client_secret)

        body = await _post_token_request(
            http_client, cfg["token_url"], form, self._server_name
        )
        token = _parse_token_response(body, self._server_name)
        token.identity = self._identity
        del password
        if client_secret:
            del client_secret
        return token

    async def _exchange_refresh_token(
        self,
        http_client: Any,
        refresh_token: str,
    ) -> Optional[_CachedToken]:
        """Try a refresh_token grant; return None on failure."""
        cfg = self._cfg
        client_secret = _resolve_client_secret(cfg)
        form: dict = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": cfg["client_id"],
        }
        if cfg.get("scope"):
            form["scope"] = cfg["scope"]
        if client_secret:
            form["client_secret"] = client_secret
        try:
            body = await _post_token_request(
                http_client, cfg["token_url"], form, self._server_name
            )
            token = _parse_token_response(body, self._server_name)
            # RFC 6749 §6: the refresh response MAY omit refresh_token, which
            # means "keep using the one you have". Parsing it as None would
            # persist a token with no refresh credential and force a full
            # service-account exchange (re-reading the long-lived password) on
            # the very next renewal — the opposite of what the AS asked for.
            if not token.refresh_token:
                token.refresh_token = refresh_token
            token.identity = self._identity
            return token
        except ValueError:
            logger.debug(
                "MCP service-account '%s': refresh_token grant failed, "
                "falling back to service-account exchange",
                self._server_name,
            )
            return None

    async def _acquire_token(self, http_client: Any) -> _CachedToken:
        """Acquire a fresh token, using refresh_token if available.

        Protected by a per-server asyncio.Lock so concurrent requests only
        trigger one exchange.
        """
        async with self._refresh_lock:
            # Re-check under lock — another coroutine may have refreshed.
            cached = self._get_cached_token()
            if cached is not None:
                return cached

            # Try refresh_token first.
            existing = self._load_from_disk() or self._mem_token
            if existing and existing.refresh_token:
                new_token = await self._exchange_refresh_token(
                    http_client, existing.refresh_token
                )
                if new_token is not None:
                    self._mem_token = new_token
                    self._save_to_disk(new_token)
                    logger.debug(
                        "MCP service-account '%s': renewed via refresh_token",
                        self._server_name,
                    )
                    return new_token

            # Fall back to service-account exchange.
            token = await self._exchange_service_account(http_client)
            self._mem_token = token
            self._save_to_disk(token)
            logger.debug(
                "MCP service-account '%s': acquired new access token",
                self._server_name,
            )
            return token

    # -- httpx.Auth protocol -------------------------------------------------

    def auth_flow(self, request: Any):  # type: ignore[override]
        # httpx.Auth requires a sync auth_flow stub; the async path below
        # is used exclusively in our async MCP context.  This stub is never
        # called because async_auth_flow is overridden and httpx prefers it.
        raise NotImplementedError(  # pragma: no cover
            "ServiceAccountAuth requires an async context; "
            "use an AsyncClient, not a sync Client"
        )

    async def async_auth_flow(self, request: Any):  # type: ignore[override]
        """Inject Bearer token, handle one 401 retry.

        httpx drives this generator:
          1. ``__anext__()``       → we yield the request with Authorization header
          2. ``asend(response)``  → we inspect the response
             - 2xx/other: generator returns → httpx uses that response
             - 401:  invalidate cache, fetch fresh token, yield retry request
             - ``asend(response2)`` → generator returns
        """
        # Build a small dedicated client for token-endpoint requests.  It is
        # created fresh each auth-flow invocation so it lives only as long as
        # a single MCP request (including one possible 401 retry).  We resolve
        # the correct httpx module here (same logic as _resolve_auth_base) to
        # handle both mcp 1.x (httpx) and mcp 2.x (httpx2) at runtime.
        try:
            from mcp.client import streamable_http as _transport

            _httpx_mod = getattr(_transport, "httpx2", None) or getattr(
                _transport, "httpx", None
            )
        except ImportError:
            _httpx_mod = None
        if _httpx_mod is None:
            try:
                import httpx2 as _httpx_mod  # type: ignore[no-redef]
            except ImportError:
                import httpx as _httpx_mod  # type: ignore[no-redef]

        # follow_redirects=False is a security requirement, not a default:
        # 307/308 preserve the method and body, so following one would replay
        # the service-account password at whatever origin the token endpoint
        # names.  _post_token_request turns any 3xx into an error.
        async with _httpx_mod.AsyncClient(follow_redirects=False) as token_client:
            token = await self._acquire_token(token_client)

            # Inject Authorization header without logging the value.
            request.headers["Authorization"] = f"Bearer {token.access_token}"
            response = yield request

            if response.status_code != 401:
                return

            # 401: invalidate and retry once.
            #
            # Only invalidate if the token we actually SENT is still the
            # cached one. Under concurrency a 401 can arrive late: request A
            # goes out on token T1, T1 expires, request B refreshes to T2, and
            # only then does A's 401 (for T1) come back. Unconditionally
            # clearing here would throw away the perfectly good T2 that B just
            # minted — and every in-flight delayed 401 would do it again,
            # producing an exchange storm where one refresh was needed. When
            # the cache has already moved on, just retry with the current
            # token.
            stale = token.access_token
            logger.debug(
                "MCP service-account '%s': received 401, refreshing token",
                self._server_name,
            )
            current = self._mem_token
            if current is None or current.access_token == stale:
                self._mem_token = None
                _delete_token_cache(self._cache_path)

            token = await self._acquire_token(token_client)
            request.headers["Authorization"] = f"Bearer {token.access_token}"
            yield request


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def build_service_account_auth(
    server_name: str,
    sa_config: dict,
    *,
    hermes_home: Optional[str | Path] = None,
) -> "ServiceAccountAuth":
    """Build and return a :class:`ServiceAccountAuth` for *server_name*.

    ``sa_config`` is the value of the ``service_account:`` sub-key in the
    MCP server config dict.  Call this once per server and cache the result;
    it manages its own token state.

    ``hermes_home`` should be the OWNING profile's canonical home. It pins
    the token cache file and the refresh lock, so two profiles configuring
    the same logical server name never share a minted token. Callers inside
    the multiplexed gateway pass it explicitly (``MCPServerTask`` passes the
    identity it was constructed with); when omitted it falls back to the
    ambient ``get_hermes_home()``, which is correct for single-profile use.

    Raises ``ValueError`` if the config is missing required fields.
    """
    errors = validate_service_account_config(server_name, sa_config)
    if errors:
        raise ValueError("; ".join(errors))
    auth = ServiceAccountAuth(server_name, sa_config, hermes_home=hermes_home)
    # Diagnostics: profile identity + server name only. password_env is a
    # variable NAME (not a value) and is the exact field that made the
    # cross-profile bug visible, so it is safe and useful to record; no
    # secret, token or header value is ever logged here.
    logger.debug(
        "MCP service-account '%s': bound to profile home %s "
        "(password_env=%s, cache=%s)",
        server_name,
        auth._hermes_home,
        sa_config.get("password_env", ""),
        auth._cache_path,
    )
    return auth


def remove_service_account_tokens(
    server_name: str,
    *,
    hermes_home: Optional[str | Path] = None,
    sa_config: Optional[dict] = None,
) -> None:
    """Delete the on-disk service-account token cache for *server_name*.

    Cache filenames carry an identity digest, so removing "the" cache for a
    server means removing every identity ever cached under that name unless
    the caller pins one via ``sa_config``. Sweeping by prefix keeps ``hermes
    mcp logout``-style flows from leaving a token behind after the operator
    edited ``username`` or ``scope``.

    Only ever deletes inside this profile's service-account directory.
    """
    if sa_config is not None:
        _delete_token_cache(
            _get_sa_token_path(
                server_name, hermes_home, sa_identity_fingerprint(sa_config)
            )
        )
        logger.info("MCP service-account '%s': removed token cache", server_name)
        return

    from tools.mcp_oauth import _safe_filename

    token_dir = _get_sa_token_dir(hermes_home)
    prefix = f"{_safe_filename(server_name)}-"
    removed = 0
    try:
        candidates = list(token_dir.glob(f"{prefix}*.json"))
    except OSError:
        candidates = []
    for path in candidates:
        _delete_token_cache(path)
        removed += 1
    # Legacy layout (mcp-tokens/<server>-sa.json) written by earlier builds.
    legacy = _get_sa_token_dir(hermes_home).parent / f"{_safe_filename(server_name)}-sa.json"
    if legacy.exists():
        _delete_token_cache(legacy)
        removed += 1
    logger.info(
        "MCP service-account '%s': removed %d token cache file(s)",
        server_name,
        removed,
    )
