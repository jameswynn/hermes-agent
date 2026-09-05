"""Tests for tools/mcp_service_account.py — service-account M2M OAuth provider.

Each test uses a local mock token endpoint (aiohttp or http.server) and a
mock MCP server.  No real credentials are used.  The HERMES_HOME env var
is monkeypatched to a tmp_path so no real profile is touched.

Tests cover:
- Config validation (required fields, env-var name checks)
- First token acquisition and Authorization header injection
- Cached token reuse (no second exchange)
- Proactive renewal before expiry (token near-expired → refresh)
- refresh_token path and fallback to service-account exchange
- One 401 retry: fresh token acquired, request retried
- Concurrent requests: only one token exchange
- Missing password env var → clear error, no secret in message
- Malformed token response → clear error
- Token endpoint failure (HTTP 500) → clear error
- No secret values in log output
- Profile/home cache isolation (two homes → separate caches)
- Backwards compatibility: header auth and browser OAuth modes still work
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import types
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sa_config(
    *,
    grant_type: str | None = "authentik_app_password",
    token_url: str = "https://idp.example/token/",
    client_id: str = "toolhive",
    username: str = "zug",
    password_env: str = "TEST_SA_PASSWORD",
    scope: str = "openid profile",
    client_secret_env: str = "",
) -> dict:
    cfg: dict = {
        "token_url": token_url,
        "client_id": client_id,
        "username": username,
        "password_env": password_env,
        "scope": scope,
    }
    # ``grant_type=None`` builds a config that omits the discriminator, so
    # tests can assert it is required rather than defaulted.
    if grant_type is not None:
        cfg["grant_type"] = grant_type
    if client_secret_env:
        cfg["client_secret_env"] = client_secret_env
    return cfg


def _sa_cache_path(server_name: str, home, cfg: dict | None = None):
    """Cache path for *server_name* under *home*, bound to *cfg*'s identity.

    Filenames carry an identity digest (token_url/client_id/username/scope/…),
    so a test that wants "the cache file this config writes" must supply the
    same config the provider was built with.
    """
    from tools.mcp_service_account import (
        _get_sa_token_path,
        sa_identity_fingerprint,
    )

    cfg = _make_sa_config() if cfg is None else cfg
    return _get_sa_token_path(server_name, home, sa_identity_fingerprint(cfg))


def _fake_token_response(
    access_token: str = "ACCESS",
    expires_in: int = 3600,
    refresh_token: str | None = None,
) -> dict:
    d = {"access_token": access_token, "token_type": "Bearer", "expires_in": expires_in}
    if refresh_token:
        d["refresh_token"] = refresh_token
    return d


class _FakeResponse:
    """Minimal fake httpx.Response."""

    def __init__(self, status_code: int, body: dict | None = None):
        self.status_code = status_code
        self._body = body or {}

    def json(self) -> dict:
        return self._body


class _FakeHttpxClient:
    """Records calls and returns configurable responses."""

    def __init__(self, responses: list[_FakeResponse]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        pass

    async def post(self, url: str, *, data: dict, headers: dict) -> _FakeResponse:
        self.calls.append({"url": url, "data": dict(data)})
        if not self._responses:
            raise RuntimeError("No more fake responses")
        return self._responses.pop(0)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestValidateServiceAccountConfig:
    def test_valid_config(self):
        from tools.mcp_service_account import validate_service_account_config

        cfg = _make_sa_config()
        assert validate_service_account_config("srv", cfg) == []

    def test_missing_required_fields(self):
        """An empty config reports the discriminator and the common fields.

        Per-strategy fields (username/password_env) are deliberately *not*
        reported here: without a grant_type there is no strategy yet, so the
        validator cannot know which extra fields apply. Naming them anyway
        would re-introduce the inference this schema exists to remove.
        """
        from tools.mcp_service_account import validate_service_account_config

        errors = validate_service_account_config("srv", {})
        assert any("grant_type" in e for e in errors)
        assert any("token_url" in e for e in errors)
        assert any("client_id" in e for e in errors)

    def test_missing_grant_type_specific_fields(self):
        from tools.mcp_service_account import validate_service_account_config

        cfg = {"grant_type": "authentik_app_password"}
        errors = validate_service_account_config("srv", cfg)
        assert any("username" in e for e in errors)
        assert any("password_env" in e for e in errors)

    def test_invalid_password_env_name(self):
        from tools.mcp_service_account import validate_service_account_config

        cfg = _make_sa_config(password_env="bad name!")
        errors = validate_service_account_config("srv", cfg)
        assert any("password_env" in e for e in errors)

    def test_invalid_client_secret_env_name(self):
        from tools.mcp_service_account import validate_service_account_config

        cfg = _make_sa_config(client_secret_env="bad name!")
        errors = validate_service_account_config("srv", cfg)
        assert any("client_secret_env" in e for e in errors)

    def test_invalid_token_url_scheme(self):
        from tools.mcp_service_account import validate_service_account_config

        cfg = _make_sa_config(token_url="ftp://bad.example/token")
        errors = validate_service_account_config("srv", cfg)
        assert any("token_url" in e for e in errors)

    def test_not_a_dict(self):
        from tools.mcp_service_account import validate_service_account_config

        errors = validate_service_account_config("srv", "string")  # type: ignore
        assert errors


# ---------------------------------------------------------------------------
# Credential-egress boundary: the token endpoint is the ONLY secret sink
# ---------------------------------------------------------------------------


class TestTokenEndpointIsTheCredentialBoundary:
    """The token request carries the service-account password.

    Two properties keep that credential pinned to exactly the sink the config
    proves: the endpoint must be https://, and a redirect away from it is
    never followed.
    """

    def test_plaintext_http_token_url_rejected_at_validation(self):
        from tools.mcp_service_account import validate_service_account_config

        cfg = _make_sa_config(token_url="http://idp.example/token/")
        errors = validate_service_account_config("srv", cfg)
        assert any("token_url" in e for e in errors), (
            "http:// token_url must be rejected — it would put the "
            "service-account password on the wire in the clear"
        )

    def test_build_refuses_plaintext_http_token_url(self):
        from tools.mcp_service_account import build_service_account_auth

        cfg = _make_sa_config(token_url="http://idp.example/token/")
        with pytest.raises(ValueError, match="token_url"):
            build_service_account_auth("srv", cfg)

    @pytest.mark.asyncio
    async def test_post_refuses_plaintext_http_even_if_validation_bypassed(self):
        """Consumption-time guard, not just a validation-time one.

        A caller can construct ServiceAccountAuth directly, or mutate the
        config after validation. The last check before the credential leaves
        the process must stand on its own.
        """
        from tools.mcp_service_account import _post_token_request

        client = _FakeHttpxClient([_FakeResponse(200, _fake_token_response())])
        with pytest.raises(ValueError, match="plaintext non-loopback"):
            await _post_token_request(
                client,
                "http://idp.example/token/",
                {"password": "hunter2"},
                "srv",
            )
        assert client.calls == [], "No request may be issued to a plaintext endpoint"

    @pytest.mark.asyncio
    async def test_loopback_plaintext_http_is_allowed(self):
        """Local development IdPs stay usable; the network case does not.

        http:// on loopback never reaches a network, so there is no path on
        which the password could be observed (cf. RFC 8252 §8.3). Any other
        host must still be https://.
        """
        from tools.mcp_service_account import _post_token_request

        for url in (
            "http://localhost:9000/token/",
            "http://127.0.0.1:9000/token/",
            "http://[::1]:9000/token/",
        ):
            client = _FakeHttpxClient([_FakeResponse(200, _fake_token_response())])
            body = await _post_token_request(
                client, url, {"password": "hunter2"}, "srv"
            )
            assert body["access_token"] == "ACCESS"
            assert len(client.calls) == 1

    @pytest.mark.asyncio
    async def test_loopback_lookalike_hosts_are_still_refused(self):
        """A hostname that merely *contains* localhost is not loopback."""
        from tools.mcp_service_account import _post_token_request

        for url in (
            "http://localhost.evil.example/token/",
            "http://127.0.0.1.evil.example/token/",
            "http://notlocalhost/token/",
        ):
            client = _FakeHttpxClient([_FakeResponse(200, _fake_token_response())])
            with pytest.raises(ValueError, match="plaintext non-loopback"):
                await _post_token_request(
                    client, url, {"password": "hunter2"}, "srv"
                )
            assert client.calls == []

    @pytest.mark.asyncio
    async def test_redirect_is_refused_on_loopback_too(self):
        """The redirect ban is not conditional on the scheme."""
        from tools.mcp_service_account import _post_token_request

        client = _FakeHttpxClient([_FakeResponse(307, {})])
        with pytest.raises(ValueError, match="redirect"):
            await _post_token_request(
                client,
                "http://localhost:9000/token/",
                {"password": "hunter2"},
                "srv",
            )
        assert len(client.calls) == 1

    @pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
    @pytest.mark.asyncio
    async def test_redirect_from_token_endpoint_is_not_followed(self, status):
        """307/308 preserve method and body — following one replays the password.

        The fake client returns the redirect rather than transparently
        chasing it, mirroring follow_redirects=False. Exactly one request must
        have been made, to the configured origin only.
        """
        from tools.mcp_service_account import _post_token_request

        client = _FakeHttpxClient([_FakeResponse(status, {})])
        with pytest.raises(ValueError, match="redirect"):
            await _post_token_request(
                client,
                "https://idp.example/token/",
                {"grant_type": "client_credentials", "password": "hunter2"},
                "srv",
            )

        assert len(client.calls) == 1, "The form must not be replayed after a 3xx"
        assert client.calls[0]["url"] == "https://idp.example/token/"

    @pytest.mark.asyncio
    async def test_redirect_error_does_not_leak_the_location_or_password(self):
        from tools.mcp_service_account import _post_token_request

        client = _FakeHttpxClient([_FakeResponse(307, {})])
        with pytest.raises(ValueError) as exc:
            await _post_token_request(
                client,
                "https://idp.example/token/",
                {"password": "hunter2"},
                "srv",
            )
        assert "hunter2" not in str(exc.value)
        assert "evil.example" not in str(exc.value)

    @pytest.mark.asyncio
    async def test_token_client_is_built_with_redirects_disabled(
        self, tmp_path, monkeypatch
    ):
        """The live auth flow must construct its client with follow_redirects=False.

        Asserted at the construction site because the fake client in the other
        tests cannot itself prove which flag the real one was given.
        """
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("TEST_SA_PASSWORD", "hunter2")

        from tools.mcp_service_account import (
            ServiceAccountAuth,
            _clear_refresh_locks_for_tests,
        )

        _clear_refresh_locks_for_tests()
        auth = ServiceAccountAuth("srv", _make_sa_config(), hermes_home=tmp_path)

        captured_kwargs: list[dict] = []

        def _fake_async_client(**kwargs):
            captured_kwargs.append(kwargs)
            return _FakeHttpxClient([])

        fake_httpx = types.SimpleNamespace(AsyncClient=_fake_async_client)

        async def _ok_post(http_client, token_url, form, server_name):
            return _fake_token_response("TOK")

        with patch.dict(sys.modules, {"httpx2": fake_httpx}):
            with patch("tools.mcp_service_account._post_token_request", _ok_post):
                with patch(
                    "mcp.client.streamable_http.httpx2", fake_httpx, create=True
                ):
                    request = MagicMock()
                    request.headers = {}
                    gen = auth.async_auth_flow(request)
                    await gen.__anext__()
                    resp = MagicMock()
                    resp.status_code = 200
                    try:
                        await gen.asend(resp)
                    except StopAsyncIteration:
                        pass

        assert captured_kwargs, "No AsyncClient was constructed"
        assert captured_kwargs[0].get("follow_redirects") is False


# ---------------------------------------------------------------------------
# Grant strategy is explicit, never inferred
# ---------------------------------------------------------------------------


class TestGrantStrategyIsExplicit:
    def test_grant_type_is_required(self):
        from tools.mcp_service_account import validate_service_account_config

        cfg = _make_sa_config(grant_type=None)
        errors = validate_service_account_config("srv", cfg)
        assert any("grant_type" in e for e in errors), (
            "grant_type must not be inferred from the presence of "
            "username/password_env"
        )

    def test_unknown_grant_type_rejected(self):
        from tools.mcp_service_account import validate_service_account_config

        cfg = _make_sa_config(grant_type="client_credentials")
        errors = validate_service_account_config("srv", cfg)
        assert any("grant_type" in e for e in errors), (
            "The standards-conforming client_credentials strategy is not "
            "implemented; it must be rejected rather than silently treated "
            "as the Authentik app-password extension"
        )

    def test_build_rejects_unknown_grant_type(self):
        from tools.mcp_service_account import build_service_account_auth

        cfg = _make_sa_config(grant_type="totally_made_up")
        with pytest.raises(ValueError, match="grant_type"):
            build_service_account_auth("srv", cfg)

    @pytest.mark.asyncio
    async def test_authentik_app_password_wire_witness(self, tmp_path, monkeypatch):
        """Pin the exact form Authentik's service-account extension expects.

        Authentik reuses the ``client_credentials`` wire grant name while
        adding a resource-owner username/password pair, so the config-level
        strategy name and the wire value deliberately differ.
        """
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("TEST_SA_PASSWORD", "hunter2")
        monkeypatch.setenv("TEST_SA_CLIENT_SECRET", "cs3cr3t")

        from tools.mcp_service_account import (
            ServiceAccountAuth,
            _clear_refresh_locks_for_tests,
        )

        _clear_refresh_locks_for_tests()
        cfg = _make_sa_config(client_secret_env="TEST_SA_CLIENT_SECRET")
        auth = ServiceAccountAuth("srv", cfg, hermes_home=tmp_path)

        form = auth._build_exchange_form("hunter2", "cs3cr3t")

        assert form == {
            "grant_type": "client_credentials",
            "client_id": "toolhive",
            "username": "zug",
            "password": "hunter2",
            "scope": "openid profile",
            "client_secret": "cs3cr3t",
        }

    def test_exchange_form_rejects_unsupported_strategy_at_runtime(self, tmp_path):
        from tools.mcp_service_account import ServiceAccountAuth

        cfg = _make_sa_config(grant_type="client_credentials")
        auth = ServiceAccountAuth("srv", cfg, hermes_home=tmp_path)
        with pytest.raises(ValueError, match="grant_type"):
            auth._build_exchange_form("hunter2", None)


# ---------------------------------------------------------------------------
# Password resolution
# ---------------------------------------------------------------------------


class TestResolvePassword:
    def test_reads_from_env(self, monkeypatch):
        from tools.mcp_service_account import _resolve_password

        monkeypatch.setenv("MY_PASS", "secret123")
        assert _resolve_password({"password_env": "MY_PASS"}, "srv") == "secret123"

    def test_missing_env_raises_with_env_name_not_value(self, monkeypatch):
        from tools.mcp_service_account import _resolve_password

        monkeypatch.delenv("MISSING_VAR", raising=False)
        with pytest.raises(ValueError) as exc:
            _resolve_password({"password_env": "MISSING_VAR"}, "srv")
        msg = str(exc.value)
        assert "MISSING_VAR" in msg
        # The error message should include the env-var *name*, not attempt to show any value
        assert "secret" not in msg.lower()

    def test_no_password_env_raises(self):
        from tools.mcp_service_account import _resolve_password

        with pytest.raises(ValueError, match="password_env is required"):
            _resolve_password({}, "srv")


# ---------------------------------------------------------------------------
# Token acquisition — first call
# ---------------------------------------------------------------------------


class TestFirstTokenAcquisition:
    @pytest.mark.asyncio
    async def test_injects_bearer_header(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("TEST_SA_PASSWORD", "hunter2")

        from tools.mcp_service_account import (
            ServiceAccountAuth,
            _clear_refresh_locks_for_tests,
        )

        _clear_refresh_locks_for_tests()

        cfg = _make_sa_config()
        auth = ServiceAccountAuth("srv", cfg, hermes_home=tmp_path)

        fake_client = _FakeHttpxClient([
            _FakeResponse(200, _fake_token_response("TOKEN_A")),
        ])

        request = MagicMock()
        request.headers = {}
        request.status_code = 200

        # Patch the httpx client constructor used inside async_auth_flow
        with patch(
            "tools.mcp_service_account.ServiceAccountAuth._exchange_service_account",
            new=AsyncMock(return_value=_parse_token("TOKEN_A")),
        ):
            gen = auth.async_auth_flow(request)
            sent_request = await gen.__anext__()
            # Feed a 200 response — no retry needed
            resp = MagicMock()
            resp.status_code = 200
            with pytest.raises(StopAsyncIteration):
                await gen.asend(resp)

        assert sent_request.headers.get("Authorization") == "Bearer TOKEN_A"

    @pytest.mark.asyncio
    async def test_posts_correct_grant_form(self, tmp_path, monkeypatch):
        """Verify the client_credentials form fields via _post_token_request."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("TEST_SA_PASSWORD", "hunter2")

        from tools.mcp_service_account import (
            ServiceAccountAuth,
            _clear_refresh_locks_for_tests,
        )

        _clear_refresh_locks_for_tests()

        cfg = _make_sa_config()
        auth = ServiceAccountAuth("srv", cfg, hermes_home=tmp_path)
        posted_forms: list[dict] = []

        async def _capture_post(http_client, token_url, form, server_name):
            posted_forms.append(dict(form))
            # _post_token_request returns the raw JSON dict from the token endpoint
            return _fake_token_response("TOK")

        with patch("tools.mcp_service_account._post_token_request", _capture_post):
            request = MagicMock()
            request.headers = {}
            gen = auth.async_auth_flow(request)
            await gen.__anext__()
            resp = MagicMock()
            resp.status_code = 200
            try:
                await gen.asend(resp)
            except StopAsyncIteration:
                pass

        assert posted_forms, "No token request was made"
        form = posted_forms[0]
        assert form["grant_type"] == "client_credentials"
        assert form["client_id"] == "toolhive"
        assert form["username"] == "zug"
        assert form["password"] == "hunter2"
        assert "scope" in form


def _parse_token(
    access_token: str, expires_in: int = 3600, refresh_token: str | None = None
):
    """Build a _CachedToken for test use."""
    from tools.mcp_service_account import _CachedToken

    return _CachedToken(access_token, time.time() + expires_in, refresh_token)


# ---------------------------------------------------------------------------
# Token caching
# ---------------------------------------------------------------------------


class TestTokenCaching:
    @pytest.mark.asyncio
    async def test_cached_token_reused_no_second_exchange(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("TEST_SA_PASSWORD", "pw")

        from tools.mcp_service_account import (
            ServiceAccountAuth,
            _clear_refresh_locks_for_tests,
        )

        _clear_refresh_locks_for_tests()

        cfg = _make_sa_config()
        auth = ServiceAccountAuth("srv", cfg, hermes_home=tmp_path)

        exchange_count = 0

        async def _fake_exchange(self_inner, http_client):
            nonlocal exchange_count
            exchange_count += 1
            return _parse_token("CACHED_TOK")

        with patch.object(
            ServiceAccountAuth, "_exchange_service_account", _fake_exchange
        ):
            # First request
            for _ in range(2):
                req = MagicMock()
                req.headers = {}
                gen = auth.async_auth_flow(req)
                await gen.__anext__()
                resp = MagicMock()
                resp.status_code = 200
                try:
                    await gen.asend(resp)
                except StopAsyncIteration:
                    pass

        assert exchange_count == 1, "Token should be reused, not re-fetched"

    @pytest.mark.asyncio
    async def test_expired_token_triggers_renewal(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("TEST_SA_PASSWORD", "pw")

        from tools.mcp_service_account import (
            ServiceAccountAuth,
            _CachedToken,
            _clear_refresh_locks_for_tests,
        )

        _clear_refresh_locks_for_tests()

        cfg = _make_sa_config()
        auth = ServiceAccountAuth("srv", cfg, hermes_home=tmp_path)
        # Seed an already-expired token
        auth._mem_token = _CachedToken("OLD_TOK", time.time() - 10)

        exchange_count = 0

        async def _fake_exchange(self_inner, http_client):
            nonlocal exchange_count
            exchange_count += 1
            return _parse_token("FRESH_TOK")

        with patch.object(
            ServiceAccountAuth, "_exchange_service_account", _fake_exchange
        ):
            req = MagicMock()
            req.headers = {}
            gen = auth.async_auth_flow(req)
            sent = await gen.__anext__()
            resp = MagicMock()
            resp.status_code = 200
            try:
                await gen.asend(resp)
            except StopAsyncIteration:
                pass

        assert sent.headers["Authorization"] == "Bearer FRESH_TOK"
        assert exchange_count == 1

    @pytest.mark.asyncio
    async def test_near_expiry_triggers_proactive_renewal(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("TEST_SA_PASSWORD", "pw")

        from tools.mcp_service_account import (
            ServiceAccountAuth,
            _CachedToken,
            _PROACTIVE_RENEW_BUFFER_SECONDS,
            _clear_refresh_locks_for_tests,
        )

        _clear_refresh_locks_for_tests()

        cfg = _make_sa_config()
        auth = ServiceAccountAuth("srv", cfg, hermes_home=tmp_path)
        # Token expiring within the buffer window
        auth._mem_token = _CachedToken(
            "NEAR_EXPIRY_TOK",
            time.time() + _PROACTIVE_RENEW_BUFFER_SECONDS - 5,
        )

        async def _fake_exchange(self_inner, http_client):
            return _parse_token("RENEWED_TOK")

        with patch.object(
            ServiceAccountAuth, "_exchange_service_account", _fake_exchange
        ):
            req = MagicMock()
            req.headers = {}
            gen = auth.async_auth_flow(req)
            sent = await gen.__anext__()
            resp = MagicMock()
            resp.status_code = 200
            try:
                await gen.asend(resp)
            except StopAsyncIteration:
                pass

        assert sent.headers["Authorization"] == "Bearer RENEWED_TOK"

    @pytest.mark.asyncio
    async def test_disk_cache_written_and_read(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("TEST_SA_PASSWORD", "pw")

        from tools.mcp_service_account import (
            ServiceAccountAuth,
            _get_sa_token_path,
            _clear_refresh_locks_for_tests,
        )

        _clear_refresh_locks_for_tests()

        cfg = _make_sa_config()
        auth = ServiceAccountAuth("srv", cfg, hermes_home=tmp_path)

        async def _fake_exchange(self_inner, http_client):
            return _parse_token("DISK_TOK")

        with patch.object(
            ServiceAccountAuth, "_exchange_service_account", _fake_exchange
        ):
            req = MagicMock()
            req.headers = {}
            gen = auth.async_auth_flow(req)
            await gen.__anext__()
            resp = MagicMock()
            resp.status_code = 200
            try:
                await gen.asend(resp)
            except StopAsyncIteration:
                pass

        cache_path = _sa_cache_path("srv", tmp_path)
        assert cache_path.exists()
        data = json.loads(cache_path.read_text())
        assert data["access_token"] == "DISK_TOK"
        # File must be mode 0600
        mode = cache_path.stat().st_mode & 0o777
        assert mode == 0o600, f"Expected 0600, got {oct(mode)}"

    @pytest.mark.asyncio
    async def test_disk_cache_cold_load(self, tmp_path, monkeypatch):
        """A fresh ServiceAccountAuth instance reads the disk cache."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("TEST_SA_PASSWORD", "pw")

        from tools.mcp_service_account import (
            ServiceAccountAuth,
            _get_sa_token_path,
            _write_token_cache,
            _clear_refresh_locks_for_tests,
        )

        _clear_refresh_locks_for_tests()

        # Write a valid token cache manually. The identity must match the
        # config the provider is built with — a cached token is bound to the
        # token_url/client_id/username/scope that minted it.
        from tools.mcp_service_account import sa_identity_fingerprint

        cache_path = _sa_cache_path("srv", tmp_path)
        _write_token_cache(
            cache_path,
            {
                "access_token": "COLD_TOK",
                "expires_at": time.time() + 3600,
                "identity": sa_identity_fingerprint(_make_sa_config()),
            },
        )

        exchange_count = 0

        async def _fake_exchange(self_inner, http_client):
            nonlocal exchange_count
            exchange_count += 1
            return _parse_token("NEW_TOK")

        cfg = _make_sa_config()
        auth = ServiceAccountAuth("srv", cfg, hermes_home=tmp_path)

        with patch.object(
            ServiceAccountAuth, "_exchange_service_account", _fake_exchange
        ):
            req = MagicMock()
            req.headers = {}
            gen = auth.async_auth_flow(req)
            sent = await gen.__anext__()
            resp = MagicMock()
            resp.status_code = 200
            try:
                await gen.asend(resp)
            except StopAsyncIteration:
                pass

        assert sent.headers["Authorization"] == "Bearer COLD_TOK"
        assert exchange_count == 0


# ---------------------------------------------------------------------------
# Refresh-token path
# ---------------------------------------------------------------------------


class TestRefreshTokenPath:
    @pytest.mark.asyncio
    async def test_refresh_token_used_before_service_account(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("TEST_SA_PASSWORD", "pw")

        from tools.mcp_service_account import (
            ServiceAccountAuth,
            _CachedToken,
            _clear_refresh_locks_for_tests,
        )

        _clear_refresh_locks_for_tests()

        cfg = _make_sa_config()
        auth = ServiceAccountAuth("srv", cfg, hermes_home=tmp_path)
        # Expired token with a refresh_token
        auth._mem_token = _CachedToken("OLD_TOK", time.time() - 10, "REFRESH_ME")

        refresh_called = []
        exchange_called = []

        async def _fake_refresh(self_inner, http_client, refresh_token):
            refresh_called.append(refresh_token)
            return _parse_token("REFRESHED_TOK")

        async def _fake_exchange(self_inner, http_client):
            exchange_called.append(True)
            return _parse_token("SA_TOK")

        with (
            patch.object(ServiceAccountAuth, "_exchange_refresh_token", _fake_refresh),
            patch.object(
                ServiceAccountAuth, "_exchange_service_account", _fake_exchange
            ),
        ):
            req = MagicMock()
            req.headers = {}
            gen = auth.async_auth_flow(req)
            sent = await gen.__anext__()
            resp = MagicMock()
            resp.status_code = 200
            try:
                await gen.asend(resp)
            except StopAsyncIteration:
                pass

        assert sent.headers["Authorization"] == "Bearer REFRESHED_TOK"
        assert refresh_called == ["REFRESH_ME"]
        assert exchange_called == []

    @pytest.mark.asyncio
    async def test_refresh_token_failure_falls_back_to_service_account(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("TEST_SA_PASSWORD", "pw")

        from tools.mcp_service_account import (
            ServiceAccountAuth,
            _CachedToken,
            _clear_refresh_locks_for_tests,
        )

        _clear_refresh_locks_for_tests()

        cfg = _make_sa_config()
        auth = ServiceAccountAuth("srv", cfg, hermes_home=tmp_path)
        auth._mem_token = _CachedToken("OLD_TOK", time.time() - 10, "BAD_REFRESH")

        async def _fake_refresh(self_inner, http_client, refresh_token):
            return None  # simulate failure

        async def _fake_exchange(self_inner, http_client):
            return _parse_token("FALLBACK_TOK")

        with (
            patch.object(ServiceAccountAuth, "_exchange_refresh_token", _fake_refresh),
            patch.object(
                ServiceAccountAuth, "_exchange_service_account", _fake_exchange
            ),
        ):
            req = MagicMock()
            req.headers = {}
            gen = auth.async_auth_flow(req)
            sent = await gen.__anext__()
            resp = MagicMock()
            resp.status_code = 200
            try:
                await gen.asend(resp)
            except StopAsyncIteration:
                pass

        assert sent.headers["Authorization"] == "Bearer FALLBACK_TOK"


# ---------------------------------------------------------------------------
# 401 retry
# ---------------------------------------------------------------------------


class TestFourOhOneRetry:
    @pytest.mark.asyncio
    async def test_401_triggers_retry_with_new_token(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("TEST_SA_PASSWORD", "pw")

        from tools.mcp_service_account import (
            ServiceAccountAuth,
            _CachedToken,
            _clear_refresh_locks_for_tests,
        )

        _clear_refresh_locks_for_tests()

        cfg = _make_sa_config()
        auth = ServiceAccountAuth("srv", cfg, hermes_home=tmp_path)
        # Pre-populate a valid in-memory token so the first request uses it
        # without calling exchange. Exchange is only called on the 401 retry.
        auth._mem_token = _CachedToken("STALE_TOK", time.time() + 3600)

        async def _fake_exchange(self_inner, http_client):
            return _parse_token("FRESH_TOK")

        with patch.object(
            ServiceAccountAuth, "_exchange_service_account", _fake_exchange
        ):
            req = MagicMock()
            req.headers = {}
            gen = auth.async_auth_flow(req)

            # First yield — stale token is in cache, injected
            first_req = await gen.__anext__()
            assert first_req.headers["Authorization"] == "Bearer STALE_TOK"

            # Feed a 401 response
            resp_401 = MagicMock()
            resp_401.status_code = 401
            retry_req = await gen.asend(resp_401)
            assert retry_req.headers["Authorization"] == "Bearer FRESH_TOK"

            # End the flow
            resp_200 = MagicMock()
            resp_200.status_code = 200
            try:
                await gen.asend(resp_200)
            except StopAsyncIteration:
                pass


# ---------------------------------------------------------------------------
# Concurrent refresh deduplication
# ---------------------------------------------------------------------------


class TestConcurrentRefresh:
    @pytest.mark.asyncio
    async def test_concurrent_requests_only_one_exchange(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("TEST_SA_PASSWORD", "pw")

        from tools.mcp_service_account import (
            ServiceAccountAuth,
            _clear_refresh_locks_for_tests,
        )

        _clear_refresh_locks_for_tests()

        cfg = _make_sa_config()
        exchange_count = [0]

        async def _slow_exchange(self_inner, http_client):
            exchange_count[0] += 1
            await asyncio.sleep(0.01)
            return _parse_token("SHARED_TOK")

        async def _run_one():
            auth = ServiceAccountAuth("srv", cfg, hermes_home=tmp_path)
            req = MagicMock()
            req.headers = {}
            gen = auth.async_auth_flow(req)
            await gen.__anext__()
            resp = MagicMock()
            resp.status_code = 200
            try:
                await gen.asend(resp)
            except StopAsyncIteration:
                pass

        with patch.object(
            ServiceAccountAuth, "_exchange_service_account", _slow_exchange
        ):
            await asyncio.gather(*[_run_one() for _ in range(5)])

        # Only one exchange should have fired (others waited on the lock and
        # found the cached token after the lock was released).
        assert exchange_count[0] == 1


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestErrorCases:
    @pytest.mark.asyncio
    async def test_missing_password_env_raises_actionable_error(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("NO_SUCH_ENV", raising=False)

        from tools.mcp_service_account import (
            ServiceAccountAuth,
            _clear_refresh_locks_for_tests,
        )

        _clear_refresh_locks_for_tests()

        cfg = _make_sa_config(password_env="NO_SUCH_ENV")
        auth = ServiceAccountAuth("srv", cfg, hermes_home=tmp_path)

        req = MagicMock()
        req.headers = {}
        gen = auth.async_auth_flow(req)
        with pytest.raises(ValueError) as exc:
            await gen.__anext__()
        msg = str(exc.value)
        assert "NO_SUCH_ENV" in msg
        # No secret should appear — env var is not set so there's nothing to leak,
        # but let's verify the message structure is about the missing var name.
        assert (
            "not set" in msg.lower()
            or "missing" in msg.lower()
            or "empty" in msg.lower()
        )

    @pytest.mark.asyncio
    async def test_malformed_token_response_raises_clear_error(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("TEST_SA_PASSWORD", "pw")

        from tools.mcp_service_account import (
            ServiceAccountAuth,
            _clear_refresh_locks_for_tests,
            _post_token_request,
        )

        _clear_refresh_locks_for_tests()

        cfg = _make_sa_config()
        auth = ServiceAccountAuth("srv", cfg, hermes_home=tmp_path)

        async def _bad_post(http_client, token_url, form, server_name):
            return {"not_an_access_token": True}

        with patch("tools.mcp_service_account._post_token_request", _bad_post):
            req = MagicMock()
            req.headers = {}
            gen = auth.async_auth_flow(req)
            with pytest.raises(ValueError, match="access_token"):
                await gen.__anext__()

    @pytest.mark.asyncio
    async def test_token_endpoint_http_500_raises_clear_error(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("TEST_SA_PASSWORD", "pw")

        from tools.mcp_service_account import (
            ServiceAccountAuth,
            _clear_refresh_locks_for_tests,
        )

        _clear_refresh_locks_for_tests()

        cfg = _make_sa_config()
        auth = ServiceAccountAuth("srv", cfg, hermes_home=tmp_path)

        async def _bad_post(http_client, token_url, form, server_name):
            raise ValueError(
                f"MCP service-account '{server_name}': token endpoint returned HTTP 500"
            )

        with patch("tools.mcp_service_account._post_token_request", _bad_post):
            req = MagicMock()
            req.headers = {}
            gen = auth.async_auth_flow(req)
            with pytest.raises(ValueError, match="HTTP 500"):
                await gen.__anext__()


# ---------------------------------------------------------------------------
# No secrets in logs
# ---------------------------------------------------------------------------


class TestNoSecretsInLogs:
    @pytest.mark.asyncio
    async def test_password_not_in_logs_or_errors(self, tmp_path, monkeypatch, caplog):
        secret_password = "super_secret_hunter2"
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("TEST_SA_PASSWORD", secret_password)

        from tools.mcp_service_account import (
            ServiceAccountAuth,
            _clear_refresh_locks_for_tests,
        )

        _clear_refresh_locks_for_tests()

        cfg = _make_sa_config()
        auth = ServiceAccountAuth("srv", cfg, hermes_home=tmp_path)

        async def _fake_exchange(self_inner, http_client):
            return _parse_token("TOK")

        with caplog.at_level(logging.DEBUG, logger="tools.mcp_service_account"):
            with patch.object(
                ServiceAccountAuth, "_exchange_service_account", _fake_exchange
            ):
                req = MagicMock()
                req.headers = {}
                gen = auth.async_auth_flow(req)
                await gen.__anext__()
                resp = MagicMock()
                resp.status_code = 200
                try:
                    await gen.asend(resp)
                except StopAsyncIteration:
                    pass

        for record in caplog.records:
            assert secret_password not in record.getMessage(), (
                f"Password leaked in log: {record.getMessage()}"
            )

    @pytest.mark.asyncio
    async def test_access_token_not_in_logs(self, tmp_path, monkeypatch, caplog):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("TEST_SA_PASSWORD", "pw")

        from tools.mcp_service_account import (
            ServiceAccountAuth,
            _clear_refresh_locks_for_tests,
        )

        _clear_refresh_locks_for_tests()

        secret_token = "very_secret_access_token_xyz"
        cfg = _make_sa_config()
        auth = ServiceAccountAuth("srv", cfg, hermes_home=tmp_path)

        async def _fake_exchange(self_inner, http_client):
            return _parse_token(secret_token)

        with caplog.at_level(logging.DEBUG, logger="tools.mcp_service_account"):
            with patch.object(
                ServiceAccountAuth, "_exchange_service_account", _fake_exchange
            ):
                req = MagicMock()
                req.headers = {}
                gen = auth.async_auth_flow(req)
                await gen.__anext__()
                resp = MagicMock()
                resp.status_code = 200
                try:
                    await gen.asend(resp)
                except StopAsyncIteration:
                    pass

        for record in caplog.records:
            assert secret_token not in record.getMessage(), (
                f"Access token leaked in log: {record.getMessage()}"
            )


# ---------------------------------------------------------------------------
# Profile / home isolation
# ---------------------------------------------------------------------------


class TestProfileIsolation:
    @pytest.mark.asyncio
    async def test_two_profiles_separate_caches(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TEST_SA_PASSWORD", "pw")

        from tools.mcp_service_account import (
            ServiceAccountAuth,
            _get_sa_token_path,
            _clear_refresh_locks_for_tests,
        )

        _clear_refresh_locks_for_tests()

        home_a = tmp_path / "profile-a"
        home_b = tmp_path / "profile-b"
        cfg = _make_sa_config()

        tokens_issued = []

        async def _fake_exchange(self_inner, http_client):
            tok = f"TOK_{len(tokens_issued)}"
            tokens_issued.append(tok)
            return _parse_token(tok)

        with patch.object(
            ServiceAccountAuth, "_exchange_service_account", _fake_exchange
        ):
            for home in (home_a, home_b):
                auth = ServiceAccountAuth("srv", cfg, hermes_home=home)
                req = MagicMock()
                req.headers = {}
                gen = auth.async_auth_flow(req)
                await gen.__anext__()
                resp = MagicMock()
                resp.status_code = 200
                try:
                    await gen.asend(resp)
                except StopAsyncIteration:
                    pass

        # Each profile has its own cache file
        path_a = _sa_cache_path("srv", home_a)
        path_b = _sa_cache_path("srv", home_b)
        assert path_a.exists()
        assert path_b.exists()
        assert path_a != path_b

        data_a = json.loads(path_a.read_text())
        data_b = json.loads(path_b.read_text())
        assert data_a["access_token"] != data_b["access_token"]


# ---------------------------------------------------------------------------
# Backwards compatibility — header and oauth modes still work
# ---------------------------------------------------------------------------


class TestBackwardsCompatibility:
    def test_header_mode_config_unchanged(self):
        """auth: header config still passes through mcp_config helpers."""
        from hermes_cli.mcp_config import _bearer_auth_headers, _env_key_for_server

        key = _env_key_for_server("myserver")
        headers = _bearer_auth_headers("myserver")
        assert headers == {"Authorization": f"Bearer ${{{key}}}"}

    def test_service_account_mode_does_not_break_oauth_manager(
        self, tmp_path, monkeypatch
    ):
        """build_service_account_auth is independent of MCPOAuthManager."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from tools.mcp_service_account import (
            build_service_account_auth,
            validate_service_account_config,
        )
        from tools.mcp_oauth_manager import get_manager

        cfg = _make_sa_config()
        assert validate_service_account_config("srv", cfg) == []
        # Manager should still work independently
        mgr = get_manager()
        assert mgr is not None

    def test_build_service_account_auth_raises_on_bad_config(self):
        from tools.mcp_service_account import build_service_account_auth

        with pytest.raises(ValueError):
            build_service_account_auth("srv", {})  # missing required fields


# ---------------------------------------------------------------------------
# Token cache file permissions
# ---------------------------------------------------------------------------


class TestTokenCacheFilePermissions:
    @pytest.mark.asyncio
    async def test_cache_file_is_mode_0600(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("TEST_SA_PASSWORD", "pw")

        from tools.mcp_service_account import (
            ServiceAccountAuth,
            _get_sa_token_path,
            _clear_refresh_locks_for_tests,
        )

        _clear_refresh_locks_for_tests()

        cfg = _make_sa_config()
        auth = ServiceAccountAuth("srv", cfg, hermes_home=tmp_path)

        async def _fake_exchange(self_inner, http_client):
            return _parse_token("TOK")

        with patch.object(
            ServiceAccountAuth, "_exchange_service_account", _fake_exchange
        ):
            req = MagicMock()
            req.headers = {}
            gen = auth.async_auth_flow(req)
            await gen.__anext__()
            resp = MagicMock()
            resp.status_code = 200
            try:
                await gen.asend(resp)
            except StopAsyncIteration:
                pass

        cache_path = _sa_cache_path("srv", tmp_path)
        mode = cache_path.stat().st_mode & 0o777
        assert mode == 0o600, f"Expected 0600, got {oct(mode)}"


# ---------------------------------------------------------------------------
# Profile secret-scope isolation (multiplexing bug fix)
# ---------------------------------------------------------------------------


class TestProfileSecretScopeIsolation:
    """Verify _resolve_password/_resolve_client_secret use the profile scope."""

    def teardown_method(self, _method):
        from agent.secret_scope import set_multiplex_active

        set_multiplex_active(False)

    # -- _resolve_password ----------------------------------------------------

    def test_reads_from_active_scope_not_environ(self, monkeypatch):
        """Scope value takes priority over os.environ in non-multiplex mode."""
        from agent.secret_scope import set_secret_scope, reset_secret_scope
        from tools.mcp_service_account import _resolve_password

        monkeypatch.setenv("TEST_SA_PASSWORD", "environ_value")
        token = set_secret_scope({"TEST_SA_PASSWORD": "scope_value"})
        try:
            result = _resolve_password({"password_env": "TEST_SA_PASSWORD"}, "srv")
        finally:
            reset_secret_scope(token)
        assert result == "scope_value"

    def test_non_multiplex_no_scope_falls_through_to_environ(self, monkeypatch):
        """Non-multiplex + no scope: falls back to os.environ (backwards compat)."""
        from tools.mcp_service_account import _resolve_password

        monkeypatch.setenv("TEST_SA_PASSWORD", "from_environ")
        assert (
            _resolve_password({"password_env": "TEST_SA_PASSWORD"}, "srv")
            == "from_environ"
        )

    def test_multiplex_no_scope_raises_unscoped_error(self, monkeypatch):
        """Multiplex active + no scope → UnscopedSecretError (fail closed)."""
        from agent.secret_scope import set_multiplex_active, UnscopedSecretError
        from tools.mcp_service_account import _resolve_password

        monkeypatch.setenv("TEST_SA_PASSWORD", "should_not_be_read")
        set_multiplex_active(True)
        with pytest.raises(UnscopedSecretError):
            _resolve_password({"password_env": "TEST_SA_PASSWORD"}, "srv")

    def test_multiplex_scope_miss_raises_value_error_not_environ(self, monkeypatch):
        """Multiplex + scope installed but key absent → ValueError, not os.environ leak."""
        from agent.secret_scope import (
            set_multiplex_active,
            set_secret_scope,
            reset_secret_scope,
        )
        from tools.mcp_service_account import _resolve_password

        monkeypatch.setenv("TEST_SA_PASSWORD", "should_not_leak")
        set_multiplex_active(True)
        token = set_secret_scope({})  # scope present but empty
        try:
            with pytest.raises(ValueError, match="TEST_SA_PASSWORD"):
                _resolve_password({"password_env": "TEST_SA_PASSWORD"}, "srv")
        finally:
            reset_secret_scope(token)

    def test_profile_ab_password_isolation_via_scopes(self):
        """Profile A and B each obtain their own password from their respective scopes."""
        from agent.secret_scope import set_secret_scope, reset_secret_scope
        from tools.mcp_service_account import _resolve_password

        cfg = {"password_env": "AUTHENTIK_APP_PASSWORD"}
        results = []
        for pw in ("pw_for_profile_a", "pw_for_profile_b"):
            token = set_secret_scope({"AUTHENTIK_APP_PASSWORD": pw})
            try:
                results.append(_resolve_password(cfg, "toolhive"))
            finally:
                reset_secret_scope(token)
        assert results == ["pw_for_profile_a", "pw_for_profile_b"]

    # -- _resolve_client_secret -----------------------------------------------

    def test_client_secret_reads_from_scope(self, monkeypatch):
        """client_secret_env is also read from the profile scope, not os.environ."""
        from agent.secret_scope import set_secret_scope, reset_secret_scope
        from tools.mcp_service_account import _resolve_client_secret

        monkeypatch.setenv("MY_CLIENT_SECRET", "environ_secret")
        token = set_secret_scope({"MY_CLIENT_SECRET": "scope_secret"})
        try:
            result = _resolve_client_secret({"client_secret_env": "MY_CLIENT_SECRET"})
        finally:
            reset_secret_scope(token)
        assert result == "scope_secret"

    def test_client_secret_scope_miss_in_multiplex_returns_none(self, monkeypatch):
        """Multiplex + scope missing client_secret_env → None, not environ leak."""
        from agent.secret_scope import (
            set_multiplex_active,
            set_secret_scope,
            reset_secret_scope,
        )
        from tools.mcp_service_account import _resolve_client_secret

        monkeypatch.setenv("MY_CLIENT_SECRET", "should_not_leak")
        set_multiplex_active(True)
        token = set_secret_scope({})
        try:
            result = _resolve_client_secret({"client_secret_env": "MY_CLIENT_SECRET"})
            assert result is None
        finally:
            reset_secret_scope(token)

    # -- full auth-flow with scoped password ----------------------------------

    @pytest.mark.asyncio
    async def test_auth_flow_uses_scoped_password_not_environ(
        self, tmp_path, monkeypatch
    ):
        """ServiceAccountAuth acquires token using the profile scope password."""
        from agent.secret_scope import set_secret_scope, reset_secret_scope
        from tools.mcp_service_account import (
            ServiceAccountAuth,
            _clear_refresh_locks_for_tests,
        )

        _clear_refresh_locks_for_tests()

        # Password absent from environ — must come entirely from scope.
        monkeypatch.delenv("AUTHENTIK_ZUG_APP_PASSWORD", raising=False)

        captured_forms: list[dict] = []

        async def _capture_post(http_client, token_url, form, server_name):
            captured_forms.append(dict(form))
            return _fake_token_response("SCOPED_TOKEN")

        cfg = _make_sa_config(password_env="AUTHENTIK_ZUG_APP_PASSWORD")
        auth = ServiceAccountAuth("toolhive", cfg, hermes_home=tmp_path)

        scope_token = set_secret_scope({"AUTHENTIK_ZUG_APP_PASSWORD": "zug_scope_pw"})
        try:
            with patch("tools.mcp_service_account._post_token_request", _capture_post):
                req = MagicMock()
                req.headers = {}
                gen = auth.async_auth_flow(req)
                sent = await gen.__anext__()
                resp = MagicMock()
                resp.status_code = 200
                try:
                    await gen.asend(resp)
                except StopAsyncIteration:
                    pass
        finally:
            reset_secret_scope(scope_token)

        assert sent.headers["Authorization"] == "Bearer SCOPED_TOKEN"
        assert captured_forms[0]["password"] == "zug_scope_pw"
        # Sanity: the scoped password must not appear in the Authorization header.
        assert "zug_scope_pw" not in sent.headers["Authorization"]

    @pytest.mark.asyncio
    async def test_cache_path_isolation_separate_hermes_homes(
        self, tmp_path, monkeypatch
    ):
        """Two profiles with the same server name use separate cache files."""
        from tools.mcp_service_account import (
            ServiceAccountAuth,
            _get_sa_token_path,
            _clear_refresh_locks_for_tests,
        )

        _clear_refresh_locks_for_tests()

        home_a = tmp_path / "profile-a"
        home_b = tmp_path / "profile-b"
        cfg = _make_sa_config()

        async def _fake_exchange_a(self_inner, http_client):
            return _parse_token("TOKEN_PROFILE_A")

        async def _fake_exchange_b(self_inner, http_client):
            return _parse_token("TOKEN_PROFILE_B")

        for home, fake_exchange in (
            (home_a, _fake_exchange_a),
            (home_b, _fake_exchange_b),
        ):
            auth = ServiceAccountAuth("srv", cfg, hermes_home=home)
            monkeypatch.setenv("TEST_SA_PASSWORD", "pw")
            with patch.object(
                ServiceAccountAuth, "_exchange_service_account", fake_exchange
            ):
                req = MagicMock()
                req.headers = {}
                gen = auth.async_auth_flow(req)
                await gen.__anext__()
                resp = MagicMock()
                resp.status_code = 200
                try:
                    await gen.asend(resp)
                except StopAsyncIteration:
                    pass

        path_a = _sa_cache_path("srv", home_a)
        path_b = _sa_cache_path("srv", home_b)
        assert path_a.exists() and path_b.exists()
        assert path_a != path_b
        import json as _json

        assert _json.loads(path_a.read_text())["access_token"] == "TOKEN_PROFILE_A"
        assert _json.loads(path_b.read_text())["access_token"] == "TOKEN_PROFILE_B"
        # Modes must be 0600
        assert (path_a.stat().st_mode & 0o777) == 0o600
        assert (path_b.stat().st_mode & 0o777) == 0o600
