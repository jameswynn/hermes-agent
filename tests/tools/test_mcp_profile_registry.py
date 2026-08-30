"""Profile-scoped MCP registry: two profiles, one process, one server name.

Security boundary under test. The multiplexed gateway runs many Hermes
profiles in ONE process. Every piece of MCP runtime state used to be keyed by
the *logical server name* alone, which is not unique across profiles: two
profiles both configuring a server called ``toolhive`` collapsed onto one
entry, so whichever profile connected first won the slot and the others
silently reused its config. The reported symptom was Carol's turn resolving
``AUTHENTIK_ZUG_APP_PASSWORD`` — Zug's env var, from Zug's server config — and
other personas 401ing on a token minted for someone else.

The isolation key is the canonical profile home (``hermes_home_key()`` over
the resolved ``HERMES_HOME``). These tests assert that, for the same logical
server name across two profiles:

- config, sessions, tool ownership and failure state are separate;
- a call made under profile B can never reach profile A's task;
- credentials, token cache paths and refresh locks are separate;
- the identity survives the hop onto the shared MCP event loop and the
  long-lived reconnect task that inherits its context;
- single-profile behaviour is unchanged.

Adversarial notes:
- The public tool NAME is deliberately shared (``mcp__toolhive__ping``): the
  model tool registry is process-global and renaming would break prompt-cache
  stability and the public API. The boundary is therefore enforced at
  dispatch/visibility, not by namespacing — so the "B cannot reach A" tests
  below are the load-bearing ones.
- Failure state must be per-profile in BOTH directions: a broken profile must
  not park a healthy one, and a healthy one must not clear a broken one's
  backoff.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools import mcp_profile, mcp_tool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_registries():
    """Every test starts from an empty registry map and leaves none behind."""
    mcp_profile.reset_all_registries()
    yield
    mcp_profile.reset_all_registries()


def _profile_homes(tmp_path: Path) -> tuple[Path, Path]:
    home_a = tmp_path / "profile-zug"
    home_b = tmp_path / "profile-carol"
    home_a.mkdir(parents=True, exist_ok=True)
    home_b.mkdir(parents=True, exist_ok=True)
    return home_a, home_b


def _write_profile(
    home: Path,
    *,
    url: str,
    password_env: str,
    password: str,
    server_name: str = "toolhive",
    username: str = "svc",
) -> None:
    """Write a realistic profile: config.yaml + .env, same server name."""
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        f"""
mcp_servers:
  {server_name}:
    url: {url}
    auth: service_account
    service_account:
      grant_type: authentik_app_password
      token_url: https://idp.example/token/
      client_id: {server_name}
      username: {username}
      password_env: {password_env}
      scope: openid profile
""".lstrip(),
        encoding="utf-8",
    )
    (home / ".env").write_text(f"{password_env}={password}\n", encoding="utf-8")


class _FakeSession:
    """Minimal MCP ClientSession stand-in that records which task was used."""

    def __init__(self, tag: str):
        self.tag = tag
        self.calls: list[str] = []

    async def call_tool(self, name, arguments=None):
        self.calls.append(name)
        return SimpleNamespace(
            content=[SimpleNamespace(text=self.tag, type="text")],
            isError=False,
            structuredContent=None,
        )


def _fake_server(tag: str, name: str = "toolhive"):
    """A stand-in for a connected MCPServerTask."""
    server = SimpleNamespace(
        name=name,
        session=_FakeSession(tag),
        _rpc_lock=None,
        _registered_tool_names=[f"mcp__{name}__ping"],
        _tools=[],
        _inflight_tasks=set(),
        _reconnecting=False,
        _pending_call_context=None,
    )
    server.mark_tool_call = lambda: None
    server._is_recycled_stdio = lambda: False
    return server


def _run_loop(coro_or_factory, timeout=30):
    """Stand-in for ``_run_on_mcp_loop`` that runs inline on a fresh loop."""
    coro = coro_or_factory() if callable(coro_or_factory) else coro_or_factory
    loop = asyncio.new_event_loop()
    try:
        async def _install_locks():
            for reg in mcp_profile.all_registries():
                for srv in list(reg.servers.values()):
                    if getattr(srv, "_rpc_lock", None) is None:
                        srv._rpc_lock = asyncio.Lock()
            return await coro
        return loop.run_until_complete(_install_locks())
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Registry isolation
# ---------------------------------------------------------------------------


class TestRegistryIsolation:
    def test_same_server_name_gets_independent_tasks(self, tmp_path):
        """Two profiles, one server name → two distinct MCPServerTask slots."""
        home_a, home_b = _profile_homes(tmp_path)
        task_a, task_b = _fake_server("A"), _fake_server("B")

        with mcp_profile.profile_scope(home_a):
            mcp_tool._servers["toolhive"] = task_a
        with mcp_profile.profile_scope(home_b):
            # B's slot was empty despite A having connected first — this is
            # the exact collapse that made Carol reuse Zug's server config.
            assert "toolhive" not in mcp_tool._servers
            mcp_tool._servers["toolhive"] = task_b

        with mcp_profile.profile_scope(home_a):
            assert mcp_tool._servers["toolhive"] is task_a
        with mcp_profile.profile_scope(home_b):
            assert mcp_tool._servers["toolhive"] is task_b

    def test_profile_key_is_the_canonical_home(self, tmp_path):
        """Identity is the resolved home, not the server name or a nickname."""
        from hermes_constants import hermes_home_key

        home_a, home_b = _profile_homes(tmp_path)
        with mcp_profile.profile_scope(home_a):
            assert mcp_profile.current_profile_key() == hermes_home_key(home_a)
        with mcp_profile.profile_scope(home_b):
            assert mcp_profile.current_profile_key() == hermes_home_key(home_b)

        # Same home reached by a non-canonical path resolves to one registry.
        alias = Path(str(home_a) + "/./")
        with mcp_profile.profile_scope(home_a):
            mcp_tool._servers["toolhive"] = "task"
        with mcp_profile.profile_scope(alias):
            assert mcp_tool._servers.get("toolhive") == "task"

    def test_failure_state_does_not_park_the_other_profile(self, tmp_path):
        """A's connect failure/backoff/breaker must not gate B."""
        home_a, home_b = _profile_homes(tmp_path)

        with mcp_profile.profile_scope(home_a):
            mcp_tool._record_connect_failure("toolhive")
            mcp_tool._server_connect_errors["toolhive"] = "401 Unauthorized"
            mcp_tool._bump_server_error("toolhive")
            mcp_tool._bump_server_error("toolhive")
            mcp_tool._bump_server_error("toolhive")
            assert mcp_tool._connect_cooldown_active("toolhive")
            assert mcp_tool._server_error_counts["toolhive"] == 3

        with mcp_profile.profile_scope(home_b):
            assert not mcp_tool._connect_cooldown_active("toolhive")
            assert "toolhive" not in mcp_tool._server_connect_errors
            assert mcp_tool._server_error_counts.get("toolhive", 0) == 0
            # B succeeding must not clear A's backoff either.
            mcp_tool._clear_connect_failure("toolhive")
            mcp_tool._reset_server_error("toolhive")

        with mcp_profile.profile_scope(home_a):
            assert mcp_tool._connect_cooldown_active("toolhive")
            assert mcp_tool._server_error_counts["toolhive"] == 3

    def test_lazy_metadata_and_fingerprints_are_per_profile(self, tmp_path):
        """Lazy schema-cache registration state is not shared."""
        home_a, home_b = _profile_homes(tmp_path)

        with mcp_profile.profile_scope(home_a):
            mcp_tool._lazy_server_configs["toolhive"] = {"url": "https://a.example"}
            mcp_tool._lazy_server_fingerprints["toolhive"] = "fp-a"
            mcp_tool._lazy_server_tool_names["toolhive"] = ["mcp__toolhive__ping"]

        with mcp_profile.profile_scope(home_b):
            assert mcp_tool._lazy_server_configs == {}
            assert mcp_tool._lazy_server_fingerprints.get("toolhive") is None
            mcp_tool._lazy_server_fingerprints["toolhive"] = "fp-b"

        with mcp_profile.profile_scope(home_a):
            assert mcp_tool._lazy_server_fingerprints["toolhive"] == "fp-a"
            assert mcp_tool._lazy_server_configs["toolhive"]["url"] == "https://a.example"

    def test_trust_and_tool_provenance_are_per_profile(self, tmp_path):
        """Operator trust config and tool ownership don't cross profiles."""
        home_a, home_b = _profile_homes(tmp_path)

        with mcp_profile.profile_scope(home_a):
            mcp_tool._record_tool_trust_metadata(
                "toolhive", {"trust": "untrusted"}, []
            )
            mcp_tool._track_mcp_tool_server("mcp__toolhive__ping", "toolhive")
            mcp_tool._parallel_safe_servers.add("toolhive")

        with mcp_profile.profile_scope(home_b):
            # B never marked it untrusted; A's operator decision must not
            # leak in either direction.
            assert mcp_tool._server_trust_levels.get("toolhive") is None
            assert mcp_tool._mcp_tool_server_names == {}
            assert "toolhive" not in mcp_tool._parallel_safe_servers
            assert mcp_tool.get_registered_mcp_server_names() == set()
            assert not mcp_tool.has_registered_mcp_tools()

        with mcp_profile.profile_scope(home_a):
            assert mcp_tool._server_trust_levels["toolhive"] == "untrusted"
            assert mcp_tool.get_registered_mcp_server_names() == {"toolhive"}
            assert mcp_tool.is_mcp_tool_parallel_safe("mcp__toolhive__ping")

    def test_discovery_lock_path_is_per_profile(self, tmp_path):
        """Each profile locks its own home, not whichever resolved first."""
        home_a, home_b = _profile_homes(tmp_path)

        with mcp_profile.profile_scope(home_a):
            cookie_a = mcp_tool._try_acquire_mcp_discovery_lock()
            path_a = mcp_profile.current_registry().discovery_lock_path
        try:
            with mcp_profile.profile_scope(home_b):
                # B must NOT be blocked by A holding its own profile's lock.
                cookie_b = mcp_tool._try_acquire_mcp_discovery_lock()
                path_b = mcp_profile.current_registry().discovery_lock_path
                assert cookie_b is not None
                if cookie_b is not mcp_tool._LOCK_UNAVAILABLE:
                    cookie_b.release()
        finally:
            if cookie_a is not None and cookie_a is not mcp_tool._LOCK_UNAVAILABLE:
                cookie_a.release()

        assert path_a != path_b
        assert Path(path_a).parent == home_a
        assert Path(path_b).parent == home_b


# ---------------------------------------------------------------------------
# Dispatch boundary — the load-bearing security assertion
# ---------------------------------------------------------------------------


class TestDispatchBoundary:
    def test_check_fn_hides_the_other_profiles_server(self, tmp_path):
        """Tool visibility follows the caller's profile, not the registrant's."""
        home_a, home_b = _profile_homes(tmp_path)
        with mcp_profile.profile_scope(home_a):
            mcp_tool._servers["toolhive"] = _fake_server("A")

        check = mcp_tool._make_check_fn("toolhive")
        with mcp_profile.profile_scope(home_a):
            assert check() is True
        with mcp_profile.profile_scope(home_b):
            assert check() is False

    def test_call_from_b_cannot_reach_a_task(self, tmp_path):
        """The public tool name is shared; the connection must not be."""
        home_a, home_b = _profile_homes(tmp_path)
        task_a = _fake_server("A")
        with mcp_profile.profile_scope(home_a):
            mcp_tool._servers["toolhive"] = task_a

        handler = mcp_tool._make_tool_handler("toolhive", "ping", 30.0)

        with patch("tools.mcp_tool._run_on_mcp_loop", side_effect=_run_loop):
            with mcp_profile.profile_scope(home_b):
                result = handler({})
            # B gets a clean "not connected", NOT A's session.
            assert "not connected" in result
            assert task_a.session.calls == []

            with mcp_profile.profile_scope(home_a):
                ok = handler({})
        assert task_a.session.calls == ["ping"]
        assert "A" in ok

    def test_each_profile_dispatches_to_its_own_task(self, tmp_path):
        """Same tool name, two profiles, two distinct sessions."""
        home_a, home_b = _profile_homes(tmp_path)
        task_a, task_b = _fake_server("A"), _fake_server("B")
        with mcp_profile.profile_scope(home_a):
            mcp_tool._servers["toolhive"] = task_a
        with mcp_profile.profile_scope(home_b):
            mcp_tool._servers["toolhive"] = task_b

        handler = mcp_tool._make_tool_handler("toolhive", "ping", 30.0)
        with patch("tools.mcp_tool._run_on_mcp_loop", side_effect=_run_loop):
            with mcp_profile.profile_scope(home_a):
                out_a = handler({})
            with mcp_profile.profile_scope(home_b):
                out_b = handler({})

        assert task_a.session.calls == ["ping"]
        assert task_b.session.calls == ["ping"]
        assert "A" in out_a and "B" in out_b

    def test_breaker_open_in_a_does_not_short_circuit_b(self, tmp_path):
        """A tripped breaker is profile-local; B's calls still go through."""
        home_a, home_b = _profile_homes(tmp_path)
        task_b = _fake_server("B")
        with mcp_profile.profile_scope(home_a):
            for _ in range(mcp_tool._CIRCUIT_BREAKER_THRESHOLD):
                mcp_tool._bump_server_error("toolhive")
        with mcp_profile.profile_scope(home_b):
            mcp_tool._servers["toolhive"] = task_b

        handler = mcp_tool._make_tool_handler("toolhive", "ping", 30.0)
        with patch("tools.mcp_tool._run_on_mcp_loop", side_effect=_run_loop):
            with mcp_profile.profile_scope(home_a):
                blocked = handler({})
            with mcp_profile.profile_scope(home_b):
                allowed = handler({})

        assert "unreachable" in blocked
        assert task_b.session.calls == ["ping"]
        assert "B" in allowed


# ---------------------------------------------------------------------------
# Config loading — the original failure
# ---------------------------------------------------------------------------


class TestConfigLoading:
    def test_each_profile_loads_its_own_config(self, tmp_path, monkeypatch):
        """`_load_mcp_config` reads the ACTIVE profile's config.yaml."""
        from hermes_constants import (
            reset_hermes_home_override,
            set_hermes_home_override,
        )

        home_a, home_b = _profile_homes(tmp_path)
        _write_profile(
            home_a,
            url="https://toolhive.zug.example/mcp",
            password_env="AUTHENTIK_ZUG_APP_PASSWORD",
            password="zug-secret",
            username="zug",
        )
        _write_profile(
            home_b,
            url="https://toolhive.carol.example/mcp",
            password_env="AUTHENTIK_CAROL_APP_PASSWORD",
            password="carol-secret",
            username="carol",
        )
        monkeypatch.setenv("HERMES_HOME", str(home_a))

        def _load_for(home):
            token = set_hermes_home_override(str(home))
            try:
                return mcp_tool._load_mcp_config()
            finally:
                reset_hermes_home_override(token)

        cfg_a = _load_for(home_a)
        cfg_b = _load_for(home_b)

        assert cfg_a["toolhive"]["url"] == "https://toolhive.zug.example/mcp"
        assert cfg_b["toolhive"]["url"] == "https://toolhive.carol.example/mcp"
        # The reported bug in one assertion: Carol must never be handed Zug's
        # password_env, even though Zug's profile loaded first.
        assert (
            cfg_b["toolhive"]["service_account"]["password_env"]
            == "AUTHENTIK_CAROL_APP_PASSWORD"
        )
        assert (
            cfg_a["toolhive"]["service_account"]["password_env"]
            == "AUTHENTIK_ZUG_APP_PASSWORD"
        )

    def test_profile_cannot_borrow_the_other_profiles_password(
        self, tmp_path, monkeypatch
    ):
        """Under multiplexing, A's scope cannot resolve B's password_env."""
        from agent.secret_scope import (
            build_profile_secret_scope,
            reset_secret_scope,
            set_multiplex_active,
            set_secret_scope,
        )
        from tools.mcp_service_account import _resolve_password

        home_a, home_b = _profile_homes(tmp_path)
        _write_profile(
            home_a,
            url="https://a.example/mcp",
            password_env="AUTHENTIK_ZUG_APP_PASSWORD",
            password="zug-secret",
        )
        _write_profile(
            home_b,
            url="https://b.example/mcp",
            password_env="AUTHENTIK_CAROL_APP_PASSWORD",
            password="carol-secret",
        )
        # Simulate the leak precondition: Zug's env is in os.environ because
        # Zug launched the process.
        monkeypatch.setenv("AUTHENTIK_ZUG_APP_PASSWORD", "zug-secret")
        set_multiplex_active(True)
        try:
            token = set_secret_scope(build_profile_secret_scope(home_b))
            try:
                assert (
                    _resolve_password(
                        {"password_env": "AUTHENTIK_CAROL_APP_PASSWORD"}, "toolhive"
                    )
                    == "carol-secret"
                )
                # Carol's scope must not reach Zug's value even though it is
                # sitting in os.environ.
                with pytest.raises(ValueError) as excinfo:
                    _resolve_password(
                        {"password_env": "AUTHENTIK_ZUG_APP_PASSWORD"}, "toolhive"
                    )
                assert "zug-secret" not in str(excinfo.value)
            finally:
                reset_secret_scope(token)
        finally:
            set_multiplex_active(False)


# ---------------------------------------------------------------------------
# Credentials, caches and the MCPServerTask identity binding
# ---------------------------------------------------------------------------


class TestCredentialIsolation:
    def test_token_cache_paths_are_per_profile(self, tmp_path):
        """Same server name → different 0600 cache files under each home."""
        from tools.mcp_service_account import _get_sa_token_path

        home_a, home_b = _profile_homes(tmp_path)
        path_a = _get_sa_token_path("toolhive", home_a)
        path_b = _get_sa_token_path("toolhive", home_b)

        assert path_a != path_b
        assert str(path_a).startswith(str(home_a))
        assert str(path_b).startswith(str(home_b))

    def test_server_task_binds_and_keeps_its_owning_profile(self, tmp_path):
        """Reconnects run later, on the loop — identity must be bound at birth."""
        home_a, home_b = _profile_homes(tmp_path)
        from hermes_constants import hermes_home_key

        with mcp_profile.profile_scope(home_a):
            task = mcp_tool.MCPServerTask("toolhive")

        assert task._profile_key == hermes_home_key(home_a)
        # Resolving the owner from inside another profile's scope (what a
        # reconnect on the shared loop would do) still lands on A.
        with mcp_profile.profile_scope(home_b):
            assert task._registry().key == hermes_home_key(home_a)
            task._registry().servers["toolhive"] = task
        with mcp_profile.profile_scope(home_a):
            assert mcp_tool._servers["toolhive"] is task
        with mcp_profile.profile_scope(home_b):
            assert "toolhive" not in mcp_tool._servers

    def test_service_account_auth_pins_the_passed_home(self, tmp_path):
        """`build_service_account_auth` uses the explicit profile home."""
        from tools.mcp_service_account import build_service_account_auth

        home_a, home_b = _profile_homes(tmp_path)
        cfg = {
            "grant_type": "authentik_app_password",
            "token_url": "https://idp.example/token/",
            "client_id": "toolhive",
            "username": "svc",
            "password_env": "AUTHENTIK_ZUG_APP_PASSWORD",
        }
        auth_a = build_service_account_auth("toolhive", cfg, hermes_home=home_a)
        auth_b = build_service_account_auth("toolhive", cfg, hermes_home=home_b)

        assert auth_a._cache_path != auth_b._cache_path
        assert str(auth_a._cache_path).startswith(str(home_a))
        assert str(auth_b._cache_path).startswith(str(home_b))
        # Same server name, different profile → different refresh lock.
        assert auth_a._refresh_lock is not auth_b._refresh_lock

    def test_build_service_account_auth_logs_no_secret(self, tmp_path, caplog):
        """Diagnostics name the profile and server, never the credential."""
        from tools.mcp_service_account import build_service_account_auth

        home_a, _ = _profile_homes(tmp_path)
        cfg = {
            "grant_type": "authentik_app_password",
            "token_url": "https://idp.example/token/",
            "client_id": "toolhive",
            "username": "svc",
            "password_env": "AUTHENTIK_ZUG_APP_PASSWORD",
        }
        with caplog.at_level(logging.DEBUG, logger="tools.mcp_service_account"):
            build_service_account_auth("toolhive", cfg, hermes_home=home_a)

        text = caplog.text
        assert "toolhive" in text
        assert str(home_a) in text
        # The env-var NAME is diagnostic; no value may appear.
        assert "AUTHENTIK_ZUG_APP_PASSWORD" in text
        assert "zug-secret" not in text

    def test_cache_is_namespaced_away_from_browser_oauth(self, tmp_path):
        """A server named ``foo-sa`` must not alias ``foo``'s SA cache.

        The old layout wrote ``mcp-tokens/<server>-sa.json`` next to browser
        OAuth's ``mcp-tokens/<server>.json``, so server ``foo-sa``'s OAuth
        cache and server ``foo``'s service-account cache were the same file.
        """
        from tools.mcp_oauth import _get_token_dir
        from tools.mcp_service_account import _get_sa_token_path

        home, _ = _profile_homes(tmp_path)
        sa_path = _get_sa_token_path("foo", home, "ident")
        oauth_dir = _get_token_dir(home)

        assert sa_path.parent == oauth_dir / "service-account"
        assert sa_path.parent != oauth_dir
        # And the collision that motivated the change is gone.
        assert sa_path != oauth_dir / "foo-sa.json"

    def test_cache_filenames_survive_sanitizer_collisions(self, tmp_path):
        """Lossy name sanitizing must not merge two servers' caches."""
        from tools.mcp_service_account import _get_sa_token_path

        home, _ = _profile_homes(tmp_path)
        paths = {
            _get_sa_token_path(name, home, "ident")
            for name in ("tool/hive", "tool.hive", "tool_hive", "tool-hive")
        }
        assert len(paths) == 4

    def test_cached_token_is_bound_to_its_identity(self, tmp_path):
        """Changing username/scope/client_id invalidates the cached token."""
        from tools.mcp_service_account import (
            ServiceAccountAuth,
            _CachedToken,
            sa_identity_fingerprint,
        )

        home, _ = _profile_homes(tmp_path)
        base = {
            "grant_type": "authentik_app_password",
            "token_url": "https://idp.example/token/",
            "client_id": "toolhive",
            "username": "zug",
            "password_env": "AUTHENTIK_ZUG_APP_PASSWORD",
            "scope": "openid",
        }
        auth = ServiceAccountAuth("toolhive", base, hermes_home=home)
        auth._save_to_disk(_CachedToken("TOK", __import__("time").time() + 3600))
        assert auth._load_from_disk() is not None

        for field, value in (
            ("username", "carol"),
            ("scope", "openid admin"),
            ("client_id", "other"),
            ("token_url", "https://idp2.example/token/"),
            ("password_env", "AUTHENTIK_CAROL_APP_PASSWORD"),
        ):
            changed = dict(base, **{field: value})
            assert sa_identity_fingerprint(changed) != sa_identity_fingerprint(base)
            other = ServiceAccountAuth("toolhive", changed, hermes_home=home)
            # Different identity → different file, and nothing to load.
            assert other._cache_path != auth._cache_path
            assert other._load_from_disk() is None

    def test_identityless_legacy_cache_is_discarded(self, tmp_path):
        """A cache written before identity binding must not be trusted."""
        import time as _time

        from tools.mcp_service_account import (
            ServiceAccountAuth,
            _write_token_cache,
        )

        home, _ = _profile_homes(tmp_path)
        cfg = {
            "grant_type": "authentik_app_password",
            "token_url": "https://idp.example/token/",
            "client_id": "toolhive",
            "username": "zug",
            "password_env": "AUTHENTIK_ZUG_APP_PASSWORD",
        }
        auth = ServiceAccountAuth("toolhive", cfg, hermes_home=home)
        _write_token_cache(
            auth._cache_path,
            {"access_token": "LEGACY", "expires_at": _time.time() + 3600},
        )
        assert auth._load_from_disk() is None
        # The unusable file is cleaned up rather than left to be re-read.
        assert not auth._cache_path.exists()

    @pytest.mark.asyncio
    async def test_each_profile_posts_its_own_credential(self, tmp_path):
        """End-to-end auth flow: two profiles, one server name, two passwords."""
        from tools.mcp_service_account import (
            ServiceAccountAuth,
            _clear_refresh_locks_for_tests,
        )
        from agent.secret_scope import (
            reset_secret_scope,
            set_multiplex_active,
            set_secret_scope,
        )

        _clear_refresh_locks_for_tests()
        home_a, home_b = _profile_homes(tmp_path)
        posted: list[dict] = []

        async def _capture(http_client, token_url, form, server_name):
            posted.append(dict(form))
            return {
                "access_token": f"TOK-{form['password']}",
                "token_type": "Bearer",
                "expires_in": 3600,
            }

        cache_paths: dict[str, Path] = {}

        async def _exchange(home, password_env, password):
            cfg = {
                "grant_type": "authentik_app_password",
                "token_url": "https://idp.example/token/",
                "client_id": "toolhive",
                "username": "svc",
                "password_env": password_env,
                "scope": "openid",
            }
            auth = ServiceAccountAuth("toolhive", cfg, hermes_home=home)
            cache_paths[str(home)] = auth._cache_path
            token = set_secret_scope({password_env: password})
            try:
                request = MagicMock()
                request.headers = {}
                gen = auth.async_auth_flow(request)
                sent = await gen.__anext__()
                resp = MagicMock()
                resp.status_code = 200
                try:
                    await gen.asend(resp)
                except StopAsyncIteration:
                    pass
                return sent
            finally:
                reset_secret_scope(token)

        set_multiplex_active(True)
        try:
            with patch("tools.mcp_service_account._post_token_request", _capture):
                sent_a = await _exchange(
                    home_a, "AUTHENTIK_ZUG_APP_PASSWORD", "zug-secret"
                )
                sent_b = await _exchange(
                    home_b, "AUTHENTIK_CAROL_APP_PASSWORD", "carol-secret"
                )
        finally:
            set_multiplex_active(False)
            _clear_refresh_locks_for_tests()

        assert [f["password"] for f in posted] == ["zug-secret", "carol-secret"]
        assert sent_a.headers["Authorization"] == "Bearer TOK-zug-secret"
        assert sent_b.headers["Authorization"] == "Bearer TOK-carol-secret"
        # Each profile's token landed under its OWN home, mode 0600.
        cache_a = cache_paths[str(home_a)]
        cache_b = cache_paths[str(home_b)]
        assert str(cache_a).startswith(str(home_a))
        assert str(cache_b).startswith(str(home_b))
        assert cache_a.exists() and cache_b.exists()
        assert (cache_a.stat().st_mode & 0o777) == 0o600
        assert (cache_b.stat().st_mode & 0o777) == 0o600
        assert "zug-secret" not in cache_b.read_text(encoding="utf-8")
        assert "carol-secret" not in cache_a.read_text(encoding="utf-8")


class TestTokenLifecycleEdgeCases:
    """Review-comment cases: short-lived tokens, refresh retention, 401 races."""

    def test_short_lived_token_is_still_usable(self):
        """expires_in below the flat renew buffer must not be born invalid.

        With a flat 60s buffer a 30s token satisfies ``is_valid()`` never, so
        every request re-exchanges and nothing is ever cached.
        """
        from tools.mcp_service_account import _parse_token_response

        token = _parse_token_response(
            {"access_token": "SHORT", "expires_in": 30}, "srv"
        )
        assert token.lifetime == 30.0
        assert token.renew_buffer() == 15.0
        assert token.is_valid() is True

    def test_short_lived_token_still_renews_in_its_second_half(self):
        from tools.mcp_service_account import _parse_token_response

        token = _parse_token_response(
            {"access_token": "SHORT", "expires_in": 30},
            "srv",
            now=__import__("time").time() - 20,
        )
        assert token.is_valid() is False

    def test_unknown_lifetime_keeps_the_flat_buffer(self):
        """A directly-built token (lifetime unknown) keeps historical behaviour."""
        import time as _time

        from tools.mcp_service_account import (
            _CachedToken,
            _PROACTIVE_RENEW_BUFFER_SECONDS,
        )

        near = _CachedToken("T", _time.time() + 30)
        assert near.lifetime is None
        assert near.renew_buffer() == float(_PROACTIVE_RENEW_BUFFER_SECONDS)
        # A long-lived token seen 30s from expiry must still renew.
        assert near.is_valid() is False

    @pytest.mark.asyncio
    async def test_refresh_response_without_refresh_token_keeps_the_old_one(
        self, tmp_path
    ):
        """RFC 6749 §6: omitted refresh_token means 'keep the one you have'."""
        from tools.mcp_service_account import ServiceAccountAuth

        home, _ = _profile_homes(tmp_path)
        cfg = {
            "grant_type": "authentik_app_password",
            "token_url": "https://idp.example/token/",
            "client_id": "toolhive",
            "username": "svc",
            "password_env": "PW",
        }
        auth = ServiceAccountAuth("toolhive", cfg, hermes_home=home)

        async def _post(http_client, token_url, form, server_name):
            # Deliberately no refresh_token in the response.
            return {"access_token": "NEW", "token_type": "Bearer", "expires_in": 3600}

        with patch("tools.mcp_service_account._post_token_request", _post):
            token = await auth._exchange_refresh_token(MagicMock(), "KEEP_ME")

        assert token is not None
        assert token.access_token == "NEW"
        assert token.refresh_token == "KEEP_ME"

    @pytest.mark.asyncio
    async def test_refresh_response_with_new_refresh_token_rotates(self, tmp_path):
        from tools.mcp_service_account import ServiceAccountAuth

        home, _ = _profile_homes(tmp_path)
        cfg = {
            "grant_type": "authentik_app_password",
            "token_url": "https://idp.example/token/",
            "client_id": "toolhive",
            "username": "svc",
            "password_env": "PW",
        }
        auth = ServiceAccountAuth("toolhive", cfg, hermes_home=home)

        async def _post(http_client, token_url, form, server_name):
            return {
                "access_token": "NEW",
                "expires_in": 3600,
                "refresh_token": "ROTATED",
            }

        with patch("tools.mcp_service_account._post_token_request", _post):
            token = await auth._exchange_refresh_token(MagicMock(), "OLD")

        assert token.refresh_token == "ROTATED"

    @pytest.mark.asyncio
    async def test_delayed_401_does_not_discard_a_concurrent_refresh(self, tmp_path):
        """A late 401 for an OLD token must not nuke the token that replaced it.

        Request A goes out on T1, T1 expires, request B refreshes to T2, and
        only then does A's 401 (for T1) arrive. Clearing unconditionally would
        throw T2 away and force another exchange — once per delayed 401.
        """
        import time as _time

        from tools.mcp_service_account import ServiceAccountAuth, _CachedToken

        home, _ = _profile_homes(tmp_path)
        cfg = {
            "grant_type": "authentik_app_password",
            "token_url": "https://idp.example/token/",
            "client_id": "toolhive",
            "username": "svc",
            "password_env": "PW",
        }
        auth = ServiceAccountAuth("toolhive", cfg, hermes_home=home)

        t1 = _CachedToken("T1", _time.time() + 3600, lifetime=3600.0)
        t2 = _CachedToken("T2", _time.time() + 3600, lifetime=3600.0)
        auth._mem_token = t1

        acquisitions: list[str] = []

        async def _acquire(self_inner, http_client):
            acquisitions.append("call")
            return self_inner._mem_token

        deleted: list = []
        with patch.object(ServiceAccountAuth, "_acquire_token", _acquire), patch(
            "tools.mcp_service_account._delete_token_cache",
            side_effect=lambda p: deleted.append(p),
        ):
            request = MagicMock()
            request.headers = {}
            gen = auth.async_auth_flow(request)
            sent = await gen.__anext__()
            assert sent.headers["Authorization"] == "Bearer T1"

            # B's refresh lands while A's request is in flight.
            auth._mem_token = t2

            resp = MagicMock()
            resp.status_code = 401
            retry = await gen.asend(resp)

        # T2 survived: not cleared from memory, cache file not deleted.
        assert auth._mem_token is t2
        assert deleted == []
        assert retry.headers["Authorization"] == "Bearer T2"

    @pytest.mark.asyncio
    async def test_401_for_the_current_token_does_invalidate(self, tmp_path):
        """The ordinary case must still clear the token that was rejected."""
        import time as _time

        from tools.mcp_service_account import ServiceAccountAuth, _CachedToken

        home, _ = _profile_homes(tmp_path)
        cfg = {
            "grant_type": "authentik_app_password",
            "token_url": "https://idp.example/token/",
            "client_id": "toolhive",
            "username": "svc",
            "password_env": "PW",
        }
        auth = ServiceAccountAuth("toolhive", cfg, hermes_home=home)
        t1 = _CachedToken("T1", _time.time() + 3600, lifetime=3600.0)
        auth._mem_token = t1

        cleared: list = []
        # Exactly ONE exchange is expected. The first leg is served from the
        # valid in-memory token without opening a token client at all (see
        # ``_acquire_token_with_client``); only the 401 retry mints.
        acquisitions: list[str] = []
        t3 = _CachedToken("T3", _time.time() + 3600, lifetime=3600.0)

        async def _acquire(self_inner, http_client):
            acquisitions.append("call")
            return t3

        with patch.object(ServiceAccountAuth, "_acquire_token", _acquire), patch(
            "tools.mcp_service_account._delete_token_cache",
            side_effect=lambda p: cleared.append(p),
        ):
            request = MagicMock()
            request.headers = {}
            gen = auth.async_auth_flow(request)
            sent = await gen.__anext__()
            assert sent.headers["Authorization"] == "Bearer T1"
            resp = MagicMock()
            resp.status_code = 401
            retry = await gen.asend(resp)

        assert cleared == [auth._cache_path]
        assert retry.headers["Authorization"] == "Bearer T3"
        assert acquisitions == ["call"]


# ---------------------------------------------------------------------------
# Context propagation onto the shared MCP event loop
# ---------------------------------------------------------------------------


class TestLoopPropagation:
    def test_run_on_mcp_loop_carries_home_secret_and_profile(self, tmp_path):
        """The thread hop must not drop home, secret scope, or identity."""
        from agent.secret_scope import (
            current_secret_scope,
            reset_secret_scope,
            set_secret_scope,
        )
        from hermes_constants import (
            get_hermes_home,
            reset_hermes_home_override,
            set_hermes_home_override,
        )

        home_a, _ = _profile_homes(tmp_path)
        captured: dict = {}

        async def _probe():
            captured["home"] = str(get_hermes_home())
            captured["scope"] = dict(current_secret_scope() or {})
            captured["profile"] = mcp_profile.current_profile_key()
            captured["thread"] = threading.current_thread().name
            return True

        mcp_tool._ensure_mcp_loop()
        home_token = set_hermes_home_override(str(home_a))
        secret_token = set_secret_scope({"AUTHENTIK_ZUG_APP_PASSWORD": "zug-secret"})
        try:
            assert mcp_tool._run_on_mcp_loop(_probe, timeout=10) is True
        finally:
            reset_secret_scope(secret_token)
            reset_hermes_home_override(home_token)

        from hermes_constants import hermes_home_key

        assert captured["thread"] == "mcp-event-loop"
        assert captured["home"] == str(home_a)
        assert captured["scope"] == {"AUTHENTIK_ZUG_APP_PASSWORD": "zug-secret"}
        assert captured["profile"] == hermes_home_key(home_a)

    def test_loop_scope_does_not_leak_between_profiles(self, tmp_path):
        """Two profiles' loop work is task-local, not last-writer-wins."""
        from agent.secret_scope import (
            current_secret_scope,
            reset_secret_scope,
            set_secret_scope,
        )
        from hermes_constants import (
            hermes_home_key,
            reset_hermes_home_override,
            set_hermes_home_override,
        )

        home_a, home_b = _profile_homes(tmp_path)
        seen: list[tuple] = []

        async def _probe():
            await asyncio.sleep(0.01)
            seen.append(
                (
                    mcp_profile.current_profile_key(),
                    dict(current_secret_scope() or {}),
                )
            )
            return True

        mcp_tool._ensure_mcp_loop()

        def _call(home, secret):
            home_token = set_hermes_home_override(str(home))
            secret_token = set_secret_scope(secret)
            try:
                mcp_tool._run_on_mcp_loop(_probe, timeout=10)
            finally:
                reset_secret_scope(secret_token)
                reset_hermes_home_override(home_token)

        threads = [
            threading.Thread(
                target=_call, args=(home_a, {"AUTHENTIK_ZUG_APP_PASSWORD": "zug"})
            ),
            threading.Thread(
                target=_call, args=(home_b, {"AUTHENTIK_CAROL_APP_PASSWORD": "carol"})
            ),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=20)

        assert len(seen) == 2
        assert dict(seen) == {
            hermes_home_key(home_a): {"AUTHENTIK_ZUG_APP_PASSWORD": "zug"},
            hermes_home_key(home_b): {"AUTHENTIK_CAROL_APP_PASSWORD": "carol"},
        }

    def test_no_scope_active_is_a_passthrough(self):
        """Single-profile callers keep the un-wrapped coroutine."""

        async def _plain():
            return 1

        coro = _plain()
        with patch(
            "hermes_constants.get_hermes_home_override", return_value=None
        ), patch("agent.secret_scope.current_secret_scope", return_value=None):
            assert mcp_tool._wrap_with_profile_scope(coro) is coro
        coro.close()

    def test_secret_scope_alone_is_enough_to_wrap(self):
        """A cron-style scope with no home override must still propagate."""

        async def _plain():
            return 1

        coro = _plain()
        with patch(
            "hermes_constants.get_hermes_home_override", return_value=None
        ), patch(
            "agent.secret_scope.current_secret_scope", return_value={"K": "v"}
        ):
            wrapped = mcp_tool._wrap_with_profile_scope(coro)
        assert wrapped is not coro
        wrapped.close()
        coro.close()


# ---------------------------------------------------------------------------
# Backwards compatibility
# ---------------------------------------------------------------------------


class TestSingleProfileCompatibility:
    def test_state_views_behave_like_the_old_globals(self):
        """No profile scope active → ordinary dict/set semantics."""
        mcp_tool._servers["srv"] = "task"
        assert mcp_tool._servers["srv"] == "task"
        assert "srv" in mcp_tool._servers
        assert list(mcp_tool._servers) == ["srv"]
        assert len(mcp_tool._servers) == 1
        assert dict(mcp_tool._servers) == {"srv": "task"}
        assert mcp_tool._servers.copy() == {"srv": "task"}
        assert mcp_tool._servers.get("nope") is None
        assert mcp_tool._servers == {"srv": "task"}
        del mcp_tool._servers["srv"]
        assert mcp_tool._servers == {}

        mcp_tool._server_connecting.add("srv")
        assert "srv" in mcp_tool._server_connecting
        assert set(mcp_tool._server_connecting) == {"srv"}
        mcp_tool._server_connecting.update(["a", "b"])
        mcp_tool._server_connecting.difference_update(["a"])
        assert set(mcp_tool._server_connecting) == {"srv", "b"}
        mcp_tool._server_connecting.discard("b")
        mcp_tool._server_connecting.clear()
        assert set(mcp_tool._server_connecting) == set()

    def test_patch_dict_still_works_on_the_views(self):
        """Existing tests patch these containers directly — keep that working."""
        server = _fake_server("A")
        with patch.dict(mcp_tool._servers, {"srv": server}):
            assert mcp_tool._servers["srv"] is server
        assert "srv" not in mcp_tool._servers

        with patch.dict(mcp_tool._server_error_counts, {"srv": 7}, clear=True):
            assert mcp_tool._server_error_counts["srv"] == 7
        assert mcp_tool._server_error_counts == {}

    def test_static_header_and_oauth_modes_are_untouched(self, tmp_path):
        """Non-service-account auth paths keep their existing behaviour."""
        from tools.mcp_service_account import validate_service_account_config

        # A header-auth server has no service_account block at all; nothing in
        # the profile work should start requiring one.
        cfg = {"url": "https://x.example/mcp", "headers": {"Authorization": "Bearer x"}}
        assert "service_account" not in cfg
        # And SA validation still rejects a missing grant_type as before.
        assert validate_service_account_config("srv", {}) != []

    def test_shutdown_sweeps_every_profile(self, tmp_path):
        """Process teardown must not strand other profiles' state."""
        home_a, home_b = _profile_homes(tmp_path)
        with mcp_profile.profile_scope(home_a):
            mcp_tool._record_connect_failure("toolhive")
        with mcp_profile.profile_scope(home_b):
            mcp_tool._record_connect_failure("toolhive")

        with patch("tools.mcp_tool._stop_mcp_loop"):
            mcp_tool.shutdown_mcp_servers()

        for home in (home_a, home_b):
            with mcp_profile.profile_scope(home):
                assert mcp_tool._server_connect_retry_after == {}
                assert mcp_tool._server_connect_failures == {}


# ---------------------------------------------------------------------------
# Real-path integration: two profile homes, real discovery/registration
# ---------------------------------------------------------------------------


class TestRealPathIntegration:
    def test_two_profiles_register_and_dispatch_independently(
        self, tmp_path, monkeypatch, caplog
    ):
        """Real ``_load_mcp_config`` → real registration → real dispatch.

        Only the transport is faked: ``_connect_server`` returns a stub whose
        session records the URL and credential it was built from. Everything
        else — config loading, secret resolution, registry writes, tool
        registration, handler/check_fn dispatch — is the production path.
        """
        from agent.secret_scope import (
            build_profile_secret_scope,
            reset_secret_scope,
            set_multiplex_active,
            set_secret_scope,
        )
        from hermes_constants import (
            reset_hermes_home_override,
            set_hermes_home_override,
        )
        from tools.registry import registry

        home_a, home_b = _profile_homes(tmp_path)
        _write_profile(
            home_a,
            url="https://toolhive.zug.example/mcp",
            password_env="AUTHENTIK_ZUG_APP_PASSWORD",
            password="zug-secret",
            username="zug",
        )
        _write_profile(
            home_b,
            url="https://toolhive.carol.example/mcp",
            password_env="AUTHENTIK_CAROL_APP_PASSWORD",
            password="carol-secret",
            username="carol",
        )
        monkeypatch.setenv("HERMES_HOME", str(home_a))
        monkeypatch.setenv("AUTHENTIK_ZUG_APP_PASSWORD", "zug-secret")

        connected: list[dict] = []

        async def _fake_connect(name, config):
            """Stand in for the transport; resolve the credential for real."""
            from tools.mcp_service_account import _resolve_password

            sa = config.get("service_account") or {}
            password = _resolve_password(sa, name)
            server = mcp_tool.MCPServerTask(name)
            server.session = _FakeSession(config["url"])
            server._rpc_lock = asyncio.Lock()
            server._tools = [
                SimpleNamespace(
                    name="ping",
                    description="ping",
                    inputSchema={"type": "object", "properties": {}},
                    annotations=None,
                )
            ]
            server._ready.set()
            server._config = config
            connected.append(
                {
                    "profile": server._profile_key,
                    "url": config["url"],
                    "password_env": sa.get("password_env"),
                    "password": password,
                }
            )
            return server

        registered: dict[str, list[str]] = {}

        def _run_profile(home):
            home_token = set_hermes_home_override(str(home))
            secret_token = set_secret_scope(build_profile_secret_scope(home))
            try:
                servers = mcp_tool._load_mcp_config()
                names = mcp_tool.register_mcp_servers(servers)
                registered[str(home)] = names
            finally:
                reset_secret_scope(secret_token)
                reset_hermes_home_override(home_token)

        set_multiplex_active(True)
        try:
            with caplog.at_level(logging.DEBUG), patch(
                "tools.mcp_tool._ensure_mcp_sdk", return_value=True
            ), patch("tools.mcp_tool._connect_server", _fake_connect):
                _run_profile(home_a)
                _run_profile(home_b)
        finally:
            set_multiplex_active(False)

        # 1. Each profile connected with ITS OWN url + credential.
        from hermes_constants import hermes_home_key

        by_profile = {c["profile"]: c for c in connected}
        assert set(by_profile) == {hermes_home_key(home_a), hermes_home_key(home_b)}
        zug = by_profile[hermes_home_key(home_a)]
        carol = by_profile[hermes_home_key(home_b)]
        assert zug["url"] == "https://toolhive.zug.example/mcp"
        assert carol["url"] == "https://toolhive.carol.example/mcp"
        assert zug["password_env"] == "AUTHENTIK_ZUG_APP_PASSWORD"
        # The acceptance criterion: Carol never attempts Zug's env var, and
        # never receives Zug's value even though it is in os.environ.
        assert carol["password_env"] == "AUTHENTIK_CAROL_APP_PASSWORD"
        assert carol["password"] == "carol-secret"
        assert zug["password"] == "zug-secret"

        # 2. Each profile owns its own task and its own registered tools.
        with mcp_profile.profile_scope(home_a):
            task_a = mcp_tool._servers["toolhive"]
            assert mcp_tool._mcp_tool_server_names.get("mcp__toolhive__ping") == "toolhive"
        with mcp_profile.profile_scope(home_b):
            task_b = mcp_tool._servers["toolhive"]
        assert task_a is not task_b
        assert task_a._profile_key != task_b._profile_key
        assert registered[str(home_a)] == registered[str(home_b)] == [
            "mcp__toolhive__ping"
        ]

        # 3. Dispatch under each profile reaches only that profile's session.
        handler = mcp_tool._make_tool_handler("toolhive", "ping", 30.0)
        with patch("tools.mcp_tool._run_on_mcp_loop", side_effect=_run_loop):
            with mcp_profile.profile_scope(home_a):
                out_a = handler({})
            with mcp_profile.profile_scope(home_b):
                out_b = handler({})
        assert "toolhive.zug.example" in out_a
        assert "toolhive.carol.example" in out_b
        assert task_a.session.calls == ["ping"] and task_b.session.calls == ["ping"]

        # 4. A failure injected into A does not park or block B.
        with mcp_profile.profile_scope(home_a):
            mcp_tool._servers.pop("toolhive")
            mcp_tool._server_connect_errors["toolhive"] = "401 Unauthorized"
            mcp_tool._record_connect_failure("toolhive")
        check = mcp_tool._make_check_fn("toolhive")
        with mcp_profile.profile_scope(home_a):
            assert check() is False
        with mcp_profile.profile_scope(home_b):
            assert check() is True
            assert "toolhive" not in mcp_tool._server_connect_errors

        # 5. No credential values in captured logs.
        text = caplog.text
        assert "zug-secret" not in text
        assert "carol-secret" not in text

        # 6. Cache files, if any, live only under their own profile home.
        for home, foreign in ((home_a, home_b), (home_b, home_a)):
            for path in home.rglob("*"):
                assert str(foreign) not in str(path)

        for name in registered[str(home_a)]:
            registry.deregister(name)


# ---------------------------------------------------------------------------
# The MCP event loop is process-global; idleness is not
# ---------------------------------------------------------------------------


class TestLoopIdlenessIsProcessWide:
    """``_stop_mcp_loop_if_idle`` must consider EVERY profile's registry.

    ``_mcp_loop`` is one loop for the whole process, shared by every profile's
    long-lived ``MCPServerTask.run()`` and every stdio child those tasks own.
    The idle check, though, used to read ``_servers`` / ``_server_connecting``,
    which are views onto the *ambient* profile.  A dashboard or CLI probe
    running under profile B therefore saw an empty registry, concluded the loop
    was idle, and stopped it — cancelling profile A's server tasks and killing
    A's stdio children.  A must be able to veto B's teardown.
    """

    @pytest.fixture(autouse=True)
    def _restore_loop(self):
        yield
        mcp_tool._stop_mcp_loop()

    def test_profile_b_probe_cannot_stop_profile_a_loop(self, tmp_path):
        home_a, home_b = _profile_homes(tmp_path)

        mcp_tool._ensure_mcp_loop()
        loop = mcp_tool._mcp_loop
        assert loop is not None and loop.is_running()

        with mcp_profile.profile_scope(home_a):
            mcp_tool._servers["toolhive"] = MagicMock()

        with mcp_profile.profile_scope(home_b):
            assert mcp_tool._servers == {}
            stopped = mcp_tool._stop_mcp_loop_if_idle()

        assert stopped is False, (
            "profile B's probe stopped the shared MCP loop while profile A "
            "still owns a registered server on it"
        )
        assert mcp_tool._mcp_loop is loop
        assert loop.is_running()

    def test_profile_b_probe_cannot_stop_loop_while_a_is_connecting(self, tmp_path):
        """An in-flight connect in another profile also vetoes the stop.

        ``_server_connecting`` is the window where a task exists on the loop
        but is not yet in ``_servers`` — the exact interval a concurrent probe
        is most likely to land in.
        """
        home_a, home_b = _profile_homes(tmp_path)

        mcp_tool._ensure_mcp_loop()
        loop = mcp_tool._mcp_loop

        with mcp_profile.profile_scope(home_a):
            mcp_tool._server_connecting.add("toolhive")

        with mcp_profile.profile_scope(home_b):
            stopped = mcp_tool._stop_mcp_loop_if_idle()

        assert stopped is False
        assert mcp_tool._mcp_loop is loop
        assert loop.is_running()

    def test_genuinely_idle_loop_still_stops(self, tmp_path):
        """The veto is not blanket: no profile holding state means stop."""
        home_a, home_b = _profile_homes(tmp_path)

        mcp_tool._ensure_mcp_loop()
        assert mcp_tool._mcp_loop is not None

        # Touch both registries so they exist but stay empty of server state.
        for home in (home_a, home_b):
            with mcp_profile.profile_scope(home):
                mcp_tool._server_connect_errors["toolhive"] = "boom"

        with mcp_profile.profile_scope(home_b):
            assert mcp_tool._stop_mcp_loop_if_idle() is True
        assert mcp_tool._mcp_loop is None


# ---------------------------------------------------------------------------
# The state views must behave like the built-ins they replace
# ---------------------------------------------------------------------------


class TestProfileScopedSetOperators:
    """``ProfileScopedSet`` stands in for ``set`` at 60+ MCP call sites.

    ``collections.abc.Set`` builds every operator result through
    ``cls._from_iterable(<generator>)``. The default implementation is
    ``cls(iterable)``, but this class's ``__init__`` takes a registry FIELD
    NAME — so ``_server_connecting | {...}`` raised ``TypeError: attribute
    name must be string, not 'generator'`` instead of returning a set, and so
    did every other operator, on BOTH operand orders (``set.__or__`` returns
    ``NotImplemented`` for a non-``set``, after which Python calls the view's
    reflected method).

    The result type matters as much as the absence of the crash: a view is an
    alias for one field of one profile's registry, so a derived value must be
    a detached plain ``set``. Returning another view would re-bind to whatever
    profile happened to be active when it was next read — the exact
    cross-profile aliasing this module exists to prevent.
    """

    @pytest.fixture
    def view(self, tmp_path):
        home_a, _home_b = _profile_homes(tmp_path)
        with mcp_profile.profile_scope(home_a):
            v = mcp_profile.ProfileScopedSet("server_connecting")
            v.add("toolhive")
            yield v

    def test_binary_operators_both_orders(self, view):
        assert view | {"other"} == {"toolhive", "other"}
        assert {"other"} | view == {"toolhive", "other"}
        assert view & {"toolhive", "nope"} == {"toolhive"}
        assert {"toolhive", "nope"} & view == {"toolhive"}
        assert view - {"toolhive"} == set()
        assert {"toolhive", "other"} - view == {"other"}
        assert view ^ {"other"} == {"toolhive", "other"}
        assert {"other"} ^ view == {"toolhive", "other"}

    def test_operator_results_are_detached_plain_sets(self, view, tmp_path):
        """A derived set must not follow the ambient profile around."""
        home_a, home_b = _profile_homes(tmp_path)
        derived = view | {"other"}
        assert type(derived) is set
        with mcp_profile.profile_scope(home_b):
            # B's registry is empty; the derived value must not notice.
            assert derived == {"toolhive", "other"}
            assert mcp_profile.ProfileScopedSet("server_connecting") == set()

    def test_in_place_operators_mutate_this_profiles_registry(self, tmp_path):
        home_a, home_b = _profile_homes(tmp_path)
        with mcp_profile.profile_scope(home_a):
            view = mcp_profile.ProfileScopedSet("server_connecting")
            view |= {"toolhive", "other"}
            assert view == {"toolhive", "other"}
            # ``MutableSet.__iand__`` is implemented as ``self - it``, so it
            # went down the same broken ``_from_iterable`` path.
            view &= {"toolhive"}
            assert view == {"toolhive"}
            view ^= {"other"}
            assert view == {"toolhive", "other"}
            view -= {"other"}
            assert view == {"toolhive"}
            assert mcp_profile.registry_for(home_a).server_connecting == {"toolhive"}
        with mcp_profile.profile_scope(home_b):
            assert mcp_profile.ProfileScopedSet("server_connecting") == set()

    def test_named_set_methods(self, view):
        assert view.union({"other"}) == {"toolhive", "other"}
        assert view.intersection({"toolhive", "nope"}) == {"toolhive"}
        assert view.difference({"toolhive"}) == set()
        assert view.symmetric_difference({"other"}) == {"toolhive", "other"}
        assert view.issubset({"toolhive", "other"}) is True
        assert view.issuperset({"toolhive"}) is True
        assert view.isdisjoint({"other"}) is True
        view.intersection_update({"toolhive", "other"})
        assert view == {"toolhive"}
        view.symmetric_difference_update({"toolhive"})
        assert view == set()

    def test_comparisons_and_membership_still_work(self, view):
        assert view == {"toolhive"}
        assert view != {"other"}
        assert "toolhive" in view
        assert view <= {"toolhive", "other"}
        assert view < {"toolhive", "other"}
        assert view >= {"toolhive"}
        assert len(view) == 1
