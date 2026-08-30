"""Tool-availability caching must not leak across profiles or pin stale state.

``model_tools.get_tool_definitions`` filters the catalog through each entry's
``check_fn``, and ``tools.registry`` TTL-caches those verdicts for 30s. That
cache is keyed by ``(check_fn, check_fn_cache_scope())``.

For MCP that key is dangerous in both of its parts:

* **The ``check_fn`` is shared across profiles by design.** Two profiles
  configuring a server called ``toolhive`` both register the public tool name
  ``mcp__toolhive__ping``, so there is ONE registry entry and ONE ``check_fn``
  object. Only the state that function reads is profile-scoped
  (``tools/mcp_profile.py``). A cached verdict against that shared object is
  therefore a cross-profile answer unless the scope half of the key separates
  them.
* **The scope half used to be gated on ``is_multiplex_active()``**, which only
  ``gateway/run.py`` ever sets. The desktop/TUI gateway serves many profiles
  from one process too — it binds ``session["profile_home"]`` with
  ``set_hermes_home_override`` for each agent build — but never sets that
  flag, so every desktop profile collapsed onto the single ``(check_fn,
  None)`` key.

And even within one profile, caching a two-dict-lookup probe for 30s means a
transport blip pins "unavailable" long after the reconnect landed, so every
agent built in that window comes up without the server's tools, with no error
and no log line.

No network, no credentials, no live state: temp profile homes and registry
state built in-process.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools import mcp_profile, mcp_tool, registry as registry_mod

_SERVER = "toolhive"


@pytest.fixture(autouse=True)
def _clean():
    from agent.secret_scope import is_multiplex_active, set_multiplex_active

    was_multiplex = is_multiplex_active()
    mcp_profile.reset_all_registries()
    registry_mod.invalidate_check_fn_cache()
    try:
        yield
    finally:
        mcp_profile.reset_all_registries()
        registry_mod.invalidate_check_fn_cache()
        set_multiplex_active(was_multiplex)


@pytest.fixture
def homes(tmp_path: Path) -> tuple[Path, Path]:
    a = tmp_path / "jonathon"
    b = tmp_path / "carol"
    a.mkdir()
    b.mkdir()
    return a, b


class _HomeOverride:
    """``set_hermes_home_override`` as a context manager."""

    def __init__(self, home) -> None:
        self._home = home
        self._token = None

    def __enter__(self):
        self._token = set_hermes_home_override(str(self._home))
        return self

    def __exit__(self, *exc):
        reset_hermes_home_override(self._token)
        return False


# ---------------------------------------------------------------------------
# 1. The cache scope must follow the profile, not the multiplex flag
# ---------------------------------------------------------------------------


class TestCacheScopeIdentity:
    def test_home_override_scopes_the_cache_without_the_multiplex_flag(self, homes):
        """The desktop gateway's isolation boundary is the override, not a flag."""
        from agent.secret_scope import set_multiplex_active

        home_a, home_b = homes
        set_multiplex_active(False)

        with _HomeOverride(home_a):
            scope_a = registry_mod.check_fn_cache_scope()
        with _HomeOverride(home_b):
            scope_b = registry_mod.check_fn_cache_scope()

        assert scope_a and scope_b
        assert scope_a != scope_b, (
            "two profiles shared one availability-cache key, so one profile's "
            "verdict answered for the other"
        )
        assert scope_a == str(home_a.resolve())

    def test_single_profile_process_keeps_the_process_wide_cache(self):
        """No override and no multiplexing → historical behaviour (None)."""
        from agent.secret_scope import set_multiplex_active

        set_multiplex_active(False)
        assert registry_mod.check_fn_cache_scope() is None

    def test_multiplex_without_an_override_still_bypasses(self):
        """Unresolved identity must never be aliased onto a shared key."""
        from agent.secret_scope import set_multiplex_active

        set_multiplex_active(True)
        assert registry_mod.check_fn_cache_scope() == registry_mod.CHECK_FN_CACHE_BYPASS

    def test_two_profiles_do_not_share_a_cached_verdict(self, homes):
        """End to end through ``_check_fn_cached`` with one shared check_fn."""
        from agent.secret_scope import set_multiplex_active

        home_a, home_b = homes
        set_multiplex_active(False)

        calls: list[str] = []

        def _probe() -> bool:
            # Deliberately NOT marked no-cache: this asserts the scope half of
            # the key, independently of the MCP opt-out below.
            key = mcp_profile.current_profile_key()
            calls.append(key)
            return key == mcp_profile.canonical_profile_key(str(home_a))

        with _HomeOverride(home_a):
            assert registry_mod._check_fn_cached(_probe) is True
        with _HomeOverride(home_b):
            assert registry_mod._check_fn_cached(_probe) is False, (
                "profile A's cached True answered for profile B"
            )
        # Same profile again → served from cache, not re-probed.
        with _HomeOverride(home_a):
            assert registry_mod._check_fn_cached(_probe) is True
        assert len(calls) == 2


# ---------------------------------------------------------------------------
# 2. MCP availability is live, never cached
# ---------------------------------------------------------------------------


class TestMcpAvailabilityIsUncached:
    def test_make_check_fn_is_marked_no_cache(self):
        check = mcp_tool._make_check_fn(_SERVER)
        assert registry_mod.is_no_cache_check_fn(check) is True

    def test_reconnect_is_visible_immediately(self, homes):
        """A transport blip must not hide a recovered server for the TTL.

        ``_check`` is two dict lookups against live state; caching it means an
        agent built seconds after a reconnect still comes up without the
        server's tools.
        """
        home_a, _home_b = homes
        check = mcp_tool._make_check_fn(_SERVER)

        with _HomeOverride(home_a):
            # Down: nothing registered under this profile.
            assert registry_mod._check_fn_cached(check) is False
            # Recovered: the lazy config lands, exactly as discovery does.
            mcp_profile.current_registry().lazy_server_configs[_SERVER] = {
                "url": "https://toolhive.example/mcp"
            }
            assert registry_mod._check_fn_cached(check) is True, (
                "a stale 'unavailable' verdict survived the reconnect"
            )

    def test_no_cached_verdict_is_reported_for_mcp(self, homes):
        """Read-only surfaces must report "unknown", not a stale verdict."""
        home_a, _home_b = homes
        check = mcp_tool._make_check_fn(_SERVER)

        with _HomeOverride(home_a):
            mcp_profile.current_registry().lazy_server_configs[_SERVER] = {"url": "u"}
            assert registry_mod._check_fn_cached(check) is True
            assert registry_mod.get_cached_check_fn_result(check) is None

    def test_one_shared_check_fn_answers_per_profile(self, homes):
        """The registry entry is shared; the answer must not be.

        This is the shape the live failure had: both profiles register the
        same public tool name, so there is one ``check_fn`` object. Only
        profile A has the server.
        """
        from agent.secret_scope import set_multiplex_active

        home_a, home_b = homes
        set_multiplex_active(False)
        check = mcp_tool._make_check_fn(_SERVER)

        with _HomeOverride(home_a):
            mcp_profile.current_registry().lazy_server_configs[_SERVER] = {"url": "a"}
            assert registry_mod._check_fn_cached(check) is True

        with _HomeOverride(home_b):
            assert registry_mod._check_fn_cached(check) is False, (
                "profile B was told it has an MCP server it never discovered"
            )

    def test_no_cache_marker_does_not_grow_a_process_global_set(self):
        """One closure per server per discovery pass must not accumulate."""
        before = len(registry_mod._NO_CACHE_CHECK_FNS)
        for i in range(50):
            mcp_tool._make_check_fn(f"server-{i}")
        assert len(registry_mod._NO_CACHE_CHECK_FNS) == before

    def test_decorator_form_still_works(self):
        """``no_cache_check_fn`` keeps its existing contract."""

        @registry_mod.no_cache_check_fn
        def _probe() -> bool:
            return True

        assert registry_mod.is_no_cache_check_fn(_probe) is True
        assert _probe in registry_mod._NO_CACHE_CHECK_FNS
