"""Startup MCP discovery must run per profile, not once per process.

The multiplexed gateway serves many Hermes profiles from ONE process. Commit
``02f39e700`` made the MCP *runtime* state profile-scoped
(``tools/mcp_profile.py``), but the thing that decides whether discovery runs
at all -- ``hermes_cli.mcp_startup.start_background_mcp_discovery`` -- still
gated on a process-global ``_mcp_discovery_started`` / ``_mcp_discovery_thread``
pair. The first profile to reach that gate owned it, so every other profile
returned without ever populating its own registry.

Two halves are exercised here, both on the REAL startup entry point
(``tui_gateway.entry.ensure_mcp_discovery_started`` -- the call the multiplexed
gateway makes from ``server._build`` after binding the session's HERMES_HOME
and secret scope):

1. **The gate.** Two profiles configuring the same server name (``toolhive``)
   must each get their own discovery run, reading their own ``config.yaml``.
2. **The scope.** ContextVars do not cross a bare ``threading.Thread``, so the
   discovery thread has to re-install the caller's secret scope as well as the
   HERMES_HOME override. Without it every ``${VAR}`` in the profile's MCP
   config resolves through ``agent.secret_scope.get_secret`` with no scope --
   which under multiplexing raises ``UnscopedSecretError``, gets swallowed by
   ``_load_mcp_config``'s broad except, and silently yields an EMPTY config for
   that profile.

Secret scopes are installed here as explicit literal dicts rather than via
``build_profile_secret_scope`` so no ``.env`` file is written and no profile
secret is ever unioned into ``os.environ`` -- the isolation the fix exists to
protect.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import pytest

from agent import secret_scope as secret_scope_mod
from hermes_cli import mcp_startup
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools import mcp_profile, mcp_tool

# Same logical server name in both profiles -- the collapsing key.
_SERVER = "toolhive"

_ZUG_ENV = "AUTHENTIK_ZUG_APP_PASSWORD"
_CAROL_ENV = "AUTHENTIK_CAROL_APP_PASSWORD"

_JOIN_TIMEOUT_S = 10.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_profile(home: Path, password_env: str, *, url: str | None = None) -> None:
    """Write a realistic profile home: config.yaml only, no .env on disk."""
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        f"""
mcp_servers:
  {_SERVER}:
    url: {url or "https://toolhive.example/mcp"}
    auth: service_account
    service_account:
      grant_type: authentik_app_password
      token_url: https://idp.example/token/
      client_id: {_SERVER}
      username: svc
      password_env: {password_env}
""".lstrip(),
        encoding="utf-8",
    )


def _key(home: Path) -> str:
    return mcp_profile.canonical_profile_key(str(home))


def _start_discovery_for(home: Path, secrets: dict[str, str] | None = None) -> None:
    """Enter the REAL gateway startup path under *home*'s profile context.

    Mirrors ``tui_gateway/server.py::_build``: bind the session profile's
    HERMES_HOME override and secret scope, THEN call
    ``ensure_mcp_discovery_started()``.
    """
    from tui_gateway import entry as tui_entry

    home_token = set_hermes_home_override(str(home))
    secret_token = secret_scope_mod.set_secret_scope(secrets if secrets is not None else {})
    try:
        tui_entry.ensure_mcp_discovery_started()
    finally:
        secret_scope_mod.reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)


def _start_coordinator_for(home: Path, secrets: dict[str, str] | None = None) -> None:
    """Call the coordinator DIRECTLY under *home*'s profile context.

    ``_start_discovery_for`` goes through ``tui_gateway.entry``, which applies
    its own config gate before delegating. The lock-scope tests need the
    coordinator itself, with nothing in front of it.
    """
    home_token = set_hermes_home_override(str(home))
    secret_token = secret_scope_mod.set_secret_scope(
        secrets if secrets is not None else {}
    )
    try:
        mcp_startup.start_background_mcp_discovery(
            logger=logging.getLogger("tui_gateway.entry"),
            thread_name=f"test-mcp-discovery-{home.name}",
        )
    finally:
        secret_scope_mod.reset_secret_scope(secret_token)
        reset_hermes_home_override(home_token)


def _join_all_discovery(timeout: float = _JOIN_TIMEOUT_S) -> None:
    """Join every discovery thread this module may have spawned."""
    deadline = time.monotonic() + timeout
    for thread in mcp_startup.discovery_threads():
        remaining = max(0.0, deadline - time.monotonic())
        thread.join(timeout=remaining)


def _wait_for(predicate, timeout: float = _JOIN_TIMEOUT_S) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


class _Recorder:
    """Stands in for ``register_mcp_servers`` and records what discovery saw."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.seen: list[tuple[str, dict]] = []
        self.entered: dict[str, threading.Event] = {}
        self.gates: dict[str, threading.Event] = {}

    def gate(self, home: Path) -> threading.Event:
        """Block this profile's discovery inside register until released."""
        event = threading.Event()
        self.gates[_key(home)] = event
        return event

    def entered_event(self, home: Path) -> threading.Event:
        event = self.entered.setdefault(_key(home), threading.Event())
        return event

    def __call__(self, servers: dict) -> list:
        key = mcp_profile.current_profile_key()
        with self.lock:
            self.seen.append((key, servers.get(_SERVER) or {}))
        self.entered.setdefault(key, threading.Event()).set()
        gate = self.gates.get(key)
        if gate is not None:
            gate.wait(timeout=_JOIN_TIMEOUT_S)
        return []

    def configs_for(self, home: Path) -> list[dict]:
        with self.lock:
            return [cfg for key, cfg in self.seen if key == _key(home)]

    def keys_seen(self) -> list[str]:
        with self.lock:
            return [key for key, _ in self.seen]

    def password_envs_for(self, home: Path) -> list[str]:
        return [
            (cfg.get("service_account") or {}).get("password_env")
            for cfg in self.configs_for(home)
        ]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_discovery_state():
    """Start from a clean coordinator + registry map, leave no thread behind.

    ``ensure_mcp_discovery_started`` latches ``tui_gateway.entry
    ._mcp_discovery_enabled`` for the rest of the PROCESS (by design: later
    agent builds re-invoke the idempotent spawn so the retry-after-zero-
    connected allowance can fire). Left set, it makes
    ``entry.wait_for_mcp_discovery`` delegate to ``mcp_startup`` in every
    later test file too — which is what turned
    ``test_make_agent_waits_for_shared_mcp_discovery`` red when this file ran
    before it. Restore it here rather than weakening the production latch.
    """
    from tui_gateway import entry as tui_entry

    was_enabled = tui_entry._mcp_discovery_enabled
    mcp_startup.reset_discovery_state()
    mcp_profile.reset_all_registries()
    try:
        yield
    finally:
        _join_all_discovery()
        mcp_startup.reset_discovery_state()
        mcp_profile.reset_all_registries()
        tui_entry._mcp_discovery_enabled = was_enabled


@pytest.fixture
def recorder(monkeypatch) -> _Recorder:
    """Run real discovery up to (not including) the connect step."""
    rec = _Recorder()
    # Pretend the optional MCP SDK is importable so discover_mcp_tools gets past
    # its gate; everything before register_mcp_servers stays real (config load,
    # env interpolation, the per-profile cross-process discovery lock).
    monkeypatch.setattr(mcp_tool, "_ensure_mcp_sdk", lambda: True)
    monkeypatch.setattr(mcp_tool, "register_mcp_servers", rec)
    return rec


@pytest.fixture
def profiles(tmp_path: Path) -> tuple[Path, Path]:
    home_a = tmp_path / "profile-zug"
    home_b = tmp_path / "profile-carol"
    _write_profile(home_a, _ZUG_ENV)
    _write_profile(home_b, _CAROL_ENV)
    return home_a, home_b


# ---------------------------------------------------------------------------
# 1. The one-shot gate must not suppress a second profile
# ---------------------------------------------------------------------------


def test_second_profile_discovers_while_first_is_still_in_flight(profiles, recorder):
    """Profile A holding the discovery slot must not starve profile B.

    This is the reported failure: Zug's agent builds first, its discovery
    thread is still connecting, and Carol's build returns from
    ``start_background_mcp_discovery`` without ever loading Carol's config.
    """
    home_a, home_b = profiles
    release_a = recorder.gate(home_a)
    a_entered = recorder.entered_event(home_a)

    _start_discovery_for(home_a)
    assert a_entered.wait(timeout=_JOIN_TIMEOUT_S), "profile A never ran discovery"

    _start_discovery_for(home_b)
    b_ran = _wait_for(lambda: _key(home_b) in recorder.keys_seen())
    release_a.set()

    assert b_ran, (
        "profile B was denied startup discovery while profile A's thread was "
        "alive -- the process-global one-shot gate collapsed both profiles"
    )


def test_each_profile_reads_its_own_config(profiles, recorder):
    """A's discovery loads A's password_env; B's loads B's. Never crossed."""
    home_a, home_b = profiles

    _start_discovery_for(home_a)
    _start_discovery_for(home_b)
    _join_all_discovery()

    assert recorder.password_envs_for(home_a) == [_ZUG_ENV]
    assert recorder.password_envs_for(home_b) == [_CAROL_ENV]


def test_profiles_cannot_borrow_each_others_password_env(profiles, recorder):
    """No discovery run may ever see the other profile's credential name."""
    home_a, home_b = profiles

    _start_discovery_for(home_a)
    _start_discovery_for(home_b)
    _join_all_discovery()

    # Both must actually have run -- "saw nothing" would satisfy the
    # non-containment checks below vacuously.
    assert recorder.password_envs_for(home_a)
    assert recorder.password_envs_for(home_b)
    assert _CAROL_ENV not in recorder.password_envs_for(home_a)
    assert _ZUG_ENV not in recorder.password_envs_for(home_b)


def test_repeated_discovery_for_one_profile_dedupes(profiles, recorder):
    """Two builds for the SAME profile share one in-flight discovery run."""
    home_a, home_b = profiles
    release_a = recorder.gate(home_a)
    a_entered = recorder.entered_event(home_a)

    _start_discovery_for(home_a)
    assert a_entered.wait(timeout=_JOIN_TIMEOUT_S)
    first_thread = mcp_startup.discovery_thread_for_profile(str(home_a))

    _start_discovery_for(home_a)
    second_thread = mcp_startup.discovery_thread_for_profile(str(home_a))
    assert second_thread is first_thread, "same profile spawned a duplicate run"

    # Dedup for A must not have consumed B's slot.
    _start_discovery_for(home_b)
    b_ran = _wait_for(lambda: _key(home_b) in recorder.keys_seen())
    release_a.set()
    assert b_ran

    _join_all_discovery()
    assert recorder.password_envs_for(home_a) == [_ZUG_ENV]


def test_failed_profile_does_not_suppress_healthy_profile(profiles, monkeypatch):
    """A raising/parked discovery for A must leave B's discovery working."""
    home_a, home_b = profiles
    seen: list[str] = []
    seen_lock = threading.Lock()

    def _register(servers: dict) -> list:
        key = mcp_profile.current_profile_key()
        if key == _key(home_a):
            raise RuntimeError("simulated connect failure for profile A")
        with seen_lock:
            seen.append(key)
        return []

    monkeypatch.setattr(mcp_tool, "_ensure_mcp_sdk", lambda: True)
    monkeypatch.setattr(mcp_tool, "register_mcp_servers", _register)

    _start_discovery_for(home_a)
    _join_all_discovery()
    _start_discovery_for(home_b)
    _join_all_discovery()

    with seen_lock:
        assert seen == [_key(home_b)]


def test_concurrent_profiles_keep_their_own_scope(profiles, recorder):
    """Both profiles in flight at once still each read their own config."""
    home_a, home_b = profiles
    release_a = recorder.gate(home_a)
    release_b = recorder.gate(home_b)
    a_entered = recorder.entered_event(home_a)
    b_entered = recorder.entered_event(home_b)

    _start_discovery_for(home_a)
    _start_discovery_for(home_b)
    try:
        assert a_entered.wait(timeout=_JOIN_TIMEOUT_S)
        assert b_entered.wait(timeout=_JOIN_TIMEOUT_S)
    finally:
        release_a.set()
        release_b.set()
    _join_all_discovery()

    assert recorder.password_envs_for(home_a) == [_ZUG_ENV]
    assert recorder.password_envs_for(home_b) == [_CAROL_ENV]


# ---------------------------------------------------------------------------
# 2. The discovery thread must carry the caller's secret scope
# ---------------------------------------------------------------------------


def test_discovery_thread_carries_caller_secret_scope(tmp_path, recorder, monkeypatch):
    """``${VAR}`` in a profile's MCP config resolves from THAT profile's scope.

    ContextVars do not cross a bare thread. With multiplexing on and no scope
    re-installed inside the discovery thread, ``get_secret`` fails closed,
    ``_load_mcp_config`` swallows it, and the profile silently discovers
    nothing.
    """
    monkeypatch.setattr(secret_scope_mod, "_MULTIPLEX_ACTIVE", True)

    home_a = tmp_path / "profile-zug"
    home_b = tmp_path / "profile-carol"
    _write_profile(home_a, _ZUG_ENV, url="${TOOLHIVE_URL}")
    _write_profile(home_b, _CAROL_ENV, url="${TOOLHIVE_URL}")

    _start_discovery_for(home_a, {"TOOLHIVE_URL": "https://zug.toolhive.example/mcp"})
    _start_discovery_for(home_b, {"TOOLHIVE_URL": "https://carol.toolhive.example/mcp"})
    _join_all_discovery()

    urls_a = [cfg.get("url") for cfg in recorder.configs_for(home_a)]
    urls_b = [cfg.get("url") for cfg in recorder.configs_for(home_b)]
    assert urls_a == ["https://zug.toolhive.example/mcp"], (
        "profile A's discovery thread lost its secret scope"
    )
    assert urls_b == ["https://carol.toolhive.example/mcp"], (
        "profile B's discovery thread lost its secret scope"
    )


def test_unscoped_caller_stays_unscoped_in_the_thread(tmp_path, recorder, monkeypatch):
    """Single-profile CLI: no scope in, no scope invented in the thread."""
    monkeypatch.setattr(secret_scope_mod, "_MULTIPLEX_ACTIVE", False)
    monkeypatch.setenv("TOOLHIVE_URL", "https://from-process-env.example/mcp")

    home = tmp_path / "profile-solo"
    _write_profile(home, _ZUG_ENV, url="${TOOLHIVE_URL}")

    from tui_gateway import entry as tui_entry

    observed: list[object] = []
    original = mcp_tool._load_mcp_config

    def _spy() -> dict:
        observed.append(secret_scope_mod.current_secret_scope())
        return original()

    monkeypatch.setattr(mcp_tool, "_load_mcp_config", _spy)

    home_token = set_hermes_home_override(str(home))
    try:
        tui_entry.ensure_mcp_discovery_started()
    finally:
        reset_hermes_home_override(home_token)
    _join_all_discovery()

    # Discovery loads the config once, then the post-run status probe loads it
    # again; both must stay unscoped.
    assert observed, "discovery never loaded the config"
    assert all(scope is None for scope in observed), (
        "an unscoped caller must not gain a scope inside the discovery thread"
    )
    urls = [cfg.get("url") for cfg in recorder.configs_for(home)]
    assert urls == ["https://from-process-env.example/mcp"]


# ---------------------------------------------------------------------------
# 3. Coordinator bookkeeping
# ---------------------------------------------------------------------------


def test_single_profile_startup_is_unchanged(tmp_path, recorder):
    """One profile: one thread, exposed under the legacy module globals."""
    home = tmp_path / "profile-solo"
    _write_profile(home, _ZUG_ENV)

    _start_discovery_for(home)
    assert mcp_startup._mcp_discovery_thread is not None
    assert mcp_startup._mcp_discovery_started is True
    _join_all_discovery()

    assert recorder.password_envs_for(home) == [_ZUG_ENV]


def test_waiters_are_profile_scoped(profiles, recorder):
    """``wait_for``/``in_flight``/``join`` answer for the CALLING profile."""
    home_a, home_b = profiles
    release_a = recorder.gate(home_a)
    a_entered = recorder.entered_event(home_a)

    _start_discovery_for(home_a)
    assert a_entered.wait(timeout=_JOIN_TIMEOUT_S)

    token = set_hermes_home_override(str(home_a))
    try:
        assert mcp_startup.mcp_discovery_in_flight() is True
    finally:
        reset_hermes_home_override(token)

    token = set_hermes_home_override(str(home_b))
    try:
        # B never started discovery, so B has nothing in flight and its
        # bounded join returns immediately rather than blocking on A.
        assert mcp_startup.mcp_discovery_in_flight() is False
        assert mcp_startup.join_mcp_discovery(timeout=0.1) is True
    finally:
        reset_hermes_home_override(token)

    release_a.set()
    _join_all_discovery()


def test_shutdown_closes_every_profile_discovery(profiles, recorder):
    """No discovery thread is orphaned once every profile's run completes."""
    home_a, home_b = profiles

    _start_discovery_for(home_a)
    _start_discovery_for(home_b)
    _join_all_discovery()

    assert mcp_startup.discovery_threads() == []
    for home in (home_a, home_b):
        assert mcp_startup.discovery_thread_for_profile(str(home)) is None


def test_zero_connected_retry_is_per_profile(profiles, recorder, caplog):
    """A's completed-but-empty run may retry without touching B's state."""
    home_a, home_b = profiles

    with caplog.at_level(logging.WARNING, logger="tui_gateway.entry"):
        _start_discovery_for(home_a)
        _join_all_discovery()
        # A finished with zero connected servers -> a later build for A is
        # allowed to retry (this is the existing allowance, now per-profile).
        _start_discovery_for(home_a)
        _join_all_discovery()

    assert recorder.password_envs_for(home_a) == [_ZUG_ENV, _ZUG_ENV]
    assert recorder.password_envs_for(home_b) == []


def test_profile_without_mcp_servers_is_done_not_failed(tmp_path, recorder, caplog):
    """"Nothing configured" must not read as "discovery failed" on every call.

    The retry allowance exists for a run that *tried* and came back empty. A
    profile with no ``mcp_servers`` never had a run to fail, and there is
    nothing a retry could find. Since ``GatewayRunner._ensure_profile_mcp_tools``
    reaches this gate once per turn, mistaking the two warned on every message
    such a profile ever received.
    """
    home = tmp_path / "profile-no-mcp"
    home.mkdir()
    (home / "config.yaml").write_text("model: some-model\n", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="tui_gateway.entry"):
        for _ in range(3):
            _start_discovery_for(home)
        _join_all_discovery()

    assert [r.getMessage() for r in caplog.records] == []
    assert recorder.seen == []


# ---------------------------------------------------------------------------
# 3. The coordinator lock must not span the blocking parts of the decision
# ---------------------------------------------------------------------------


def test_slow_config_probe_for_one_profile_does_not_block_another(
    profiles, recorder, monkeypatch
):
    """One profile's config/registry probe must not hold the process lock.

    ``start_background_mcp_discovery`` used to hold the module-global
    ``_mcp_discovery_lock`` across ``_has_configured_mcp_servers()`` (a
    ``config.yaml`` read) and ``_profile_mcp_is_populated()`` (which imports
    the MCP SDK stack and takes ``tools.mcp_tool``'s registry lock). Every
    other profile then blocked on that lock — including
    ``mcp_discovery_in_flight()``, which the multiplexed gateway calls once
    per turn from ``GatewayRunner._ensure_profile_mcp_tools`` while holding
    its own readiness lock. A cold MCP import for profile A therefore stalled
    profile B's turn.

    The decision now runs under a PER-PROFILE lock, so B proceeds while A is
    still probing.

    Entered through ``start_background_mcp_discovery`` rather than
    ``ensure_mcp_discovery_started``: the latter runs its OWN
    ``_has_configured_mcp_servers()`` gate first (``tui_gateway/entry.py``),
    outside any coordinator lock, so blocking there would prove nothing about
    the lock this test is about.
    """
    home_a, home_b = profiles

    a_probing = threading.Event()
    release_a = threading.Event()
    real_probe = mcp_startup._has_configured_mcp_servers

    def _slow_probe_for_a() -> bool:
        from hermes_constants import get_hermes_home

        if str(get_hermes_home()) == str(home_a):
            a_probing.set()
            # Bounded so a regression fails the assertions below rather than
            # hanging the suite.
            release_a.wait(timeout=_JOIN_TIMEOUT_S)
        return real_probe()

    monkeypatch.setattr(
        mcp_startup, "_has_configured_mcp_servers", _slow_probe_for_a
    )

    a_thread = threading.Thread(
        target=_start_coordinator_for, args=(home_a,), daemon=True
    )
    a_thread.start()
    assert a_probing.wait(timeout=_JOIN_TIMEOUT_S), "profile A never reached the probe"

    # A is parked inside its own decision. B must still be able to ask about
    # itself and to start its own run, both without waiting on A.
    b_done = threading.Event()

    def _run_b() -> None:
        token = set_hermes_home_override(str(home_b))
        try:
            assert mcp_startup.mcp_discovery_in_flight() is False
        finally:
            reset_hermes_home_override(token)
        _start_coordinator_for(home_b)
        b_done.set()

    b_thread = threading.Thread(target=_run_b, daemon=True)
    b_thread.start()
    assert b_done.wait(timeout=5.0), (
        "profile B blocked behind profile A's config probe — the coordinator "
        "lock is spanning the blocking part of the decision again"
    )

    release_a.set()
    a_thread.join(timeout=_JOIN_TIMEOUT_S)
    b_thread.join(timeout=_JOIN_TIMEOUT_S)
    _join_all_discovery()

    # Both profiles really did discover, each reading its own config.
    assert {str(home) for home, _ in recorder.seen} == {str(home_a), str(home_b)}


def test_same_profile_still_deduplicates_under_the_split_lock(profiles, recorder):
    """Splitting the lock must not let one profile spawn two runs.

    Same-profile callers serialize on that profile's own lock, so the second
    caller sees the first caller's installed thread and returns.
    """
    home_a, _home_b = profiles
    release_a = recorder.gate(home_a)
    a_entered = recorder.entered_event(home_a)

    _start_discovery_for(home_a)
    assert a_entered.wait(timeout=_JOIN_TIMEOUT_S)

    for _ in range(3):
        _start_discovery_for(home_a)

    assert len(mcp_startup.discovery_threads()) == 1

    release_a.set()
    _join_all_discovery()
    assert [str(home) for home, _ in recorder.seen] == [str(home_a)]
