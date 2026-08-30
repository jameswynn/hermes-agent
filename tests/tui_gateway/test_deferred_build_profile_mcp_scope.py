"""The deferred TUI/desktop agent build must expose ITS profile's MCP tools.

Companion to ``tests/gateway/test_multiplex_mcp_discovery_profile_scope.py``,
which covers the *messaging* gateway (``GatewayRunner._run_agent``). This file
covers the other production surface — the desktop/TUI path:

    ``session.create`` (methods_session) → ``_schedule_agent_build``
    → ``server._start_agent_build`` (deferred build THREAD)
    → ``tui_gateway.entry.ensure_mcp_discovery_started``
    → ``hermes_cli.mcp_startup.start_background_mcp_discovery``
    → the tool snapshot ``AIAgent`` takes at construction time.

Why this needed its own coverage: the existing deferred-build tests in
``tests/test_tui_gateway_server.py``
(``test_profile_scoped_agent_build_starts_mcp_discovery_in_profile_home`` and
``..._installs_secret_scope``) stub BOTH ``ensure_mcp_discovery_started`` and
``_make_agent``. They prove the build thread *binds* the right profile home and
secret scope, but nothing downstream of that binding runs — so they cannot see
whether discovery actually populated this profile's registry, nor whether the
resulting tool survives ``_make_check_fn``'s availability filter into the
snapshot. That filter is exactly where the live failure was silent: a tool
registered under one profile is dropped for another with no error and no log
line.

Everything here runs against temp profile homes and a deterministic fake
discovery. No network, no real credentials, no live state.
"""

from __future__ import annotations

import threading
import uuid
from pathlib import Path

import pytest

from hermes_cli import mcp_startup
from tools import mcp_profile, mcp_tool
from tools.registry import registry
from tui_gateway import server

# Both profiles configure the SAME logical server name — the collapse that made
# "some profile discovered toolhive" look like "THIS profile discovered it".
_SERVER = "toolhive"
_TOOL = f"mcp__{_SERVER}__ping"

_JOIN_TIMEOUT_S = 10.0
_BUILD_TIMEOUT_S = 20.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_profile(home: Path, url: str) -> None:
    """A realistic profile home: config.yaml only, no .env, no secrets."""
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        f"""
mcp_servers:
  {_SERVER}:
    url: {url}
    connect_timeout: 1
""".lstrip(),
        encoding="utf-8",
    )


def _key(home: Path) -> str:
    return mcp_profile.canonical_profile_key(str(home))


def _join_all_discovery(timeout: float = _JOIN_TIMEOUT_S) -> None:
    for thread in mcp_startup.discovery_threads():
        thread.join(timeout=timeout)


def _visible_tools() -> set[str]:
    """The catalog ``AIAgent`` snapshots under the ambient profile scope.

    ``skip_tool_search_assembly=True`` reads the real availability-filtered
    catalog rather than the tool_search/tool_describe bridge the assembly step
    may collapse it into — that collapse is a presentation layer over these
    same tools, so it would hide the fact under test.
    """
    from model_tools import get_tool_definitions

    return {
        t["function"]["name"]
        for t in get_tool_definitions(quiet_mode=True, skip_tool_search_assembly=True)
        if isinstance(t, dict) and isinstance(t.get("function"), dict)
    }


class _FakeDiscovery:
    """Stands in for ``tools.mcp_tool.discover_mcp_tools``.

    Real up to the transport: loads the ambient profile's config through the
    REAL ``_load_mcp_config`` (so a wrong ``HERMES_HOME`` or a missing secret
    scope surfaces here), then registers one tool per configured server into
    the REAL process-global registry behind the REAL profile-scoped
    ``_make_check_fn`` availability probe.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.seen: list[tuple[str, dict]] = []

    def __call__(self) -> list[str]:
        key = mcp_profile.current_profile_key()
        servers = mcp_tool._load_mcp_config()
        with self.lock:
            self.seen.append((key, dict(servers.get(_SERVER) or {})))
        names: list[str] = []
        reg = mcp_profile.current_registry()
        for name, cfg in servers.items():
            # Mirror lazy registration: available to THIS profile, first call
            # would connect it — which is what ``_make_check_fn`` reports.
            reg.lazy_server_configs[name] = dict(cfg)
            tool_name = f"mcp__{name}__ping"
            registry.register(
                name=tool_name,
                toolset=f"mcp-{name}",
                schema={
                    "name": tool_name,
                    "description": f"Fake tool from MCP server '{name}'",
                    "parameters": {"type": "object", "properties": {}},
                },
                handler=lambda **_kw: "",
                check_fn=mcp_tool._make_check_fn(name),
                is_async=False,
                description=f"Fake tool from MCP server '{name}'",
            )
            registry.register_toolset_alias(name, f"mcp-{name}")
            mcp_tool._track_mcp_tool_server(tool_name, name)
            names.append(tool_name)
        return names

    def urls_for(self, home: Path) -> list[str]:
        with self.lock:
            return [cfg.get("url") for key, cfg in self.seen if key == _key(home)]


def _run_deferred_build(monkeypatch, profile_home: Path) -> dict:
    """Drive the REAL ``_start_agent_build`` on its real deferred thread.

    ``_make_agent`` stands in for ``AIAgent`` construction and records exactly
    what the agent would snapshot at that instant: the bound ``HERMES_HOME``,
    the installed secret scope, and the availability-filtered tool catalog.
    Everything upstream of it — the home/secret binding, the real
    ``ensure_mcp_discovery_started``, the real per-profile coordinator, the
    real bounded discovery join — runs unmodified.
    """
    observed: dict = {}
    built = threading.Event()

    def _fake_make_agent(*_a, **_kw):
        from agent.secret_scope import current_secret_scope
        from hermes_constants import get_hermes_home

        # The REAL ``_make_agent`` joins both discovery owners, bounded by
        # ``mcp_discovery_timeout``, immediately before constructing AIAgent —
        # ``ensure_mcp_discovery_started`` only *spawns* the thread. That join
        # is the thing that decides whether a just-discovered tool makes the
        # snapshot, so it is part of the path under test and is replicated
        # here rather than stubbed away.
        from hermes_cli.mcp_startup import wait_for_mcp_discovery as _startup_wait
        from tui_gateway.entry import wait_for_mcp_discovery as _entry_wait

        _startup_wait()
        _entry_wait()

        observed["home"] = str(get_hermes_home())
        scope = current_secret_scope()
        observed["secret_scope"] = dict(scope) if scope else None
        observed["profile_key"] = mcp_profile.current_profile_key()
        observed["tools"] = _visible_tools()
        built.set()
        return type("Agent", (), {"model": "test"})()

    monkeypatch.setattr(server, "_make_agent", _fake_make_agent)
    # Trim everything off the build thread that is orthogonal to MCP scope;
    # the MCP path itself (ensure_mcp_discovery_started → mcp_startup) is left
    # entirely real. Mirrors the existing deferred-build tests.
    monkeypatch.setattr(server, "_wire_callbacks", lambda _sid: None)
    monkeypatch.setattr(server, "_attach_worker", lambda *a, **k: None)
    monkeypatch.setattr(server, "_config_model_target", lambda: ("", ""))
    monkeypatch.setattr(server, "_start_notification_poller", lambda *a, **k: None)
    monkeypatch.setattr(server, "_schedule_mcp_late_refresh", lambda *a, **k: None)
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    monkeypatch.setattr(server, "_open_profile_session_db", lambda _h: None)

    ready = threading.Event()
    sid = f"mcp-scope-{uuid.uuid4().hex[:8]}"
    session = {
        "agent_ready": ready,
        "session_key": f"mcp-scope-key-{uuid.uuid4().hex[:8]}",
        "profile_home": str(profile_home),
    }

    # Spec Phase 1 step 2: the in-memory session must already carry the right
    # profile home BEFORE the build thread runs. A build that has to guess is
    # the failure mode this whole path exists to prevent.
    assert session["profile_home"] == str(profile_home)

    server._sessions[sid] = session
    try:
        server._start_agent_build(sid, session)
        assert built.wait(timeout=_BUILD_TIMEOUT_S), "build never reached _make_agent"
        assert ready.wait(timeout=5), "agent_ready never set after build"
    finally:
        server._sessions.pop(sid, None)
    _join_all_discovery()
    return observed


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_mcp_state():
    """Leave no coordinator entry, profile registry, or tool registration."""
    from agent.secret_scope import is_multiplex_active, set_multiplex_active
    from tui_gateway import entry as tui_entry

    was_multiplex = is_multiplex_active()
    # Running the REAL ``ensure_mcp_discovery_started`` latches this module
    # global True for the rest of the process, which changes whether
    # ``entry.wait_for_mcp_discovery`` delegates to the shared owner. Restore
    # it, or this file silently rewrites the behaviour of every later test
    # that exercises that wait (e.g. test_make_agent_waits_for_shared_mcp_
    # discovery in tests/test_tui_gateway_server.py).
    was_discovery_enabled = tui_entry._mcp_discovery_enabled
    mcp_startup.reset_discovery_state()
    mcp_profile.reset_all_registries()
    # The desktop gateway serves every profile from one process; this is the
    # flag ``registry.check_fn_cache_scope`` reads to keep tool-availability
    # caching profile-keyed. Without it one profile's ``mcp__…`` entry would
    # answer for every other profile — the isolation asserted below.
    set_multiplex_active(True)
    try:
        yield
    finally:
        _join_all_discovery()
        mcp_startup.reset_discovery_state()
        mcp_profile.reset_all_registries()
        for tool_name in list(registry.get_all_tool_names()):
            if tool_name.startswith("mcp__"):
                registry.deregister(tool_name)
        tui_entry._mcp_discovery_enabled = was_discovery_enabled
        set_multiplex_active(was_multiplex)


@pytest.fixture
def discovery(monkeypatch) -> _FakeDiscovery:
    fake = _FakeDiscovery()
    monkeypatch.setattr(mcp_tool, "discover_mcp_tools", fake)
    return fake


@pytest.fixture
def profiles(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    home_a = tmp_path / "jonathon"
    home_b = tmp_path / "carol"
    _write_profile(home_a, "https://toolhive-a.example/mcp")
    _write_profile(home_b, "https://toolhive-b.example/mcp")
    # Launch profile is a THIRD home with no mcp_servers at all — the live
    # shape, and the one that used to starve selected profiles of discovery.
    launch = tmp_path / "launch"
    launch.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(launch))
    return home_a, home_b


# ---------------------------------------------------------------------------
# 1. Session creation records the profile home the build will need
# ---------------------------------------------------------------------------


def test_session_create_records_profile_home_before_the_build(
    profiles, monkeypatch, tmp_path
):
    """``session.create`` must resolve ``profile`` → ``profile_home`` up front.

    Driven through the REAL handler and the REAL ``server._profile_home``
    resolution (only the profile-directory lookup is redirected at temp
    homes), with the deferred build intercepted so we observe the session dict
    exactly as ``_start_agent_build`` would receive it.
    """
    home_a, _home_b = profiles
    scheduled: list[str] = []
    monkeypatch.setattr(server, "_schedule_agent_build", lambda sid, *a, **k: scheduled.append(sid))
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda *a, **k: None)
    monkeypatch.setattr(
        "hermes_cli.profiles.get_profile_dir", lambda name: str(tmp_path / name)
    )

    handler = server._methods["session.create"]
    resp = handler(1, {"profile": "jonathon", "cols": 80})
    sid = resp["result"]["session_id"]

    try:
        assert scheduled == [sid], "the build must be deferred, not run inline"
        session = server._sessions[sid]
        assert session["profile_home"] == str(home_a)
        assert session["agent"] is None, "agent must not exist before the build"
    finally:
        server._sessions.pop(sid, None)


# ---------------------------------------------------------------------------
# 2. The live failure, on the deferred build path
# ---------------------------------------------------------------------------


def test_deferred_build_exposes_the_profiles_mcp_tool_in_the_snapshot(
    profiles, discovery, monkeypatch
):
    """The reported symptom: a fresh profile session with no MCP tool at all.

    Asserts the whole chain in one shot — discovery ran under the SELECTED
    profile's home, read THAT profile's config, and the resulting tool
    survived ``_make_check_fn`` into the snapshot the agent takes.
    """
    home_a, _home_b = profiles

    observed = _run_deferred_build(monkeypatch, home_a)

    assert observed["home"] == str(home_a)
    assert observed["profile_key"] == _key(home_a)
    assert discovery.urls_for(home_a) == ["https://toolhive-a.example/mcp"]
    assert _TOOL in observed["tools"], (
        "the selected profile's MCP tool is missing from the tool snapshot — "
        "this is the live 'no tool, no error, no log line' failure"
    )


def test_deferred_build_binds_the_profiles_secret_scope(
    profiles, discovery, monkeypatch
):
    """Discovery and the agent must resolve secrets from THIS profile's .env.

    Without the scope, ``${VAR}`` refs in the profile's MCP config resolve
    through an unscoped ``get_secret``; under multiplexing that fails closed
    and ``_load_mcp_config`` swallows it into an EMPTY config, so the profile
    silently discovers nothing.
    """
    home_a, _home_b = profiles
    (home_a / ".env").write_text("TOOLHIVE_TOKEN=a-only\n", encoding="utf-8")

    observed = _run_deferred_build(monkeypatch, home_a)

    assert observed["secret_scope"] == {"TOOLHIVE_TOKEN": "a-only"}


# ---------------------------------------------------------------------------
# 3. Isolation between two profiles sharing one server name
# ---------------------------------------------------------------------------


def test_two_profiles_build_independently_through_the_deferred_path(
    profiles, discovery, monkeypatch
):
    """Same server name, different configs — each build must read its own."""
    home_a, home_b = profiles

    first = _run_deferred_build(monkeypatch, home_a)
    second = _run_deferred_build(monkeypatch, home_b)

    assert discovery.urls_for(home_a) == ["https://toolhive-a.example/mcp"]
    assert discovery.urls_for(home_b) == ["https://toolhive-b.example/mcp"]
    assert first["profile_key"] != second["profile_key"]
    assert _TOOL in first["tools"]
    assert _TOOL in second["tools"]


def test_a_profile_that_never_discovered_sees_no_mcp_tool(
    profiles, discovery, monkeypatch
):
    """The isolation has teeth: A's registration must not answer for B.

    B never builds, so its registry stays empty. The process-global tool
    registry still holds the shared ``mcp__toolhive__ping`` NAME from A's
    discovery, so this asserts the availability probe — not the name — is what
    decides, which is the same filter that produced the live silence.
    """
    home_a, home_b = profiles

    _run_deferred_build(monkeypatch, home_a)

    with mcp_profile.profile_scope(home_b):
        assert _TOOL not in _visible_tools()


# ---------------------------------------------------------------------------
# 4. Lazy first-use keeps the originating profile's identity
# ---------------------------------------------------------------------------


def test_lazy_first_use_connects_with_the_originating_profiles_config(
    profiles, discovery, monkeypatch
):
    """A deferred first call must spawn THIS profile's server, not another's.

    Both profiles register a lazy ``toolhive``; the connect attempt made under
    each scope must carry that profile's own URL. A cross-profile leak here
    would send one profile's traffic (and credentials) to the other's
    endpoint.
    """
    home_a, home_b = profiles
    _run_deferred_build(monkeypatch, home_a)
    _run_deferred_build(monkeypatch, home_b)

    attempts: list[tuple[str, str]] = []

    async def _fake_connect(name, config):
        attempts.append((mcp_profile.current_profile_key(), config.get("url")))
        return None

    monkeypatch.setattr(mcp_tool, "_discover_and_register_server", _fake_connect)

    for home in (home_a, home_b):
        with mcp_profile.profile_scope(home):
            mcp_tool._ensure_lazy_server_connected(_SERVER)

    assert attempts == [
        (_key(home_a), "https://toolhive-a.example/mcp"),
        (_key(home_b), "https://toolhive-b.example/mcp"),
    ]
