"""Tests for the TUI gateway's late MCP tool-snapshot refresh.

When an MCP server connects slower than the bounded wait in ``_make_agent``,
the agent is built without its tools and the banner/tool count is stale for the
session. ``_schedule_mcp_late_refresh`` waits for discovery to land, then
rebuilds the snapshot and re-emits ``session.info`` — but only while the
session is still pre-first-turn, so it never invalidates a cached prompt.
"""

import threading
import time
import types

import model_tools
from tui_gateway import server
from tui_gateway import entry


def _make_fake_agent(initial_tools, *, user_turns=0, api_calls=0):
    agent = types.SimpleNamespace()
    agent.tools = list(initial_tools)
    agent.valid_tool_names = {t["function"]["name"] for t in initial_tools}
    agent._user_turn_count = user_turns
    agent._api_call_count = api_calls
    return agent


def _tool(name):
    return {"type": "function", "function": {"name": name, "description": "", "parameters": {}}}


def _drain_refresh_threads(timeout=5.0):
    deadline = time.time() + timeout
    for th in list(threading.enumerate()):
        if th.name.startswith("tui-mcp-late-refresh-"):
            th.join(timeout=max(0.0, deadline - time.time()))


def _install(monkeypatch, *, in_flight, join_result, new_defs):
    """Wire entry discovery accessors + get_tool_definitions, capture emits."""
    monkeypatch.setattr(entry, "mcp_discovery_in_flight", lambda: in_flight)
    monkeypatch.setattr(entry, "join_mcp_discovery", lambda timeout=None: join_result)
    monkeypatch.setattr(model_tools, "get_tool_definitions", lambda **kw: list(new_defs))
    monkeypatch.setattr(server, "_load_enabled_toolsets", lambda *_a, **_kw: None)
    monkeypatch.setattr(server, "_session_info", lambda agent, session: {"tools_len": len(agent.tools)})

    emitted = []
    monkeypatch.setattr(server, "_emit", lambda event, sid, payload=None: emitted.append((event, sid, payload)))
    return emitted


def test_late_refresh_adds_tools_and_reemits_when_pre_first_turn(monkeypatch):
    base = [_tool("read_file"), _tool("write_file")]
    full = base + [_tool("mcp__nous_support__a")]  # discovery added one tool
    agent = _make_fake_agent(base)
    sid = "sess-late-1"
    server._sessions[sid] = {"agent": agent}
    try:
        emitted = _install(monkeypatch, in_flight=True, join_result=True, new_defs=full)
        server._schedule_mcp_late_refresh(sid, agent)
        _drain_refresh_threads()

        assert len(agent.tools) == 3
        assert "mcp__nous_support__a" in agent.valid_tool_names
        assert ("session.info", sid, {"tools_len": 3}) in emitted
    finally:
        server._sessions.pop(sid, None)


def test_no_refresh_when_discovery_not_in_flight(monkeypatch):
    base = [_tool("read_file")]
    agent = _make_fake_agent(base)
    sid = "sess-late-2"
    server._sessions[sid] = {"agent": agent}
    try:
        # in_flight=False → helper returns immediately, no thread, no rebuild.
        emitted = _install(monkeypatch, in_flight=False, join_result=True, new_defs=base + [_tool("x")])
        server._schedule_mcp_late_refresh(sid, agent)
        _drain_refresh_threads()

        assert len(agent.tools) == 1
        assert emitted == []
    finally:
        server._sessions.pop(sid, None)


def test_no_refresh_once_conversation_started(monkeypatch):
    """Cache safety: never rebuild the tool list after the first turn."""
    base = [_tool("read_file")]
    full = base + [_tool("mcp__late__b")]
    agent = _make_fake_agent(base, user_turns=1)  # a turn already happened
    sid = "sess-late-3"
    server._sessions[sid] = {"agent": agent}
    try:
        emitted = _install(monkeypatch, in_flight=True, join_result=True, new_defs=full)
        server._schedule_mcp_late_refresh(sid, agent)
        _drain_refresh_threads()

        # Snapshot frozen; no re-emit that would invalidate the prompt cache.
        assert len(agent.tools) == 1
        assert emitted == []
    finally:
        server._sessions.pop(sid, None)


def test_no_reemit_when_discovery_added_nothing(monkeypatch):
    base = [_tool("read_file"), _tool("write_file")]
    agent = _make_fake_agent(base)
    sid = "sess-late-4"
    server._sessions[sid] = {"agent": agent}
    try:
        # Discovery finished but the registry is unchanged (same count) →
        # don't churn the client with a redundant session.info.
        emitted = _install(monkeypatch, in_flight=True, join_result=True, new_defs=list(base))
        server._schedule_mcp_late_refresh(sid, agent)
        _drain_refresh_threads()

        assert len(agent.tools) == 2
        assert emitted == []
    finally:
        server._sessions.pop(sid, None)


def test_no_refresh_when_join_times_out(monkeypatch):
    base = [_tool("read_file")]
    full = base + [_tool("mcp__slow__c")]
    agent = _make_fake_agent(base)
    sid = "sess-late-5"
    server._sessions[sid] = {"agent": agent}
    try:
        # Server never connected within the bound → join returns False, no rebuild.
        emitted = _install(monkeypatch, in_flight=True, join_result=False, new_defs=full)
        server._schedule_mcp_late_refresh(sid, agent)
        _drain_refresh_threads()

        assert len(agent.tools) == 1
        assert emitted == []
    finally:
        server._sessions.pop(sid, None)


def test_no_refresh_when_session_replaced(monkeypatch):
    """If the session's agent was swapped (e.g. /new) while we waited, bail."""
    base = [_tool("read_file")]
    full = base + [_tool("mcp__late__d")]
    agent = _make_fake_agent(base)
    other_agent = _make_fake_agent(base)
    sid = "sess-late-6"
    server._sessions[sid] = {"agent": agent}
    try:
        emitted = _install(monkeypatch, in_flight=True, join_result=True, new_defs=full)

        # Swap the stored agent out the moment join is awaited.
        def _swap_join(timeout=None):
            server._sessions[sid]["agent"] = other_agent
            return True

        monkeypatch.setattr(entry, "join_mcp_discovery", _swap_join)
        server._schedule_mcp_late_refresh(sid, agent)
        _drain_refresh_threads()

        # Neither agent's snapshot was rebuilt; no emit.
        assert len(agent.tools) == 1
        assert len(other_agent.tools) == 1
        assert emitted == []
    finally:
        server._sessions.pop(sid, None)


def test_late_refresh_worker_keeps_the_sessions_profile_scope(monkeypatch, tmp_path):
    """The refresh thread must run under the SESSION's profile, not the launch one.

    ``_schedule_mcp_late_refresh`` is called from the deferred build thread,
    which has the session profile's ``HERMES_HOME`` override and secret scope
    installed. ContextVars do not cross into a bare ``threading.Thread``, so
    without an explicit replay the worker ran under the launch profile — and
    every question it asks is then asked about the wrong profile: which
    discovery thread to join (``_thread_for_current_profile`` keys on the
    ambient home) and which registry ``refresh_agent_mcp_tools`` reads. On a
    multiplexed desktop gateway that either finds nothing or splices another
    profile's MCP tools into this session's agent.
    """
    from agent.secret_scope import current_secret_scope, set_secret_scope, reset_secret_scope
    from hermes_constants import (
        get_hermes_home,
        reset_hermes_home_override,
        set_hermes_home_override,
    )

    profile_home = tmp_path / "carol"
    profile_home.mkdir()
    (profile_home / ".env").write_text("TOOLHIVE_TOKEN=carol-only\n", encoding="utf-8")

    seen: dict = {}

    base = [_tool("read_file")]
    full = base + [_tool("mcp__toolhive__ping")]
    agent = _make_fake_agent(base)
    sid = "sess-late-scope"
    server._sessions[sid] = {"agent": agent}

    home_token = None
    secret_token = None
    try:
        emitted = _install(monkeypatch, in_flight=True, join_result=True, new_defs=full)

        def _observing_join(timeout=None):
            seen["home"] = str(get_hermes_home())
            scope = current_secret_scope()
            seen["secret_scope"] = dict(scope) if scope else None
            return True

        monkeypatch.setattr(entry, "join_mcp_discovery", _observing_join)

        # Stand in for the deferred build thread's bound profile scope.
        home_token = set_hermes_home_override(str(profile_home))
        secret_token = set_secret_scope({"TOOLHIVE_TOKEN": "carol-only"})

        server._schedule_mcp_late_refresh(sid, agent)
        _drain_refresh_threads()
    finally:
        if secret_token is not None:
            reset_secret_scope(secret_token)
        if home_token is not None:
            reset_hermes_home_override(home_token)
        server._sessions.pop(sid, None)

    assert seen.get("home") == str(profile_home), (
        "the late-refresh worker ran under the launch profile's HERMES_HOME"
    )
    assert seen.get("secret_scope") == {"TOOLHIVE_TOKEN": "carol-only"}
    assert ("session.info", sid, {"tools_len": 2}) in emitted
