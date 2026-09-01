"""Regression test for /reload-mcp refreshing cached agent tool lists.

Before this fix, the gateway's _execute_mcp_reload reconnected MCP servers
and updated the global _servers registry, but cached AIAgent instances kept
their original tools list. Users had to run /new (discarding conversation
history) for the agent to pick up the new tools.

This test exercises _execute_mcp_reload directly with mocked MCP discovery
and asserts that every cached agent's `tools` and `valid_tool_names`
attributes are overwritten with the freshly-discovered tool set.
"""

from __future__ import annotations

from collections import OrderedDict
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event() -> MessageEvent:
    return MessageEvent(text="/reload-mcp", source=_make_source(), message_id="m1")


def _make_runner_with_cached_agents(num_agents: int = 2):
    """Build a bare GatewayRunner with `num_agents` fake cached agents."""
    import threading

    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )

    # Session store stub — _execute_mcp_reload writes a transcript message
    # at the end; tests don't care about that side effect.
    session_entry = SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.append_to_transcript = MagicMock()

    # Build N fake cached agents with stale `tools` + `valid_tool_names`.
    runner._agent_cache = OrderedDict()
    runner._agent_cache_lock = threading.Lock()
    for i in range(num_agents):
        stale_tool = {
            "type": "function",
            "function": {"name": f"stale_tool_{i}", "description": "old"},
        }
        agent = SimpleNamespace(
            tools=[stale_tool],
            valid_tool_names={f"stale_tool_{i}"},
            enabled_toolsets=None,
            disabled_toolsets=None,
        )
        runner._agent_cache[f"session-{i}"] = (agent, f"sig-{i}")

    return runner


@pytest.mark.asyncio
async def test_reload_mcp_refreshes_cached_agent_tools():
    """After /reload-mcp succeeds, every cached agent gets its tool list
    replaced with the freshly-discovered set."""
    runner = _make_runner_with_cached_agents(num_agents=3)

    # Snapshot the stale state so we can assert it changed.
    pre_reload_tools = {
        key: list(entry[0].tools) for key, entry in runner._agent_cache.items()
    }

    # Fresh tools that get_tool_definitions() will return after the reload.
    fresh_tool_defs = [
        {
            "type": "function",
            "function": {"name": "HassTurnOn", "description": "Turns on a device"},
        },
        {
            "type": "function",
            "function": {"name": "HassTurnOff", "description": "Turns off a device"},
        },
    ]

    with (
        patch("tools.mcp_tool.shutdown_mcp_servers"),
        patch("tools.mcp_tool.discover_mcp_tools", return_value=["HassTurnOn", "HassTurnOff"]),
        patch.dict("tools.mcp_tool._servers", {"homeassistant": object()}, clear=True),
        patch("model_tools.get_tool_definitions", return_value=fresh_tool_defs),
    ):
        result = await runner._execute_mcp_reload(_make_event())

    # The reload itself returned a status string (not an exception).
    assert isinstance(result, str)

    # Every cached agent has fresh tools and the matching valid_tool_names.
    expected_names = {"HassTurnOn", "HassTurnOff"}
    for key, (agent, _sig) in runner._agent_cache.items():
        assert agent.tools == fresh_tool_defs, (
            f"Agent {key} kept stale tools: {agent.tools} != {fresh_tool_defs}"
        )
        assert agent.valid_tool_names == expected_names, (
            f"Agent {key} kept stale valid_tool_names: {agent.valid_tool_names}"
        )
        # Sanity check that the swap actually changed something.
        assert agent.tools != pre_reload_tools[key]


@pytest.mark.asyncio
async def test_reload_mcp_handles_empty_agent_cache():
    """Reload with no cached agents (e.g. fresh gateway) must not raise."""
    runner = _make_runner_with_cached_agents(num_agents=0)
    assert len(runner._agent_cache) == 0

    with (
        patch("tools.mcp_tool.shutdown_mcp_servers"),
        patch("tools.mcp_tool.discover_mcp_tools", return_value=[]),
        patch.dict("tools.mcp_tool._servers", {}, clear=True),
        patch("model_tools.get_tool_definitions", return_value=[]),
    ):
        result = await runner._execute_mcp_reload(_make_event())

    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_reload_mcp_preserves_per_agent_toolset_overrides():
    """If a cached agent was built with enabled_toolsets=["safe"], the
    refresh must pass that same list to get_tool_definitions so the agent
    doesn't silently gain disabled tools after a reload."""
    runner = _make_runner_with_cached_agents(num_agents=1)
    # Override the toolsets on the cached agent.
    agent, _sig = runner._agent_cache["session-0"]
    agent.enabled_toolsets = ["safe"]
    agent.disabled_toolsets = ["terminal"]

    captured_calls = []

    def _capture_get_tool_definitions(**kwargs):
        captured_calls.append(kwargs)
        return [{"type": "function", "function": {"name": "refreshed"}}]

    with (
        patch("tools.mcp_tool.shutdown_mcp_servers"),
        patch("tools.mcp_tool.discover_mcp_tools", return_value=["refreshed"]),
        patch.dict("tools.mcp_tool._servers", {"homeassistant": object()}, clear=True),
        patch("model_tools.get_tool_definitions", side_effect=_capture_get_tool_definitions),
    ):
        await runner._execute_mcp_reload(_make_event())

    assert captured_calls, "get_tool_definitions was never called to refresh the cache"
    assert captured_calls[0]["enabled_toolsets"] == ["safe"]
    assert captured_calls[0]["disabled_toolsets"] == ["terminal"]


@pytest.mark.asyncio
async def test_reload_mcp_scopes_shutdown_to_calling_profile(monkeypatch, tmp_path):
    """A multiplexed reload must tear down only the calling profile."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")},
        multiplex_profiles=True,
    )
    runner._agent_cache = OrderedDict()
    runner._agent_cache_lock = __import__("threading").Lock()
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key="agent:anton:matrix:dm:c1",
        session_id="sess-anton",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner.session_store.append_to_transcript = MagicMock()
    runner._profile_target_for_source = MagicMock(
        return_value=("anton", tmp_path / "anton")
    )

    calls = []

    def fake_shutdown(*, scope=None, reset_cooldown_for=None):
        calls.append(("shutdown", scope, set(reset_cooldown_for or ())))
        return ["toolhive"]

    with (
        patch("tools.mcp_tool.shutdown_mcp_servers", side_effect=fake_shutdown),
        patch("tools.mcp_tool.discover_mcp_tools", return_value=[]),
        patch.dict("tools.mcp_tool._servers", {}, clear=True),
        patch.dict("tools.mcp_tool._server_scopes", {"toolhive": str(tmp_path / "anton")}, clear=True),
        patch("tools.mcp_tool.mcp_scope_key", return_value=str(tmp_path / "anton")),
        patch("tools.mcp_tool.mcp_servers_in_scope", return_value=set()),
        patch("model_tools.get_tool_definitions", return_value=[]),
    ):
        result = await runner._execute_mcp_reload(_make_event())

    assert "anton" in result
    assert calls == [("shutdown", str(tmp_path / "anton"), set())]


@pytest.mark.asyncio
async def test_refresh_profile_agent_tools_does_not_touch_other_profile(monkeypatch):
    """Refreshing Anton cannot replace a different profile's snapshot."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    import threading

    runner._agent_cache = OrderedDict()
    runner._agent_cache_lock = threading.Lock()
    anton = SimpleNamespace(tools=["old-a"], valid_tool_names={"old-a"})
    carol = SimpleNamespace(tools=["old-c"], valid_tool_names={"old-c"})
    runner._agent_cache["agent:anton:matrix:dm:a"] = (anton, "a")
    runner._agent_cache["agent:carol:matrix:dm:c"] = (carol, "c")
    runner._is_session_running = MagicMock(return_value=False)

    with patch(
        "tools.mcp_tool.refresh_agent_mcp_tools",
        side_effect=lambda agent, quiet_mode=True: agent.tools.__setitem__(0, "new") or True,
    ) as refresh:
        refreshed, skipped = runner._refresh_profile_agent_tools("anton")

    assert (refreshed, skipped) == (1, 0)
    assert anton.tools == ["new"]
    assert carol.tools == ["old-c"]
    refresh.assert_called_once_with(anton, quiet_mode=True)


@pytest.mark.asyncio
async def test_refresh_profile_agent_tools_skips_active_turn():
    """Reload does not mutate an agent while it is in a live turn."""
    from gateway.run import GatewayRunner
    import threading

    runner = object.__new__(GatewayRunner)
    runner._agent_cache = OrderedDict()
    runner._agent_cache_lock = threading.Lock()
    agent = SimpleNamespace(tools=["old"], valid_tool_names={"old"})
    runner._agent_cache["agent:anton:matrix:dm:a"] = (agent, "a")
    runner._is_session_running = MagicMock(return_value=True)

    with patch("tools.mcp_tool.refresh_agent_mcp_tools") as refresh:
        refreshed, skipped = runner._refresh_profile_agent_tools("anton")

    assert (refreshed, skipped) == (0, 1)
    assert agent.tools == ["old"]
    refresh.assert_not_called()
