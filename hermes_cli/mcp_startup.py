"""Shared CLI/TUI-safe helpers for background MCP discovery.

The multiplexed gateway serves many Hermes profiles from ONE process, so the
coordinator state below is keyed by **canonical profile home** rather than
being a single process-wide slot. ``tools/mcp_profile.py`` made the MCP
*runtime* state (server sessions, cooldowns, tool ownership) profile-scoped;
this module is the thing that decides whether discovery runs at all, and it
has to be scoped the same way or the first profile to reach the gate owns it
and every other profile's registry is never populated.

The key is computed the same way ``tools.mcp_profile.canonical_profile_key``
computes it (``hermes_home_key()`` over the resolved ``HERMES_HOME``), but via
``hermes_constants`` directly so the non-MCP startup path never imports the
MCP stack just to look up its own slot.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager, nullcontext
from typing import Dict, List, Optional, Tuple

_mcp_discovery_lock = threading.Lock()

# Legacy process-global mirror of the MOST RECENTLY started discovery run.
# Retained because external callers (and several tests) read/inject these
# directly; the per-profile map below is what actually gates the work. See
# ``_thread_for_current_profile`` for how an injected legacy thread is told
# apart from one this module spawned for a specific profile.
_mcp_discovery_started = False
_mcp_discovery_thread: Optional[threading.Thread] = None


class _ProfileDiscovery:
    """Background-discovery coordinator state owned by exactly one profile.

    ``lock`` serializes the *decision* to spawn (the config probe, the
    populated-registry probe, and the thread install) for THIS profile only.
    It exists so those steps no longer run under the module-global
    ``_mcp_discovery_lock``: they read ``config.yaml`` from disk and import
    the MCP SDK stack, and holding a process-wide lock across that blocked
    every other profile — including the per-turn ``mcp_discovery_in_flight()``
    probe on the gateway's hot path, which needs the global lock only to read
    one dict entry.

    Lock order is always ``lock`` → ``_mcp_discovery_lock`` and never the
    reverse; no code path takes a profile lock while holding the global one.
    """

    __slots__ = ("home", "started", "thread", "lock", "evaluating")

    def __init__(self, home: str) -> None:
        self.home = home
        self.started = False
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        # True while a caller is between "decided to look" and "installed a
        # thread". Keeps the pruner from evicting a state that is mid-decision
        # (its ``thread`` is still None, so it would otherwise look finished).
        self.evaluating = False


# canonical profile key -> that profile's coordinator state.
_profile_discovery: Dict[str, _ProfileDiscovery] = {}

# A normal deployment has a handful of profiles, but ephemeral profile homes
# (one per session) would otherwise accumulate forever in a long-lived
# gateway. Above this many entries, finished ones are dropped; the worst case
# for a dropped entry is that its profile re-runs discovery.
_PROFILE_DISCOVERY_MAX = 256


def _current_profile() -> Tuple[str, str]:
    """Return ``(canonical_key, resolved_home)`` for the CALLING context.

    Falls back to a shared empty key if home resolution fails, which degrades
    to the historical single-slot behavior rather than crashing startup.
    """
    try:
        from hermes_constants import get_hermes_home, hermes_home_key

        home = str(get_hermes_home())
        return hermes_home_key(home), home
    except Exception:
        return "", ""


def _prune_finished_locked() -> None:
    """Drop coordinator entries with no live thread. Caller holds the lock.

    An entry whose owner is mid-decision (``evaluating``) is kept even though
    its ``thread`` is still ``None``: evicting it would let the next caller
    for that profile build a SECOND ``_ProfileDiscovery`` with its own lock,
    and the two would each spawn a discovery thread for the same profile.
    """
    if len(_profile_discovery) < _PROFILE_DISCOVERY_MAX:
        return
    for key in [
        key
        for key, state in _profile_discovery.items()
        if not state.evaluating
        and (state.thread is None or not state.thread.is_alive())
    ]:
        _profile_discovery.pop(key, None)


def _thread_for_current_profile() -> Optional[threading.Thread]:
    """Return the discovery thread the CALLING profile should wait on.

    A profile with its own coordinator entry always gets its own thread (or
    ``None``): profile B must never block on, or report itself in flight
    because of, profile A's discovery.

    When the calling profile has no entry, fall back to the legacy module
    global ONLY if that thread was injected from outside (no profile owns it).
    ``tui_gateway.entry`` and the late-refresh owner probes rely on that
    fallback; a thread this module spawned belongs to one profile and is
    deliberately invisible to the others. Either way this is a *wait* target
    only -- it grants no access to another profile's registry.
    """
    key, _home = _current_profile()
    with _mcp_discovery_lock:
        state = _profile_discovery.get(key)
        if state is not None:
            return state.thread
        legacy = _mcp_discovery_thread
        if legacy is None:
            return None
        if any(s.thread is legacy for s in _profile_discovery.values()):
            return None
        return legacy


def _capture_caller_scope() -> Tuple[Optional[str], Optional[object]]:
    """Snapshot the caller's profile scope for replay inside the thread."""
    try:
        from hermes_constants import get_hermes_home_override

        home_override = get_hermes_home_override()
    except Exception:
        home_override = None
    try:
        from agent.secret_scope import current_secret_scope

        secret_scope = current_secret_scope()
    except Exception:
        secret_scope = None
    return home_override, secret_scope


@contextmanager
def _caller_profile_scope(home_override, secret_scope):
    """Re-install the caller's profile scope inside the discovery thread.

    ContextVars do not propagate into a bare ``threading.Thread``, so both
    halves of the profile identity have to be replayed here:

    * the ``HERMES_HOME`` override, or discovery reads the LAUNCH profile's
      ``mcp_servers`` instead of the selected profile's (#67605);
    * the **secret scope**, or every ``${VAR}`` in that profile's MCP config
      resolves through ``agent.secret_scope.get_secret`` with no scope. Under
      multiplexing that fails closed with ``UnscopedSecretError``, which
      ``tools.mcp_tool._load_mcp_config`` swallows into an EMPTY config -- the
      profile silently discovers nothing. The same scope is what the
      service-account token exchange needs when discovery hands work to the
      shared MCP event loop (``_wrap_with_profile_scope`` propagates whatever
      scope is active on the scheduling thread, which is this one).

    A caller with no scope installs none: single-profile CLI/TUI behavior is
    unchanged, and we never invent a scope that would mask a missing one.
    """
    home_token = None
    secret_token = None
    try:
        from hermes_constants import set_hermes_home_override

        home_token = set_hermes_home_override(home_override)
    except Exception:
        home_token = None
    if secret_scope is not None:
        try:
            from agent.secret_scope import set_secret_scope

            secret_token = set_secret_scope(secret_scope)
        except Exception:
            secret_token = None
    try:
        yield
    finally:
        if secret_token is not None:
            try:
                from agent.secret_scope import reset_secret_scope

                reset_secret_scope(secret_token)
            except Exception:
                pass
        if home_token is not None:
            try:
                from hermes_constants import reset_hermes_home_override

                reset_hermes_home_override(home_token)
            except Exception:
                pass


# Public aliases. Any surface that hands MCP work to a bare ``threading.Thread``
# needs exactly this capture/replay pair — ``tui_gateway.server``'s late tool
# refresh does — and duplicating the fail-open import dance there would just be
# a second place to get it subtly wrong.
capture_caller_scope = _capture_caller_scope
caller_profile_scope = _caller_profile_scope


def discovery_threads() -> List[threading.Thread]:
    """Return every live discovery thread this module owns, across profiles.

    Process-wide teardown/diagnostics: a shutdown path must be able to reach
    all of them, not just the ambient profile's.
    """
    with _mcp_discovery_lock:
        threads = [
            state.thread
            for state in _profile_discovery.values()
            if state.thread is not None
        ]
        if _mcp_discovery_thread is not None and not any(
            t is _mcp_discovery_thread for t in threads
        ):
            threads.append(_mcp_discovery_thread)
    return [t for t in threads if t.is_alive()]


def discovery_thread_for_profile(home) -> Optional[threading.Thread]:
    """Return the discovery thread owned by *home*, or None."""
    try:
        from hermes_constants import hermes_home_key

        key = hermes_home_key(str(home))
    except Exception:
        return None
    with _mcp_discovery_lock:
        state = _profile_discovery.get(key)
        return state.thread if state is not None else None


def reset_discovery_state() -> None:
    """Forget all coordinator state for every profile.

    Teardown helper: does NOT join running threads (they are daemons and
    detach cleanly -- each one only touches the state object it captured).
    """
    global _mcp_discovery_started, _mcp_discovery_thread
    with _mcp_discovery_lock:
        _profile_discovery.clear()
        _mcp_discovery_started = False
        _mcp_discovery_thread = None


def _profile_mcp_is_populated() -> bool:
    """True when the CALLING profile ended a discovery run with usable tools.

    "Usable" is deliberately broader than "connected": a server registered
    from the on-disk schema cache (``lazy``, #56832) has real, callable tools
    in the registry and no live session — ``get_mcp_status`` reports it as
    ``configured``, not ``connected``. Gating the retry allowance on
    connection state alone therefore treated every lazily-registered profile
    as a failed discovery and re-ran the whole pass on every subsequent agent
    build, with a WARNING each time. Connection status and cache-backed
    availability are separate facts; this asks the question the retry actually
    cares about.

    Raises nothing: callers treat an unreadable state as "cannot tell".
    """
    from tools.mcp_tool import get_mcp_status, get_registered_mcp_server_names

    if any((entry or {}).get("connected") for entry in (get_mcp_status() or [])):
        return True
    return bool(get_registered_mcp_server_names())


def _has_configured_mcp_servers() -> bool:
    """Cheap config probe so non-MCP users avoid importing the MCP stack."""
    try:
        from hermes_cli.config import read_raw_config

        raw_config = read_raw_config() or {}
        mcp_servers = raw_config.get("mcp_servers")
        if isinstance(mcp_servers, dict) and len(mcp_servers) > 0:
            return True
        from hermes_cli.agent_plugins import has_enabled_agent_plugin_mcp

        return has_enabled_agent_plugin_mcp(raw_config)
    except Exception:
        # Be conservative: if config probing fails, try discovery in the
        # background so startup still can't block.
        return True


def start_background_mcp_discovery(*, logger, thread_name: str) -> None:
    """Spawn one background MCP discovery thread PER PROFILE.

    Deduplication, the "already started" gate and the retry allowance are all
    keyed by the CALLING profile's canonical home. Two profiles configuring
    the same server name therefore each get their own discovery run reading
    their own ``config.yaml``; a profile whose run is still in flight (or that
    parked/failed) neither blocks nor suppresses any other profile. Repeated
    calls for the SAME profile still collapse onto the one live thread.

    If a profile's discovery run exits without connecting any MCP server (for
    example after startup cancellation / OOM restart), later calls for THAT
    profile are allowed to retry instead of permanently pinning it in a
    "discovery already started" state with zero MCP tools. The status probe
    behind that decision reads the calling profile's registry, so one
    profile's healthy servers can no longer satisfy another profile's gate.

    The module-global lock is held only long enough to look up that
    coordinator; the decision itself runs under the profile's own lock. See
    :class:`_ProfileDiscovery` for why.
    """
    key, home = _current_profile()

    # Global lock: just enough to find (or create) THIS profile's coordinator.
    # Everything expensive below runs under that profile's own lock instead,
    # so one profile's config read / MCP-SDK import cannot stall another
    # profile's discovery start or its per-turn in-flight probe.
    with _mcp_discovery_lock:
        state = _profile_discovery.get(key)
        if state is None:
            _prune_finished_locked()
            state = _ProfileDiscovery(home)
            _profile_discovery[key] = state
        # Claimed before releasing the global lock so the pruner cannot evict
        # this entry between here and acquiring ``state.lock``.
        state.evaluating = True

    try:
        _start_discovery_for_profile(
            state, home, logger=logger, thread_name=thread_name
        )
    finally:
        with _mcp_discovery_lock:
            state.evaluating = False


def _start_discovery_for_profile(
    state: "_ProfileDiscovery", home: str, *, logger, thread_name: str
) -> None:
    """Decide and spawn for ONE profile, holding only that profile's lock.

    Split out of :func:`start_background_mcp_discovery` so the blocking parts
    of the decision — ``_has_configured_mcp_servers`` (reads ``config.yaml``)
    and ``_profile_mcp_is_populated`` (imports ``tools.mcp_tool`` and takes
    its registry lock) — happen off the module-global lock. Same-profile
    callers still serialize here, so the dedup guarantee is unchanged.
    """
    global _mcp_discovery_started, _mcp_discovery_thread

    with state.lock:
        if state.started:
            thread = state.thread
            if thread is not None and thread.is_alive():
                return
            try:
                # Config first, and not only to keep the non-MCP startup path
                # from importing the MCP stack: a profile with nothing
                # configured is DONE, not failed. There is nothing for a retry
                # to find, and the gateway now reaches this gate once per turn
                # (``GatewayRunner._ensure_profile_mcp_tools``) — so treating
                # it as a failure warned on every message an MCP-less profile
                # ever received.
                if not _has_configured_mcp_servers():
                    return
                if _profile_mcp_is_populated():
                    return
            except Exception:
                return
            logger.warning(
                "Background MCP discovery for profile %s previously exited "
                "with no usable MCP tools; retrying discovery thread",
                home or "<default>",
            )
            state.started = False
            state.thread = None

        state.started = True
        _mcp_discovery_started = True
        if not _has_configured_mcp_servers():
            return

        # Capture the caller's profile scope and replay it inside the thread:
        # the context-local HERMES_HOME override (so discovery reads the
        # SELECTED profile's mcp_servers, not the launch profile's -- #67605)
        # and the profile secret scope (so that config's ${VAR} refs and the
        # service-account token exchange resolve against the right profile's
        # credentials instead of failing closed). ContextVars do not propagate
        # into bare threads; see _caller_profile_scope. The config gate above
        # already runs on the caller's thread, so it sees the same override.
        home_override, secret_scope = _capture_caller_scope()

        def _discover() -> None:
            try:
                with _caller_profile_scope(home_override, secret_scope):
                    try:
                        _discover_mcp_tools_without_interactive_oauth()
                        try:
                            if not _profile_mcp_is_populated():
                                # Profile-stamped: on a multiplexed gateway an
                                # unattributed "zero servers" line is not
                                # actionable — it names neither the profile
                                # whose config was read nor the one whose
                                # agent will come up without MCP tools.
                                logger.warning(
                                    "Background MCP discovery for profile %s "
                                    "completed with no usable MCP tools",
                                    home or "<default>",
                                )
                        except Exception:
                            logger.debug("Failed to inspect MCP status after background discovery", exc_info=True)
                    except Exception:
                        # WARNING, not DEBUG: a discovery pass that raised
                        # leaves this profile with NO MCP tools for every
                        # agent built under it. At the gateway's default log
                        # level a DEBUG line makes that indistinguishable
                        # from "this profile has no MCP servers" — the exact
                        # silence this investigation started from.
                        logger.warning(
                            "Background MCP tool discovery failed for profile %s",
                            home or "<default>",
                            exc_info=True,
                        )
            finally:
                global _mcp_discovery_thread
                me = threading.current_thread()
                with _mcp_discovery_lock:
                    # Identity-checked: a retry may already have installed a
                    # newer thread for this profile, and the legacy mirror may
                    # belong to a different profile entirely.
                    #
                    # ``state.thread`` is written here under the GLOBAL lock
                    # but under ``state.lock`` on the spawn path. That is safe
                    # precisely because of the identity check, not by luck: the
                    # only interleaving is "retry installs a new thread while
                    # the old one is exiting", and ``state.thread is me`` is
                    # then False, so the exiting thread cannot clear a
                    # successor. Cleanup deliberately does NOT take
                    # ``state.lock`` — a finishing thread must never block
                    # behind another caller's slow config/registry probe.
                    if state.thread is me:
                        state.thread = None
                    if _mcp_discovery_thread is me:
                        _mcp_discovery_thread = None

        thread = threading.Thread(
            target=_discover,
            name=thread_name,
            daemon=True,
        )
        state.thread = thread
        _mcp_discovery_thread = thread
        thread.start()


def _resolve_discovery_timeout(
    explicit: "float | None", *, single_query: bool = False
) -> float:
    """Resolve the MCP discovery wait bound: explicit arg > config > default.

    Reads ``mcp_discovery_timeout`` from config.yaml, defaulting to the value in
    ``DEFAULT_CONFIG`` (single source of truth) when the key is absent. Kept lazy
    and fail-safe — a missing/invalid value or a broken config falls back to a
    short safe bound so startup can never hang or crash.

    When ``single_query`` is True (``hermes -z "..."`` / ``-q``), the larger
    ``mcp_single_query_discovery_timeout`` bound is used instead. In single-query
    mode there is only ONE turn, so the between-turns late-binding refresh never
    runs — a server that misses the small interactive bound would be invisible to
    the LLM for the whole session. The wait still returns the instant discovery
    completes (see ``wait_for_mcp_discovery``), so fast servers pay ~0s; the
    larger bound only caps how long a genuinely slow cold-start may block.
    """
    if explicit is not None:
        return explicit
    key = (
        "mcp_single_query_discovery_timeout"
        if single_query
        else "mcp_discovery_timeout"
    )
    fallback = 15.0 if single_query else 1.5
    try:
        from hermes_cli.config import load_config, DEFAULT_CONFIG

        default = float(DEFAULT_CONFIG.get(key, fallback))
        try:
            raw = (load_config() or {}).get(key, default)
            val = float(raw)
            return val if val > 0 else default
        except Exception:
            return default
    except Exception:
        return fallback


def _discover_mcp_tools_without_interactive_oauth() -> None:
    """Run MCP discovery without letting OAuth read from the user's stdin."""
    try:
        from tools.mcp_oauth import suppress_interactive_oauth
    except Exception:
        suppress_interactive_oauth = nullcontext

    with suppress_interactive_oauth():
        from tools.mcp_tool import discover_mcp_tools

        discover_mcp_tools()


def wait_for_mcp_discovery(
    timeout: "float | None" = None, *, single_query: bool = False
) -> None:
    """Wait for background MCP discovery before the first tool snapshot.

    ``thread.join(timeout)`` returns the INSTANT discovery completes, so this
    only ever blocks for the real connect time of a still-pending server —
    users with no MCP servers or fast servers pay ~0s.  The bound (from
    ``mcp_discovery_timeout`` in config) just caps the wait so a dead server
    can't freeze startup; servers that miss it are picked up by the automatic
    late-binding refresh.

    When ``single_query`` is True, the bound comes from
    ``mcp_single_query_discovery_timeout`` instead (default 15s vs 1.5s
    interactive) because one-shot sessions have no second turn to recover.

    Waits on the CALLING profile's discovery thread only, so an agent build
    for profile B is never delayed by (nor satisfied by) profile A's run.
    """
    thread = _thread_for_current_profile()
    if thread is None or not thread.is_alive():
        return
    thread.join(timeout=_resolve_discovery_timeout(timeout, single_query=single_query))


def mcp_discovery_in_flight() -> bool:
    """Return True if THIS module's background discovery thread is still running.

    Mirrors ``tui_gateway.entry.mcp_discovery_in_flight`` for the surfaces that
    start discovery through ``start_background_mcp_discovery`` here (the desktop
    app + dashboard WebSocket sidecar via ``tui_gateway/ws.py``, and
    ``hermes dashboard``).  Those processes populate THIS module's
    per-profile coordinator state, not ``tui_gateway.entry``'s thread, so the
    late-refresh scheduler must consult both to decide whether a slow server's
    tools are still pending (see #51587).

    Reports for the CALLING profile (see ``_thread_for_current_profile``): a
    late refresh for profile B must not wait on profile A's discovery.
    """
    thread = _thread_for_current_profile()
    return thread is not None and thread.is_alive()


def join_mcp_discovery(timeout: "float | None" = None) -> bool:
    """Block until THIS module's background discovery finishes, up to ``timeout``.

    Returns True if discovery has completed (thread absent or no longer alive),
    False if it is still running after the timeout.  Unlike
    ``wait_for_mcp_discovery`` this accepts an unbounded/long wait and reports
    the outcome, for the off-critical-path late-refresh waiter.

    Scoped to the CALLING profile's discovery thread.
    """
    thread = _thread_for_current_profile()
    if thread is None:
        return True
    thread.join(timeout=timeout)
    return not thread.is_alive()


def ensure_mcp_discovery_before_agent_build(
    *,
    logger,
    timeout: "float | None" = None,
    single_query: bool = False,
    thread_name: str = "cli-mcp-discovery",
) -> None:
    """Give configured MCP tools a bounded chance to register before AIAgent.

    Non-interactive first turns (``chat -q``, ``hermes -z``) can construct
    ``AIAgent`` before the normal banner or tool-list paths touch
    ``get_tool_definitions()``.  Because the agent snapshots its tool
    registry at construction time, the first and only model turn can miss
    native ``mcp__...`` tools even when the MCP server is healthy.

    ``wait_for_mcp_discovery()`` only joins an already-created discovery
    thread, so it no-ops if a direct/single-query path reaches agent
    construction before MCP startup created that thread.  This helper makes
    the construction site self-sufficient: start discovery if needed, then
    wait up to the configured bound.

    When ``single_query`` is True, the larger
    ``mcp_single_query_discovery_timeout`` bound is used (default 15s vs 1.5s
    interactive) because one-shot sessions have no second turn to recover.

    Failures are swallowed so a broken MCP config never aborts agent
    construction — the agent runs without MCP tools, same as before.
    """
    try:
        start_background_mcp_discovery(
            logger=logger,
            thread_name=thread_name,
        )
        wait_for_mcp_discovery(timeout=timeout, single_query=single_query)
    except Exception:
        logger.debug(
            "MCP discovery readiness check failed before agent build",
            exc_info=True,
        )
