"""Profile-scoped isolation for native MCP state.

The multiplexed gateway serves many Hermes profiles from ONE process. Before
this module every piece of MCP runtime state -- loaded server configs, live
``MCPServerTask`` sessions, connect errors, cooldowns, circuit breakers, trust
tiers, lazy-registration metadata and tool provenance -- lived in module-global
dicts keyed by the *logical server name* alone.

That key is not unique across profiles. Two profiles both configuring a server
called ``toolhive`` collapsed onto one entry, so whichever profile connected
first won the slot and every other profile silently reused its config. The
observed symptom: Carol's turn resolving ``AUTHENTIK_ZUG_APP_PASSWORD`` (Zug's
env var, from Zug's server config) and other personas 401ing on a token minted
for someone else.

The isolation key here is the **canonical profile home** --
``hermes_constants.hermes_home_key()`` over the resolved ``HERMES_HOME`` --
which is the same identity the rest of Hermes uses for profile-scoped state
(secret scopes, OAuth token stores, check_fn caches). All MCP state is held in
a per-profile :class:`MCPProfileRegistry`; ``tools.mcp_tool`` exposes the old
module-global names as live *views* onto the current profile's registry, so
every existing call site is profile-scoped without having to thread an extra
argument through 60+ places (and without a missed site silently reopening the
cross-profile hole).

Single-profile deployments are unaffected: one profile resolves to one
registry, which behaves exactly like the old globals.
"""
from __future__ import annotations

import contextlib
import threading
from collections.abc import MutableMapping, MutableSet
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Profile identity
# ---------------------------------------------------------------------------

# Explicit identity pin. ``_run_on_mcp_loop`` stamps the CALLER's resolved
# profile key here before handing work to the shared MCP event loop, and the
# long-lived server run task inherits it through the context copy that
# ``asyncio.ensure_future`` performs. That makes reconnects, lazy first-use
# connects and background refreshes keep the profile identity they were
# created under, instead of re-deriving it from whatever ambient home happens
# to be resolvable on the loop thread.
_PINNED_PROFILE: ContextVar[Optional[Tuple[str, str]]] = ContextVar(
    "_MCP_PINNED_PROFILE", default=None
)

# ``hermes_home_key()`` resolves + normcases a path (~80us). The MCP hot paths
# (tool dispatch, check_fn, status) hit it on every access through the state
# views, so memoize the pure string->key mapping.
_KEY_CACHE: Dict[str, str] = {}
_KEY_CACHE_LOCK = threading.Lock()
_KEY_CACHE_MAX = 256


def canonical_profile_key(home: str | Path) -> str:
    """Return the canonical isolation key for a profile home path."""
    raw = str(home)
    cached = _KEY_CACHE.get(raw)
    if cached is not None:
        return cached
    from hermes_constants import hermes_home_key

    key = hermes_home_key(raw)
    with _KEY_CACHE_LOCK:
        if len(_KEY_CACHE) >= _KEY_CACHE_MAX:
            _KEY_CACHE.clear()
        _KEY_CACHE[raw] = key
    return key


def current_profile() -> Tuple[str, str]:
    """Return ``(profile_key, profile_home)`` for the active profile.

    Resolution order: the explicit pin installed by ``_run_on_mcp_loop`` (so
    MCP-loop work keeps its originating identity), then the ambient Hermes
    home -- context-local override, ``HERMES_HOME``, platform default -- via
    ``get_hermes_home()``.
    """
    pinned = _PINNED_PROFILE.get()
    if pinned is not None:
        return pinned
    from hermes_constants import get_hermes_home

    home = str(get_hermes_home())
    return canonical_profile_key(home), home


def current_profile_key() -> str:
    """Return the canonical isolation key for the active profile."""
    return current_profile()[0]


def current_profile_home() -> str:
    """Return the resolved Hermes home path for the active profile."""
    return current_profile()[1]


def pin_profile(home: str | Path | None) -> Optional[Token]:
    """Pin the MCP profile identity for the current context.

    Returns a token for :func:`unpin_profile`, or ``None`` when *home* is
    falsy (nothing pinned).
    """
    if not home:
        return None
    resolved = str(home)
    return _PINNED_PROFILE.set((canonical_profile_key(resolved), resolved))


def unpin_profile(token: Optional[Token]) -> None:
    """Undo a :func:`pin_profile`."""
    if token is not None:
        _PINNED_PROFILE.reset(token)


@contextlib.contextmanager
def profile_scope(home: str | Path):
    """Context manager pinning the MCP profile identity to *home*.

    Used by tests and by any caller that must act on a specific profile's MCP
    state without mutating the process-global ``HERMES_HOME``.
    """
    token = pin_profile(home)
    try:
        yield current_profile_key()
    finally:
        unpin_profile(token)


# ---------------------------------------------------------------------------
# The per-profile registry
# ---------------------------------------------------------------------------

# Field name -> kind. The views in ``tools.mcp_tool`` are built from this map,
# so adding profile-scoped MCP state is a one-line change here.
_DICT_FIELDS: Tuple[str, ...] = (
    "servers",
    "server_connect_errors",
    "lazy_server_configs",
    "lazy_server_fingerprints",
    "lazy_server_tool_names",
    "server_connect_retry_after",
    "server_connect_failures",
    "server_error_counts",
    "server_breaker_opened_at",
    "server_trust_levels",
    "tool_read_only_hints",
    "mcp_tool_server_names",
)
_SET_FIELDS: Tuple[str, ...] = (
    "server_connecting",
    "parallel_safe_servers",
)


class MCPProfileRegistry:
    """All MCP runtime state owned by exactly one Hermes profile."""

    __slots__ = ("key", "home", "discovery_lock_path") + _DICT_FIELDS + _SET_FIELDS

    def __init__(self, key: str, home: str) -> None:
        self.key = key
        self.home = home
        # Resolved lazily on first discovery; per-profile so two profiles
        # never serialize on (or clobber) each other's discovery lock file.
        self.discovery_lock_path: Optional[str] = None
        for field in _DICT_FIELDS:
            setattr(self, field, {})
        for field in _SET_FIELDS:
            setattr(self, field, set())

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (
            f"<MCPProfileRegistry key={self.key!r} "
            f"servers={sorted(self.servers)!r}>"
        )

    def is_empty(self) -> bool:
        """True when this profile holds no MCP state worth keeping."""
        return not any(
            getattr(self, field)
            for field in _DICT_FIELDS + _SET_FIELDS
        )


_REGISTRIES: Dict[str, MCPProfileRegistry] = {}
_REGISTRY_LOCK = threading.RLock()


def registry_for_key(key: str, home: Optional[str] = None) -> MCPProfileRegistry:
    """Return (creating if needed) the registry for a canonical profile key."""
    with _REGISTRY_LOCK:
        reg = _REGISTRIES.get(key)
        if reg is None:
            reg = MCPProfileRegistry(key, home if home is not None else key)
            _REGISTRIES[key] = reg
        return reg


def registry_for(home: str | Path) -> MCPProfileRegistry:
    """Return (creating if needed) the registry for a profile home path."""
    resolved = str(home)
    return registry_for_key(canonical_profile_key(resolved), resolved)


def current_registry() -> MCPProfileRegistry:
    """Return the registry owned by the active profile."""
    key, home = current_profile()
    return registry_for_key(key, home)


def all_registries() -> List[MCPProfileRegistry]:
    """Snapshot every live profile registry (process-wide shutdown paths)."""
    with _REGISTRY_LOCK:
        return list(_REGISTRIES.values())


def known_profile_keys() -> List[str]:
    """Return the canonical keys of every profile with MCP state."""
    with _REGISTRY_LOCK:
        return list(_REGISTRIES)


def drop_registry(key: str) -> Optional[MCPProfileRegistry]:
    """Forget a profile's registry entirely (session close / teardown)."""
    with _REGISTRY_LOCK:
        return _REGISTRIES.pop(key, None)


def prune_empty_registries() -> int:
    """Drop registries that hold no state. Returns how many were dropped."""
    with _REGISTRY_LOCK:
        stale = [k for k, reg in _REGISTRIES.items() if reg.is_empty()]
        for key in stale:
            _REGISTRIES.pop(key, None)
        return len(stale)


def reset_all_registries() -> None:
    """Drop every profile registry. Test/teardown helper."""
    with _REGISTRY_LOCK:
        _REGISTRIES.clear()
    with _KEY_CACHE_LOCK:
        _KEY_CACHE.clear()


# ---------------------------------------------------------------------------
# Live views onto the current profile's registry
# ---------------------------------------------------------------------------
#
# These exist so ``tools.mcp_tool`` can keep its historical module-global
# names (``_servers``, ``_server_connecting``, ...) while every read and write
# lands in the ACTIVE profile's registry. That matters for correctness, not
# just ergonomics: a view cannot be forgotten at a call site the way an extra
# ``profile_key=`` argument can, so there is no way to leave one MCP code path
# quietly process-global.


class ProfileScopedDict(MutableMapping):
    """Dict-like view of one field on the active profile's registry."""

    __slots__ = ("_field",)

    def __init__(self, field: str) -> None:
        self._field = field

    @property
    def _target(self) -> dict:
        return getattr(current_registry(), self._field)

    def __getitem__(self, key: Any) -> Any:
        return self._target[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        self._target[key] = value

    def __delitem__(self, key: Any) -> None:
        del self._target[key]

    def __iter__(self) -> Iterator:
        return iter(self._target)

    def __len__(self) -> int:
        return len(self._target)

    def __contains__(self, key: Any) -> bool:
        return key in self._target

    def __repr__(self) -> str:
        return repr(self._target)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, ProfileScopedDict):
            return self._target == other._target
        return self._target == other

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    __hash__ = None  # type: ignore[assignment]

    def copy(self) -> dict:
        """Plain-dict copy (``unittest.mock.patch.dict`` relies on this)."""
        return dict(self._target)


class ProfileScopedSet(MutableSet):
    """Set-like view of one field on the active profile's registry."""

    __slots__ = ("_field",)

    def __init__(self, field: str) -> None:
        self._field = field

    @classmethod
    def _from_iterable(cls, it) -> set:
        """Build the RESULT of a set operation as a plain ``set``.

        ``collections.abc.Set`` implements ``|``, ``&``, ``-``, ``^`` (and
        their reflected forms, plus ``MutableSet.__iand__``/``__ixor__``) by
        calling ``self._from_iterable(<generator>)``, whose default is
        ``cls(iterable)``. Our ``__init__`` takes a registry FIELD NAME, so the
        default handed the generator straight to ``getattr`` and every operator
        died with ``TypeError: attribute name must be string, not 'generator'``
        — on both operand orders, since ``set.__or__`` returns
        ``NotImplemented`` for a non-``set`` and Python then calls our
        ``__ror__``.

        A plain ``set`` is also the semantically right result type: these views
        are *aliases* for one field of one profile's registry, so a derived
        value must be an ordinary detached set, never a second view that would
        silently re-bind to whichever profile is active when it is next read.
        """
        return set(it)

    @property
    def _target(self) -> set:
        return getattr(current_registry(), self._field)

    def __contains__(self, value: Any) -> bool:
        return value in self._target

    def __iter__(self) -> Iterator:
        return iter(self._target)

    def __len__(self) -> int:
        return len(self._target)

    def add(self, value: Any) -> None:
        self._target.add(value)

    def discard(self, value: Any) -> None:
        self._target.discard(value)

    def clear(self) -> None:
        self._target.clear()

    def update(self, *others: Any) -> None:
        self._target.update(*others)

    def difference_update(self, *others: Any) -> None:
        self._target.difference_update(*others)

    def intersection_update(self, *others: Any) -> None:
        self._target.intersection_update(*others)

    def symmetric_difference_update(self, other: Any) -> None:
        self._target.symmetric_difference_update(other)

    # ``MutableSet`` supplies the operators but none of ``set``'s NAMED
    # methods. These views stand in for plain ``set`` objects at every MCP
    # call site, so a caller reaching for ``.union(...)`` must not get an
    # AttributeError that only fires on the profile-scoped build.
    def union(self, *others: Any) -> set:
        return self._target.union(*others)

    def intersection(self, *others: Any) -> set:
        return self._target.intersection(*others)

    def difference(self, *others: Any) -> set:
        return self._target.difference(*others)

    def symmetric_difference(self, other: Any) -> set:
        return self._target.symmetric_difference(other)

    def issubset(self, other: Any) -> bool:
        return self._target.issubset(other)

    def issuperset(self, other: Any) -> bool:
        return self._target.issuperset(other)

    def isdisjoint(self, other: Any) -> bool:
        return self._target.isdisjoint(other)

    def copy(self) -> set:
        return set(self._target)

    def __repr__(self) -> str:
        return repr(self._target)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, ProfileScopedSet):
            return self._target == other._target
        if isinstance(other, (set, frozenset)):
            return self._target == other
        return NotImplemented

    def __ne__(self, other: Any) -> bool:
        result = self.__eq__(other)
        return result if result is NotImplemented else not result

    __hash__ = None  # type: ignore[assignment]


def make_state_views() -> Dict[str, Any]:
    """Build the ``{module_global_name: view}`` mapping for ``mcp_tool``."""
    views: Dict[str, Any] = {}
    for field in _DICT_FIELDS:
        views[field] = ProfileScopedDict(field)
    for field in _SET_FIELDS:
        views[field] = ProfileScopedSet(field)
    return views


# ---------------------------------------------------------------------------
# Diagnostics (never log secrets)
# ---------------------------------------------------------------------------

def describe_profile_state() -> List[dict]:
    """Return a secret-free summary of per-profile MCP state.

    Reports profile identity + server names only. Deliberately carries no
    config values, headers, env-var *values*, tokens or cache contents --
    callers log this.
    """
    summary: List[dict] = []
    for reg in all_registries():
        summary.append(
            {
                "profile_key": reg.key,
                "connected": sorted(reg.servers),
                "connecting": sorted(reg.server_connecting),
                "failed": sorted(reg.server_connect_errors),
                "lazy": sorted(reg.lazy_server_configs),
                "tools": len(reg.mcp_tool_server_names),
            }
        )
    return summary


__all__ = [
    "MCPProfileRegistry",
    "ProfileScopedDict",
    "ProfileScopedSet",
    "all_registries",
    "canonical_profile_key",
    "current_profile",
    "current_profile_home",
    "current_profile_key",
    "current_registry",
    "describe_profile_state",
    "drop_registry",
    "known_profile_keys",
    "make_state_views",
    "pin_profile",
    "profile_scope",
    "prune_empty_registries",
    "registry_for",
    "registry_for_key",
    "reset_all_registries",
    "unpin_profile",
]
