"""Pydantic request/response models for the Hermes dashboard web server.

Extracted verbatim from ``hermes_cli/web_server.py`` (pure schema move).
``web_server`` re-exports every name here, so existing imports like
``from hermes_cli.web_server import ConfigUpdate`` keep working.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, SecretStr, field_validator, model_validator


# --- from web_server.py (originally lines 1273-1372) ---

class ConfigUpdate(BaseModel):
    config: dict
    profile: Optional[str] = None


class EnvVarUpdate(BaseModel):
    key: str
    value: str
    profile: Optional[str] = None
    # Optional bearer key for the connectivity probe of a custom/local endpoint
    # (``key == "OPENAI_BASE_URL"``). Self-hosted endpoints that gate
    # ``/v1/models`` behind auth otherwise look "reachable but empty"; sending
    # the key lets the probe enumerate the served models. Ignored for the
    # regular PUT /api/env path (which only reads key/value).
    api_key: str = ""


class EnvVarDelete(BaseModel):
    key: str
    profile: Optional[str] = None


class EnvVarReveal(BaseModel):
    key: str
    profile: Optional[str] = None


class MemoryProviderConfigUpdate(BaseModel):
    values: Dict[str, Any] = {}


class MemoryProviderSetupRequest(BaseModel):
    values: Dict[str, Any] = {}


class CustomEndpointUpdate(BaseModel):
    id: str = ""
    name: str
    base_url: str
    model: str
    api_key: Optional[str] = None
    context_length: Optional[int] = None
    discover_models: bool = True
    make_default: bool = False
    models: Optional[List[str]] = None


class MessagingPlatformUpdate(BaseModel):
    enabled: Optional[bool] = None
    env: Dict[str, str] = {}
    clear_env: List[str] = []
    # Explicit body profile beats the query param injected by the global
    # dashboard profile switcher (same precedence as other scoped writes).
    profile: Optional[str] = None


class TelegramOnboardingStart(BaseModel):
    bot_name: Optional[str] = None


class TelegramOnboardingApply(BaseModel):
    allowed_user_ids: List[str]
    profile: Optional[str] = None


class WhatsAppOnboardingStart(BaseModel):
    mode: Optional[str] = "bot"
    allowed_users: Optional[str] = ""
    profile: Optional[str] = None


class WhatsAppOnboardingApply(BaseModel):
    mode: Optional[str] = None
    allowed_users: Optional[str] = None
    profile: Optional[str] = None


class AudioTranscriptionRequest(BaseModel):
    data_url: str
    mime_type: Optional[str] = None


class ManagedFileUpload(BaseModel):
    path: str
    data_url: str
    overwrite: bool = True


class ChatImageUpload(BaseModel):
    data_url: str
    filename: Optional[str] = None


class ManagedDirectoryCreate(BaseModel):
    path: str


class ManagedFileDelete(BaseModel):
    path: str
    recursive: bool = False


# --- from web_server.py (originally lines 1398-1491) ---

class ModelAssignment(BaseModel):
    """Payload for POST /api/model/set — assign a provider/model to a slot.

    scope="main"        → writes model.provider + model.default
    scope="auxiliary"   → writes auxiliary.<task>.provider + auxiliary.<task>.model
    scope="auxiliary" with task=""  → applied to every auxiliary.* slot
    scope="auxiliary" with task="__reset__"  → resets every slot to provider="auto"
    """
    scope: str
    provider: str
    model: str
    task: str = ""
    # Optional OpenAI-compatible endpoint URL. Honored for custom/local
    # providers on the main slot AND on auxiliary slots — lets the GUI wire a
    # self-hosted endpoint (vLLM, llama.cpp, Ollama, …) that needs no API key.
    # The runtime resolvers read model.base_url / auxiliary.<task>.base_url
    # from config (they ignore OPENAI_BASE_URL), so this is the path that
    # actually wires a local endpoint into resolution.
    base_url: str = ""
    # Optional API key for a custom/local endpoint. Persisted to
    # ``model.api_key`` (main slot) or ``auxiliary.<task>.api_key`` (aux
    # slots) — where the runtime resolvers read it — so a self-hosted
    # endpoint that requires auth works from the GUI. Mirrors the key the
    # ``hermes model`` custom flow collects.
    api_key: str = ""
    confirm_expensive_model: bool = False
    profile: Optional[str] = None


class MoaModelSlot(BaseModel):
    provider: str = ""
    model: str = ""
    # Optional per-slot reasoning effort. Declared so a client round-tripping
    # the GET payload doesn't have it stripped at parse time and wiped on save.
    reasoning_effort: Optional[str] = None
    enabled: bool = True


class _MoaReferenceControls(BaseModel):
    # None = no per-preset override; the fan-out inherits
    # auxiliary.moa_reference.timeout (900s default).
    reference_timeout: Optional[float] = None
    degraded_reference_policy: Literal["loud", "silent"] = "loud"

    @field_validator("reference_timeout", mode="before")
    @classmethod
    def _validate_reference_timeout(cls, value: Any) -> Optional[float]:
        """Reject JSON booleans/non-finite values before float coercion."""
        if value is None or value == "":
            return None
        if isinstance(value, bool):
            raise ValueError("reference_timeout must be a finite positive number")
        try:
            timeout = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "reference_timeout must be a finite positive number"
            ) from exc
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("reference_timeout must be a finite positive number")
        return timeout


class MoaPresetPayload(_MoaReferenceControls):
    reference_models: list[MoaModelSlot] = []
    aggregator: MoaModelSlot = MoaModelSlot()
    # None = temperature omitted from API calls (provider default), matching
    # single-model agent behavior.
    reference_temperature: Optional[float] = None
    aggregator_temperature: Optional[float] = None
    max_tokens: int = 4096
    # Newer per-preset knobs (see moa_config._normalize_preset). Optional so
    # older clients that never send them keep working; declared so clients
    # that round-trip the GET payload don't silently erase hand-set values.
    reference_max_tokens: Optional[int] = None
    fanout: Optional[str] = None
    enabled: bool = True


class MoaConfigPayload(_MoaReferenceControls):
    default_preset: str = "default"
    active_preset: str = ""
    presets: dict[str, MoaPresetPayload] = {}
    # Backward-compatible flat payload fields used by older dashboard/desktop
    # clients during this PR's transition window.
    reference_models: list[MoaModelSlot] = []
    aggregator: MoaModelSlot = MoaModelSlot()
    reference_temperature: Optional[float] = None
    aggregator_temperature: Optional[float] = None
    max_tokens: int = 4096
    reference_max_tokens: Optional[int] = None
    fanout: Optional[str] = None
    enabled: bool = True
    profile: Optional[str] = None


# --- from web_server.py (originally lines 2718-2720) ---

class FsWriteText(BaseModel):
    path: str
    content: str


# --- from web_server.py (originally lines 2826-2856) ---

class GitPathBody(BaseModel):
    path: str


class GitFileBody(BaseModel):
    path: str
    file: Optional[str] = None


class GitPrListBody(BaseModel):
    path: str
    branches: List[str] = []
    # PRs a session recovered from its transcript, which we know by number
    # rather than by the branch it came from.
    numbers: List[int] = []


class SessionPrScanBody(BaseModel):
    ids: List[str] = []


class GitCommitBody(BaseModel):
    path: str
    message: str
    push: bool = False


class GitWorktreeAddBody(BaseModel):
    path: str
    name: Optional[str] = None
    branch: Optional[str] = None
    base: Optional[str] = None
    existingBranch: Optional[str] = None


class GitWorktreeRemoveBody(BaseModel):
    path: str
    worktreePath: str
    force: bool = False


class GitBranchSwitchBody(BaseModel):
    path: str
    branch: str


# --- from web_server.py (originally lines 3610-3611) ---

class CuratorPause(BaseModel):
    paused: bool


# --- from web_server.py (originally lines 3649-3657) ---

class LearningNodeRef(BaseModel):
    id: str
    profile: Optional[str] = None


class LearningNodeEdit(BaseModel):
    id: str
    content: str
    profile: Optional[str] = None


# --- from web_server.py (originally lines 3786-3792) ---

class DebugShareRequest(BaseModel):
    # Redaction is ON by default — force-mode scrubs credential-shaped tokens
    # out of log content before it leaves the machine. The toggle exists so an
    # operator who knows the logs are clean can opt out for fuller fidelity.
    redact: bool = True
    # Recent log lines included in the summary tail (full logs are separate).
    lines: int = 200


# --- from web_server.py (originally lines 4492-4493) ---

class TTSSpeakRequest(BaseModel):
    text: str


# --- from web_server.py (originally lines 11549-11551) ---

class OAuthSubmitBody(BaseModel):
    session_id: str
    code: str


# --- from web_server.py (originally lines 11708-11715) ---

class BulkDeleteSessions(BaseModel):
    ids: List[str]
    profile: Optional[str] = None


class SessionImport(BaseModel):
    sessions: List[Dict[str, Any]]
    profile: Optional[str] = None


# --- from web_server.py (originally lines 12082-12090) ---

class SessionRename(BaseModel):
    title: Optional[str] = None
    archived: Optional[bool] = None
    # Generic visibility flag. This is also used by process-light cross-profile
    # reconciliation, where the primary backend opens the owner's state.db.
    hidden: Optional[bool] = None
    # Durable "keep" flag mirrored from the Desktop sidebar's pins; pinned
    # sessions are exempt from the sessions.auto_archive stale sweep.
    pinned: Optional[bool] = None
    # Read-state watermark toggle (sessions.last_read_at): True marks the
    # session explicitly unread, False marks it read up to now. Mirrored from
    # the Desktop sidebar's "Mark as unread"/"Mark as read". None = leave alone.
    unread: Optional[bool] = None
    # Mutate a session belonging to another profile (opens its state.db). Omit
    # for the current/default profile.
    profile: Optional[str] = None


class SessionOwnerBackfill(BaseModel):
    """Body for POST /api/sessions/owner-backfill (#94724 legacy migration).

    ``profile`` scopes WHICH profile's state.db is stamped (same semantics as
    every other session route); the stamped value is always that store's own
    serving-profile identity — the caller cannot inject an arbitrary owner.
    """

    profile: Optional[str] = None


# --- from web_server.py (originally lines 12149-12174) ---

class SessionPrune(BaseModel):
    older_than_days: Optional[float] = 90
    source: Optional[str] = None
    profile: Optional[str] = None
    # Extended filters (all optional, AND together — mirrors the CLI flags)
    started_before: Optional[float] = None  # epoch seconds
    started_after: Optional[float] = None  # epoch seconds
    title_like: Optional[str] = None
    end_reason: Optional[str] = None
    cwd_prefix: Optional[str] = None
    min_messages: Optional[int] = None
    max_messages: Optional[int] = None
    model_like: Optional[str] = None
    provider: Optional[str] = None
    user_id: Optional[str] = None
    chat_id: Optional[str] = None
    chat_type: Optional[str] = None
    branch_like: Optional[str] = None
    min_tokens: Optional[int] = None
    max_tokens: Optional[int] = None
    min_cost: Optional[float] = None
    max_cost: Optional[float] = None
    min_tool_calls: Optional[int] = None
    max_tool_calls: Optional[int] = None
    include_archived: bool = False
    dry_run: bool = False


# --- from web_server.py (originally lines 12335-12352) ---

class CronJobCreate(BaseModel):
    prompt: str = ""
    schedule: str
    name: str = ""
    deliver: str = "local"
    skills: Optional[List[str]] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    base_url: Optional[str] = None
    script: Optional[str] = None
    context_from: Optional[Any] = None
    enabled_toolsets: Optional[List[str]] = None
    workdir: Optional[str] = None
    no_agent: bool = False


class CronJobUpdate(BaseModel):
    updates: dict


# --- from web_server.py (originally lines 12924-12926) ---

class AutomationBlueprintInstantiate(BaseModel):
    blueprint: str                      # blueprint key, e.g. "morning-brief"
    values: Dict[str, Any] = {}      # filled slot values from the form


# --- from web_server.py (originally lines 13002-13019) ---

#: Fields whose NAME says the client tried to send a secret VALUE. Checked
#: case-insensitively before ``extra="forbid"`` reports them, so the operator
#: gets the actionable "use <field>_env" message rather than a generic
#: "extra inputs are not permitted".
_SA_SECRET_VALUE_FIELDS: Dict[str, str] = {
    "password": "password_env",
    "client_secret": "client_secret_env",
}


class MCPServiceAccountConfig(BaseModel):
    """The dashboard-submitted ``service_account:`` block. Env-var NAMES only.

    This was ``Dict[str, Any]``, which meant the dashboard would persist
    whatever the client sent into ``config.yaml`` verbatim and echo it back on
    read. Two things follow from a strict model instead:

    * **Nothing secret can be stored.** Every field here is a non-secret
      configuration reference — ``password_env`` and ``client_secret_env``
      name an environment variable, they never carry its value — and
      ``extra="forbid"`` means no other key survives. A caller that sends
      ``password`` or ``client_secret`` (or any key at all that this contract
      does not define) is refused rather than having it written to disk and
      returned by ``GET /api/mcp/servers``.
    * **Malformed input fails as a 422 with a readable message**, not as an
      opaque ``AttributeError``/``TypeError`` 500 deeper in the config
      writer. A non-mapping body, a nested object where a string belongs, a
      wrong type — all are reported at the boundary.

    ``token_url``/``client_id``/``username``/``password_env`` are typed as
    optional here on purpose: *presence* is a domain rule that
    ``tools.mcp_service_account.validate_service_account_config`` owns for the
    CLI, the dashboard and the runtime alike, and duplicating it here would
    let the two drift.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    grant_type: str = "authentik_app_password"
    token_url: str = ""
    client_id: str = ""
    username: str = ""
    #: Env-var NAME holding the service-account password — never the password.
    password_env: str = ""
    scope: str = ""
    #: Env-var NAME holding an optional client secret — never the secret.
    client_secret_env: str = ""

    @model_validator(mode="before")
    @classmethod
    def _reject_secret_values_and_non_mappings(cls, data: Any) -> Any:
        if data is None or isinstance(data, cls):
            return data
        if not isinstance(data, dict):
            raise ValueError(
                "service_account must be an object with configuration "
                "references (token_url, client_id, username, password_env, "
                f"scope, client_secret_env); got {type(data).__name__}"
            )
        for key in data:
            replacement = _SA_SECRET_VALUE_FIELDS.get(str(key).strip().lower())
            if replacement:
                raise ValueError(
                    f"service_account.{key} must not be sent to the server. "
                    f"Use {replacement} (an environment-variable name) "
                    "instead, and set the value in your profile's .env file."
                )
        return data

    def to_config_dict(self) -> Dict[str, str]:
        """Serialize to the ``service_account:`` mapping written to config.yaml.

        Empty fields are dropped rather than written as ``""`` so
        ``validate_service_account_config`` reports them as *missing* (with
        the flag/field name to supply) instead of as malformed, and so an
        untouched optional field never appears in the file at all.
        """
        return {
            field: value
            for field, value in (
                (name, getattr(self, name)) for name in type(self).model_fields
            )
            if isinstance(value, str) and value
        }

    @classmethod
    def summarize_stored(cls, stored: Any) -> Dict[str, str]:
        """Non-secret view of an ON-DISK block, for read endpoints.

        An allowlist, not a denylist. ``config.yaml`` is hand-editable, so the
        stored block can hold anything; filtering out a fixed pair of names
        (the previous behaviour) still echoed back ``passwd``, ``secret``, a
        mis-cased ``Password``, or any other key someone put there. Returning
        only the fields this contract defines cannot leak one. Non-mapping
        input yields ``{}`` rather than raising, so one hand-broken entry
        cannot 500 the whole server list.
        """
        if not isinstance(stored, dict):
            return {}
        out: Dict[str, str] = {}
        for name in cls.model_fields:
            value = stored.get(name)
            if isinstance(value, str) and value.strip():
                out[name] = value.strip()
        return out


class MCPServerCreate(BaseModel):
    name: str
    url: Optional[str] = None
    command: Optional[str] = None
    args: List[str] = []
    # env: KEY=VALUE map for stdio servers (API keys, etc.)
    env: Dict[str, str] = {}
    # auth: "none" | "oauth" | "header" | "service_account" | None
    auth: Optional[str] = None
    # One-time provisioning input; persisted only to the profile's .env.
    bearer_token: Optional[SecretStr] = None
    # Non-secret service-account fields (token_url, client_id, username, scope,
    # password_env, client_secret_env). Secrets are referenced by env-var name only.
    service_account: Optional[MCPServiceAccountConfig] = None
    profile: Optional[str] = None


class MCPServersReplace(BaseModel):
    # Whole-map replace (name → raw server config) for the GUI mcp.json editor.
    servers: Dict[str, Dict[str, Any]] = {}
    profile: Optional[str] = None


# --- from web_server.py (originally lines 13518-13520) ---

class MCPEnabledToggle(BaseModel):
    enabled: bool
    profile: Optional[str] = None


# --- from web_server.py (originally lines 13622-13627) ---

class MCPCatalogInstall(BaseModel):
    name: str
    # env: KEY=VALUE map for catalog entries that declare required env vars.
    env: Dict[str, str] = {}
    enable: bool = True
    profile: Optional[str] = None


# --- from web_server.py (originally lines 13716-13723) ---

class PairingApprove(BaseModel):
    platform: str
    code: str = ""
    request_id: str = ""
    profile: Optional[str] = None


class PairingRevoke(BaseModel):
    platform: str
    user_id: str
    profile: Optional[str] = None


# --- from web_server.py (originally lines 13793-13804) ---

class WebhookCreate(BaseModel):
    name: str
    description: Optional[str] = None
    events: List[str] = []
    prompt: Optional[str] = None
    script: Optional[str] = None
    skills: List[str] = []
    deliver: str = "log"
    deliver_only: bool = False
    deliver_chat_id: Optional[str] = None
    # secret: omit to auto-generate
    secret: Optional[str] = None


# --- from web_server.py (originally lines 13930-13931) ---

class WebhookEnabledToggle(BaseModel):
    enabled: bool


# --- from web_server.py (originally lines 13997-14002) ---

class CredentialPoolAdd(BaseModel):
    provider: str
    # api_key for API-key providers; OAuth pooling stays CLI-only (it needs
    # an interactive browser flow that doesn't belong in a single POST).
    api_key: str
    label: Optional[str] = None


# --- from web_server.py (originally lines 14171-14178) ---

class MemoryProviderSelect(BaseModel):
    # "" or "built-in" disables the external provider (built-in only).
    provider: str


class MemoryReset(BaseModel):
    # "all" | "memory" | "user"
    target: str = "all"


# --- from web_server.py (originally lines 14274-14276) ---

class BackupRequest(BaseModel):
    # Optional output path; defaults to a timestamped zip in the home dir.
    output: Optional[str] = None


# --- from web_server.py (originally lines 14339-14348) ---

class ImportRequest(BaseModel):
    archive: str
    # Pass --force to `hermes import`. The spawned action runs with
    # stdin=DEVNULL, so the CLI's interactive "Continue? [y/N]" overwrite
    # prompt hits EOF and auto-aborts ("Aborted.", exit 1) whenever the
    # target already has a config — which it always does when the dashboard
    # itself is running from it. The dashboard shows its own confirm modal
    # before calling this endpoint, then sends force=True so the restore
    # proceeds non-interactively.
    force: bool = False


# --- from web_server.py (originally lines 14505-14513) ---

class HookCreate(BaseModel):
    event: str
    command: str
    matcher: Optional[str] = None
    timeout: Optional[int] = None
    # approve: write the consent allowlist entry too (the operator using the
    # authenticated dashboard is giving consent). Without it the hook is
    # configured but won't fire until approved.
    approve: bool = True


# --- from web_server.py (originally lines 14573-14575) ---

class HookDelete(BaseModel):
    event: str
    command: str


# --- from web_server.py (originally lines 14667-14669) ---

class SkillInstallRequest(BaseModel):
    identifier: str
    profile: Optional[str] = None


# --- from web_server.py (originally lines 14724-14726) ---

class SkillUninstallRequest(BaseModel):
    name: str
    profile: Optional[str] = None


# --- from web_server.py (originally lines 14748-14749) ---

class SkillsUpdateRequest(BaseModel):
    profile: Optional[str] = None


# --- from web_server.py (originally lines 15116-15166) ---

class ProfileCreate(BaseModel):
    name: str
    clone_from: Optional[str] = None
    # Backward compatibility for older dashboard/desktop clients. New clients
    # send clone_from="default" (or another profile name) explicitly.
    clone_from_default: bool = False
    clone_all: bool = False
    no_skills: bool = False
    description: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    # Profile-builder additions — all optional, all applied best-effort AFTER
    # the profile directory exists, so a hiccup in any of them never 500s the
    # create (the user can fix it from the relevant dashboard page afterward).
    # MCP servers to write into the new profile's config.yaml.
    mcp_servers: List["MCPServerCreate"] = []
    # Built-in / optional skills to KEEP active. When this list is non-empty,
    # the builder uses "replace" semantics: the bundle is seeded, then every
    # seeded skill NOT in this list is added to the profile's disabled list.
    # Empty list = leave the seeded bundle untouched (legacy behaviour).
    keep_skills: List[str] = []
    # Skills-hub identifiers to install into the new profile. Installed async
    # via a subprocess scoped to the profile (`hermes -p <name> skills install`)
    # because skills_hub.SKILLS_DIR is import-time-bound and the HERMES_HOME
    # override can't redirect it. Returns spawned PIDs for the UI to poll.
    hub_skills: List[str] = []


class ProfileRename(BaseModel):
    new_name: str


class ProfileExport(BaseModel):
    # Optional extra root-level files to stage into the archive, filename →
    # text content (e.g. desktop.json — the desktop appearance overlay).
    extra_files: Dict[str, str] = {}
    # Where to write the archive. Empty → a staging path under HERMES_HOME.
    output: str = ""


class ProfileImport(BaseModel):
    # Path to a profile .tar.gz on the backend's filesystem (the desktop's
    # local/pooled backends share the machine with the picker dialog).
    archive: str
    # Override the profile name inferred from the archive root.
    name: Optional[str] = None


class ProfileSoulUpdate(BaseModel):
    content: str


class ProfileActiveUpdate(BaseModel):
    name: str


class ProfileDescriptionUpdate(BaseModel):
    description: str = ""


class ProfileModelUpdate(BaseModel):
    provider: str
    model: str


class ProfileDescribeAuto(BaseModel):
    overwrite: bool = False


# --- from web_server.py (originally lines 15831-15834) ---

class SkillToggle(BaseModel):
    name: str
    enabled: bool
    profile: Optional[str] = None


# --- from web_server.py (originally lines 15883-15893) ---

class SkillCreate(BaseModel):
    name: str
    content: str
    category: Optional[str] = None
    profile: Optional[str] = None


class SkillContentUpdate(BaseModel):
    name: str
    content: str
    profile: Optional[str] = None


# --- from web_server.py (originally lines 16022-16024) ---

class ToolsetToggle(BaseModel):
    enabled: bool
    profile: Optional[str] = None


# --- from web_server.py (originally lines 16199-16204) ---

class ToolsetProviderSelect(BaseModel):
    provider: str
    # Web-only capability scope: 'search' | 'extract'. Omitted → whole-provider
    # selection through the legacy apply_provider_selection path (web.backend).
    capability: Optional[str] = None
    profile: Optional[str] = None


# --- from web_server.py (originally lines 16324-16327) ---

class ToolsetModelSelect(BaseModel):
    model: str
    provider: Optional[str] = None
    profile: Optional[str] = None


# --- from web_server.py (originally lines 16510-16512) ---

class ToolsetEnvUpdate(BaseModel):
    env: Dict[str, str]
    profile: Optional[str] = None


# --- from web_server.py (originally lines 16570-16572) ---

class ToolsetPostSetup(BaseModel):
    key: str
    profile: Optional[str] = None


# --- from web_server.py (originally lines 16823-16825) ---

class TerminalBackendSelect(BaseModel):
    backend: str
    profile: Optional[str] = None


# --- from web_server.py (originally lines 16919-16921) ---

class RawConfigUpdate(BaseModel):
    yaml_text: str
    profile: Optional[str] = None


# --- from web_server.py (originally lines 19410-19411) ---

class ThemeSetBody(BaseModel):
    name: str


# --- from web_server.py (originally lines 19449-19450) ---

class FontSetBody(BaseModel):
    font: str


# --- from web_server.py (originally lines 19681-19684) ---

class _AgentPluginInstallBody(BaseModel):
    identifier: str
    force: bool = False
    enable: bool = True


# --- from web_server.py (originally lines 19896-19898) ---

class _PluginProvidersPutBody(BaseModel):
    memory_provider: Optional[str] = None
    context_engine: Optional[str] = None


# --- from web_server.py (originally lines 19919-19920) ---

class _PluginVisibilityBody(BaseModel):
    hidden: bool

