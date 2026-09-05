---
sidebar_position: 8
title: "MCP Config Reference"
description: "Reference for Hermes Agent MCP configuration keys, filtering semantics, and utility-tool policy"
---

# MCP Config Reference

This page is the compact reference companion to the main MCP docs.

For conceptual guidance, see:
- [MCP (Model Context Protocol)](/user-guide/features/mcp)
- [Use MCP with Hermes](/guides/use-mcp-with-hermes)

## Root config shape

```yaml
mcp_servers:
  <server_name>:
    command: "..."      # stdio servers
    args: []
    env: {}

    # OR
    url: "..."          # HTTP servers
    headers: {}

    # Optional HTTP/SSE TLS settings:
    ssl_verify: true                # bool or path to a CA bundle (PEM)
    client_cert: "/path/to/cert.pem"  # mTLS client certificate (see below)
    # client_key: "/path/to/key.pem"  # optional, when key lives in a separate file

    enabled: true
    timeout: 120
    connect_timeout: 60
    supports_parallel_tool_calls: false
    tools:
      include: []
      exclude: []
      resources: true
      prompts: true
```

## Server keys

| Key | Type | Applies to | Meaning |
|---|---|---|---|
| `command` | string | stdio | Executable to launch |
| `args` | list | stdio | Arguments for the subprocess |
| `env` | mapping | stdio | Environment passed to the subprocess |
| `url` | string | HTTP | Remote MCP endpoint |
| `headers` | mapping | HTTP | Headers for remote server requests |
| `ssl_verify` | bool or string | HTTP | TLS verification. `true` (default) uses system CAs, `false` disables verification (insecure), or a string path to a custom CA bundle (PEM) |
| `client_cert` | string or list | HTTP | mTLS client certificate. String = path to a PEM file containing cert + key. List `[cert, key]` = separate files. List `[cert, key, password]` = encrypted key |
| `client_key` | string | HTTP | Path to the client private key, when `client_cert` is a string and the key is in a separate file |
| `enabled` | bool | both | Skip the server entirely when false |
| `timeout` | number | both | Tool call timeout in seconds (default: `300`) |
| `connect_timeout` | number | both | Initial connection timeout in seconds (default: `60`) |
| `protocol` | string | both | Protocol-era negotiation: `auto` (default — legacy `initialize` handshake first, falling back to the 2026-07-28 `server/discover` stateless probe when the server rejects the handshake as modern-only), `stateless` (probe `server/discover` first; one legacy retry), or `legacy` (handshake only, no fallback) |
| `supports_parallel_tool_calls` | bool | both | Allow tools from this server to run concurrently |
| `skip_preflight` | bool | HTTP | Bypass the fail-fast content-type probe for valid Streamable HTTP endpoints whose HEAD/GET answers a non-MCP content type (default: `false`) |
| `transport` | string | HTTP | Set to `sse` to use the SSE transport instead of Streamable HTTP |
| `keepalive_interval` | number | both | Liveness ping cadence in seconds (default: `180`, floored at 5s). Set below the server's session TTL for servers that GC idle sessions quickly |
| `idle_timeout_seconds` | number | stdio | Optional stdio server recycle after idle time (`0` disables). May also live under a `lifecycle:` mapping |
| `max_lifetime_seconds` | number | stdio | Optional stdio server recycle after age (`0` disables). May also live under a `lifecycle:` mapping |
| `tools` | mapping | both | Filtering and utility-tool policy |
| `auth` | string | HTTP | Authentication method. `oauth` = OAuth 2.1 PKCE (browser login). `service_account` = machine-to-machine token exchange, strategy chosen by `service_account.grant_type` (see below). |
| `service_account` | mapping | HTTP | Service-account config block (required when `auth: service_account`). See `service_account` sub-keys below. |
| `sampling` | mapping | both | Server-initiated LLM request policy (see MCP guide) |
| `elicitation` | mapping | both | Server-initiated user-input requests. `enabled` (default `true`) and `timeout` in seconds (default `300`). Form-mode requests route through the approval surface; URL-mode is declined (see MCP guide) |
| `trust` | string | both | Trust tier: `full` (default) or `untrusted`. On an `untrusted` server, every write-capable tool call (any tool without a `readOnlyHint: true` annotation) requires user approval through the standard approval surface before it runs. `readOnlyHint` is a server-supplied *hint* — a lying server can at most skip approval for tools it claims are read-only, never gain extra access — so mark any server you don't fully control as `untrusted`. Unrecognized values are treated as `untrusted` (fail-closed) |

## Environment variable references

String values anywhere in a server entry (`env`, `headers`, `args`, `url`, …) may reference environment variables with `${VAR}` or the Cursor-style SecretRef form `${env:VAR}` — both resolve to the same variable, so MCP snippets copied from Cursor / Claude configs work unchanged:

```yaml
mcp_servers:
  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "${env:GITHUB_TOKEN}"   # same as "${GITHUB_TOKEN}"
```

Values resolve from the active profile's secret scope (falling back to the process environment), so put the secret in `~/.hermes/.env`. An unset variable keeps its literal placeholder.

### Context variables

Beyond env vars, the Cursor-style context variables are interpolated too (names are case-sensitive):

| Variable | Resolves to |
|---|---|
| `${userHome}` | The current user's home directory |
| `${workspaceFolder}` | The session workspace root (the session's terminal cwd when known, else the process cwd) |
| `${workspaceFolderBasename}` | The basename of `${workspaceFolder}` |
| `${pathSeparator}` / `${/}` | The OS path separator (`os.sep`) |

```yaml
mcp_servers:
  filesystem:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "${workspaceFolder}"]
    env:
      CACHE_DIR: "${userHome}${/}.cache${/}mcp"
```

Any other `${...}` reference falls through to the env-var lookup above.

## `tools` policy keys

| Key | Type | Meaning |
|---|---|---|
| `include` | string or list | Whitelist server-native MCP tools. Entries may be exact names or fnmatch-style globs (`*_radar_*`, `get_zones_*`) |
| `exclude` | string or list | Blacklist server-native MCP tools. Same exact-name / glob semantics as `include` |
| `resources` | bool-like | Enable/disable `list_resources` + `read_resource` |
| `prompts` | bool-like | Enable/disable `list_prompts` + `get_prompt` |

## Filtering semantics

### `include`

If `include` is set, only those server-native MCP tools are registered.

```yaml
tools:
  include: [create_issue, list_issues]
```

### `exclude`

If `exclude` is set and `include` is not, every server-native MCP tool except those names is registered.

```yaml
tools:
  exclude: [delete_customer]
```

### Precedence

If both are set, `include` wins.

```yaml
tools:
  include: [create_issue]
  exclude: [create_issue, delete_issue]
```

Result:
- `create_issue` is still allowed
- `delete_issue` is ignored because `include` takes precedence

## Utility-tool policy

Hermes may register these utility wrappers per MCP server:

Resources:
- `list_resources`
- `read_resource`

Prompts:
- `list_prompts`
- `get_prompt`

### Disable resources

```yaml
tools:
  resources: false
```

### Disable prompts

```yaml
tools:
  prompts: false
```

### Capability-aware registration

Even when `resources: true` or `prompts: true`, Hermes only registers those utility tools if the MCP session actually exposes the corresponding capability.

So this is normal:
- you enable prompts
- but no prompt utilities appear
- because the server does not support prompts

## `enabled: false`

```yaml
mcp_servers:
  legacy:
    url: "https://mcp.legacy.internal"
    enabled: false
```

Behavior:
- no connection attempt
- no discovery
- no tool registration
- config remains in place for later reuse

## Empty result behavior

If filtering removes all server-native tools and no utility tools are registered, Hermes does not create an empty MCP runtime toolset for that server.

## Example configs

### Safe GitHub allowlist

```yaml
mcp_servers:
  github:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-github"]
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "***"
    tools:
      include: [list_issues, create_issue, update_issue, search_code]
      resources: false
      prompts: false
```

### Stripe blacklist

```yaml
mcp_servers:
  stripe:
    url: "https://mcp.stripe.com"
    headers:
      Authorization: "Bearer ***"
    tools:
      exclude: [delete_customer, refund_payment]
```

### Resource-only docs server

```yaml
mcp_servers:
  docs:
    url: "https://mcp.docs.example.com"
    tools:
      include: []
      resources: true
      prompts: false
```

### TLS client certificate (mTLS)

For HTTP/SSE servers that require a client certificate, set `client_cert` (and optionally `client_key`):

```yaml
mcp_servers:
  # Combined cert + key in a single PEM file
  internal_api:
    url: "https://mcp.internal.example.com/mcp"
    client_cert: "~/secrets/mcp-client.pem"

  # Separate cert and key files
  partner_api:
    url: "https://mcp.partner.example.com/mcp"
    client_cert: "~/secrets/client.crt"
    client_key: "~/secrets/client.key"

  # Encrypted key with a passphrase (3-element list form)
  bank_api:
    url: "https://mcp.bank.example.com/mcp"
    client_cert: ["~/secrets/client.crt", "~/secrets/client.key", "my-passphrase"]

  # Custom CA bundle (private CA / self-signed server)
  lab_api:
    url: "https://mcp.lab.local/mcp"
    ssl_verify: "~/secrets/lab-ca.pem"
    client_cert: "~/secrets/lab-client.pem"
```

Notes:
- Paths support `~` expansion. Missing files fail fast at connect time with a server-scoped error message.
- `ssl_verify: false` disables server certificate verification entirely. Don't use this with real services.
- Works on both Streamable HTTP and SSE transports.

## Reloading config

After changing MCP config, reload servers with:

```text
/reload-mcp
```

## Tool naming

Server-native MCP tools become:

```text
mcp__<server>__<tool>
```

Examples:
- `mcp__github__create_issue`
- `mcp__filesystem__read_file`
- `mcp__my_api__query_data`

Utility tools follow the same prefixing pattern:
- `mcp__<server>__list_resources`
- `mcp__<server>__read_resource`
- `mcp__<server>__list_prompts`
- `mcp__<server>__get_prompt`

The double-underscore delimiter (`mcp__…__…`) matches the convention used by Claude Code, Codex, and OpenCode, and disambiguates the server/tool boundary even when either component contains underscores.

### Name sanitization

Any character that is not a letter, digit, or underscore (hyphens, dots, spaces, etc.) in both server names and tool names is replaced with an underscore before registration. This ensures tool names are valid identifiers for LLM function-calling APIs.

For example, a server named `my-api` exposing a tool called `list-items.v2` becomes:

```text
mcp__my_api__list_items_v2
```

Keep this in mind when writing `include` / `exclude` filters — use the **original** MCP tool name (with hyphens/dots), not the sanitized version.

## OAuth 2.1 authentication

For HTTP servers that require OAuth, set `auth: oauth` on the server entry:

```yaml
mcp_servers:
  protected_api:
    url: "https://mcp.example.com/mcp"
    auth: oauth
```

Behavior:
- Hermes uses the MCP SDK's OAuth 2.1 PKCE flow (metadata discovery, client identification, token exchange, and refresh)
- On first connect, a browser window opens for authorization
- Tokens are persisted to `~/.hermes/mcp-tokens/<server>.json` and reused across sessions
- Token refresh is automatic; re-authorization only happens when refresh fails
- Only applies to HTTP/StreamableHTTP transport (`url`-based servers)

### Client identification: CIMD and DCR

Hermes identifies itself to authorization servers with a **Client ID Metadata Document** (CIMD), the mechanism the MCP `2026-07-28` spec adopted in place of Dynamic Client Registration. The document is published at
`https://nousresearch.github.io/hermes-agent/docs/oauth/client-metadata.json`, and that URL *is* the `client_id` — the authorization server fetches it to learn Hermes' name, logo, and permitted redirect URIs. Nothing is registered per install, and nothing is user-specific.

The final choice belongs to the authorization server: the SDK sends the document URL as the `client_id` only when the server advertises `client_id_metadata_document_supported: true` in its metadata, and otherwise registers via DCR exactly as before. DCR is deprecated in the MCP spec but still what almost every deployed server uses today.

#### Callback ports

The document declares a fixed set of loopback redirect URIs, and the spec requires the redirect URI in an authorization request to be an *exact string match* against one of them — so a CIMD flow cannot use the random high port Hermes normally picks. Hermes therefore pins the callback to one of ports `27890`–`27894`.

That pin has to be chosen before the server's capabilities are known, because the redirect URI is fixed at the start of the flow while the server's metadata only arrives partway through. So Hermes pins the port for any flow that *could* end up using CIMD, and reverts to a random port for the rest:

- A server Hermes has connected to before, whose cached metadata does not advertise CIMD, keeps the random port it has always used.
- A server Hermes has never reached gets a pinned port on that first login, since guessing is the only way CIMD can ever be used.
- Anything that would move the callback elsewhere reverts too: a pre-registered `oauth.client_id`, an `oauth.client_secret`, a custom `oauth.client_name` or `oauth.token_endpoint_auth_method`, an `oauth.redirect_uri` or `oauth.redirect_port` override, a dashboard- or desktop-driven login, an existing client registration on disk, or all five ports being held by other processes.

Each pinned port is bound as soon as it is chosen and held until the browser redirect arrives, so two concurrent logins — a second profile, or another server in the same process — cannot land on the same listener.

#### When a server rejects the document

If a server fetches the document and refuses it at the *token* endpoint (`invalid_client`), Hermes logs the rejection, records it under `~/.hermes/mcp-tokens/<server>.cimd-off`, and uses DCR for that server from then on.

A server that cannot fetch or validate the document at all aborts at the *authorization* endpoint instead, before any redirect happens. There is no signal Hermes can observe there, so the browser shows an invalid-client error and the login times out after five minutes. The timeout message names the document and points at `cimd: false`. Running `hermes mcp login <server>` clears the recorded rejection, so a corrected document gets another chance.

#### Optional per-server keys

```yaml
mcp_servers:
  protected_api:
    url: "https://mcp.example.com/mcp"
    auth: oauth
    oauth:
      client_metadata_url: "https://example.com/my-cimd.json"  # self-hosted document
      cimd: false                                              # force DCR
      user_agent: "My-MCP-Client/1.0"                          # token-request User-Agent
```

`client_metadata_url` must be an HTTPS URL with a path (no bare origin, no fragment, no userinfo, no `.`/`..` segments) that returns `200` and `Content-Type: application/json` with **no redirect** — authorization servers are forbidden from following redirects when fetching it. Hermes still pins its callback to the same `27890`–`27894` range, so a self-hosted document must declare all ten loopback URIs (`http://127.0.0.1:<port>/callback` and `http://localhost:<port>/callback` for each port), and its `client_id` must be its own URL.

`user_agent` replaces the HTTP library's default `User-Agent` on **token-endpoint requests only** (authorization-code exchange and refresh) — some authorization servers and WAFs reject the default `python-httpx/...` value there. It never applies to MCP traffic or OAuth discovery, and no other token-request headers are configurable. Empty or null values are ignored.

## Service-account (M2M) authentication

For HTTP MCP servers that authenticate machine identities rather than human users, use `auth: service_account`. This is distinct from `auth: oauth` (which opens a browser window for user login) — no browser interaction is needed. Hermes exchanges a long-lived service-account credential for a short-lived Bearer access token and renews it automatically.

The grant strategy is selected explicitly by `service_account.grant_type` and is never inferred from which fields you happen to set. One strategy is implemented today:

| `grant_type` | What it does |
|---|---|
| `authentik_app_password` | Authentik's service-account extension: posts `grant_type=client_credentials` **plus** a resource-owner `username`/`password` pair. |

:::caution Not a generic client-credentials client
`authentik_app_password` is a provider extension that reuses the `client_credentials` wire name. It is **not** the RFC 6749 §4.4.2 client-credentials request, which carries no username or password. Identity providers whose M2M flow is plain client authentication — Keycloak service accounts, Auth0 M2M — do not work with this strategy. Support for a standards-conforming `client_credentials` strategy would be an additive change; for those providers today, use a static token via `headers:` instead.
:::

```yaml
mcp_servers:
  toolhive:
    url: https://mcp.example.com/mcp
    auth: service_account
    service_account:
      grant_type: authentik_app_password         # required — no default
      token_url: https://idp.example.com/application/o/toolhive/token/
      client_id: toolhive
      username: zug
      password_env: AUTHENTIK_ZUG_APP_PASSWORD   # env-var NAME, not the value
      scope: "openid profile groups toolhive-audience"
      client_secret_env: MY_CLIENT_SECRET        # optional
```

**Secret values must never appear in `config.yaml`**. Only environment-variable *names* belong in the config. The values are resolved at runtime through the active profile's secret scope, falling back to the process environment, which is populated from `$HERMES_HOME/.env` before any MCP connection is made. In a multi-profile process each profile resolves its own value, so two profiles can use the same env-var name for different credentials. Put the actual secrets there:

```sh
# ~/.hermes/.env  (or the active profile's .env)
AUTHENTIK_ZUG_APP_PASSWORD=your-app-password-here
```

### `service_account` sub-keys

| Key | Required | Meaning |
|---|---|---|
| `grant_type` | yes | Grant strategy. Only `authentik_app_password` is supported; there is no default |
| `token_url` | yes | OAuth token endpoint URL. **`https://` is required** for any host reachable over a network, and enforced twice — at config validation and again immediately before every token request. Plain `http://` is accepted only for loopback (`localhost`, `127.0.0.1`, `::1`), which never leaves the machine; it logs a warning on every exchange and is meant for local development IdPs only |
| `client_id` | yes | Client ID registered at the IdP |
| `username` | yes | Service-account username (`authentik_app_password` only) |
| `password_env` | yes | Name of the environment variable holding the password (`authentik_app_password` only) |
| `scope` | no | Space-separated OAuth scopes to request |
| `client_secret_env` | no | Name of the environment variable holding the optional client secret |

### Behavior

- Tokens are cached at `$HERMES_HOME/mcp-tokens/service-account/<server>-<digest>.json` (mode `0600`, atomic write). The path is rooted at the profile's own home, so two profiles configuring the same server name never share a token.
- A cached token is **bound to the identity that minted it** — `grant_type`, `token_url`, `client_id`, `username`, `scope` and the credential env-var *names*. Change any of them and the cached token is discarded and re-minted instead of being presented for the previous identity. Only env-var names are hashed; no secret value is.
- A valid token is reused across reconnects; a new token is fetched proactively 60 seconds before expiry, or at half the token's lifetime when the server issues one shorter than 120 seconds.
- If the server returns a `refresh_token`, Hermes uses it on the next renewal before falling back to a fresh service-account exchange. A refresh response that omits `refresh_token` means "keep the one you have" (RFC 6749 §6) and the existing one is retained.
- A single `401` response triggers one immediate re-fetch; if the re-fetch also fails, the error is surfaced to the model.
- Concurrent requests share a single in-process lock so only one token exchange fires at a time.
- Passwords and access tokens are never logged or written to `config.yaml`.
- TLS verification is always on; there is no option to disable it for service-account auth.
- **Token-endpoint redirects are not followed.** A `307`/`308` preserves the method and body, so following one would replay the password — and the client secret — at an origin your config never authorised. Any `3xx` from `token_url` is reported as an error; point `token_url` at the authorization server's final token endpoint.

### Credential rotation

The password is read from the environment on every token exchange, but editing `$HERMES_HOME/.env` does **not** change an already-running process's environment. Rotating a service-account password therefore requires restarting Hermes (or the gateway); automatic token renewal renews the *access token*, not the source credential. Until the restart, renewal and reconnect keep presenting the old password.

### Setting up via CLI

```sh
# 1. Set the secret in your profile's .env
echo 'AUTHENTIK_ZUG_APP_PASSWORD=my-app-password' >> ~/.hermes/.env

# 2. Add the server with auth: service_account already in config.yaml,
#    then validate and probe it
hermes mcp add toolhive \
  --url https://mcp.example.com/mcp \
  --auth service_account
```

Because the password is read from an environment variable, `hermes mcp add` will validate the config and report any missing env-var names without ever prompting for or storing the secret itself.

### Comparison with other auth modes

| Mode | When to use |
|---|---|
| `auth: oauth` | Human user login via browser (PKCE). IdP manages sessions. |
| `auth: service_account` | Machine identity (M2M) against Authentik. Long-lived app password exchanged for short-lived Bearer token. No browser. |
| `headers:` with `${VAR}` | Static API key injected directly (no exchange, no expiry). |

## Add to Hermes link

MCP vendors and docs can offer a one-click **"Add to Hermes"** button that opens the Hermes desktop app with a pre-filled server config, mirroring Cursor's `cursor://anysphere.cursor-deeplink/mcp/install` scheme:

```text
hermes://mcp/install?name=NAME&config=BASE64
```

- `name` — the server name. Must match `^[A-Za-z0-9._-]{1,64}$`.
- `config` — the server config object as **base64url-encoded JSON** (standard base64 is also accepted). The decoded JSON must be an object with either a string `url` field (`http://`/`https://` only) or a string `command` field, and may carry any of the server keys documented above. Payloads over 32KB are rejected.

Example (JavaScript):

```js
const config = { url: 'https://mcp.example.com/mcp' }
const link = `hermes://mcp/install?name=example&config=${btoa(JSON.stringify(config))
  .replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')}`
```

Opening the link never installs anything by itself: the desktop app shows a confirmation dialog with the server name and the full pretty-printed config (with an extra caution for `command`-based servers, which run a local process), and the user must explicitly confirm. Existing server names are never overwritten — the user is asked to rename or cancel.
