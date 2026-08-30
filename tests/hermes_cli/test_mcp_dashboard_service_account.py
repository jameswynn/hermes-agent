"""The dashboard's ``service_account`` contract is references-only and strict.

``POST /api/mcp/servers`` writes straight into the profile's ``config.yaml``
and ``GET /api/mcp/servers`` reads it back, so whatever this endpoint accepts
becomes persistent state that is echoed to every dashboard client. The block
used to be typed ``Dict[str, Any]``, which meant:

* any key at all was persisted verbatim, with only ``password`` and
  ``client_secret`` filtered on read — so ``passwd``, a mis-cased
  ``Password``, or any other operator-invented key round-tripped;
* a malformed body (non-mapping, nested object where a string belongs) reached
  the config writer and surfaced as an opaque 500 instead of a validation
  error naming the field.

Every field in the contract is a non-secret configuration *reference*:
``password_env`` and ``client_secret_env`` name an environment variable, never
its value. Nothing here should be able to put a credential into config.yaml.

Runs entirely against an isolated ``HERMES_HOME``; no network, no real
credentials.
"""

import pytest


def _client():
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")
    import hermes_state
    from hermes_constants import get_hermes_home
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    hermes_state.DEFAULT_DB_PATH = get_hermes_home() / "state.db"
    return client


_VALID_SA = {
    "grant_type": "authentik_app_password",
    "token_url": "https://idp.example/application/o/toolhive/token/",
    "client_id": "toolhive",
    "username": "svc",
    "password_env": "AUTHENTIK_SVC_APP_PASSWORD",
    "scope": "openid profile groups",
}


def _body(**overrides):
    body = {
        "name": "toolhive",
        "url": "https://toolhive.example/mcp",
        "auth": "service_account",
        "service_account": dict(_VALID_SA),
    }
    body.update(overrides)
    return body


class TestServiceAccountCreate:
    @pytest.fixture(autouse=True)
    def _setup(self, _isolate_hermes_home):
        self.client = _client()

    def test_valid_block_is_persisted_verbatim(self):
        resp = self.client.post("/api/mcp/servers", json=_body())
        assert resp.status_code == 200, resp.text
        assert resp.json()["auth"] == "service_account"
        assert resp.json()["service_account"] == _VALID_SA

        from hermes_cli.mcp_config import _get_mcp_servers

        stored = _get_mcp_servers()["toolhive"]
        assert stored["auth"] == "service_account"
        assert stored["service_account"] == _VALID_SA

    def test_no_secret_value_reaches_config(self):
        from hermes_constants import get_hermes_home

        self.client.post("/api/mcp/servers", json=_body())
        config_text = (get_hermes_home() / "config.yaml").read_text(encoding="utf-8")
        # Only the env-var NAME, never a value, and no value-taking field.
        assert "AUTHENTIK_SVC_APP_PASSWORD" in config_text
        assert "password:" not in config_text
        assert "client_secret:" not in config_text

    @pytest.mark.parametrize(
        "field,replacement",
        [("password", "password_env"), ("client_secret", "client_secret_env")],
    )
    def test_secret_value_fields_are_refused(self, field, replacement):
        sa = dict(_VALID_SA)
        sa[field] = "hunter2-should-never-be-stored"
        resp = self.client.post("/api/mcp/servers", json=_body(service_account=sa))

        assert resp.status_code == 422, resp.text
        assert replacement in resp.text
        assert "hunter2-should-never-be-stored" not in resp.text
        self._assert_nothing_written()

    def test_mixed_case_secret_field_is_refused(self):
        """A denylist on exact keys missed ``Password``; the check is folded."""
        sa = dict(_VALID_SA)
        sa["Password"] = "hunter2"
        resp = self.client.post("/api/mcp/servers", json=_body(service_account=sa))

        assert resp.status_code == 422, resp.text
        assert "password_env" in resp.text
        self._assert_nothing_written()

    def test_unknown_fields_are_refused_not_persisted(self):
        """``extra='forbid'``: an operator-invented key cannot reach disk."""
        sa = dict(_VALID_SA)
        sa["passwd"] = "hunter2"
        sa["notes"] = "anything"
        resp = self.client.post("/api/mcp/servers", json=_body(service_account=sa))

        assert resp.status_code == 422, resp.text
        self._assert_nothing_written()

    @pytest.mark.parametrize(
        "value",
        ["not-a-mapping", ["a", "b"], 7, True],
        ids=["string", "list", "int", "bool"],
    )
    def test_non_mapping_block_fails_as_validation_not_500(self, value):
        resp = self.client.post("/api/mcp/servers", json=_body(service_account=value))

        assert resp.status_code == 422, resp.text
        assert resp.status_code != 500
        self._assert_nothing_written()

    def test_nested_object_in_a_string_field_fails_as_validation(self):
        sa = dict(_VALID_SA)
        sa["token_url"] = {"evil": "https://attacker.example/token"}
        resp = self.client.post("/api/mcp/servers", json=_body(service_account=sa))

        assert resp.status_code == 422, resp.text
        self._assert_nothing_written()

    def test_missing_required_domain_fields_are_a_400(self):
        """Presence rules stay in the shared validator, surfaced as a 400."""
        sa = {k: v for k, v in _VALID_SA.items() if k != "client_id"}
        resp = self.client.post("/api/mcp/servers", json=_body(service_account=sa))

        assert resp.status_code == 400, resp.text
        assert "client_id" in resp.json()["detail"]
        self._assert_nothing_written()

    def test_plaintext_token_url_is_refused(self):
        sa = dict(_VALID_SA, token_url="http://idp.example/token/")
        resp = self.client.post("/api/mcp/servers", json=_body(service_account=sa))

        assert resp.status_code == 400, resp.text
        assert "https://" in resp.json()["detail"]
        self._assert_nothing_written()

    def test_service_account_without_the_auth_mode_is_refused(self):
        """Silently dropping it would save a server that authenticates to nothing."""
        resp = self.client.post(
            "/api/mcp/servers", json=_body(auth="none")
        )

        assert resp.status_code == 400, resp.text
        assert "service_account" in resp.json()["detail"]
        self._assert_nothing_written()

    def test_service_account_auth_without_a_block_is_refused(self):
        resp = self.client.post(
            "/api/mcp/servers",
            json={
                "name": "toolhive",
                "url": "https://toolhive.example/mcp",
                "auth": "service_account",
            },
        )

        assert resp.status_code == 400, resp.text
        self._assert_nothing_written()

    def test_bearer_token_is_rejected_for_service_account(self):
        resp = self.client.post(
            "/api/mcp/servers", json=_body(bearer_token="Bearer nope")
        )

        assert resp.status_code == 400, resp.text
        self._assert_nothing_written()

    def _assert_nothing_written(self):
        from hermes_cli.mcp_config import _get_mcp_servers

        assert "toolhive" not in _get_mcp_servers()


class TestServiceAccountRead:
    """``GET`` must never echo a secret, and never 500 on a hand-edited block."""

    @pytest.fixture(autouse=True)
    def _setup(self, _isolate_hermes_home):
        self.client = _client()

    def _store(self, service_account):
        from hermes_cli.mcp_config import _save_mcp_server

        _save_mcp_server(
            "toolhive",
            {
                "url": "https://toolhive.example/mcp",
                "auth": "service_account",
                "service_account": service_account,
            },
        )

    def test_hand_edited_secret_fields_are_not_echoed(self):
        """An allowlist: only contract fields come back, whatever is stored."""
        self._store(
            {
                **_VALID_SA,
                "password": "plaintext-in-config",
                "passwd": "also-plaintext",
                "Password": "third-plaintext",
                "client_secret": "cs3cr3t",
                "notes": "operator scribble",
            }
        )
        resp = self.client.get("/api/mcp/servers")

        assert resp.status_code == 200, resp.text
        assert "plaintext-in-config" not in resp.text
        assert "also-plaintext" not in resp.text
        assert "third-plaintext" not in resp.text
        assert "cs3cr3t" not in resp.text
        assert "operator scribble" not in resp.text
        srv = resp.json()["servers"][0]
        assert srv["service_account"] == _VALID_SA

    @pytest.mark.parametrize(
        "stored", ["oops", ["a"], 5, None], ids=["string", "list", "int", "null"]
    )
    def test_non_mapping_stored_block_does_not_500_the_list(self, stored):
        self._store(stored)
        resp = self.client.get("/api/mcp/servers")

        assert resp.status_code == 200, resp.text
        assert resp.json()["servers"][0]["service_account"] == {}

    def test_non_string_values_are_dropped_not_serialized(self):
        self._store({**_VALID_SA, "scope": {"nested": True}})
        resp = self.client.get("/api/mcp/servers")

        assert resp.status_code == 200, resp.text
        assert "scope" not in resp.json()["servers"][0]["service_account"]


class TestServiceAccountModel:
    """Unit-level contract for the model itself."""

    def test_to_config_dict_drops_empties_and_strips(self):
        from hermes_cli.web_models import MCPServiceAccountConfig

        cfg = MCPServiceAccountConfig(
            token_url="  https://idp.example/token/  ",
            client_id="toolhive",
            username="svc",
            password_env="P",
        )
        assert cfg.to_config_dict() == {
            "grant_type": "authentik_app_password",
            "token_url": "https://idp.example/token/",
            "client_id": "toolhive",
            "username": "svc",
            "password_env": "P",
        }

    def test_summarize_stored_is_total(self):
        from hermes_cli.web_models import MCPServiceAccountConfig

        for bad in (None, "x", [1], 3, object()):
            assert MCPServiceAccountConfig.summarize_stored(bad) == {}

    def test_every_model_field_is_a_non_secret_reference(self):
        """A field taking a secret VALUE must never be added to this model."""
        from hermes_cli.web_models import MCPServiceAccountConfig

        assert set(MCPServiceAccountConfig.model_fields) == {
            "grant_type",
            "token_url",
            "client_id",
            "username",
            "password_env",
            "scope",
            "client_secret_env",
        }


class TestValidationErrorsDoNotEchoInput:
    """A 422 must name the bad field without replaying its value.

    FastAPI's default handler copies the offending value into each error entry
    as ``input``. Several of these endpoints exist to receive
    credential-adjacent input, so refusing to STORE a secret and then
    reflecting it into the response body — and from there into dashboard
    logs, devtools and HAR captures — undoes the refusal.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, _isolate_hermes_home):
        self.client = _client()

    def test_rejected_secret_is_not_reflected(self):
        sa = dict(_VALID_SA, password="super-secret-value-42")
        resp = self.client.post("/api/mcp/servers", json=_body(service_account=sa))

        assert resp.status_code == 422
        assert "super-secret-value-42" not in resp.text

    def test_error_entries_still_name_the_field_and_the_reason(self):
        sa = dict(_VALID_SA, token_url={"nested": "https://evil.example"})
        resp = self.client.post("/api/mcp/servers", json=_body(service_account=sa))

        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert isinstance(detail, list) and detail
        entry = detail[0]
        # The contract clients rely on is preserved; only the value is gone.
        assert "loc" in entry and "msg" in entry and "type" in entry
        assert "input" not in entry
        assert "token_url" in entry["loc"]
        assert "evil.example" not in resp.text
