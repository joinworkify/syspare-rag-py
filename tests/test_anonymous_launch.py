import json
from types import SimpleNamespace

import rag_server


def _request(headers=None, host="203.0.113.10"):
    return SimpleNamespace(
        headers=headers or {},
        client=SimpleNamespace(host=host),
    )


def setup_function():
    rag_server._rate_limit_events.clear()


def teardown_function():
    rag_server._rate_limit_events.clear()


def test_anonymous_request_has_no_chat_history_context(monkeypatch):
    monkeypatch.delenv("CHAT_HISTORY_DEV_USER", raising=False)

    assert rag_server._chat_history_auth_context(_request()) is None


def test_authenticated_header_creates_chat_history_context():
    ctx = rag_server._chat_history_auth_context(
        _request(headers={"X-User-Id": "farmer@example.com"})
    )

    assert ctx is not None
    assert ctx["identity"] == "farmer@example.com"
    assert ctx["user_key"] != "farmer@example.com"


def test_rate_limit_uses_proxy_ip_when_enabled(monkeypatch):
    monkeypatch.setattr(rag_server, "TRUST_PROXY_HEADERS", True)

    request = _request(
        headers={"X-Forwarded-For": "198.51.100.25, 10.0.0.1"},
        host="127.0.0.1",
    )

    assert rag_server._rate_limit_key(request) == "ip:198.51.100.25"


def test_rate_limit_blocks_after_configured_window(monkeypatch):
    monkeypatch.setattr(rag_server, "RATE_LIMIT_ENABLED", True)
    request = _request()

    assert (
        rag_server._check_rate_limit(
            request,
            scope="test_chat",
            policies=[("minute", 2, 60)],
        )
        is None
    )
    assert (
        rag_server._check_rate_limit(
            request,
            scope="test_chat",
            policies=[("minute", 2, 60)],
        )
        is None
    )

    response = rag_server._check_rate_limit(
        request,
        scope="test_chat",
        policies=[("minute", 2, 60)],
    )

    assert response is not None
    assert response.status_code == 429
    assert int(response.headers["Retry-After"]) > 0
    body = json.loads(response.body.decode("utf-8"))
    assert body["retry_after_seconds"] > 0
