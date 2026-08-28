# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import http.server
import ipaddress
import socket
import threading

import pytest
import requests
from fastapi import HTTPException

from llamafactory.api.common import check_ssrf_url, fetch_safe_url


class _CountingHandler(http.server.BaseHTTPRequestHandler):
    r"""Base test HTTP handler that records how many requests it served."""

    response_body = b"OK"
    hit_count = 0

    def do_GET(self):  # noqa: N802
        type(self).hit_count += 1
        self.send_response(200)
        self.end_headers()
        self.wfile.write(self.response_body)

    def log_message(self, *args):
        pass  # keep test output clean


def _start_server(bind_ip: str, handler_cls: type[http.server.BaseHTTPRequestHandler], port: int = 0):
    server = http.server.HTTPServer((bind_ip, port), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


@pytest.fixture
def pretend_loopback_is_global(monkeypatch):
    r"""Make `ipaddress.ip_address("127.0.0.1").is_global` return True.

    All of 127.0.0.0/8 is loopback and therefore never actually global, so real SSRF payloads
    can't be built against it directly. This fixture lets tests stand a local HTTP server in for
    what would, in a real attack, be a public IP address the attacker controls (the first hop that
    `check_ssrf_url` is expected to allow), while 127.0.0.2 is left as a genuinely private/rejected
    address representing the internal target the attacker is trying to reach.
    """
    original_is_global = ipaddress.IPv4Address.is_global

    def patched_is_global(self):
        if str(self) == "127.0.0.1":
            return True
        return original_is_global.fget(self)

    monkeypatch.setattr(ipaddress.IPv4Address, "is_global", property(patched_is_global))


def test_check_ssrf_url_rejects_private_ip():
    with pytest.raises(HTTPException) as exc_info:
        check_ssrf_url("http://127.0.0.1/")
    assert exc_info.value.status_code == 403


def test_check_ssrf_url_rejects_non_http_scheme():
    with pytest.raises(HTTPException) as exc_info:
        check_ssrf_url("file:///etc/passwd")
    assert exc_info.value.status_code == 400


def test_check_ssrf_url_returns_resolved_ip(pretend_loopback_is_global):
    ip = check_ssrf_url("http://127.0.0.1:1234/")
    assert ip == "127.0.0.1"


def test_naive_redirect_follow_is_vulnerable_to_ssrf(pretend_loopback_is_global):
    r"""Demonstrates the bug in #10646: validating only the first hop is not enough.

    This mirrors the old, vulnerable call pattern (`check_ssrf_url(url)` followed by a plain
    `requests.get(url, stream=True)`, which follows redirects by default): the first hop passes
    the check, but nothing stops the HTTP client from then following a redirect straight into a
    private address.
    """
    _CountingHandler.hit_count = 0

    class InternalHandler(_CountingHandler):
        response_body = b"SECRET_INTERNAL_DATA"

    internal = _start_server("127.0.0.2", InternalHandler)
    internal_port = internal.server_address[1]

    class RedirectHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.2:{internal_port}/")
            self.end_headers()

        def log_message(self, *args):
            pass

    public = _start_server("127.0.0.1", RedirectHandler)
    url = f"http://127.0.0.1:{public.server_address[1]}/"

    check_ssrf_url(url)  # passes: 127.0.0.1 is "global" under this fixture
    response = requests.get(url, stream=True, timeout=5)  # old code's actual fetch call
    assert response.raw.read() == b"SECRET_INTERNAL_DATA"
    assert InternalHandler.hit_count == 1

    internal.shutdown()
    public.shutdown()


def test_fetch_safe_url_blocks_redirect_to_private_ip(pretend_loopback_is_global):
    r"""fetch_safe_url() must re-validate (and refuse to follow) a redirect into a private IP."""
    _CountingHandler.hit_count = 0

    class InternalHandler(_CountingHandler):
        response_body = b"SECRET_INTERNAL_DATA"

    internal = _start_server("127.0.0.2", InternalHandler)
    internal_port = internal.server_address[1]

    class RedirectHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.2:{internal_port}/")
            self.end_headers()

        def log_message(self, *args):
            pass

    public = _start_server("127.0.0.1", RedirectHandler)
    url = f"http://127.0.0.1:{public.server_address[1]}/"

    with pytest.raises(HTTPException) as exc_info:
        fetch_safe_url(url)

    assert exc_info.value.status_code == 403
    assert InternalHandler.hit_count == 0  # the internal server must never be reached

    internal.shutdown()
    public.shutdown()


def test_fetch_safe_url_follows_redirect_to_another_public_ip(pretend_loopback_is_global):
    r"""A redirect to a URL that also passes the SSRF check must still be followed."""
    _CountingHandler.hit_count = 0

    class TargetHandler(_CountingHandler):
        response_body = b"FINAL_DESTINATION"

    target = _start_server("127.0.0.1", TargetHandler)
    target_port = target.server_address[1]

    class RedirectHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{target_port}/final")
            self.end_headers()

        def log_message(self, *args):
            pass

    # A second server on the same (pretend-global) loopback address handles the redirect hop.
    redirector = _start_server("127.0.0.1", RedirectHandler)
    url = f"http://127.0.0.1:{redirector.server_address[1]}/"

    response = fetch_safe_url(url)
    assert response.content == b"FINAL_DESTINATION"
    assert TargetHandler.hit_count == 1

    target.shutdown()
    redirector.shutdown()


def test_naive_request_without_pinning_is_vulnerable_to_dns_rebinding(monkeypatch, pretend_loopback_is_global):
    r"""Demonstrates the DNS-rebinding half of #10646.

    A plain `socket.getaddrinfo`-based check followed by a *separate* connection attempt can
    resolve a hostname twice: once for the SSRF check, and again when the HTTP client actually
    connects. A malicious/compromised DNS server can answer differently the second time (the
    "rebind"), pointing the real connection at a private address after the check already passed.
    """
    _CountingHandler.hit_count = 0

    class PrivateHandler(_CountingHandler):
        response_body = b"SECRET_INTERNAL_DATA"

    private = _start_server("127.0.0.2", PrivateHandler)
    port = private.server_address[1]
    public = _start_server("127.0.0.1", _CountingHandler, port=port)

    real_getaddrinfo = socket.getaddrinfo
    calls = {"n": 0}

    def rebinding_getaddrinfo(host, *args, **kwargs):
        if host == "rebind.example.test":
            calls["n"] += 1
            # 1st resolution (the SSRF check) answers with the "public" address; every
            # subsequent resolution (i.e. what the HTTP client does when it connects) answers
            # with the private one.
            target = "127.0.0.1" if calls["n"] == 1 else "127.0.0.2"
            return real_getaddrinfo(target, *args, **kwargs)
        return real_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", rebinding_getaddrinfo)

    url = f"http://rebind.example.test:{port}/"
    check_ssrf_url(url)  # 1st resolution: passes, resolves to the "public" 127.0.0.1
    response = requests.get(url, timeout=5)  # 2nd resolution happens here: rebinds to 127.0.0.2
    assert response.content == b"SECRET_INTERNAL_DATA"
    assert PrivateHandler.hit_count == 1

    private.shutdown()
    public.shutdown()


def test_fetch_safe_url_pins_connection_against_dns_rebinding(monkeypatch, pretend_loopback_is_global):
    r"""fetch_safe_url() must connect to the IP it validated, immune to a second/different DNS answer."""
    _CountingHandler.hit_count = 0

    class SafeHandler(_CountingHandler):
        response_body = b"SAFE_DATA"

    class PrivateHandler(_CountingHandler):
        response_body = b"SECRET_INTERNAL_DATA"

    private = _start_server("127.0.0.2", PrivateHandler)
    port = private.server_address[1]
    safe = _start_server("127.0.0.1", SafeHandler, port=port)

    real_getaddrinfo = socket.getaddrinfo
    calls = {"n": 0}

    def rebinding_getaddrinfo(host, *args, **kwargs):
        if host == "rebind.example.test":
            calls["n"] += 1
            target = "127.0.0.1" if calls["n"] == 1 else "127.0.0.2"
            return real_getaddrinfo(target, *args, **kwargs)
        return real_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", rebinding_getaddrinfo)

    url = f"http://rebind.example.test:{port}/"
    response = fetch_safe_url(url)

    assert response.content == b"SAFE_DATA"
    assert PrivateHandler.hit_count == 0  # the connection must never reach the rebound address

    private.shutdown()
    safe.shutdown()
