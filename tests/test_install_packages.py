"""Exercise real pip against a local index serving intentionally damaged wheels."""
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import os
import threading
import zipfile

import pytest

from scripts.install_packages import install


@pytest.fixture
def package_index(tmp_path, monkeypatch):
    # Restrict the integration test to its local index, target and cache.
    for name in list(os.environ):
        if name.startswith("PIP_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("PIP_CONFIG_FILE", os.devnull)
    monkeypatch.setenv("PIP_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    wheel = io.BytesIO()
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("zap_download_probe/__init__.py", "VALUE = 123\n")
        info = "zap_download_probe-0.0.1.dist-info/"
        archive.writestr(info + "METADATA", "Metadata-Version: 2.1\nName: zap-download-probe\nVersion: 0.0.1\n")
        archive.writestr(info + "WHEEL", "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
        archive.writestr(info + "RECORD", "")
    good = wheel.getvalue()
    digest = hashlib.sha256(good).hexdigest()
    state = {"bad_downloads": 1, "wheel_requests": 0, "index_requests": 0}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path.endswith(".whl"):
                state["wheel_requests"] += 1
                body = b"truncated wheel" if state["wheel_requests"] <= state["bad_downloads"] else good
                content_type = "application/octet-stream"
            elif self.path == "/simple/zap-download-probe/":
                state["index_requests"] += 1
                body = (f'<a href="/zap_download_probe-0.0.1-py3-none-any.whl#sha256={digest}">'
                        'zap_download_probe-0.0.1-py3-none-any.whl</a>').encode()
                content_type = "text/html"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "public, max-age=3600")
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/simple", state
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_bad_hash_is_redownloaded_and_only_good_wheel_installed(package_index, tmp_path):
    index, state = package_index
    target = tmp_path / "target"
    log = tmp_path / "install.log"
    result = install(["zap-download-probe==0.0.1", "--target", str(target), "--no-deps"], index, log)
    assert result == 0
    assert state["wheel_requests"] == 2
    assert (target / "zap_download_probe/__init__.py").read_text() == "VALUE = 123\n"
    assert "do not match the hashes" in log.read_text().lower()


def test_persistent_bad_hash_stops_without_installing(package_index, tmp_path):
    index, state = package_index
    state["bad_downloads"] = 100
    target = tmp_path / "target"
    result = install(["zap-download-probe==0.0.1", "--target", str(target), "--no-deps"],
                     index, tmp_path / "install.log")
    assert result != 0
    assert state["wheel_requests"] == 3
    assert not (target / "zap_download_probe").exists()


def test_unavailable_version_is_not_retried(package_index, tmp_path):
    index, state = package_index
    result = install(["zap-download-probe==9.9.9", "--target", str(tmp_path / "target"), "--no-deps"],
                     index, tmp_path / "install.log")
    assert result != 0
    assert state["index_requests"] == 1
    assert state["wheel_requests"] == 0
