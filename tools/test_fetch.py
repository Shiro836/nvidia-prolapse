#!/usr/bin/env python3
"""Unit tests of tools/fetch.py (no network: file:// sources): python3 -m pytest -q tools/test_fetch.py"""
import hashlib
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch  # noqa: E402


@pytest.fixture(autouse=True)
def no_wait(monkeypatch):
    monkeypatch.setattr(fetch, 'BACKOFF', (0,))
    monkeypatch.setattr(fetch.time, 'sleep', lambda s: None)


def src(tmp_path, name, data):
    p = tmp_path / name
    p.write_bytes(data)
    return p.as_uri()


def test_second_mirror_used_when_first_fails(tmp_path):
    good = src(tmp_path, 'good', b'payload')
    out = fetch.fetch([(tmp_path / 'missing').as_uri(), good], tmp_path / 'out',
                      hashlib.sha256(b'payload').hexdigest(), hashlib.md5(b'payload').hexdigest(), say=lambda t: None)
    assert out.read_bytes() == b'payload' and not list(tmp_path.glob('out.part*'))


def test_wrong_bytes_never_kept(tmp_path):
    bad = src(tmp_path, 'bad', b'other')
    with pytest.raises(SystemExit, match='no source gave the pinned bytes'):
        fetch.fetch([bad], tmp_path / 'out', hashlib.sha256(b'payload').hexdigest(), say=lambda t: None)
    assert not (tmp_path / 'out').exists() and not list(tmp_path.glob('out.part*'))


def test_md5_pin_checked(tmp_path):
    good = src(tmp_path, 'good', b'payload')
    with pytest.raises(SystemExit):
        fetch.fetch([good], tmp_path / 'out', hashlib.sha256(b'payload').hexdigest(), '0' * 32, say=lambda t: None)


def test_existing_verified_file_kept(tmp_path):
    (tmp_path / 'out').write_bytes(b'payload')
    assert fetch.fetch(['file:///nonexistent'], tmp_path / 'out', hashlib.sha256(b'payload').hexdigest(),
                       say=lambda t: None).read_bytes() == b'payload'


def test_retry_run_rounds_and_cleanup(tmp_path):
    marker, junk = tmp_path / 'n', tmp_path / 'junk'
    junk.mkdir()
    # fails twice, succeeds on the third round; cleanup removes the partial directory between rounds
    script = f'n=$(cat {marker} 2>/dev/null || echo 0); echo $((n+1)) > {marker}; [ $n -ge 2 ]'
    fetch.retry_run(['sh', '-c', script], cleanup=junk, say=lambda t: None)
    assert marker.read_text().strip() == '3' and not junk.exists()
    with pytest.raises(SystemExit, match='failed 4 times'):
        fetch.retry_run(['false'], say=lambda t: None)


def test_sha256_required(tmp_path):
    with pytest.raises(SystemExit, match='sha256 pin is required'):
        fetch.fetch(['file:///x'], tmp_path / 'out', '')


def test_trickling_source_given_up_for_the_next(tmp_path, monkeypatch):
    """A source that keeps the connection alive but sends below the floor is abandoned (the hosted hwcomposer
    job once waited 42 min on such a SourceForge mirror); the next source delivers."""
    import http.server
    import threading
    payload = b'x' * (256 * 1024)

    class Trickle(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            for i in range(0, len(payload), 1024):     # 1 KiB every 50 ms: about 20 KiB/s
                self.wfile.write(payload[i:i + 1024])
                self.wfile.flush()
                threading.Event().wait(0.05)    # time.sleep is patched out by no_wait

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Trickle)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(fetch, 'RATE_WINDOW', 0.5)
    monkeypatch.setenv('FETCH_MIN_KIBPS', '100')
    said = []
    try:
        out = fetch.fetch([f'http://127.0.0.1:{server.server_port}/img', src(tmp_path, 'good', payload)],
                          tmp_path / 'out', hashlib.sha256(payload).hexdigest(), say=said.append)
    finally:
        server.shutdown()
    assert out.read_bytes() == payload and any('below the 100 KiB/s floor' in line for line in said)
