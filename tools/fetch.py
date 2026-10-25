#!/usr/bin/env python3
"""Pinned downloads that one flaky host cannot fail: mirrors, retries with backoff, verified before kept.

  python3 tools/fetch.py --out FILE --sha256 HEX [--md5 HEX] URL [URL ...]
  tools/fetch.py retry [--cleanup PATH] -- CMD ...        a network command (git fetch/clone, cargo, meson),
                                                          same backoff; --cleanup PATH is removed between tries

From Python: `from fetch import fetch, retry_run, prefetch_wraps`.

fetch(): an existing FILE with the pinned hashes is kept as is. Otherwise every URL is tried in order, in up
to ATTEMPTS rounds with BACKOFF seconds between rounds; each download goes to FILE.part and is renamed only
after the sha256 (and md5, when pinned) match. A mismatching download is discarded and the next source tried;
when every source failed, the error lists each URL with its last error. Nothing unverified is ever kept.
A source that stalls or trickles is given up instead of being waited on: no data for TIMEOUT seconds, or less than
FETCH_MIN_KIBPS KiB/s (default 128; 0 = no floor) over any RATE_WINDOW-second stretch, ends that attempt and the
next source (or round) is tried; a redirecting source such as SourceForge's /download picks another mirror on the
next try. Long downloads print their progress every PROGRESS_EVERY seconds.
"""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request

ATTEMPTS = 4
BACKOFF = (10, 30, 90)          # seconds before round 2, 3, 4
TIMEOUT = 60                    # per connection (socket inactivity), not per file
RATE_WINDOW = 60                # seconds over which the transfer rate must stay above the floor
PROGRESS_EVERY = 30             # seconds between progress lines


def _digest(path, name):
    h = hashlib.new(name)
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _verified(path, sha256, md5):
    return _digest(path, 'sha256') == sha256 and (md5 is None or _digest(path, 'md5') == md5)


def _min_rate():
    """bytes/s floor: FETCH_MIN_KIBPS (KiB/s), default 128, 0 disables."""
    return int(float(os.environ.get('FETCH_MIN_KIBPS', '128')) * 1024)


def _copy(src, dst, name, say):
    """Copy the response body; give up on a source that sends less than the floor over RATE_WINDOW seconds."""
    floor = _min_rate()
    length = src.headers.get('Content-Length')
    size = f' of {int(length) >> 20} MiB' if length and length.isdigit() else ''
    start = window_start = last_report = time.monotonic()
    total = window_bytes = 0
    while True:
        chunk = src.read1(1 << 16)      # returns what has arrived; a trickle cannot block a whole 1 MiB read
        if not chunk:
            break
        dst.write(chunk)
        total += len(chunk)
        window_bytes += len(chunk)
        now = time.monotonic()
        if now - window_start >= RATE_WINDOW:
            rate = window_bytes / (now - window_start)
            if floor and rate < floor:
                raise TimeoutError(f'{rate / 1024:.0f} KiB/s over {RATE_WINDOW} s, below the '
                                   f'{floor // 1024} KiB/s floor (FETCH_MIN_KIBPS), after {total >> 20} MiB')
            window_start, window_bytes = now, 0
        if now - last_report >= PROGRESS_EVERY:
            say(f'fetch: {name}: {total >> 20} MiB{size}, {total / (now - start) / (1 << 20):.1f} MiB/s')
            last_report = now


def fetch(urls, out, sha256, md5=None, say=print):
    """Download `out` from the first source in `urls` that yields the pinned bytes. Returns Path(out)."""
    if not sha256 or len(sha256) != 64:
        raise SystemExit(f'fetch: {out}: a sha256 pin is required')
    urls = [urls] if isinstance(urls, str) else list(urls)
    out = Path(out)
    if out.is_file() and _verified(out, sha256, md5):
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(f'{out.name}.part-{os.getpid()}')   # concurrent builds may fetch the same file
    errors = {}
    for attempt in range(ATTEMPTS):
        if attempt:
            delay = BACKOFF[min(attempt - 1, len(BACKOFF) - 1)]
            say(f'fetch: {out.name}: round {attempt + 1}/{ATTEMPTS} in {delay} s')
            time.sleep(delay)
        for url in urls:
            try:
                request = urllib.request.Request(url, headers={'User-Agent': 'waydroid-nvidia-build/1'})
                with urllib.request.urlopen(request, timeout=TIMEOUT) as src, open(part, 'wb') as dst:
                    _copy(src, dst, out.name, say)
                if not _verified(part, sha256, md5):
                    raise ValueError(f'hash mismatch (got sha256 {_digest(part, "sha256")[:16]}...)')
                os.replace(part, out)
                return out
            except Exception as error:          # network, HTTP, TLS, hash: try the next source
                errors[url] = f'{type(error).__name__}: {error}'
                say(f'fetch: {url}: {errors[url]}')
                part.unlink(missing_ok=True)
    raise SystemExit(f'fetch: {out.name}: no source gave the pinned bytes (sha256 {sha256}) after {ATTEMPTS} '
                     'rounds:\n  ' + '\n  '.join(f'{u}: {e}' for u, e in errors.items()))


def retry_run(cmd, cleanup=None, say=print, **kwargs):
    """subprocess.run(cmd, check=True) for a network command, with the same rounds and backoff."""
    for attempt in range(ATTEMPTS):
        if attempt:
            delay = BACKOFF[min(attempt - 1, len(BACKOFF) - 1)]
            say(f'retry: {" ".join(map(str, cmd))}: round {attempt + 1}/{ATTEMPTS} in {delay} s')
            time.sleep(delay)
            if cleanup:
                shutil.rmtree(cleanup, ignore_errors=True)
        result = subprocess.run([str(c) for c in cmd], **kwargs)
        if result.returncode == 0:
            return result
    raise SystemExit(f'retry: {" ".join(map(str, cmd))} failed {ATTEMPTS} times (exit {result.returncode})')


def prefetch_wraps(source, say=print):
    """Download a Meson source tree's subprojects (wraps) up front, with retries, so `meson setup` does not
    depend on the network; each wrap is checked against its own source_hash by Meson."""
    retry_run(['meson', 'subprojects', 'download'], cwd=source, say=say)


def main(argv):
    if argv[:1] == ['retry']:
        argv = argv[1:]
        cleanup = None
        if argv[:1] == ['--cleanup']:
            cleanup, argv = argv[1], argv[2:]
        if argv[:1] == ['--']:
            argv = argv[1:]
        if not argv:
            raise SystemExit(__doc__)
        retry_run(argv, cleanup=cleanup, say=lambda t: print(t, file=sys.stderr))
        return 0
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--md5')
    parser.add_argument('urls', nargs='+')
    args = parser.parse_args(argv)
    fetch(args.urls, args.out, args.sha256, args.md5, say=lambda t: print(t, file=sys.stderr))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
