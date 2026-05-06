#!/usr/bin/env python3
"""
serve_viewer.py — Local HTTP server for the threejs_loop interactive viewer.

Usage:
  python serve_viewer.py                   # auto-loads current best version
  python serve_viewer.py rock v3           # loads candidates/rock_v3.mjs directly
  python serve_viewer.py --port 9000       # custom port

Opens http://localhost:8765/viewer/ in Chrome (avoids Safari HTTPS-only block).
Auto-refreshes when the loop writes a new version (3-second poll in the viewer).
"""

import argparse
import http.server
import json
import re
import shutil
import socketserver
import subprocess
import sys
import threading
import time
from pathlib import Path

LOOP_DIR = Path(__file__).parent
DEFAULT_PORT = 8765

CHROME_PATHS = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary",
]


def open_in_chrome(url: str) -> bool:
    """Open url in Chrome. Returns True on success, False if Chrome not found."""
    for chrome in CHROME_PATHS:
        if Path(chrome).exists():
            subprocess.Popen([chrome, url])
            return True
    # Fallback: try 'open -a' (macOS)
    if shutil.which("open"):
        result = subprocess.run(
            ["open", "-a", "Google Chrome", url],
            capture_output=True,
        )
        if result.returncode == 0:
            return True
    return False


def _build_asset_list() -> list[dict]:
    """Scan candidates/ and dist/ and return sorted asset metadata."""
    assets = []
    cand_dir = LOOP_DIR / "candidates"
    dist_dir  = LOOP_DIR / "dist"

    # candidates/rock_v2.mjs  →  name=rock  version=v2
    # candidates/rock_v2_a.mjs → name=rock  version=v2_a  (A/B variant — skip)
    for f in sorted(cand_dir.glob("*_v*.mjs")):
        m = re.fullmatch(r"(.+)_(v\d+)\.mjs", f.name)
        if m:
            assets.append({"name": m[1], "version": m[2],
                           "path": f"candidates/{f.name}", "type": "candidate"})

    # dist/rock.mjs  →  name=rock  version=dist
    for f in sorted(dist_dir.glob("*.mjs")):
        if f.stem != "manifest":
            assets.append({"name": f.stem, "version": "dist",
                           "path": f"dist/{f.name}", "type": "dist"})
    return assets


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(LOOP_DIR), **kw)

    def do_GET(self):
        if self.path.split("?")[0] == "/api/assets":
            body = json.dumps(_build_asset_list()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
        else:
            super().do_GET()

    def log_message(self, fmt, *args):
        if args and str(args[1]) not in ("200", "304"):
            super().log_message(fmt, *args)


def main():
    parser = argparse.ArgumentParser(description="threejs_loop viewer server")
    parser.add_argument("asset_name", nargs="?", help="Asset name (e.g. 'rock')")
    parser.add_argument("version",    nargs="?", help="Version (e.g. 'v3')")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    if args.asset_name and args.version:
        asset_path = f"candidates/{args.asset_name}_{args.version}.mjs"
        url = f"http://localhost:{args.port}/viewer/?asset={asset_path}"
    elif args.asset_name:
        # Try to find the latest version
        candidates = sorted(LOOP_DIR.glob(f"candidates/{args.asset_name}_v*.mjs"))
        if candidates:
            rel = candidates[-1].relative_to(LOOP_DIR)
            url = f"http://localhost:{args.port}/viewer/?asset={rel}"
        else:
            url = f"http://localhost:{args.port}/viewer/"
    else:
        url = f"http://localhost:{args.port}/viewer/"

    with socketserver.TCPServer(("", args.port), QuietHandler) as httpd:
        httpd.allow_reuse_address = True
        print(f"Viewer  →  {url}")
        print(f"Serving  {LOOP_DIR}")
        print(f"Press Ctrl-C to stop\n")

        # Open in Chrome after short delay (avoids Safari HTTPS-only block)
        def open_browser():
            time.sleep(0.4)
            if not open_in_chrome(url):
                import webbrowser
                print("Chrome not found — opening in default browser")
                webbrowser.open(url)

        threading.Thread(target=open_browser, daemon=True).start()

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
