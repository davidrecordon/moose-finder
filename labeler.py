#!/usr/bin/env python3
"""Local labelling page: mark which cached photos contain Moose.

Usage:
    python labeler.py                         # all cached photos, in date order
    python labeler.py --order candidates.json  # photos listed in the file first
    python labeler.py --order candidates.json --only  # just those photos

Serves http://127.0.0.1:8765 and saves every click to labels/labels.json as
{"2026-11-15/photo.jpg": "moose" | "lookalike" | "unsure" | "not_moose"}.
"""

import argparse
import json
import os
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
CACHE_DIR = PROJECT_DIR / "cache"
LABELS_PATH = PROJECT_DIR / "labels" / "labels.json"
VALID_LABELS = {"moose", "lookalike", "unsure", "not_moose"}
PHOTO_KEY = re.compile(
    r"^\d{4}-\d{2}-\d{2}/[A-Za-z0-9._-]+\.(jpg|jpeg|png)$", re.IGNORECASE
)

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Label Moose</title>
<style>
:root { --bg:#f6f5f2; --fg:#1d1c1a; --muted:#6b6862; --card:#fff; --line:#dcd9d2;
        --moose:#1f8a4c; --look:#c2410c; --unsure:#a16207; --not:#8a8780; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#161513; --fg:#eceae5; --muted:#a09c94; --card:#211f1c; --line:#3a3732; --not:#6f6c66; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.4 system-ui, sans-serif; }
header { position:sticky; top:0; z-index:5; background:var(--bg); border-bottom:1px solid var(--line);
         padding:10px 16px; display:flex; flex-wrap:wrap; gap:10px 18px; align-items:center; }
header h1 { font-size:16px; margin:0; }
#counts { color:var(--muted); font-variant-numeric:tabular-nums; }
.spacer { flex:1; }
button { font:inherit; color:var(--fg); background:var(--card); border:1px solid var(--line);
         border-radius:8px; padding:7px 12px; cursor:pointer; }
button.primary { background:var(--fg); color:var(--bg); border-color:var(--fg); }
button:disabled { opacity:.4; cursor:default; }
p.help { margin:12px 16px 0; color:var(--muted); max-width:70ch; }
#grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(230px, 1fr)); gap:12px; padding:14px 16px 40px; }
.tile { background:var(--card); border:3px solid var(--line); border-radius:10px; overflow:hidden; }
.tile img { display:block; width:100%; height:260px; object-fit:contain; background:#0002; cursor:pointer; }
.tile .row { display:flex; gap:6px; padding:6px; align-items:center; }
.tile .row button { padding:4px 8px; font-size:13px; }
.tile .name { color:var(--muted); font-size:11px; padding:0 8px 6px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.tile .state { margin-left:auto; font-size:12px; font-weight:600; }
.tile.moose { border-color:var(--moose); } .tile.moose .state { color:var(--moose); }
.tile.lookalike { border-color:var(--look); } .tile.lookalike .state { color:var(--look); }
.tile.unsure { border-color:var(--unsure); } .tile.unsure .state { color:var(--unsure); }
.tile.not_moose { opacity:.55; } .tile.not_moose .state { color:var(--not); }
#zoom { position:fixed; inset:0; background:#000d; display:none; z-index:10; cursor:zoom-out; }
#zoom img { width:100%; height:100%; object-fit:contain; }
</style></head><body>
<header>
  <h1>Is Moose in the photo?</h1>
  <span id="counts"></span>
  <span class="spacer"></span>
  <button id="prev">Back</button>
  <span id="pageno"></span>
  <button id="rest" class="primary">Rest of this page: not Moose, next</button>
</header>
<p class="help">Click a photo if Moose is anywhere in it. Use <b>Lookalike</b> for a different dog you could
mistake for Moose (the other Bernese), and <b>Unsure</b> if you can't tell. Then press the button at top right
to mark everything else on the page as not Moose and move on. Every click saves immediately.</p>
<div id="grid"></div>
<div id="zoom"><img alt=""></div>
<script>
const PER_PAGE = 24;
const NAMES = {moose:"Moose", lookalike:"Lookalike", unsure:"Unsure", not_moose:"Not Moose"};
let photos = [], labels = {}, page = 0;

async function save(key, label) {
  if (label) labels[key] = label; else delete labels[key];
  await fetch("/api/label", {method:"POST", headers:{"Content-Type":"application/json"},
                             body: JSON.stringify({photo:key, label:label})});
}
function counts() {
  const n = {moose:0, lookalike:0, unsure:0, not_moose:0};
  for (const k of photos) if (labels[k]) n[labels[k]]++;
  const done = n.moose + n.lookalike + n.unsure + n.not_moose;
  document.getElementById("counts").textContent =
    `${done} of ${photos.length} labelled · ${n.moose} Moose · ${n.lookalike} lookalike · ${n.unsure} unsure`;
}
function render() {
  const grid = document.getElementById("grid"); grid.textContent = "";
  const pages = Math.ceil(photos.length / PER_PAGE);
  for (const key of photos.slice(page * PER_PAGE, (page + 1) * PER_PAGE)) {
    const tile = document.createElement("div");
    const paint = () => { tile.className = "tile " + (labels[key] || ""); state.textContent = NAMES[labels[key]] || ""; };
    const set = async (l) => { await save(key, labels[key] === l ? null : l); paint(); counts(); };
    const img = document.createElement("img"); img.loading = "lazy"; img.src = "/photo/" + key; img.alt = key;
    img.onclick = () => set("moose");
    const row = document.createElement("div"); row.className = "row";
    for (const l of ["lookalike", "unsure"]) {
      const b = document.createElement("button"); b.textContent = NAMES[l]; b.onclick = () => set(l); row.append(b);
    }
    const z = document.createElement("button"); z.textContent = "Zoom";
    z.onclick = () => { const o = document.getElementById("zoom"); o.firstChild.src = img.src; o.style.display = "block"; };
    const state = document.createElement("span"); state.className = "state";
    row.append(z, state);
    const name = document.createElement("div"); name.className = "name"; name.textContent = key;
    tile.append(img, row, name); paint(); grid.append(tile);
  }
  document.getElementById("pageno").textContent = `Page ${page + 1} of ${pages}`;
  document.getElementById("prev").disabled = page === 0;
  document.getElementById("rest").textContent =
    page + 1 < pages ? "Rest of this page: not Moose, next" : "Rest of this page: not Moose, finish";
  counts(); window.scrollTo(0, 0);
}
document.getElementById("zoom").onclick = (e) => { e.currentTarget.style.display = "none"; };
document.getElementById("prev").onclick = () => { page--; render(); };
document.getElementById("rest").onclick = async () => {
  for (const key of photos.slice(page * PER_PAGE, (page + 1) * PER_PAGE))
    if (!labels[key]) await save(key, "not_moose");
  if ((page + 1) * PER_PAGE < photos.length) page++;
  render();
};
(async () => {
  const s = await (await fetch("/api/state")).json();
  photos = s.photos; labels = s.labels;
  const first = photos.findIndex(k => !labels[k]);
  page = first < 0 ? 0 : Math.floor(first / PER_PAGE);
  render();
})();
</script></body></html>
"""


def load_labels() -> dict:
    if LABELS_PATH.exists():
        return json.loads(LABELS_PATH.read_text())
    return {}


def save_labels(labels: dict):
    LABELS_PATH.parent.mkdir(exist_ok=True)
    tmp = LABELS_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(labels, indent=1, sort_keys=True))
    os.replace(tmp, LABELS_PATH)


def list_photos(order_file: Path | None, only: bool = False) -> list[str]:
    """All cached photos as 'date/filename', with any --order photos first."""
    cached = sorted(
        f"{p.parent.name}/{p.name}"
        for p in CACHE_DIR.glob("*/*")
        if PHOTO_KEY.match(f"{p.parent.name}/{p.name}")
    )
    if not order_file:
        return cached
    have = set(cached)
    first = [k for k in json.loads(order_file.read_text()) if k in have]
    seen = set(first)
    if only:
        return first
    return first + [k for k in cached if k not in seen]


def make_handler(photos: list[str]):
    known = set(photos)
    labels = load_labels()
    lock = threading.Lock()  # one writer at a time: the dict and the file are shared

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            elif self.path == "/api/state":
                with lock:
                    body = json.dumps({"photos": photos, "labels": labels}).encode()
                self._send(200, body, "application/json")
            elif self.path.startswith("/photo/") and self.path[7:] in known:
                self._send(200, (CACHE_DIR / self.path[7:]).read_bytes(), "image/jpeg")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path != "/api/label":
                return self._send(404, b"not found", "text/plain")
            # Only this page may save labels: refuse other sites' cross-origin
            # posts (which cannot send JSON without a preflight) and DNS rebinding.
            host = self.headers.get("Host", "").split(":")[0]
            json_body = self.headers.get("Content-Type", "").startswith(
                "application/json"
            )
            if host not in ("127.0.0.1", "localhost") or not json_body:
                return self._send(403, b"forbidden", "text/plain")
            try:
                data = json.loads(
                    self.rfile.read(int(self.headers.get("Content-Length", 0)))
                )
                photo, label = data["photo"], data["label"]
            except ValueError, KeyError, TypeError:
                return self._send(400, b"bad request", "text/plain")
            if photo not in known or (label is not None and label not in VALID_LABELS):
                return self._send(400, b"bad request", "text/plain")
            with lock:
                if label is None:
                    labels.pop(photo, None)
                else:
                    labels[photo] = label
                save_labels(labels)
            self._send(200, b"{}", "application/json")

        def log_message(self, *args):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser(
        description="Label which cached photos contain Moose"
    )
    parser.add_argument(
        "--order", type=Path, help="JSON list of 'date/filename' keys to show first"
    )
    parser.add_argument(
        "--only",
        action="store_true",
        help="Show only the photos listed in --order",
    )
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    photos = list_photos(args.order, args.only)
    if not photos:
        print(f"No cached photos found in {CACHE_DIR}")
        return
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(photos))
    print(
        f"Labelling {len(photos)} photos at http://127.0.0.1:{args.port}  (Ctrl-C to stop)"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
