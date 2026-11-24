#!/usr/bin/env python3
"""Moose Finder: find Moose in Pet Camp's daily photos.

Usage:
    python moose_finder.py                       # today
    python moose_finder.py --date 2026-11-24     # one day
    python moose_finder.py --since 2026-11-24    # every day from then to today
    python moose_finder.py --since 2026-11-14 --until 2026-11-20
    python moose_finder.py --no-download         # only photos already in cache/
    python moose_finder.py --download-only       # fetch photos, skip analysis
    python moose_finder.py --photo some.jpg      # explain one photo in detail

For each day it downloads new photos, describes every dog in them
(pipeline.py), decides which is Moose (matcher.py), prints what it found, and
writes results/<date>.json plus a results/<date>.html gallery.
"""

import argparse
import html
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from PIL import Image

import scraper
from matcher import MODEL_PATH, MooseModel, PhotoResult

PROJECT_DIR = Path(__file__).parent
RESULTS_DIR = PROJECT_DIR / "results"
CAMP_TIMEZONE = ZoneInfo("America/Los_Angeles")

_analyzer = None


def get_analyzer():
    """Load the models on first use only (a fully analysed day never needs them)."""
    global _analyzer
    if _analyzer is None:
        from pipeline import Analyzer

        print("Loading models...", flush=True)
        _analyzer = Analyzer()
        print(f"  using {_analyzer.device}", flush=True)
    return _analyzer


def progress(done: int, total: int):
    if done == total or done % 25 == 0:
        print(f"  analysed {done}/{total}", flush=True)


def daterange(start: date, end: date):
    while start <= end:
        yield start
        start += timedelta(days=1)


def describe(result: PhotoResult) -> str:
    parts = [f"body {result.similarity:.2f}"]
    if result.face_probability is not None:
        parts.append(f"face {result.face_probability:.0%}")
    else:
        parts.append("no face visible")
    return ", ".join(parts)


def write_gallery(day: str, results: list[PhotoResult], manifest: dict) -> Path:
    """A self-contained page of the day's matches, with the chosen dog boxed."""
    sections = []
    for verdict, title in (
        ("match", "Moose"),
        ("possible", "Possibly Moose (no clear view of his face)"),
    ):
        tiles = []
        for r in sorted(
            (r for r in results if r.verdict == verdict), key=lambda r: -r.score
        ):
            path = scraper.CACHE_DIR / day / r.photo
            try:
                with Image.open(path) as image:
                    width, height = image.size
            except OSError:  # analysed earlier, but the file has since gone
                continue
            box = ""
            if r.box:
                x1, y1, x2, y2 = r.box
                box = (
                    f'<span class="box" style="left:{x1 / width:.2%};top:{y1 / height:.2%};'
                    f'width:{(x2 - x1) / width:.2%};height:{(y2 - y1) / height:.2%}"></span>'
                )
            link = html.escape(manifest.get("photos", {}).get(r.photo, ""))
            tiles.append(
                f'<figure><a href="{link or "#"}"><span class="frame">'
                f'<img loading="lazy" src="../cache/{day}/{html.escape(r.photo)}" alt="">{box}'
                f"</span></a><figcaption>{html.escape(describe(r))}</figcaption></figure>"
            )
        if tiles:
            sections.append(
                f"<h2>{title} · {len(tiles)}</h2><div class=grid>{''.join(tiles)}</div>"
            )
    if not sections:
        sections.append("<p>No sign of Moose in this day's photos.</p>")
    page_link = html.escape(manifest.get("page", ""))
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Moose · {day}</title>
<style>
:root {{ --bg:#f6f5f2; --fg:#1d1c1a; --muted:#6b6862; --box:#1f8a4c; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#161513; --fg:#eceae5; --muted:#a09c94; --box:#4ade80; }} }}
body {{ margin:0; padding:16px; background:var(--bg); color:var(--fg); font:15px/1.4 system-ui, sans-serif; }}
h1 {{ font-size:20px; margin:0 0 4px; }} h2 {{ font-size:16px; margin:24px 0 10px; }}
p, a {{ color:var(--muted); }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(260px, 1fr)); gap:14px; }}
figure {{ margin:0; }} figcaption {{ color:var(--muted); font-size:13px; padding-top:4px; }}
.frame {{ position:relative; display:block; }} .frame img {{ display:block; width:100%; border-radius:8px; }}
.box {{ position:absolute; border:3px solid var(--box); border-radius:6px; box-sizing:border-box; }}
</style>
<h1>Moose on {day}</h1>
<p>{len(results)} photos checked. <a href="{page_link}">Pet Camp's page for this day</a></p>
{''.join(sections)}
</html>"""
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{day}.html"
    out.write_text(page)
    return out


def process_day(day: date, args, model: MooseModel | None) -> int:
    """Download, analyse and report one day. Returns the number of matches."""
    from pipeline import analyze_day

    key = day.isoformat()
    print(f"\n{key}")
    if not args.no_download:
        fetched = scraper.fetch_day(day, limit=args.download_limit)
        if not fetched["pages"] or not fetched["total"]:
            print("  no photos posted (yet)")
        else:
            print(f"  {fetched['total']} photos on the page, {fetched['new']} new")
        for error in fetched["errors"][:5]:
            print(f"  {error}")
    photos = scraper.cached_photos(key)
    if not photos or args.download_only:
        return 0

    features = analyze_day(key, get_analyzer, progress)
    results = model.score_photos(features)
    manifest_path = scraper.CACHE_DIR / key / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"{key}.json").write_text(
        json.dumps(
            {
                "date": key,
                "page": manifest.get("page"),
                "photos": [
                    {**r.to_dict(), "url": manifest.get("photos", {}).get(r.photo)}
                    for r in sorted(results, key=lambda r: -r.score)
                    if r.verdict != "no"
                ],
                "checked": len(results),
            },
            indent=1,
        )
    )
    matches = sorted(
        (r for r in results if r.verdict == "match"), key=lambda r: -r.score
    )
    possible = sorted(
        (r for r in results if r.verdict == "possible"), key=lambda r: -r.score
    )
    lookalikes = sum(r.verdict == "lookalike" for r in results)
    print(
        f"  {len(results)} photos checked: {len(matches)} Moose, {len(possible)} possible, "
        f"{lookalikes} lookalikes ruled out by face"
    )
    for title, group in (("Moose", matches), ("Possible", possible)):
        for r in group:
            url = manifest.get("photos", {}).get(r.photo, r.photo)
            print(f"  {title:8s} {describe(r):28s} {url}")
    # Always rewritten, so a re-run that finds nothing replaces an older gallery.
    gallery = write_gallery(key, results, manifest)
    if matches or possible:
        print(f"  gallery: {gallery}")
    return len(matches)


def explain_photo(path: Path, model: MooseModel):
    """Print every dog found in one photo and how each stage judged it."""
    from pipeline import Features

    with Image.open(path) as image:
        analysis = get_analyzer().analyze(image)
    features = Features()
    features.add(path.name, analysis)
    items = model.score_items(features)
    names = {3: "match", 2: "possible", 1: "lookalike", 0: "no"}
    print(f"\n{path}: {len(features.boxes) - 1} dogs found")
    for i in sorted(range(len(features.boxes)), key=lambda i: -items["score"][i]):
        box = features.boxes[i].tolist()
        where = "whole frame" if box[0] < 0 else f"dog at {box}"
        face = items["face_p"][i]
        face_text = "no usable face" if face != face else f"face {face:.0%}"
        print(
            f"  {names[int(items['verdict'][i])]:9s} body {items['similarity'][i]:.3f}  "
            f"{face_text:15s} body-classifier {items['body_p'][i]:.0%}  {where}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Find Moose in Pet Camp's daily photos"
    )
    parser.add_argument("--date", help="Day to check (YYYY-MM-DD); default today")
    parser.add_argument("--since", help="First day of a range (YYYY-MM-DD)")
    parser.add_argument("--until", help="Last day of a range; default today")
    parser.add_argument(
        "--no-download",
        "--analyze-only",
        action="store_true",
        dest="no_download",
        help="Only use photos already in cache/",
    )
    parser.add_argument(
        "--download-only", action="store_true", help="Download photos without analysing"
    )
    parser.add_argument(
        "--download-limit",
        type=int,
        help="Download at most this many new photos per day",
    )
    parser.add_argument("--photo", type=Path, help="Explain a single local photo")
    args = parser.parse_args()
    if args.until and not args.since:
        parser.error("--until needs --since")
    if args.date and args.since:
        parser.error("use either --date or --since/--until, not both")
    if args.photo and args.download_only:
        parser.error("--photo cannot be combined with --download-only")

    model = None
    if not args.download_only:
        if not MODEL_PATH.exists():
            sys.exit(f"{MODEL_PATH} is missing. Run: python train.py")
        model = MooseModel.load()

    if args.photo:
        return explain_photo(args.photo, model)

    today = datetime.now(CAMP_TIMEZONE).date()
    if args.since:
        start = date.fromisoformat(args.since)
        end = date.fromisoformat(args.until) if args.until else today
    else:
        start = end = date.fromisoformat(args.date) if args.date else today
    if start > end:
        parser.error(f"--since {start} is after --until {end}")
    total = sum(process_day(day, args, model) for day in daterange(start, end))
    if start != end:
        print(f"\n{total} Moose photos from {start} to {end}")


if __name__ == "__main__":
    main()
