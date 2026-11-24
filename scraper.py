"""Find and download Pet Camp's daily photos.

Each day's photos live on a WordPress post at campercameos.com. The post slugs
are inconsistent (weekday prefixes, "-2" duplicates, with/without "and"), so
the site's sitemap is the source of truth for which page belongs to which date.
Photos are saved as cache/<YYYY-MM-DD>/<original filename>, with a
manifest.json recording where each one came from.
"""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

import requests

PROJECT_DIR = Path(__file__).parent
CACHE_DIR = PROJECT_DIR / "cache"
SITEMAP_CACHE = CACHE_DIR / "sitemap.json"

BASE_URL = "https://campercameos.com"
PHOTO_PREFIX = "https://petcamp.s3.us-west-1.amazonaws.com/"
USER_AGENT = "Mozilla/5.0 (photo-downloader)"
MAX_DOWNLOAD_SIZE = 50 * 1024 * 1024
SITEMAP_MAX_AGE = (
    15 * 60
)  # seconds before a cached sitemap is re-fetched for a missing date
MONTHS = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
]
SLUG_DATE = re.compile(r"-([a-z]+)-(\d{1,2})(?:st|nd|rd|th)?-(\d{4})(?:-\d+)?$")
PHOTO_EXTENSIONS = (".jpg", ".jpeg", ".png")


def fetch(url: str, timeout: int = 30) -> bytes:
    """GET a URL, refusing anything over MAX_DOWNLOAD_SIZE."""
    with requests.get(
        url, headers={"User-Agent": USER_AGENT}, timeout=timeout, stream=True
    ) as response:
        response.raise_for_status()
        chunks, size = [], 0
        for chunk in response.iter_content(1 << 16):
            size += len(chunk)
            if size > MAX_DOWNLOAD_SIZE:
                raise ValueError(f"{url} is larger than {MAX_DOWNLOAD_SIZE} bytes")
            chunks.append(chunk)
    return b"".join(chunks)


def slug_date(url: str) -> date | None:
    """The date a main-campground post covers, or None for any other page."""
    if not url.startswith(BASE_URL + "/"):  # never follow a sitemap entry off-site
        return None
    slug = url.rstrip("/").rsplit("/", 1)[-1].lower()
    match = SLUG_DATE.search(slug)
    if not match or match.group(1) not in MONTHS:
        return None
    if "main" not in slug or "cat" in slug:
        return None
    try:
        return date(
            int(match.group(3)), MONTHS.index(match.group(1)) + 1, int(match.group(2))
        )
    except ValueError:
        return None


def ordinal(day: int) -> str:
    if 11 <= day <= 13:
        return f"{day}th"
    return f"{day}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th') }"


def guessed_pages(day: date) -> list[str]:
    """The usual slug patterns, for a post too new to be in the sitemap."""
    month = MONTHS[day.month - 1]
    return [
        f"{BASE_URL}/main-campground-{joiner}ranger-station-{month}-{number}-{day.year}/"
        for joiner in ("", "and-")
        for number in (str(day.day), ordinal(day.day))
    ]


def download_sitemap() -> dict[str, list[str]]:
    """Read every post sitemap and return {iso date: [page urls]}."""
    pages: dict[str, list[str]] = {}
    for number in range(1, 50):
        try:
            xml = fetch(f"{BASE_URL}/wp-sitemap-posts-post-{number}.xml").decode(
                "utf-8", "replace"
            )
        except requests.HTTPError as error:
            # Only "no such sitemap" marks the end; anything else is a failed read.
            if error.response is not None and error.response.status_code == 404:
                break
            raise
        for url in re.findall(r"<loc>([^<]+)</loc>", xml):
            day = slug_date(url)
            if day:
                pages.setdefault(day.isoformat(), []).append(url)
    return pages


def pages_for(day: date) -> list[str]:
    """Page URLs for a date, from the cached sitemap (refreshed if it's missing the date)."""
    cached = {}
    if SITEMAP_CACHE.exists():
        cached = json.loads(SITEMAP_CACHE.read_text())
        fresh = time.time() - SITEMAP_CACHE.stat().st_mtime < SITEMAP_MAX_AGE
        if day.isoformat() in cached or fresh:
            return cached.get(day.isoformat(), [])
    try:
        downloaded = download_sitemap()
    except requests.RequestException:
        downloaded = {}
    if not downloaded:  # keep the last good copy rather than caching a failure
        return cached.get(day.isoformat(), [])
    CACHE_DIR.mkdir(exist_ok=True)
    tmp = SITEMAP_CACHE.with_suffix(".tmp")
    tmp.write_text(json.dumps(downloaded))
    os.replace(tmp, SITEMAP_CACHE)
    return downloaded.get(day.isoformat(), [])


def photo_urls(html: str) -> list[str]:
    """Full-size photo URLs on a daily page, in page order, without duplicates."""
    urls, seen = [], set()
    for src in re.findall(r'<img[^>]+src="([^"]+)"', html):
        if not src.startswith(PHOTO_PREFIX) or "uploads" not in src:
            continue
        full = re.sub(r"-\d+x\d+(\.\w+)$", r"\1", src)  # thumbnail -> original
        if full not in seen and full.lower().endswith(PHOTO_EXTENSIONS):
            seen.add(full)
            urls.append(full)
    return urls


def local_names(
    urls: list[str], existing: dict[str, str] | None = None
) -> dict[str, str]:
    """Map each URL to a safe, unique local filename: {name: url}.

    Names already given out (an earlier manifest) are kept, so a file on disk
    always belongs to the URL the manifest says it does, even if the page's
    photo order changes between runs.
    """
    names: dict[str, str] = dict(existing or {})
    known = set(names.values())
    for url in urls:
        if url in known:
            continue
        name = re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(urlparse(url).path))
        stem, ext = os.path.splitext(name or "photo.jpg")
        candidate, n = stem + ext, 1
        while candidate in names:
            n += 1
            candidate = f"{stem}__{n}{ext}"
        names[candidate] = url
    return names


def _download(job: tuple[str, Path]) -> str:
    url, dest = job
    if dest.exists() and dest.stat().st_size > 0:
        return "cached"
    try:
        data = fetch(url, timeout=60)
    except (requests.RequestException, ValueError) as error:
        return f"error: {url}: {error}"
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(data)
    os.replace(tmp, dest)
    return "downloaded"


def fetch_day(day: date, limit: int | None = None) -> dict:
    """Download a day's photos. Returns {"pages", "total", "new", "errors"}."""
    pages, html = [], ""
    for url in pages_for(day) or guessed_pages(day):
        try:
            html += fetch(url).decode("utf-8", "replace")
            pages.append(url)
        except requests.RequestException:
            continue
    urls = photo_urls(html)
    if not urls:
        return {"pages": pages, "total": 0, "new": 0, "errors": []}
    out = CACHE_DIR / day.isoformat()
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    earlier = {}
    if manifest_path.exists():
        earlier = json.loads(manifest_path.read_text()).get("photos", {})
    names = local_names(urls, earlier)
    manifest = {"date": day.isoformat(), "page": pages[0], "photos": names}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    jobs = [(url, out / name) for name, url in names.items()]
    if limit is not None:
        missing = [job for job in jobs if not job[1].exists()]
        jobs = missing[:limit]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(_download, jobs))
    return {
        "pages": pages,
        "total": len(names),
        "new": results.count("downloaded"),
        "errors": [r for r in results if r.startswith("error")],
    }


def cached_photos(day: str) -> list[Path]:
    """Photos already on disk for a date, in name order."""
    folder = CACHE_DIR / day
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in PHOTO_EXTENSIONS)
