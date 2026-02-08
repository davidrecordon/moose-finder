#!/usr/bin/env python3
"""
Moose Finder - Pet Camp Photo Scraper
Scrapes campercameos.com for daily dog photos and identifies likely photos of Moose
using MegaDescriptor embeddings for individual animal re-identification.

Usage:
    python moose_finder.py                    # Check today's photos
    python moose_finder.py --date 2026-02-01  # Check specific date
    python moose_finder.py --analyze-only      # Only process already-cached images
    python moose_finder.py --download-limit 10  # Limit new downloads
    python moose_finder.py --download-only    # Download images without analyzing
    python moose_finder.py --show-results      # Show local status (no downloads or analysis)
"""

import argparse
import hashlib
import json
import os
import re
import shutil
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from PIL import Image

Image.MAX_IMAGE_PIXELS = 50_000_000  # 50MP decompression bomb guard


# Configuration
PROJECT_DIR = Path(__file__).parent
REFERENCE_DIR = PROJECT_DIR / "reference_photos"
CACHE_DIR = PROJECT_DIR / "cache"
RESULTS_DIR = PROJECT_DIR / "results"
DEBUG_DIR = PROJECT_DIR / "debug"

MAX_DOWNLOAD_SIZE = 50 * 1024 * 1024  # 50MB

BASE_URL = "https://campercameos.com"


def ensure_dirs():
    """Create necessary directories if they don't exist."""
    REFERENCE_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)


def day_ordinal(day: int) -> str:
    """Return day with ordinal suffix (1st, 2nd, 3rd, 4th, etc.)."""
    if 11 <= day <= 13:
        return f"{day}th"
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return f"{day}{suffix}"


def date_to_slug(date: datetime) -> str:
    """Convert date to URL slug format (e.g., 'february-3rd-2026')."""
    month = date.strftime("%B").lower()
    day = day_ordinal(date.day)
    year = date.year
    return f"{month}-{day}-{year}"


def get_daily_urls(date: datetime) -> list[str]:
    """Get candidate URLs for a specific day's photos, in order of likelihood."""
    month = date.strftime("%B").lower()
    day_ord = day_ordinal(date.day)
    day_plain = str(date.day)
    year = date.year

    # URL pattern variations to try
    patterns = [
        f"/main-campground-and-ranger-station-{month}-{day_ord}-{year}/",  # Primary: and + ordinal
        f"/main-campground-and-ranger-station-{month}-{day_plain}-{year}/",  # and + plain day
        f"/main-campground-ranger-station-{month}-{day_ord}-{year}/",  # no and + ordinal
        f"/main-campground-ranger-station-{month}-{day_plain}-{year}/",  # no and + plain day
    ]
    return [urljoin(BASE_URL, p) for p in patterns]


def scrape_image_urls(page_urls: list[str]) -> list[str]:
    """Scrape image URLs, trying multiple URL patterns until one works."""
    for page_url in page_urls:
        print(f"Trying: {page_url}")

        try:
            response = requests.get(page_url, timeout=30)
            response.raise_for_status()
        except requests.RequestException:
            continue  # Try next URL

        soup = BeautifulSoup(response.text, "html.parser")

        image_urls = []
        seen = set()
        for img in soup.find_all("img"):
            src = img.get("src", "")
            # Only accept Pet Camp's S3 bucket to prevent SSRF
            if not src.startswith("https://petcamp.s3.us-west-1.amazonaws.com/"):
                continue
            if "uploads" in src:
                full_src = re.sub(r"-\d+x\d+(\.\w+)$", r"\1", src)
                if full_src not in seen:
                    seen.add(full_src)
                    image_urls.append(full_src)

        if image_urls:
            return image_urls

    return []


def get_cache_path(url: str) -> Path:
    """Get cache file path for a URL."""
    url_hash = hashlib.md5(url.encode()).hexdigest()
    return CACHE_DIR / f"{url_hash}.jpg"


def is_cached(url: str) -> bool:
    """Check if an image URL is already cached."""
    return get_cache_path(url).exists()


def get_results_path(date: datetime) -> Path:
    """Get results file path for a date."""
    return RESULTS_DIR / f"{date.strftime('%Y-%m-%d')}.json"


def load_results(date: datetime) -> dict:
    """Load cached results for a date. Returns dict mapping URL -> result."""
    results_path = get_results_path(date)
    if results_path.exists():
        try:
            with open(results_path) as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"Warning: corrupted results file {results_path}, ignoring: {e}")
    return {}


def save_result(date: datetime, url: str, model_name: str, result: dict):
    """Save a single model result to the results file.

    Args:
        date: Date for the results file
        url: Image URL
        model_name: Model identifier (e.g. "MegaDescriptor-L-384")
        result: Dict with "confidence" and "classification" keys
    """
    results_path = get_results_path(date)
    results = load_results(date)
    results.setdefault(url, {})[model_name] = result
    tmp_path = results_path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(results, f, indent=2)
    os.replace(tmp_path, results_path)


def save_results_batch(date: datetime, batch: dict):
    """Save multiple results at once, deep-merging per-URL model dicts.

    Args:
        date: Date for the results file
        batch: Dict of {url: {model_name: {confidence, classification}}}
    """
    results_path = get_results_path(date)
    existing = load_results(date)
    for url, models in batch.items():
        existing.setdefault(url, {}).update(models)
    tmp_path = results_path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(existing, f, indent=2)
    os.replace(tmp_path, results_path)


def download_image(
    url: str, cache: bool = True, use_cached_only: bool = False
) -> Optional[Image.Image]:
    """Download an image and return as PIL Image.

    Args:
        url: Image URL to download
        cache: Whether to cache downloaded images
        use_cached_only: If True, return None for uncached images instead of downloading

    Returns:
        PIL Image or None if not available/failed
    """
    cache_path = get_cache_path(url)

    if cache_path.exists():
        try:
            return Image.open(cache_path)
        except Exception:
            pass

    if use_cached_only:
        return None

    try:
        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()

        # Check Content-Length if available
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > MAX_DOWNLOAD_SIZE:
            print(
                f"Skipping {url}: Content-Length {int(content_length)} exceeds {MAX_DOWNLOAD_SIZE} byte limit"
            )
            return None

        # Stream download with size enforcement
        chunks = []
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            downloaded += len(chunk)
            if downloaded > MAX_DOWNLOAD_SIZE:
                print(
                    f"Skipping {url}: download exceeded {MAX_DOWNLOAD_SIZE} byte limit"
                )
                return None
            chunks.append(chunk)
        content = b"".join(chunks)

        if cache:
            with open(cache_path, "wb") as f:
                f.write(content)
            return Image.open(cache_path)
        else:
            return Image.open(BytesIO(content))

    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return None


def log(msg: str, end: str = "\n", flush: bool = True):
    """Print with immediate flush for better progress visibility."""
    if end == "\r":
        msg += "\033[K"
    print(msg, end=end, flush=flush)


def find_moose_matches(
    image_urls: list[str],
    image_matcher,
    check_date: datetime,
    use_cached_only: bool = False,
    download_limit: Optional[int] = None,
) -> tuple[list[dict], list[dict]]:
    """Find Moose using MegaDescriptor embeddings.

    Args:
        image_urls: List of image URLs to check
        image_matcher: Initialized ImageMatcher instance
        check_date: Date being checked (for result caching)
        use_cached_only: Only process already-cached images
        download_limit: Max number of new images to download (None = unlimited)

    Returns:
        Tuple of (matches, all_results) where all_results contains every processed
        image sorted by confidence (highest first).
    """
    from image_matcher import classify_match

    matches = []
    all_results = []
    downloads = 0
    skipped = 0
    cached_hits = 0
    result_hits = 0

    total = len(image_urls)
    model_id = image_matcher.model_id

    existing_results = load_results(check_date)
    if existing_results:
        result_count = sum(
            1
            for url in image_urls
            if url in existing_results and model_id in existing_results[url]
        )
        if result_count:
            log(f"{result_count} already analyzed from previous run")

    for i, url in enumerate(image_urls):
        # Check if we already have a result for this URL and model
        if url in existing_results and model_id in existing_results[url]:
            result_hits += 1
            cached = existing_results[url][model_id]
            all_results.append(
                {
                    "url": url,
                    "confidence": cached["confidence"],
                    "source": model_id,
                    "classification": cached["classification"],
                }
            )
            if cached["classification"] == "match":
                matches.append(
                    {
                        "url": url,
                        "confidence": cached["confidence"],
                        "source": model_id,
                        "reasoning": f"High MegaDescriptor similarity: {cached['confidence']:.2f}",
                    }
                )
            log(
                f"[{i+1}/{total}] Already analyzed ({cached['confidence']:.0%})",
                end="\r",
            )
            continue

        # Check if image is already cached
        cached = is_cached(url)

        if not cached and use_cached_only:
            skipped += 1
            log(f"[{i+1}/{total}] Skipped (not cached)", end="\r")
            continue

        if not cached and download_limit is not None and downloads >= download_limit:
            skipped += 1
            log(f"[{i+1}/{total}] Skipped (download limit reached)", end="\r")
            continue

        if cached:
            cached_hits += 1
            log(f"[{i+1}/{total}] Analyzing...", end="\r")
        else:
            downloads += 1
            log(f"[{i+1}/{total}] Downloading...", end="\r")

        img = download_image(url, use_cached_only=use_cached_only)
        if img is None:
            continue

        try:
            similarity = image_matcher.get_similarity(img)
        except Exception as e:
            log(f"[{i+1}/{total}] Error: {e}")
            continue
        finally:
            img.close()

        classification = classify_match(similarity)

        result = {
            "confidence": similarity,
            "classification": classification,
        }
        all_results.append(
            {
                "url": url,
                "confidence": similarity,
                "source": model_id,
                "classification": classification,
            }
        )

        # Save result immediately for resumability
        save_result(check_date, url, model_id, result)

        if classification == "match":
            log(f"[{i+1}/{total}] FOUND MOOSE! ({similarity:.0%}) {url}")
            matches.append(
                {
                    "url": url,
                    "confidence": similarity,
                    "source": model_id,
                    "reasoning": f"High MegaDescriptor similarity: {similarity:.2f}",
                }
            )

    log("")
    log(
        f"Processed {total - skipped}/{total} images ({downloads} downloaded, {skipped} skipped)"
    )

    all_results = sorted(all_results, key=lambda x: x["confidence"], reverse=True)
    matches = sorted(matches, key=lambda x: x["confidence"], reverse=True)

    return matches, all_results


def has_reference_photos() -> bool:
    """Check for reference photos and print a message if none are found."""
    ref_files = [
        p for ext in ("*.jpg", "*.jpeg", "*.png") for p in REFERENCE_DIR.glob(ext)
    ]
    if not ref_files:
        print("No reference photos found!")
        print(f"Please add photos of Moose to: {REFERENCE_DIR}")
        return False
    return True


def init_matcher():
    """Initialize the MegaDescriptor matcher and load reference photos."""
    from image_matcher import ImageMatcher, MODEL_ID, is_model_cached

    cached = is_model_cached()
    if cached:
        print(f"Loading {MODEL_ID} model...")
    else:
        print(f"Downloading {MODEL_ID} model...")
    matcher = ImageMatcher(local_only=cached)
    if matcher.detector is not None:
        print("Dog detection enabled (YOLOv8n)")
    else:
        print("Dog detection unavailable (install ultralytics to enable)")
    num_refs = matcher.load_references(REFERENCE_DIR)
    print(f"Loaded {num_refs} reference photos")
    return matcher


def debug_single_url(url: str):
    """Analyze a single photo URL with full diagnostic output."""
    if not has_reference_photos():
        return

    matcher = init_matcher()
    print()

    # Download the image
    print(f"Processing: {url}")
    img = download_image(url)
    if img is None:
        print("Failed to download image.")
        return

    try:
        print(f"Image size: {img.size[0]}x{img.size[1]}\n")

        # Get debug info
        info = matcher.get_debug_info(img)

        # Print full image analysis
        full = info["full_image"]
        print(
            f"Full image similarity: {full['similarity']:.0%} (best ref: {full['best_ref']})"
        )
        for ref_name, sim in full["per_reference"]:
            print(f"  {ref_name}: {sim:6.0%}")

        # Prepare debug directory
        if DEBUG_DIR.exists():
            if DEBUG_DIR.is_symlink():
                DEBUG_DIR.unlink()
            else:
                shutil.rmtree(DEBUG_DIR)
        DEBUG_DIR.mkdir()

        # Print YOLO detections
        detections = info["detections"]
        if detections:
            print(f"\nYOLOv8n detected {len(detections)} dog(s):\n")
            for det in detections:
                idx = det["index"]
                bbox = det["bbox"]
                conf = det["yolo_confidence"]
                crop_w, crop_h = det["crop_size"]
                best_marker = (
                    "  \u2190 best" if info["best_source"] == f"detection_{idx}" else ""
                )

                print(
                    f"  Dog {idx}: bbox {bbox} conf={conf:.2f}, crop {crop_w}x{crop_h}"
                )
                print(
                    f"    Similarity: {det['similarity']:.0%} (best ref: {det['best_ref']}){best_marker}"
                )

                # Save crop
                crop_path = DEBUG_DIR / f"crop_{idx}.jpg"
                det["crop"].save(crop_path)
                det["crop"].close()
                print(f"    Saved: {crop_path}")
                print()
        elif matcher.detector is not None:
            print("\nYOLOv8n detected 0 dogs.\n")

        # Print best overall
        classification = info["classification"]
        print(
            f"Best overall: {info['best_similarity']:.0%} (from {info['best_source']}) \u2192 {classification}"
        )
    finally:
        img.close()


def main():
    parser = argparse.ArgumentParser(description="Find Moose in Pet Camp photos")
    parser.add_argument("--date", help="Date to check (YYYY-MM-DD), defaults to today")
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze already-cached images (no downloads)",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download images to cache without analyzing",
    )
    parser.add_argument(
        "--download-limit",
        type=int,
        default=None,
        help="Limit number of new image downloads",
    )
    parser.add_argument(
        "--show-results",
        action="store_true",
        help="Show local status for a date (no downloads or analysis)",
    )
    parser.add_argument(
        "--url", help="Debug mode: analyze a single photo URL with full diagnostics"
    )
    args = parser.parse_args()

    ensure_dirs()

    if args.url:
        debug_single_url(args.url)
        return

    if args.date:
        check_date = datetime.strptime(args.date, "%Y-%m-%d")
    else:
        check_date = datetime.now()

    print(f"Checking Pet Camp photos for {check_date.strftime('%B %d, %Y')}...")

    page_urls = get_daily_urls(check_date)
    image_urls = scrape_image_urls(page_urls)

    if not image_urls:
        if args.show_results:
            print("No photos posted yet for this date.")
        else:
            print(
                "No photos found. They may not be posted yet, or the URL format changed."
            )
        return

    cached_count = sum(1 for url in image_urls if is_cached(url))
    if cached_count:
        print(f"Found {len(image_urls)} photos ({cached_count} already downloaded)")
    else:
        print(f"Found {len(image_urls)} photos")

    # Show-results mode: display local status snapshot without downloads or analysis
    if args.show_results:
        total = len(image_urls)
        results = load_results(check_date)

        date_label = check_date.strftime("%B %d, %Y")
        print(f"\nStatus for {date_label} ({total} photos)\n")
        print(f"  Downloaded: {cached_count}/{total}")

        # Collect all model names found across all URLs
        model_names = set()
        for url in image_urls:
            if url in results:
                model_names.update(results[url].keys())

        if not model_names:
            print("\n  No analysis results yet.\n")
            return

        for model_name in sorted(model_names):
            analyzed = sum(
                1 for url in image_urls if url in results and model_name in results[url]
            )

            match_urls = []
            uncertain_count = 0
            no_match_count = 0
            for url in image_urls:
                if url not in results or model_name not in results[url]:
                    continue
                entry = results[url][model_name]
                cls = entry.get("classification", "")
                if cls == "match":
                    match_urls.append((entry["confidence"], url))
                elif cls == "uncertain":
                    uncertain_count += 1
                else:
                    no_match_count += 1

            pending = total - analyzed
            match_urls.sort(reverse=True)

            print(f"\n  {model_name} ({analyzed}/{total} analyzed):")
            print(f"    Matches:    {len(match_urls)}")
            print(f"    Uncertain:  {uncertain_count}")
            print(f"    No match:   {no_match_count}")
            print(f"    Pending:    {pending}")

            if match_urls:
                print("\n    Matches:")
                for i, (conf, url) in enumerate(match_urls, 1):
                    print(f"      {i}. [{conf:.0%}] {url}")

        print()
        return

    # Download-only mode: just cache images, no analysis
    if args.download_only:
        downloaded = 0
        already_cached = 0
        for i, url in enumerate(image_urls):
            if is_cached(url):
                already_cached += 1
                log(f"[{i+1}/{len(image_urls)}] Already cached", end="\r")
            else:
                if (
                    args.download_limit is not None
                    and downloaded >= args.download_limit
                ):
                    log(
                        f"[{i+1}/{len(image_urls)}] Skipped (download limit reached)",
                        end="\r",
                    )
                    continue
                log(f"[{i+1}/{len(image_urls)}] Downloading...", end="\r")
                img = download_image(url)
                if img:
                    img.close()
                    downloaded += 1
        log("")
        log(f"Done: {downloaded} downloaded, {already_cached} already cached")
        return

    if not has_reference_photos():
        return

    image_matcher = init_matcher()

    log(f"\nSearching for Moose among {len(image_urls)} photos...")
    if args.analyze_only:
        log("(analyze-only mode: only processing cached images)")
    if args.download_limit:
        log(f"(download limit: {args.download_limit} new images)")

    matches, all_results = find_moose_matches(
        image_urls,
        image_matcher,
        check_date,
        use_cached_only=args.analyze_only,
        download_limit=args.download_limit,
    )

    if all_results:
        log("\n=== All Results (sorted by confidence) ===\n")
        for i, result in enumerate(all_results, 1):
            conf = result["confidence"]
            classification = result["classification"]
            marker = ">>>" if classification == "match" else "   "
            log(f"{marker} {i:3d}. [{conf:.0%}] {result['url']}")
        log("")

    if matches:
        log(f"Found {len(matches)} likely photo(s) of Moose!\n")
        for i, match in enumerate(matches, 1):
            log(f"{i}. {match['url']}")
            log(f"   Confidence: {match['confidence']:.0%} ({match['source']})")
            if match.get("reasoning"):
                log(f"   {match['reasoning']}")
            log("")
    else:
        log("\nNo photos of Moose found today.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
