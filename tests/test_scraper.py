from datetime import date

import scraper


def test_slug_date_handles_the_sites_inconsistent_slugs():
    base = "https://campercameos.com/"
    cases = {
        "main-campground-and-ranger-station-february-3rd-2026/": date(2026, 2, 3),
        "main-campground-ranger-station-january-25-2026/": date(2026, 1, 25),
        "pet-camp-main-campground-ranger-station-monday-june-2nd-2025/": date(
            2025, 6, 2
        ),
        "main-and-ranger-station-august-11th-2025-2/": date(2025, 8, 11),
    }
    for slug, expected in cases.items():
        assert scraper.slug_date(base + slug) == expected


def test_slug_date_ignores_other_pages():
    base = "https://campercameos.com/"
    assert scraper.slug_date(base + "cat-safari-main-february-3rd-2026/") is None
    assert scraper.slug_date(base + "savannah-campground-february-3rd-2026/") is None
    assert scraper.slug_date(base + "main-campground-about-us/") is None
    assert scraper.slug_date(base + "main-campground-february-31st-2026/") is None


def test_photo_urls_strips_thumbnail_suffix_and_duplicates():
    prefix = scraper.PHOTO_PREFIX + "wp-content/uploads/2026/11/"
    html = (
        f'<img src="{prefix}a-300x400.jpg"><img class="x" src="{prefix}a.jpg">'
        f'<img src="{prefix}b.jpeg"><img src="https://example.com/uploads/c.jpg">'
    )
    assert scraper.photo_urls(html) == [prefix + "a.jpg", prefix + "b.jpeg"]


def test_local_names_are_safe_and_unique():
    names = scraper.local_names(
        ["https://x/1/photo one.jpg", "https://x/2/photo one.jpg", "https://x/3/b.jpg"]
    )
    assert list(names) == ["photo_one.jpg", "photo_one__2.jpg", "b.jpg"]


def test_guessed_pages_cover_plain_and_ordinal_days():
    pages = scraper.guessed_pages(date(2026, 11, 22))
    assert any(p.endswith("-november-22-2026/") for p in pages)
    assert any(p.endswith("-november-22nd-2026/") for p in pages)
    assert scraper.ordinal(11) == "11th" and scraper.ordinal(23) == "23rd"


def test_local_names_keep_earlier_assignments_when_page_order_changes():
    first = scraper.local_names(["https://x/10/IMG_1.jpg", "https://x/11/IMG_1.jpg"])
    again = scraper.local_names(
        ["https://x/11/IMG_1.jpg", "https://x/12/new.jpg"], first
    )
    assert again["IMG_1.jpg"] == "https://x/10/IMG_1.jpg"
    assert again["IMG_1__2.jpg"] == "https://x/11/IMG_1.jpg"
    assert again["new.jpg"] == "https://x/12/new.jpg"


def test_sitemap_failure_keeps_the_cached_copy(tmp_path, monkeypatch):
    import json
    import os

    import requests

    cache = tmp_path / "sitemap.json"
    cache.write_text(json.dumps({"2026-11-20": ["https://campercameos.com/a/"]}))
    os.utime(cache, (0, 0))  # stale
    monkeypatch.setattr(scraper, "SITEMAP_CACHE", cache)
    monkeypatch.setattr(scraper, "CACHE_DIR", tmp_path)

    def server_error(url, timeout=30):
        response = requests.Response()
        response.status_code = 503
        raise requests.HTTPError(response=response)

    monkeypatch.setattr(scraper, "fetch", server_error)
    assert scraper.pages_for(date(2026, 11, 21)) == []
    assert "2026-11-20" in json.loads(cache.read_text())


def test_slug_date_rejects_other_hosts():
    assert (
        scraper.slug_date("https://evil.example/main-campground-february-3rd-2026/")
        is None
    )
