# Moose Finder

Finds photos of Moose (a Bernese Mountain Dog) in Pet Camp's daily camper cameos.

## How it works

1. **Download.** Looks up the day's "Main Campground & Ranger Station" page in the [campercameos.com](https://campercameos.com) sitemap and downloads its photos to `cache/<date>/`.
2. **Find dogs.** YOLO26m boxes every dog in each photo.
3. **Body match.** Each dog crop is embedded with a SigLIP2 model fine-tuned for animal identification and compared with known crops of Moose. This finds "a Bernese that could be Moose", but can't tell him from the other Bernese at camp.
4. **Face check.** A small keypoint model finds the ears, nose and chin. When the dog is facing the camera, its face is cut out, embedded with DINOv3, and a classifier trained on Moose's face against the lookalikes' faces confirms or rejects. The white blaze between his eyes is narrower than theirs, and this is where that shows.
5. **Verdict.** Each photo is one of:
   - **Moose**: body matches and the face confirms it.
   - **Possible**: body matches, no clear view of the face.
   - **Lookalike**: body matches, but the face says it's another dog (counted, not listed).

Steps 2 to 4 are the slow part and are saved in `cache/<date>/features.npz`, so re-running a day, changing thresholds or retraining takes seconds.

## Setup

Needs [uv](https://docs.astral.sh/uv/). Everything else installs into a local `.venv`:

```bash
git clone https://github.com/davidrecordon/moose-finder && cd moose-finder
uv sync
```

It uses an NVIDIA GPU if there is one, then Apple Silicon's GPU, then the CPU. The first run downloads about 1 GB of model weights.

## Usage

```bash
uv run python moose_finder.py                      # today
uv run python moose_finder.py --date 2026-11-24    # one day
uv run python moose_finder.py --since 2026-11-24   # every day since, through today
uv run python moose_finder.py --since 2026-11-14 --until 2026-11-20
uv run python moose_finder.py --no-download        # only photos already in cache/
uv run python moose_finder.py --download-only      # fetch photos, don't analyse
uv run python moose_finder.py --photo some.jpg     # explain every dog in one photo
```

Each day prints its matches with links, and writes `results/<date>.json` and a `results/<date>.html` gallery with the chosen dog boxed.

Pet Camp usually posts in the afternoon or evening, San Francisco time. Running again later picks up only the new photos.

## Teaching it

The model learns from photo-level labels in `labels/labels.json`.

```bash
uv run python labeler.py          # label cached photos at http://127.0.0.1:8765
uv run python train.py            # evaluate, then retrain and save models/moose_model.npz
uv run python train.py --dry-run  # evaluate only
```

In the labeller, click a photo if Moose is in it, use **Lookalike** for another dog you could mistake for him, and mark the rest of the page as not Moose. Lookalike labels matter as much as Moose labels: they are what the face classifier learns to reject.

`train.py` holds out one stay at a time, so every reported result comes from a model that never saw photos from that stay.

## Files

| File | Purpose |
|---|---|
| `moose_finder.py` | Command line: download, analyse, report |
| `scraper.py` | Sitemap lookup and photo download |
| `pipeline.py` | Dog detection, body and face embeddings, cached per date |
| `matcher.py` | Body match plus face check; reads `models/moose_model.npz` |
| `train.py` | Builds `models/moose_model.npz` from labels |
| `labeler.py` | Local labelling page |
| `models/dogpose-yolo26n.pt` | Dog keypoint model, trained on the Ultralytics dog-pose dataset |
| `models/moose_model.npz` | Everything learned about Moose |
| `reference_photos/` | Hand-picked starting photos of Moose |

## Development

```bash
uv run pytest
uv run black *.py tests
uv run flake8 --select=F *.py tests
```

## Limits

- Moose side-on or far in the background has no usable face, so those photos rely on the body match alone and land in "Possible" at best.
- The face classifier only knows the dogs it has seen labelled. A new Bernese at camp may need a few Lookalike labels before it is rejected reliably.
