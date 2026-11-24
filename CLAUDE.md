# CLAUDE.md

## Project Overview

Moose Finder downloads Pet Camp's daily photos from campercameos.com and finds the ones containing Moose, a Bernese Mountain Dog. Several other Bernese attend the same camp, so it works in two stages: a body match finds Bernese that could be Moose, and a face check tells him from the lookalikes.

Runs natively through a `uv` virtualenv (no Docker: Docker on macOS cannot use the GPU). Device is picked automatically: CUDA, then MPS, then CPU.

## Architecture

```
moose_finder.py   CLI: download, analyse, score, print, write results/<date>.json and .html
scraper.py        Sitemap-based page lookup, photo download into cache/<date>/
pipeline.py       Analyzer (4 models) -> Features, saved as cache/<date>/features.npz
matcher.py        MooseModel: body similarity + face/body classifiers -> PhotoResult
train.py          labels/labels.json + cached features -> models/moose_model.npz
labeler.py        Local web page for labelling photos (stdlib only)
tests/            pytest; no models or network needed
```

The split that matters: `pipeline.py` is slow and knows nothing about Moose; `matcher.py` and `train.py` are fast and only read saved features. Changing thresholds or retraining never re-runs the models.

### Pipeline (pipeline.py)

1. **Detect**: YOLO26m, COCO class 16, imgsz 1280, conf 0.10, min box 0.15% of image area and 24 px a side. Item 0 of each photo is the full frame; crops get 10% padding.
2. **Body embedding**: `AvitoTech/SigLIP2-Base-for-animal-identification` (pinned revision). Crops are squashed to 512x512 before the processor so nothing is centre-cropped away.
3. **Face crop**: `models/dogpose-yolo26n.pt` predicts 24 keypoints per dog. The face box is the extent of ear bases, nose and chin scaled by 1.35, and requires nose and both ear bases at confidence 0.5 or more (the dataset has no eye labels). Faces under 24 px are not stored.
4. **Face embedding**: timm `vit_base_patch16_dinov3.lvd1689m` (pinned revision) at 384 px.

`FEATURES_VERSION` must be bumped when any of this changes; older feature files are then rebuilt.

### Matching (matcher.py)

Per item: `similarity` = max cosine to the body references. Items at or above `candidate_similarity` are candidates. A candidate with a usable face (at least `min_face_side` px) is `match` or `lookalike` by the face classifier; one without is `possible` if the body classifier agrees. A photo takes its strongest item. Settings live in `DEFAULT_SETTINGS` and are saved inside the model file.

### Training (train.py)

Labels are per photo. Training picks the crop the label refers to (best body match, two passes), then builds body references from Moose crops and fits two logistic regressions (face C=100, body C=10) against hard negatives: any crop with body similarity 0.75 or more in a photo labelled lookalike or not_moose. Evaluation is leave-one-stay-out (dates within 3 days of each other form a stay).

Current settings (`matcher.DEFAULT_SETTINGS`, chosen from held-out sweeps): `candidate_similarity` 0.85, `face_threshold` 0.6, `body_threshold` 0.6, `min_face_side` 40. The classifiers are not calibrated to 0.5: with weaker regularisation (C=1) face probabilities bunch around 0.5 to 0.7 and lookalikes pass, so re-sweep thresholds if C changes.

### Data and results (as of 2026-11-23)

- `cache/` holds 16,530 photos from 113 days across every stay since Feb 2025 (not in git). `labels/labels.json` has 1,246 labelled photos: 105 moose, 200 lookalike, 941 not_moose.
- Nov 14 to 20, 2026 is the only period where every photo is labelled, so it is the honest test set. Earlier stays have labels only for photos some version of the tool nominated.
- Held-out on the test week: of 8 Moose photos, 4 match, 2 possible, 2 missed; 0 of 784 other photos called a match.
- Checked by hand on unlabelled history: "match" was right 11 of 12 times, face rejections were right 147 of 150, but "possible" was Moose only 4 of 46 times. Without a visible face the body embedding cannot separate the Bernese; more labels did not help. A second marking-based check (side of head, chest) is the untried idea.
- The hand-picked `reference_photos/` come from earlier stays, so those stays are not clean test data.

## Commands

```bash
uv sync
uv run python moose_finder.py [--date D | --since D [--until D]] [--no-download] [--download-only]
uv run python moose_finder.py --photo path.jpg
uv run python labeler.py [--order list.json [--only]]
uv run python train.py [--dry-run]
uv run pytest
uv run black *.py tests
uv run flake8 --select=F *.py tests
```

## Site facts

- Daily pages are WordPress posts with inconsistent slugs (weekday names, `pet-camp-` prefix, `main-and-ranger-station`, `-2` duplicates). `scraper.slug_date` accepts any slug containing "main", not "cat", ending in `-<month>-<day>[st|nd|rd|th]-<year>[-N]`.
- `https://campercameos.com/wp-sitemap-posts-post-{1..N}.xml` lists every post back to 2018. Guessed URLs are a fallback for pages too new for the sitemap.
- Photos are on `https://petcamp.s3.us-west-1.amazonaws.com/`; thumbnails carry a `-WxH` suffix that is stripped to get the original.

## Notes

- Models tried and rejected for the body stage: MegaDescriptor-L-384 (the original; scores collapsed on new photos), MiewID, DINOv3, open-noodle pet-recognition, SigLIP2-giant (no better, far slower). For faces, DINOv3 beat SigLIP2 and MegaDescriptor.
- The keypoint model was trained with `yolo pose train data=dog-pose.yaml model=yolo26n-pose.pt epochs=30 imgsz=512 batch=16` (about 1.5 hours on an M5 Pro; larger batches exhaust MPS memory).
- Licences: Ultralytics (and the keypoint weights trained with it) is AGPL-3.0; the SigLIP2 model is Apache-2.0. The repo itself has no licence file yet.
- `scraper.py`'s download path was rewritten from the scratch scripts used to fill `cache/`; check it against the live site when changing it.
- `models/yolo26m.pt` is gitignored and re-downloaded by Ultralytics; `models/dogpose-yolo26n.pt` and `models/moose_model.npz` are meant to be committed so another machine can run without retraining.
- Commit only when asked.
