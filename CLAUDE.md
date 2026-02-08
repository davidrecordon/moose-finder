# CLAUDE.md

## Project Overview

Moose Finder is a Python script that scrapes daily pet daycare photos from campercameos.com and identifies photos containing a specific Bernese Mountain Dog named Moose using MegaDescriptor (purpose-built for individual animal re-identification) and YOLOv8n (dog detection for crop-then-re-ID).

## Architecture

```
moose_finder.py     # Main script: scraping, orchestration, CLI, output
image_matcher.py    # MegaDescriptor embeddings for animal re-ID
dog_detector.py     # YOLOv8n dog detection for crop-then-re-ID pipeline
```

### Detection Pipeline

1. **YOLOv8n Detection**: Detects individual dogs in each photo, producing bounding box crops
   - COCO class 16 (dog), confidence threshold 0.25, minimum area 1% of image
   - 10% padding around each crop
2. **MegaDescriptor Matching**: Full image + each crop compared against reference embeddings
   - MegaDescriptor is a foundation model specifically designed for individual animal re-ID
   - Trained on DogFaceNet and MPDD datasets, outperforms CLIP/DINOv2 on animal re-ID
   - Best score across full image + all crops is used
   - >= 0.90 similarity: match
   - 0.50-0.90 similarity: uncertain
   - < 0.50 similarity: rejected

## Key Files

- `moose_finder.py` - Entry point, CLI args, web scraping, result display, `--url` debug mode
- `image_matcher.py` - `ImageMatcher` class using MegaDescriptor-L-384, `get_debug_info()` for diagnostics
- `dog_detector.py` - `DogDetector` class wrapping YOLOv8n, `DogDetection` dataclass with crop method
- `reference_photos/` - Reference images of Moose (currently 16 photos covering various poses)
- `cache/` - Downloaded image cache (MD5-hashed filenames)
- `results/` - Per-date JSON files with analysis results (for resumability)
- `debug/` - YOLO crop images from `--url` debug mode

## Dependencies

- `timm` - PyTorch Image Models, loads MegaDescriptor from HuggingFace
- `torch`, `torchvision` - PyTorch framework
- `ultralytics` - YOLOv8n dog detection
- `requests`, `beautifulsoup4` - Web scraping
- `Pillow` - Image processing

## Common Commands

```bash
# Run detection for today
python moose_finder.py

# Run for specific date
python moose_finder.py --date 2026-02-03

# Only process cached images (no downloads)
python moose_finder.py --analyze-only

# Download images without analyzing
python moose_finder.py --download-only

# Limit new downloads
python moose_finder.py --download-limit 10

# Show local status for a date (no downloads or analysis)
python moose_finder.py --show-results
python moose_finder.py --show-results --date 2026-02-03

# Debug a single photo with full diagnostic breakdown
python moose_finder.py --url "https://petcamp.s3.us-west-1.amazonaws.com/..."

# Format code
black moose_finder.py image_matcher.py dog_detector.py

# Lint (F-codes are errors; E501 line length is handled by black)
flake8 --select=F moose_finder.py image_matcher.py dog_detector.py

# Syntax check
python -m py_compile moose_finder.py && python -m py_compile image_matcher.py && python -m py_compile dog_detector.py
```

## Thresholds

Defined in `image_matcher.py`:
- Match threshold: 0.90 (high confidence)
- Uncertain threshold: 0.50 (low confidence boundary)

Defined in `dog_detector.py`:
- YOLO confidence: 0.25
- Minimum crop area: 1% of image

## URL Patterns

Pet Camp posts photos at URLs like:
- `/main-campground-and-ranger-station-february-3rd-2026/`
- `/main-campground-ranger-station-january-25-2026/`
- Variations with/without "and", ordinal/plain day numbers

## Notes

- Photos are typically posted in the afternoon/evening SF time
- Images are cached by URL hash to avoid re-downloading
- Results are saved per-date as JSON for resumability — re-running skips already-analyzed photos
- The MegaDescriptor model downloads on first run (~400MB for L-384)
- YOLOv8n weights (yolov8n.pt) download on first run (~6.5MB)
- Known limitation: Moose in the background (<10% of frame) typically won't match due to small crop resolution
- Another Bernese Mountain Dog at the same daycare can score up to ~58% — the 0.90 threshold prevents false matches

## Future Features

These features were previously implemented and removed to keep the codebase focused. Notes for re-adding:

### Email Notifications
- Needs `smtplib` and `email.mime.multipart`/`email.mime.text`/`email.mime.image` imports
- Add `config.json` with SMTP credentials (`email_from`, `email_to`, `smtp_server`, `smtp_port`, `smtp_username`, `smtp_password`)
- Add `--send-email` and `--setup-email` CLI args
- Implement `load_config()`, `save_config()`, `send_email()`, `setup_email()` functions
- Add `config.json` to `.gitignore`

### Claude Vision Fallback
- Needs `anthropic` package (`pip install anthropic`)
- Create `claude_vision.py` with `verify_with_claude()` function that sends reference + candidate images to Claude
- Route uncertain matches (0.50-0.90 similarity) to Claude for verification
- Add `--no-claude` flag to disable API fallback
- Requires `ANTHROPIC_API_KEY` environment variable
- Claude confirmation threshold was 0.70
