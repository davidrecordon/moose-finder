# Moose Finder

Automatically finds photos of Moose (a Bernese Mountain Dog) from Pet Camp's daily camper cameos.

## How It Works

1. Scrapes the daily "Main Campground & Ranger Station" photos from [campercameos.com](https://campercameos.com)
2. Detects individual dogs in each photo using YOLOv8n
3. Compares each dog crop (plus the full image) against reference photos of Moose using MegaDescriptor embeddings
4. Classifies results as match (>90%), uncertain (50-90%), or no match (<50%)

MegaDescriptor is a foundation model specifically designed for individual animal re-identification, trained on dog-specific datasets. Combined with YOLOv8n crop detection, it reliably identifies Moose even in multi-dog photos.

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Add Reference Photos

Place clear photos of Moose in `reference_photos/`. More photos covering different poses and angles improve accuracy. The current set includes 16 reference photos covering front-face, full body, side view, and outdoor poses.

### 3. Run

```bash
python moose_finder.py
```

## Usage

```bash
# Check today's photos
python moose_finder.py

# Check a specific date
python moose_finder.py --date 2026-02-01

# Download images without analyzing
python moose_finder.py --download-only

# Only analyze already-cached images (no downloads)
python moose_finder.py --analyze-only

# Limit new downloads
python moose_finder.py --download-limit 10

# Show local status for a date (no downloads or analysis)
python moose_finder.py --show-results
python moose_finder.py --show-results --date 2026-02-03

# Debug a single photo with full diagnostic breakdown
python moose_finder.py --url "https://petcamp.s3.us-west-1.amazonaws.com/..."
```

### Debug Mode

The `--url` flag analyzes a single photo and shows the full diagnostic breakdown: per-reference similarity scores, YOLO detections with bounding boxes, crop dimensions, and saves all dog crops to `debug/` for visual inspection.

```
Full image similarity: 21% (best ref: moose_ref_4)
  moose_ref_1:     6%
  moose_ref_2:    21%
  ...

YOLOv8n detected 4 dog(s):

  Dog 1: bbox [703, 593, 1177, 1105] conf=0.94, crop 559x614
    Similarity: 26% (best ref: moose_ref_2)

  Dog 3: bbox [118, 552, 608, 988] conf=0.83, crop 588x522
    Similarity: 94% (best ref: moose_ref_15)  <- best

Best overall: 94% (from detection_3) -> match
```

## Tips

- Pet Camp usually posts photos in the afternoon/evening (SF time)
- Add more reference photos for better accuracy, especially for poses where Moose is missed
- Results are saved per-date, so re-running a date skips already-analyzed photos
- Downloaded images are cached in `cache/` to avoid re-downloading

## Troubleshooting

**"No photos found for this date"**
- Photos might not be posted yet (usually by evening)
- The URL format may have changed — check campercameos.com manually

**"No reference photos found"**
- Add photos of Moose to `reference_photos/`

**Moose not detected in a photo**
- Use `--url` to debug the specific photo
- If Moose is small in the background, YOLO may not detect it (known limitation)
- If the crop is good but similarity is low, add the photo as a new reference to cover that pose
