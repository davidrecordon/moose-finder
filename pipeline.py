"""Turn photos into features: dog crops, body embeddings, face crops, face embeddings.

This is the slow, model-heavy half of Moose Finder. It knows nothing about
Moose: it only describes every dog in every photo. Results are saved per date
in cache/<date>/features.npz, so deciding which dog is Moose (matcher.py) and
retraining (train.py) never need to re-run the models.

Per photo:
  1. YOLO26m finds dogs (full frame is always kept as an item too).
  2. SigLIP2 (fine-tuned for animal identification) embeds each item: "body".
  3. A dog keypoint model finds ear bases, nose and chin in each dog crop; if
     the dog is facing the camera, the face is cut out.
  4. DINOv3 embeds each face crop: "face".
"""

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402
import torchvision.transforms as T  # noqa: E402

from scraper import CACHE_DIR, cached_photos  # noqa: E402

Image.MAX_IMAGE_PIXELS = 50_000_000  # decompression bomb guard

PROJECT_DIR = Path(__file__).parent
MODELS_DIR = PROJECT_DIR / "models"

DETECTOR_WEIGHTS = MODELS_DIR / "yolo26m.pt"  # downloaded by Ultralytics if missing
KEYPOINT_WEIGHTS = MODELS_DIR / "dogpose-yolo26n.pt"  # trained here, see README
BODY_MODEL = "AvitoTech/SigLIP2-Base-for-animal-identification"
BODY_REVISION = "8164be99dd6c1d00c9649181d0b82cc3fe3d97ea"
FACE_MODEL = "hf-hub:timm/vit_base_patch16_dinov3.lvd1689m@c6a5fb7d12bbd3cf3b0079253141c3332aaed7da"
# Bump when anything above (or the constants below) changes, so old features are rebuilt.
FEATURES_VERSION = "v2-yolo26m-siglip2base-dogpose26n-dinov3b"

DOG_CLASS = 16  # COCO
DETECT_SIZE = 1280
DETECT_CONFIDENCE = 0.10
MIN_BOX_AREA = 0.0015  # fraction of the photo
MIN_BOX_SIDE = 24  # pixels
CROP_PADDING = 0.10
BODY_INPUT = 512
FACE_INPUT = 384
EAR_L, EAR_R, NOSE, CHIN = 14, 15, 16, 17  # indices in the 24-keypoint dog-pose layout
KEYPOINT_CONFIDENCE = 0.5
FACE_MARGIN = 1.35  # face box = keypoint extent scaled by this
MIN_FACE_SIDE = 24  # pixels; smaller faces are not stored at all
EMBED_DIM = 768


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pad_box(box, width: int, height: int, padding: float = CROP_PADDING) -> list[int]:
    """Grow a box by a fraction of its size on every side, clipped to the image."""
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * padding, (y2 - y1) * padding
    return [
        int(max(0, x1 - px)),
        int(max(0, y1 - py)),
        int(min(width, x2 + px)),
        int(min(height, y2 + py)),
    ]


def face_box_from_keypoints(keypoints: np.ndarray) -> list[float] | None:
    """Square face box from one dog's keypoints (rows of x, y, confidence).

    Needs the nose and both ear bases, which in practice means the dog is
    facing the camera enough for its face markings to show.
    """
    if min(keypoints[i, 2] for i in (EAR_L, EAR_R, NOSE)) < KEYPOINT_CONFIDENCE:
        return None
    points = np.array(
        [
            keypoints[i, :2]
            for i in (EAR_L, EAR_R, NOSE, CHIN)
            if keypoints[i, 2] >= KEYPOINT_CONFIDENCE
        ]
    )
    (x1, y1), (x2, y2) = points.min(0), points.max(0)
    side = max(x2 - x1, y2 - y1) * FACE_MARGIN
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return [cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2]


class Analyzer:
    """Loads the four models once and describes photos with them."""

    def __init__(self, device: str | None = None):
        from transformers import AutoImageProcessor, AutoModel
        from ultralytics import YOLO
        import timm

        self.device = device or pick_device()
        self.detector = YOLO(str(DETECTOR_WEIGHTS))
        self.keypoints = YOLO(str(KEYPOINT_WEIGHTS))
        self.body_processor = AutoImageProcessor.from_pretrained(
            BODY_MODEL, revision=BODY_REVISION
        )
        self.body_model = (
            AutoModel.from_pretrained(BODY_MODEL, revision=BODY_REVISION)
            .eval()
            .to(self.device)
        )
        self.face_model = (
            timm.create_model(FACE_MODEL, pretrained=True, num_classes=0)
            .eval()
            .to(self.device)
        )
        self.face_transform = T.Compose(
            [
                T.Resize((FACE_INPUT, FACE_INPUT)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    @torch.no_grad()
    def embed_bodies(self, images: list[Image.Image]) -> np.ndarray:
        out = []
        for i in range(0, len(images), 16):
            # Squash to a square first so the processor never crops the dog.
            batch = [
                im.resize((BODY_INPUT, BODY_INPUT), Image.BICUBIC)
                for im in images[i : i + 16]
            ]
            inputs = self.body_processor(images=batch, return_tensors="pt").to(
                self.device
            )
            features = self.body_model.get_image_features(**inputs)
            if not torch.is_tensor(features):
                features = features.pooler_output
            out.append(torch.nn.functional.normalize(features.float(), dim=-1).cpu())
        return torch.cat(out).numpy()

    @torch.no_grad()
    def embed_faces(self, images: list[Image.Image]) -> np.ndarray:
        if not images:
            return np.zeros((0, EMBED_DIM), np.float32)
        out = []
        for i in range(0, len(images), 16):
            batch = torch.stack([self.face_transform(im) for im in images[i : i + 16]])
            features = self.face_model(batch.to(self.device))
            out.append(torch.nn.functional.normalize(features.float(), dim=-1).cpu())
        return torch.cat(out).numpy()

    def detect_dogs(self, image: Image.Image) -> list[list[int]]:
        result = self.detector(
            image,
            verbose=False,
            conf=DETECT_CONFIDENCE,
            imgsz=DETECT_SIZE,
            classes=[DOG_CLASS],
            device=self.device,
        )[0]
        boxes = []
        for box in result.boxes.xyxy.tolist():
            w, h = box[2] - box[0], box[3] - box[1]
            big_enough = w * h >= MIN_BOX_AREA * image.width * image.height
            if big_enough and min(w, h) >= MIN_BOX_SIDE:
                boxes.append([round(v) for v in box])
        return boxes

    def find_faces(self, crops: list[Image.Image]) -> list[list[float] | None]:
        """Face box (in crop pixels) for the main dog in each crop, or None."""
        if not crops:
            return []
        results = self.keypoints(
            crops, verbose=False, imgsz=512, conf=0.25, device=self.device
        )
        faces = []
        for result in results:
            if len(result.boxes) == 0:
                faces.append(None)
                continue
            xyxy = result.boxes.xyxy
            areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
            main_dog = int(areas.argmax())
            faces.append(
                face_box_from_keypoints(result.keypoints.data[main_dog].cpu().numpy())
            )
        return faces

    def analyze(self, image: Image.Image) -> dict:
        """Describe one photo. Item 0 is the full frame; the rest are dog crops."""
        image = image.convert("RGB")
        boxes = self.detect_dogs(image)
        padded = [pad_box(b, image.width, image.height) for b in boxes]
        crops = [image.crop(b) for b in padded]
        body = self.embed_bodies([image] + crops)
        face_images, face_item, face_boxes = [], [], []
        for index, (crop, origin, face) in enumerate(
            zip(crops, padded, self.find_faces(crops))
        ):
            if face is None:
                continue
            box = [
                max(0, round(face[0] + origin[0])),
                max(0, round(face[1] + origin[1])),
                min(image.width, round(face[2] + origin[0])),
                min(image.height, round(face[3] + origin[1])),
            ]
            if min(box[2] - box[0], box[3] - box[1]) < MIN_FACE_SIDE:
                continue
            face_images.append(image.crop(box))
            face_item.append(index + 1)
            face_boxes.append(box)
        return {
            "boxes": np.array([[-1, -1, -1, -1]] + boxes, np.int32).reshape(-1, 4),
            "body": body.astype(np.float16),
            "face_item": np.array(face_item, np.int32),
            "face_boxes": np.array(face_boxes, np.int32).reshape(-1, 4),
            "face": self.embed_faces(face_images).astype(np.float16),
        }


class Features:
    """Every item of every analysed photo for one date, as flat arrays.

    photos      list of filenames
    item_photo  index into photos for each item
    boxes       dog box per item in photo pixels (-1s for the full frame)
    body        body embedding per item
    face_item   index into items for each face
    face_boxes  face box in photo pixels
    face        face embedding
    unreadable  filenames that could not be opened, so they are not retried
    """

    FIELDS = ("item_photo", "boxes", "body", "face_item", "face_boxes", "face")

    def __init__(self, photos=None, **arrays):
        self.photos: list[str] = photos or []
        self.unreadable: list[str] = arrays.get("unreadable") or []
        self.item_photo = arrays.get("item_photo", np.zeros(0, np.int32))
        self.boxes = arrays.get("boxes", np.zeros((0, 4), np.int32))
        self.body = arrays.get("body", np.zeros((0, EMBED_DIM), np.float16))
        self.face_item = arrays.get("face_item", np.zeros(0, np.int32))
        self.face_boxes = arrays.get("face_boxes", np.zeros((0, 4), np.int32))
        self.face = arrays.get("face", np.zeros((0, EMBED_DIM), np.float16))

    def add(self, name: str, analysis: dict):
        first_item = len(self.item_photo)
        count = len(analysis["boxes"])
        self.item_photo = np.concatenate(
            [self.item_photo, np.full(count, len(self.photos), np.int32)]
        )
        self.photos.append(name)
        self.boxes = np.concatenate([self.boxes, analysis["boxes"]])
        self.body = np.concatenate([self.body, analysis["body"]])
        self.face_item = np.concatenate(
            [self.face_item, analysis["face_item"] + first_item]
        )
        self.face_boxes = np.concatenate([self.face_boxes, analysis["face_boxes"]])
        self.face = np.concatenate([self.face, analysis["face"]])

    def save(self, path: Path):
        tmp = path.with_suffix(".tmp.npz")
        np.savez(
            tmp,
            version=FEATURES_VERSION,
            photos=json.dumps(self.photos),
            unreadable=json.dumps(self.unreadable),
            **{field: getattr(self, field) for field in self.FIELDS},
        )
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> "Features":
        """Saved features, or an empty set if missing or from another pipeline version."""
        if not path.exists():
            return cls()
        with np.load(path) as data:
            if str(data["version"]) != FEATURES_VERSION:
                return cls()
            unreadable = (
                json.loads(str(data["unreadable"])) if "unreadable" in data else []
            )
            return cls(
                photos=json.loads(str(data["photos"])),
                unreadable=unreadable,
                **{field: data[field] for field in cls.FIELDS},
            )


def features_path(day: str) -> Path:
    return CACHE_DIR / day / "features.npz"


def load_features(day: str) -> Features:
    return Features.load(features_path(day))


def analyze_day(day: str, analyzer_factory, progress=None) -> Features:
    """Analyse any photos of a date that aren't in its features file yet.

    analyzer_factory is called only if there is work to do, so a fully
    analysed date costs nothing and loads no models.
    """
    features = load_features(day)
    done = set(features.photos) | set(features.unreadable)
    todo = [p for p in cached_photos(day) if p.name not in done]
    if not todo:
        return features
    analyzer = analyzer_factory()
    try:
        for count, path in enumerate(todo, 1):
            try:
                with Image.open(path) as image:
                    image.load()
                    rgb = image.convert("RGB")
            except (OSError, ValueError, Image.DecompressionBombError) as error:
                # A broken file stays broken: remember it instead of retrying every run.
                print(f"  skipping {path.name}: {error}")
                features.unreadable.append(path.name)
            else:
                features.add(path.name, analyzer.analyze(rgb))
            if progress:
                progress(count, len(todo))
    finally:
        # Keep whatever was finished, even after Ctrl-C or a model error.
        features.save(features_path(day))
    return features
