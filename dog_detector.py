"""Dog detection using YOLOv8 for crop-then-re-ID pipeline."""

from dataclasses import dataclass
from PIL import Image

# COCO class index for "dog"
DOG_CLASS_ID = 16
# Minimum detection area as fraction of image area
MIN_AREA_FRACTION = 0.01
# YOLO confidence threshold
CONFIDENCE_THRESHOLD = 0.25


@dataclass
class DogDetection:
    """A detected dog bounding box."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    def crop(self, image: Image.Image, padding: float = 0.10) -> Image.Image:
        """Crop the detected dog from the image with padding.

        Args:
            image: Source image to crop from
            padding: Fraction of box size to add as padding on each side
        """
        w = self.x2 - self.x1
        h = self.y2 - self.y1
        pad_x = w * padding
        pad_y = h * padding

        img_w, img_h = image.size
        left = max(0, self.x1 - pad_x)
        top = max(0, self.y1 - pad_y)
        right = min(img_w, self.x2 + pad_x)
        bottom = min(img_h, self.y2 + pad_y)

        return image.crop((left, top, right, bottom))


class DogDetector:
    """Detect dogs in images using YOLOv8n."""

    def __init__(self):
        from ultralytics import YOLO

        self.model = YOLO("yolov8n.pt")

    def detect(self, image: Image.Image) -> list[DogDetection]:
        """Detect dogs in an image.

        Returns list of DogDetection objects for dogs found, filtered by
        confidence and minimum area.
        """
        img_w, img_h = image.size
        min_area = img_w * img_h * MIN_AREA_FRACTION

        results = self.model(image, verbose=False, conf=CONFIDENCE_THRESHOLD)

        detections = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id != DOG_CLASS_ID:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                area = (x2 - x1) * (y2 - y1)
                if area < min_area:
                    continue

                detections.append(
                    DogDetection(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        confidence=float(box.conf[0]),
                    )
                )

        return detections
