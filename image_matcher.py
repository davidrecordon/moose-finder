"""Local image matching using MegaDescriptor embeddings for individual animal re-identification."""

import timm
import torch
import torchvision.transforms as T
from huggingface_hub import try_to_load_from_cache
from PIL import Image
from pathlib import Path

MODEL_REPO = "BVRA/MegaDescriptor-L-384"
MODEL_NAME = f"hf-hub:{MODEL_REPO}"
MODEL_ID = "MegaDescriptor-L-384"
IMAGE_SIZE = 384


def is_model_cached() -> bool:
    """Check if model weights are already in the HuggingFace Hub cache."""
    for filename in ("model.safetensors", "pytorch_model.bin"):
        result = try_to_load_from_cache(MODEL_REPO, filename)
        if isinstance(result, str):
            return True
    return False


class ImageMatcher:
    """Match images using MegaDescriptor embeddings for individual animal re-identification.

    MegaDescriptor is a foundation model specifically designed for individual animal
    re-identification, trained on datasets including DogFaceNet and MPDD. It significantly
    outperforms general-purpose models like CLIP and DINOv2 on animal re-ID tasks.

    When ultralytics is installed, uses YOLOv8n to detect individual dogs and runs
    MegaDescriptor on each crop, returning the max score across full image + all crops.
    """

    def __init__(self, local_only=False):
        self.model_id = MODEL_ID
        if local_only:
            import huggingface_hub.constants

            _prev = huggingface_hub.constants.HF_HUB_OFFLINE
            huggingface_hub.constants.HF_HUB_OFFLINE = True
        self.model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)
        if local_only:
            huggingface_hub.constants.HF_HUB_OFFLINE = _prev
        self.model.eval()
        self.transforms = T.Compose(
            [
                T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.reference_embeddings = []

        # Initialize dog detector if available
        self.detector = None
        try:
            from dog_detector import DogDetector

            self.detector = DogDetector()
            self.model_id = MODEL_ID + "+YOLOv8n"
        except ImportError:
            pass

    def load_references(self, reference_dir: Path) -> int:
        """Load and embed reference photos. Returns count loaded."""
        all_paths = []
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            all_paths.extend(reference_dir.glob(ext))
        for img_path in sorted(all_paths):
            try:
                img = Image.open(img_path).convert("RGB")
                embedding = self._embed_image(img)
                self.reference_embeddings.append((embedding, img_path.stem))
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
        return len(self.reference_embeddings)

    def _embed_image(self, img: Image.Image) -> torch.Tensor:
        """Generate embedding for an image."""
        with torch.no_grad():
            img_tensor = self.transforms(img).unsqueeze(0)
            embedding = self.model(img_tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding

    def _cosine_similarity(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Compute cosine similarity between two embeddings, clamped to [0, 1]."""
        sim = torch.nn.functional.cosine_similarity(a, b).item()
        return max(0.0, min(1.0, sim))

    def _get_per_reference_similarities(
        self, candidate: Image.Image
    ) -> tuple[float, list[tuple[str, float]]]:
        """Get per-reference similarities for a candidate image.

        Returns:
            Tuple of (max_similarity, [(ref_name, similarity), ...])
        """
        candidate_emb = self._embed_image(candidate.convert("RGB"))
        per_ref = [
            (ref_name, self._cosine_similarity(candidate_emb, ref_emb))
            for ref_emb, ref_name in self.reference_embeddings
        ]
        max_sim = max((sim for _, sim in per_ref), default=0.0)
        return max_sim, per_ref

    def get_similarity(self, candidate: Image.Image) -> float:
        """Get max similarity score vs all reference photos (0.0-1.0).

        When detection is active, computes similarity for the full image and
        each detected dog crop, returning the maximum across all of them.
        """
        if not self.reference_embeddings:
            return 0.0

        # Always compute full-image similarity as baseline
        full_sim, _ = self._get_per_reference_similarities(candidate)

        # If no detector, return full-image score
        if self.detector is None:
            return full_sim

        # Detect dogs and compute per-crop similarity
        best_sim = full_sim
        for det in self.detector.detect(candidate):
            crop_sim, _ = self._get_per_reference_similarities(det.crop(candidate))
            best_sim = max(best_sim, crop_sim)

        return best_sim

    def get_debug_info(self, candidate: Image.Image) -> dict:
        """Get full diagnostic breakdown for a candidate image.

        Returns dict with per-reference scores, YOLO detections, and classification.
        """
        if not self.reference_embeddings:
            return {
                "full_image": {"similarity": 0.0, "per_reference": []},
                "detections": [],
                "best_similarity": 0.0,
                "best_source": "full_image",
                "classification": "no_match",
            }

        # Full image analysis
        full_sim, full_per_ref = self._get_per_reference_similarities(candidate)
        full_info = {
            "similarity": full_sim,
            "best_ref": max(full_per_ref, key=lambda x: x[1])[0],
            "per_reference": full_per_ref,
        }

        best_sim = full_sim
        best_source = "full_image"
        detections_info = []

        # YOLO detections
        if self.detector is not None:
            detections = self.detector.detect(candidate)
            for i, det in enumerate(detections):
                crop = det.crop(candidate)
                crop_w, crop_h = crop.size
                crop_sim, crop_per_ref = self._get_per_reference_similarities(crop)
                crop_best_ref = max(crop_per_ref, key=lambda x: x[1])[0]

                det_info = {
                    "index": i + 1,
                    "bbox": [
                        round(det.x1),
                        round(det.y1),
                        round(det.x2),
                        round(det.y2),
                    ],
                    "yolo_confidence": det.confidence,
                    "crop_size": [crop_w, crop_h],
                    "crop": crop,
                    "similarity": crop_sim,
                    "best_ref": crop_best_ref,
                    "per_reference": crop_per_ref,
                }
                detections_info.append(det_info)

                if crop_sim > best_sim:
                    best_sim = crop_sim
                    best_source = f"detection_{i + 1}"

        return {
            "full_image": full_info,
            "detections": detections_info,
            "best_similarity": best_sim,
            "best_source": best_source,
            "classification": classify_match(best_sim),
        }


def classify_match(
    similarity: float, match_threshold: float = 0.90, uncertain_threshold: float = 0.50
) -> str:
    """Classify based on similarity score.

    Args:
        similarity: MegaDescriptor similarity score (0.0-1.0)
        match_threshold: Score above this is auto-match (default 0.90)
        uncertain_threshold: Score above this needs Claude verification (default 0.50)

    Note: MegaDescriptor is trained specifically for individual animal re-identification
    and should provide better discrimination than general-purpose models.
    """
    if similarity >= match_threshold:
        return "match"
    if similarity >= uncertain_threshold:
        return "uncertain"
    return "no_match"
