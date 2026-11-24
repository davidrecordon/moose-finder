"""Decide which dog is Moose, from the features pipeline.py produced.

Two stages, both cheap (a few matrix products):

  body  How similar is each dog crop to known Moose crops? This finds
        "Bernese that could be Moose" but cannot tell him from the other
        Bernese at camp.
  face  When the dog faces the camera, a small classifier trained on Moose's
        face versus the lookalikes' faces confirms or rejects. When no face is
        visible, a weaker classifier on the body embedding gives a second
        opinion instead.

Everything learned lives in models/moose_model.npz, written by train.py.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from pipeline import Features

PROJECT_DIR = Path(__file__).parent
MODEL_PATH = PROJECT_DIR / "models" / "moose_model.npz"

VERDICT_ORDER = {"match": 3, "possible": 2, "lookalike": 1, "no": 0}

DEFAULT_SETTINGS = {
    # Body similarity a crop needs before it is considered at all.
    "candidate_similarity": 0.85,
    # Face classifier probability at or above which a visible face confirms Moose.
    "face_threshold": 0.6,
    # Faces smaller than this many pixels a side are too blurry to judge.
    "min_face_side": 40,
    # Without a face: body classifier probability needed to stay "possible".
    "body_threshold": 0.6,
    # How far the second-stage opinion moves the ranking score.
    "face_weight": 0.3,
    "body_weight": 0.1,
}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class PhotoResult:
    photo: str
    verdict: str  # match | possible | lookalike | no
    score: float  # ranking score: similarity nudged by the second-stage opinion
    similarity: float  # body similarity of the deciding item
    face_probability: float | None  # None when no usable face
    body_probability: float
    box: list[int] | None  # dog box in photo pixels; None = whole frame
    face_box: list[int] | None

    def to_dict(self) -> dict:
        return asdict(self)


class MooseModel:
    def __init__(
        self,
        body_refs: np.ndarray,
        face_w: np.ndarray,
        face_b: float,
        body_w: np.ndarray,
        body_b: float,
        settings: dict | None = None,
        info: dict | None = None,
    ):
        self.body_refs = body_refs.astype(np.float32)
        self.face_w, self.face_b = face_w.astype(np.float32), float(face_b)
        self.body_w, self.body_b = body_w.astype(np.float32), float(body_b)
        self.settings = {**DEFAULT_SETTINGS, **(settings or {})}
        self.info = info or {}

    def save(self, path: Path = MODEL_PATH):
        path.parent.mkdir(exist_ok=True)
        np.savez(
            path,
            body_refs=self.body_refs.astype(np.float16),
            face_w=self.face_w,
            face_b=self.face_b,
            body_w=self.body_w,
            body_b=self.body_b,
            settings=json.dumps(self.settings),
            info=json.dumps(self.info),
        )

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "MooseModel":
        with np.load(path) as data:
            return cls(
                data["body_refs"],
                data["face_w"],
                float(data["face_b"]),
                data["body_w"],
                float(data["body_b"]),
                json.loads(str(data["settings"])),
                json.loads(str(data["info"])),
            )

    def score_items(self, features: Features) -> dict[str, np.ndarray]:
        """Per-item arrays: similarity, body_p, face_p (NaN = no usable face), score, verdict."""
        s = self.settings
        body = features.body.astype(np.float32)
        count = len(body)
        if count == 0:
            empty = np.zeros(0, np.float32)
            return {
                "similarity": empty,
                "body_p": empty,
                "face_p": empty,
                "face_index": np.zeros(0, np.int32),
                "score": empty,
                "verdict": np.zeros(0, np.int8),
            }
        similarity = (body @ self.body_refs.T).max(1)
        body_p = sigmoid(body @ self.body_w + self.body_b)
        face_p = np.full(count, np.nan, np.float32)
        face_index = np.full(count, -1, np.int32)
        if len(features.face):
            sides = np.minimum(
                features.face_boxes[:, 2] - features.face_boxes[:, 0],
                features.face_boxes[:, 3] - features.face_boxes[:, 1],
            )
            usable = sides >= s["min_face_side"]
            probabilities = sigmoid(
                features.face.astype(np.float32) @ self.face_w + self.face_b
            )
            face_p[features.face_item[usable]] = probabilities[usable]
            face_index[features.face_item[usable]] = np.nonzero(usable)[0]
        has_face = ~np.isnan(face_p)
        score = np.where(
            has_face,
            similarity + s["face_weight"] * (np.nan_to_num(face_p) - 0.5),
            similarity + s["body_weight"] * (body_p - 0.5),
        )
        candidate = similarity >= s["candidate_similarity"]
        verdict = np.zeros(count, np.int8)  # no
        face_yes = has_face & (np.nan_to_num(face_p) >= s["face_threshold"])
        verdict[candidate & has_face & ~face_yes] = VERDICT_ORDER["lookalike"]
        verdict[candidate & ~has_face & (body_p >= s["body_threshold"])] = (
            VERDICT_ORDER["possible"]
        )
        verdict[candidate & face_yes] = VERDICT_ORDER["match"]
        return {
            "similarity": similarity,
            "body_p": body_p,
            "face_p": face_p,
            "face_index": face_index,
            "score": score,
            "verdict": verdict,
        }

    def score_photos(self, features: Features) -> list[PhotoResult]:
        """One result per photo: its strongest verdict, then its highest score."""
        items = self.score_items(features)
        names = {v: k for k, v in VERDICT_ORDER.items()}
        best: dict[int, int] = {}
        for i, photo in enumerate(features.item_photo):
            j = best.get(photo)
            key = (items["verdict"][i], items["score"][i])
            if j is None or key > (items["verdict"][j], items["score"][j]):
                best[int(photo)] = i
        results = []
        for photo, i in sorted(best.items()):
            box = features.boxes[i].tolist()
            face = items["face_index"][i]
            results.append(
                PhotoResult(
                    photo=features.photos[photo],
                    verdict=names[int(items["verdict"][i])],
                    score=float(items["score"][i]),
                    similarity=float(items["similarity"][i]),
                    face_probability=(None if face < 0 else float(items["face_p"][i])),
                    body_probability=float(items["body_p"][i]),
                    box=None if box[0] < 0 else box,
                    face_box=None if face < 0 else features.face_boxes[face].tolist(),
                )
            )
        return results
