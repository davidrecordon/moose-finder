#!/usr/bin/env python3
"""Learn what Moose looks like from labels/labels.json and write models/moose_model.npz.

Usage:
    python train.py              # evaluate, then train on every label and save
    python train.py --dry-run    # evaluate only

Labels are per photo ("moose", "lookalike", "not_moose"), made with labeler.py.
Training works out which dog in each labelled photo the label refers to, then
fits:
  - body references: the crop of Moose from every confirmed photo
  - a face classifier: Moose's face against hard negatives
  - a body classifier: the same, on body embeddings, for dogs with no visible face

Evaluation holds out one stay at a time (so a model is never tested on photos
from a stay it learned from) and reports how each labelled photo was judged.
"""

import argparse
import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression

from matcher import DEFAULT_SETTINGS, MODEL_PATH, MooseModel
from pipeline import Analyzer, Features, load_features
from scraper import CACHE_DIR

PROJECT_DIR = Path(__file__).parent
REFERENCE_DIR = PROJECT_DIR / "reference_photos"
LABELS_PATH = PROJECT_DIR / "labels" / "labels.json"
REFERENCE_FEATURES = CACHE_DIR / "reference_features.npz"

MIN_PICK_SIMILARITY = (
    0.75  # a labelled photo's chosen crop must look at least this Moose-like
)
HARD_NEGATIVE_SIMILARITY = (
    0.75  # non-Moose crops this similar are what the classifiers must reject
)
STAY_GAP_DAYS = 3
FACE_REGULARISATION = 100.0  # C for the face classifier; chosen by held-out AUC
BODY_REGULARISATION = 10.0


def reference_features() -> Features:
    """Features of the hand-picked photos in reference_photos/ (analysed once)."""
    names = sorted(p.name for p in REFERENCE_DIR.glob("*.jp*g"))
    features = Features.load(REFERENCE_FEATURES)
    if features.photos == names:
        return features
    analyzer, features = Analyzer(), Features()
    for name in names:
        with Image.open(REFERENCE_DIR / name) as image:
            features.add(name, analyzer.analyze(image))
    features.save(REFERENCE_FEATURES)
    return features


def stays(days: list[str]) -> dict[str, int]:
    """Group dates into stays: consecutive dates no more than STAY_GAP_DAYS apart."""
    groups, current = {}, 0
    ordered = sorted(days)
    for previous, day in zip([None] + ordered, ordered):
        if previous:
            gap = (date.fromisoformat(day) - date.fromisoformat(previous)).days
            if gap > STAY_GAP_DAYS:
                current += 1
        groups[day] = current
    return groups


class NotEnoughLabels(Exception):
    pass


class Examples:
    """Labelled photos resolved to specific dog crops."""

    def __init__(self, labels: dict[str, str]):
        self.labels = {k: v for k, v in labels.items() if v != "unsure"}
        self.days = sorted({k[:10] for k in self.labels})
        self.features = {day: load_features(day) for day in self.days}
        self.reference = reference_features()
        if not self.reference.photos:
            raise SystemExit(
                f"No photos in {REFERENCE_DIR}: add a few clear photos of Moose"
            )
        # Hand-picked reference photos are single-dog portraits: use the full frame.
        seed_items = [
            np.nonzero(self.reference.item_photo == i)[0][0]
            for i in range(len(self.reference.photos))
        ]
        self.seed_body = self.reference.body[seed_items].astype(np.float32)
        self.seed_face = self.reference.face.astype(np.float32)
        self.items = {}  # photo key -> (day, [item indexes])
        for key in self.labels:
            day, name = key.split("/", 1)
            f = self.features[day]
            if name in f.photos:
                rows = np.nonzero(f.item_photo == f.photos.index(name))[0]
                self.items[key] = (day, rows)
        self.missing = [k for k in self.labels if k not in self.items]
        self.moose = self._pick_moose()

    def body(self, day: str, rows) -> np.ndarray:
        return self.features[day].body[rows].astype(np.float32)

    def _best_crop(self, key: str, refs: np.ndarray) -> tuple[int, float]:
        """The item in a photo most like the references (crops preferred over the full frame)."""
        day, rows = self.items[key]
        crops = rows[1:] if len(rows) > 1 else rows
        sims = (self.body(day, crops) @ refs.T).max(1)
        return int(crops[sims.argmax()]), float(sims.max())

    def _pick_moose(self) -> dict[str, int]:
        """Which item is Moose in each photo labelled "moose".

        Two passes: the seed references find the obvious ones, which then help
        find him in photos where he looks less like the seeds. Each photo is
        judged against references from other photos only.
        """
        keys = [k for k, v in self.labels.items() if v == "moose" and k in self.items]
        picks: dict[str, int] = {}
        for _ in range(2):
            new = {}
            for key in keys:
                others = [
                    self.body(k[:10], [i])[0] for k, i in picks.items() if k != key
                ]
                refs = (
                    np.concatenate([self.seed_body, others])
                    if others
                    else self.seed_body
                )
                item, sim = self._best_crop(key, refs)
                if sim >= MIN_PICK_SIMILARITY:
                    new[key] = item
            picks = new
        return picks

    def face_of(self, day: str, item: int) -> np.ndarray | None:
        f = self.features[day]
        hits = np.nonzero(f.face_item == item)[0]
        return f.face[hits[0]].astype(np.float32) if len(hits) else None

    def training_set(self, exclude_days: set[str] = frozenset()):
        """(body refs, face X/y, body X/y) from every labelled day not excluded."""
        moose = {k: i for k, i in self.moose.items() if k[:10] not in exclude_days}
        refs = np.concatenate(
            [self.seed_body] + [self.body(k[:10], [i]) for k, i in moose.items()]
        )
        face_x, face_y = list(self.seed_face), [1] * len(self.seed_face)
        body_x, body_y = list(self.seed_body), [1] * len(self.seed_body)
        for key, item in moose.items():
            body_x.append(self.body(key[:10], [item])[0])
            body_y.append(1)
            face = self.face_of(key[:10], item)
            if face is not None:
                face_x.append(face)
                face_y.append(1)
        # Hard negatives: any Moose-like crop in a photo known not to contain Moose.
        for key, label in self.labels.items():
            if label == "moose" or key not in self.items or key[:10] in exclude_days:
                continue
            day, rows = self.items[key]
            crops = rows[1:]
            if not len(crops):
                continue
            sims = (self.body(day, crops) @ refs.T).max(1)
            for item in crops[sims >= HARD_NEGATIVE_SIMILARITY]:
                body_x.append(self.body(day, [item])[0])
                body_y.append(0)
                face = self.face_of(day, int(item))
                if face is not None:
                    face_x.append(face)
                    face_y.append(0)
        return (
            refs,
            (np.array(face_x), np.array(face_y)),
            (np.array(body_x), np.array(body_y)),
        )

    def fit(self, exclude_days: set[str] = frozenset(), settings=None) -> MooseModel:
        refs, (face_x, face_y), (body_x, body_y) = self.training_set(exclude_days)
        classifiers = []
        for name, x, y, c in (
            ("face", face_x, face_y, FACE_REGULARISATION),
            ("body", body_x, body_y, BODY_REGULARISATION),
        ):
            if len(set(y.tolist())) < 2:
                raise NotEnoughLabels(
                    f"the {name} classifier needs both Moose and non-Moose examples; "
                    f"label more photos (including lookalikes) with labeler.py"
                )
            clf = LogisticRegression(C=c, class_weight="balanced", max_iter=5000).fit(
                x, y
            )
            classifiers.append((clf.coef_[0], float(clf.intercept_[0])))
        info = {
            "body_references": int(len(refs)),
            "moose_faces": int(face_y.sum()),
            "other_faces": int((1 - face_y).sum()),
            "moose_bodies": int(body_y.sum()),
            "other_bodies": int((1 - body_y).sum()),
            "labelled_photos": len(self.labels),
            "labelled_days": len(self.days),
        }
        (fw, fb), (bw, bb) = classifiers
        return MooseModel(refs, fw, fb, bw, bb, settings, info)

    def cross_validate(self, settings=None) -> dict[str, dict]:
        """Judge every labelled photo with a model that never saw its stay."""
        groups = stays(self.days)
        results = {}
        for stay in sorted(set(groups.values())):
            held_out = {d for d, s in groups.items() if s == stay}
            try:
                model = self.fit(held_out, settings)
            except NotEnoughLabels:
                continue  # too few labels outside this stay to judge it
            for day in held_out:
                for result in model.score_photos(self.features[day]):
                    key = f"{day}/{result.photo}"
                    if key in self.labels:
                        results[key] = result
        return results


def summarise(title: str, labels: dict, results: dict, keys: list[str]):
    table = defaultdict(Counter)
    for key in keys:
        if key in results:
            table[labels[key]][results[key].verdict] += 1
    print(f"\n{title}")
    print(
        f"  {'label':10s} {'match':>6s} {'possible':>9s} {'lookalike':>10s} {'no':>6s}"
    )
    for label in ("moose", "lookalike", "not_moose"):
        row = table[label]
        print(
            f"  {label:10s} {row['match']:6d} {row['possible']:9d} {row['lookalike']:10d} {row['no']:6d}"
        )
    scored = sorted((k for k in keys if k in results), key=lambda k: -results[k].score)
    ranks = [i + 1 for i, k in enumerate(scored) if labels[k] == "moose"]
    print(f"  Moose photo ranks by score, of {len(scored)}: {ranks}")


def main():
    parser = argparse.ArgumentParser(description="Train Moose Finder from labels")
    parser.add_argument(
        "--dry-run", action="store_true", help="Evaluate without saving"
    )
    parser.add_argument(
        "--test-from",
        default="2026-11-14",
        help="Start of the fully labelled test period",
    )
    parser.add_argument("--test-to", default="2026-11-20")
    args = parser.parse_args()

    examples = Examples(json.loads(LABELS_PATH.read_text()))
    labels = examples.labels
    counts = Counter(labels.values())
    print(
        f"{len(labels)} labelled photos over {len(examples.days)} days: {dict(counts)}"
    )
    if examples.missing:
        print(
            f"  {len(examples.missing)} labelled photos have no features yet (run moose_finder.py first)"
        )
    print(
        f"Moose located in {len(examples.moose)} of {counts['moose']} Moose photos "
        f"(the rest have no crop that looks enough like him)"
    )

    results = examples.cross_validate()
    in_test = [k for k in labels if args.test_from <= k[:10] <= args.test_to]
    other = [k for k in labels if k not in set(in_test)]
    summarise(
        f"Held-out results, {args.test_from} to {args.test_to} (every photo labelled):",
        labels,
        results,
        in_test,
    )
    summarise(
        "Held-out results, earlier stays (only likely candidates were labelled):",
        labels,
        results,
        other,
    )

    if args.dry_run:
        return
    try:
        model = examples.fit()
    except NotEnoughLabels as error:
        raise SystemExit(f"Cannot train: {error}")
    model.save()
    print(f"\nSaved {MODEL_PATH.relative_to(PROJECT_DIR)}: {model.info}")
    print(f"Settings: {DEFAULT_SETTINGS}")


if __name__ == "__main__":
    main()
