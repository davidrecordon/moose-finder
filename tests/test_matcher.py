import numpy as np

from matcher import MooseModel
from pipeline import EMBED_DIM, Features, face_box_from_keypoints, pad_box
from train import stays


def unit(index):
    v = np.zeros(EMBED_DIM, np.float32)
    v[index] = 1
    return v


def make_model(**settings):
    # Body reference is axis 0. Face classifier: axis 1 = Moose, axis 2 = lookalike.
    return MooseModel(
        body_refs=unit(0)[None],
        face_w=10 * (unit(1) - unit(2)),
        face_b=0.0,
        body_w=10 * (unit(3) - unit(4)),
        body_b=0.0,
        settings=settings,
    )


def photo(body, faces=()):
    """One photo: a full frame unlike Moose, then one crop per body vector."""
    return {
        "boxes": np.array([[-1] * 4] + [[10, 10, 200, 200]] * len(body), np.int32),
        "body": np.array([unit(9)] + list(body), np.float16),
        "face_item": np.array([i for i, _ in faces], np.int32),
        "face_boxes": np.array(
            [[20, 20, 20 + s, 20 + s] for _, (s, _) in faces], np.int32
        ).reshape(-1, 4),
        "face": np.array([f for _, (_, f) in faces], np.float16).reshape(-1, EMBED_DIM),
    }


def verdicts(*photos):
    features = Features()
    for i, p in enumerate(photos):
        features.add(f"{i}.jpg", p)
    return [r.verdict for r in make_model().score_photos(features)]


def test_face_confirms_or_rejects_a_bernese():
    moose = photo([unit(0)], [(1, (80, unit(1)))])
    lookalike = photo([unit(0)], [(1, (80, unit(2)))])
    assert verdicts(moose, lookalike) == ["match", "lookalike"]


def test_no_face_falls_back_to_the_body_classifier():
    likely = photo([unit(0) * 0.9 + unit(3) * 0.4])
    unlikely = photo([unit(0) * 0.9 + unit(4) * 0.4])
    assert verdicts(likely, unlikely) == ["possible", "no"]


def test_tiny_faces_are_ignored():
    blurry = photo([unit(0) * 0.9 + unit(3) * 0.4], [(1, (30, unit(2)))])
    assert verdicts(blurry) == ["possible"]


def test_other_dogs_are_never_candidates():
    other = photo([unit(7)], [(1, (80, unit(1)))])
    assert verdicts(other) == ["no"]


def test_a_match_anywhere_in_the_photo_wins():
    both = photo([unit(0), unit(0)], [(1, (80, unit(2))), (2, (80, unit(1)))])
    features = Features()
    features.add("group.jpg", both)
    result = make_model().score_photos(features)[0]
    assert result.verdict == "match" and result.face_probability > 0.9


def test_model_round_trips_through_disk(tmp_path):
    model = make_model(candidate_similarity=0.7)
    model.save(tmp_path / "m.npz")
    loaded = MooseModel.load(tmp_path / "m.npz")
    assert loaded.settings["candidate_similarity"] == 0.7
    assert np.allclose(loaded.face_w, model.face_w)


def test_features_round_trip_and_item_offsets(tmp_path):
    features = Features()
    features.add("a.jpg", photo([unit(0)], [(1, (80, unit(1)))]))
    features.add("b.jpg", photo([unit(0)], [(1, (80, unit(2)))]))
    assert features.face_item.tolist() == [1, 3]
    features.save(tmp_path / "f.npz")
    loaded = Features.load(tmp_path / "f.npz")
    assert loaded.photos == ["a.jpg", "b.jpg"] and loaded.item_photo.tolist() == [
        0,
        0,
        1,
        1,
    ]


def test_pad_box_clips_to_the_image():
    assert pad_box([0, 10, 100, 110], 105, 300) == [0, 0, 105, 120]


def test_face_box_needs_nose_and_both_ears():
    k = np.zeros((24, 3), np.float32)
    k[14], k[15], k[16], k[17] = (
        (10, 10, 0.9),
        (50, 10, 0.9),
        (30, 40, 0.9),
        (30, 50, 0.9),
    )
    x1, y1, x2, y2 = face_box_from_keypoints(k)
    assert abs((x2 - x1) - 54) < 1e-3 and abs((x1 + x2) / 2 - 30) < 1e-3
    k[15, 2] = 0.1  # one ear hidden: side-on
    assert face_box_from_keypoints(k) is None


def test_stays_split_on_gaps():
    groups = stays(["2026-01-01", "2026-01-02", "2026-01-05", "2026-01-09"])
    assert [groups[d] for d in sorted(groups)] == [0, 0, 0, 1]


def test_analyze_day_remembers_broken_photos_and_saves_partial_work(
    tmp_path, monkeypatch
):
    import pipeline
    import scraper
    from PIL import Image

    monkeypatch.setattr(pipeline, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(scraper, "CACHE_DIR", tmp_path)
    day = tmp_path / "2026-01-01"
    day.mkdir()
    Image.new("RGB", (64, 64)).save(day / "a.jpg")
    (day / "b.jpg").write_bytes(b"not a jpeg")
    Image.new("RGB", (64, 64)).save(day / "c.jpg")

    class Stub:
        calls = 0

        def analyze(self, image):
            Stub.calls += 1
            if Stub.calls == 2:
                raise RuntimeError("out of memory")
            return photo([unit(0)])

    try:
        pipeline.analyze_day("2026-01-01", Stub)
    except RuntimeError:
        pass
    saved = pipeline.load_features("2026-01-01")
    assert saved.photos == ["a.jpg"] and saved.unreadable == ["b.jpg"]

    done = pipeline.analyze_day("2026-01-01", Stub)
    assert done.photos == ["a.jpg", "c.jpg"]

    def never():
        raise AssertionError("models should not load when nothing is left to do")

    assert pipeline.analyze_day("2026-01-01", never).photos == ["a.jpg", "c.jpg"]
