from predictivecare.features import (
    TRACK1_FEATURES, TRACK1_LABELS, TRACK2_FEATURES, TRACK2_LABELS,
)


def test_feature_counts():
    assert len(TRACK1_FEATURES) == 12
    assert len(TRACK2_FEATURES) == 12


def test_label_counts():
    assert len(TRACK1_LABELS) == 8
    assert len(TRACK2_LABELS) == 4


def test_no_duplicate_features():
    assert len(set(TRACK1_FEATURES)) == len(TRACK1_FEATURES)
    assert len(set(TRACK2_FEATURES)) == len(TRACK2_FEATURES)
