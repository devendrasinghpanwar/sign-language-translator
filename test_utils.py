"""
tests/test_utils.py

Unit tests for utils.py.

KEY IDEA FOR BEGINNERS: we don't need a real webcam or a real
MediaPipe detection to test get_landmark_row(). We just need any
object that "looks like" what MediaPipe returns (has a `.landmark`
list where each item has .x, .y, .z). This is called a "mock" or
"fake" object, and it's a core testing technique — it lets us test
our logic in isolation, fast and reliably, without depending on
external hardware or randomness.

Run with:
  pytest tests/test_utils.py -v
"""

import pytest
from utils import get_landmark_row, validate_feature_row


class FakeLandmark:
    """A tiny stand-in for MediaPipe's landmark object."""
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class FakeHandLandmarks:
    """A tiny stand-in for MediaPipe's hand_landmarks object."""
    def __init__(self, landmark_list):
        self.landmark = landmark_list


def make_fake_hand(num_points=21):
    """Build a fake hand with `num_points` landmarks, each with
    predictable, easy-to-check coordinate values."""
    landmarks = [
        FakeLandmark(x=i * 0.01, y=i * 0.02, z=i * 0.03)
        for i in range(num_points)
    ]
    return FakeHandLandmarks(landmarks)


class TestGetLandmarkRow:

    def test_returns_63_values_for_21_landmarks(self):
        """MediaPipe hands always report 21 landmarks x 3 coords = 63."""
        fake_hand = make_fake_hand(num_points=21)
        row = get_landmark_row(fake_hand)
        assert len(row) == 63

    def test_values_are_in_correct_order(self):
        """The row should be [x0, y0, z0, x1, y1, z1, ...] in landmark order."""
        fake_hand = make_fake_hand(num_points=2)
        row = get_landmark_row(fake_hand)
        # landmark 0: x=0.0, y=0.0, z=0.0
        # landmark 1: x=0.01, y=0.02, z=0.03
        assert row == [0.0, 0.0, 0.0, 0.01, 0.02, 0.03]

    def test_returns_list_of_floats(self):
        fake_hand = make_fake_hand()
        row = get_landmark_row(fake_hand)
        assert all(isinstance(v, float) for v in row)

    def test_empty_landmark_list_returns_empty_row(self):
        """Edge case: a hand with zero landmarks should give an empty row,
        not crash."""
        fake_hand = FakeHandLandmarks([])
        row = get_landmark_row(fake_hand)
        assert row == []


class TestValidateFeatureRow:

    def test_valid_row_of_63_numbers(self):
        row = [0.1] * 63
        assert validate_feature_row(row) is True

    def test_wrong_length_is_invalid(self):
        row = [0.1] * 60  # missing 3 values
        assert validate_feature_row(row) is False

    def test_empty_row_is_invalid(self):
        assert validate_feature_row([]) is False

    def test_non_numeric_value_is_invalid(self):
        row = [0.1] * 62 + ["not_a_number"]
        assert validate_feature_row(row) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
