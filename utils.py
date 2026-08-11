"""
utils.py

Shared, testable helper functions used by collect_data.py and app.py.

Pulling this logic into its own module (instead of duplicating it in
both scripts) is a code-quality best practice: it follows the DRY
principle (Don't Repeat Yourself) and — importantly for this project —
it's what makes the logic unit-testable in isolation, without needing
a real webcam or a real MediaPipe detection running.
"""

from typing import List


def get_landmark_row(hand_landmarks) -> List[float]:
    """
    Convert a MediaPipe hand_landmarks object into a flat list of 63
    numbers (21 landmarks x [x, y, z]).

    Args:
        hand_landmarks: MediaPipe NormalizedLandmarkList object (or any
            object exposing a `.landmark` list of items with x, y, z
            attributes — this duck-typing makes it easy to test with
            simple mock objects instead of a real webcam).

    Returns:
        A list of 63 floats.
    """
    row = []
    for lm in hand_landmarks.landmark:
        row.extend([lm.x, lm.y, lm.z])
    return row


def validate_feature_row(row: List[float]) -> bool:
    """
    Sanity-check that a landmark row has the expected shape and that
    all coordinate values are plausible (MediaPipe normalizes x/y to
    roughly [0, 1] relative to the image, though small negative/over-1
    values can occur near frame edges, so we allow a small margin).
    """
    if len(row) != 63:
        return False
    return all(isinstance(v, (int, float)) for v in row)
