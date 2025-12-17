"""Configuration constants for the trust and stability layer."""

# Frame Quality Thresholds
BLUR_MIN = 80.0  # Laplacian variance threshold (tune as needed)
BRIGHTNESS_MIN = 40.0  # Mean grayscale luminance threshold (0-255)

# Temporal Aggregation
N = 20  # Rolling window size
EMA_ALPHA = 0.8  # EMA smoothing factor for probabilities

# Decision Engine Thresholds
# Lowered thresholds for easier locking (can be adjusted)
LOCK_MIN_CONF = 0.60  # Minimum EMA confidence to lock (lowered from 0.70)
LOCK_MIN_MARGIN = 0.10  # Minimum margin (p1 - p2) to lock (lowered from 0.15)
LOCK_MIN_VOTE_RATIO = 0.65  # Minimum vote ratio to lock (lowered from 0.75, e.g., 13/20 agree)
UNLOCK_VOTE_RATIO = 0.55  # If drops below this, unlock
UNKNOWN_MIN_CONF = 0.55  # Below this tends to UNKNOWN
MAX_BAD_QUALITY_FRAMES = 10  # If too many blurry/dark frames in window → UNKNOWN

# Stability computation weights
STABILITY_VOTE_WEIGHT = 0.7
STABILITY_MARGIN_WEIGHT = 0.3

