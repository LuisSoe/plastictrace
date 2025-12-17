"""Unit tests for the trust and stability layer."""

import numpy as np
import cv2
from trust.frame_quality import assess_frame_quality, compute_blur_score, compute_brightness
from trust.temporal_aggregator import TemporalAggregator
from trust.decision_engine import DecisionEngine, DecisionState
from ml.config import CLASSES


def test_blur_score():
    """Test blur score computation."""
    # Create a sharp image (high variance)
    sharp = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    blur_score_sharp = compute_blur_score(sharp)
    
    # Create a blurry image (low variance)
    blurry = cv2.GaussianBlur(sharp, (15, 15), 0)
    blur_score_blurry = compute_blur_score(blurry)
    
    assert blur_score_sharp > blur_score_blurry, "Sharp image should have higher blur score"
    print("✓ Blur score test passed")


def test_brightness():
    """Test brightness computation."""
    # Create bright image
    bright = np.ones((100, 100, 3), dtype=np.uint8) * 200
    brightness_bright = compute_brightness(bright)
    
    # Create dark image
    dark = np.ones((100, 100, 3), dtype=np.uint8) * 30
    brightness_dark = compute_brightness(dark)
    
    assert brightness_bright > brightness_dark, "Bright image should have higher brightness"
    print("✓ Brightness test passed")


def test_frame_quality():
    """Test frame quality assessment."""
    # Create a good quality frame
    good_frame = np.random.randint(100, 200, (100, 100, 3), dtype=np.uint8)
    quality = assess_frame_quality(good_frame)
    
    assert isinstance(quality.is_blurry, bool)
    assert isinstance(quality.is_too_dark, bool)
    assert quality.blur_score > 0
    assert quality.brightness > 0
    print("✓ Frame quality test passed")


def test_temporal_aggregator_stable():
    """Test temporal aggregator with stable predictions."""
    num_classes = len(CLASSES)
    agg = TemporalAggregator(num_classes, window_size=20)
    
    # Simulate stable predictions (class 0 with high confidence)
    stable_probs = np.array([0.85, 0.05, 0.05, 0.05])
    
    for _ in range(20):
        result = agg.update(stable_probs)
    
    assert result["vote_label"] == 0, "Should vote for class 0"
    assert result["vote_ratio"] >= 0.9, "Should have high vote ratio"
    assert result["ema_conf"] > 0.8, "Should have high EMA confidence"
    print("✓ Temporal aggregator stable test passed")


def test_temporal_aggregator_alternating():
    """Test temporal aggregator with alternating predictions."""
    num_classes = len(CLASSES)
    agg = TemporalAggregator(num_classes, window_size=20)
    
    # Simulate alternating predictions
    probs_a = np.array([0.7, 0.1, 0.1, 0.1])
    probs_b = np.array([0.1, 0.7, 0.1, 0.1])
    
    for i in range(20):
        probs = probs_a if i % 2 == 0 else probs_b
        result = agg.update(probs)
    
    # Should have lower vote ratio due to alternation
    assert result["vote_ratio"] < 0.6, "Alternating predictions should have lower vote ratio"
    print("✓ Temporal aggregator alternating test passed")


def test_decision_engine_stable_lock():
    """Test decision engine locks on stable object."""
    engine = DecisionEngine()
    
    # Create a good quality frame
    good_frame = np.random.randint(100, 200, (100, 100, 3), dtype=np.uint8)
    
    # Simulate stable high-confidence predictions
    stable_probs = np.array([0.85, 0.05, 0.05, 0.05])
    model_output = {"probs": stable_probs.tolist()}
    
    # Process many frames
    for _ in range(30):
        result = engine.process(good_frame, model_output)
    
    # Should eventually lock
    assert result.state == DecisionState.LOCKED, "Should lock on stable predictions"
    assert result.locked_label == CLASSES[0], "Should lock on correct label"
    assert result.locked_confidence > 0.7, "Should have high locked confidence"
    print("✓ Decision engine stable lock test passed")


def test_decision_engine_alternating_unstable():
    """Test decision engine remains unstable with alternating predictions."""
    engine = DecisionEngine()
    
    good_frame = np.random.randint(100, 200, (100, 100, 3), dtype=np.uint8)
    
    # Simulate alternating predictions
    probs_a = np.array([0.6, 0.2, 0.1, 0.1])
    probs_b = np.array([0.2, 0.6, 0.1, 0.1])
    
    for i in range(30):
        probs = probs_a if i % 2 == 0 else probs_b
        model_output = {"probs": probs.tolist()}
        result = engine.process(good_frame, model_output)
    
    # Should remain unstable or scanning, not locked
    assert result.state != DecisionState.LOCKED, "Should not lock on alternating predictions"
    print("✓ Decision engine alternating unstable test passed")


def test_decision_engine_low_light_unknown():
    """Test decision engine goes to UNKNOWN with low light."""
    engine = DecisionEngine()
    
    # Create a dark frame
    dark_frame = np.ones((100, 100, 3), dtype=np.uint8) * 20
    
    probs = np.array([0.6, 0.2, 0.1, 0.1])
    model_output = {"probs": probs.tolist()}
    
    # Process many dark frames
    for _ in range(15):
        result = engine.process(dark_frame, model_output)
    
    # Should go to UNKNOWN due to bad quality
    assert result.state == DecisionState.UNKNOWN, "Should go to UNKNOWN with persistent low light"
    assert result.reason == "bad_quality", "Reason should be bad_quality"
    print("✓ Decision engine low light unknown test passed")


def test_decision_engine_lock_then_unlock():
    """Test decision engine locks then unlocks on drift."""
    engine = DecisionEngine()
    
    good_frame = np.random.randint(100, 200, (100, 100, 3), dtype=np.uint8)
    
    # First, stable predictions to lock
    stable_probs = np.array([0.85, 0.05, 0.05, 0.05])
    model_output = {"probs": stable_probs.tolist()}
    
    for _ in range(25):
        result = engine.process(good_frame, model_output)
        if result.state == DecisionState.LOCKED:
            break
    
    assert result.state == DecisionState.LOCKED, "Should lock first"
    
    # Then, drift to different predictions
    drift_probs = np.array([0.2, 0.7, 0.05, 0.05])
    model_output = {"probs": drift_probs.tolist()}
    
    for _ in range(15):
        result = engine.process(good_frame, model_output)
        if result.state != DecisionState.LOCKED:
            break
    
    # Should unlock
    assert result.state != DecisionState.LOCKED, "Should unlock on drift"
    print("✓ Decision engine lock then unlock test passed")


def test_decision_engine_low_confidence_unknown():
    """Test decision engine goes to UNKNOWN with low confidence."""
    engine = DecisionEngine()
    
    good_frame = np.random.randint(100, 200, (100, 100, 3), dtype=np.uint8)
    
    # Low confidence predictions (high entropy)
    low_conf_probs = np.array([0.3, 0.25, 0.25, 0.2])
    model_output = {"probs": low_conf_probs.tolist()}
    
    for _ in range(25):
        result = engine.process(good_frame, model_output)
    
    # Should go to UNKNOWN
    assert result.state == DecisionState.UNKNOWN, "Should go to UNKNOWN with low confidence"
    assert result.reason in ["low_confidence", "high_entropy"], "Reason should indicate low confidence"
    print("✓ Decision engine low confidence unknown test passed")


def run_all_tests():
    """Run all tests."""
    print("Running trust layer tests...\n")
    
    try:
        test_blur_score()
        test_brightness()
        test_frame_quality()
        test_temporal_aggregator_stable()
        test_temporal_aggregator_alternating()
        test_decision_engine_stable_lock()
        test_decision_engine_alternating_unstable()
        test_decision_engine_low_light_unknown()
        test_decision_engine_lock_then_unlock()
        test_decision_engine_low_confidence_unknown()
        
        print("\n✅ All tests passed!")
        return True
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Test error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    run_all_tests()

