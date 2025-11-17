"""
Unit tests for weighted ensemble module.

Tests multi-analyst signal aggregation and disagreement tracking
without requiring database access.
"""

from otonom_trader.patron.ensemble import (
    AnalystSignal,
    combine_signals,
    apply_disagreement_penalty,
    get_analyst_weights,
)


def test_combine_signals_buy_bias_and_low_disagreement():
    """Test that multiple BUY signals produce BUY direction with low disagreement."""
    signals = [
        AnalystSignal(
            name="Tech",
            direction="BUY",
            p_up=0.7,
            confidence=0.9,
            weight=1.0,
        ),
        AnalystSignal(
            name="News",
            direction="BUY",
            p_up=0.65,
            confidence=0.8,
            weight=1.0,
        ),
    ]

    ens = combine_signals(signals)

    assert ens.direction == "BUY"
    assert ens.p_up > 0.6
    assert ens.disagreement < 0.7  # Similar direction -> reasonable disagreement


def test_combine_signals_strong_disagreement():
    """Test that conflicting BUY/SELL signals produce high disagreement."""
    signals = [
        AnalystSignal(
            name="Tech",
            direction="BUY",
            p_up=0.8,
            confidence=0.9,
            weight=1.0,
        ),
        AnalystSignal(
            name="News",
            direction="SELL",
            p_up=0.2,  # SELL with high confidence
            confidence=0.9,
            weight=1.0,
        ),
    ]

    ens = combine_signals(signals)

    # BUY / SELL conflict should result in some disagreement
    assert ens.disagreement > 0.3  # Conflict produces disagreement


def test_combine_signals_single_signal():
    """Test that single signal produces expected output."""
    signals = [
        AnalystSignal(
            name="Tech",
            direction="BUY",
            p_up=0.75,
            confidence=0.9,
            weight=1.0,
        ),
    ]

    ens = combine_signals(signals)

    assert ens.direction == "BUY"
    # Single signal: p_up should be preserved
    assert abs(ens.p_up - 0.75) < 0.1  # Should be close to original p_up


def test_combine_signals_empty():
    """Test that empty signal list returns HOLD with high disagreement."""
    signals = []

    ens = combine_signals(signals)

    assert ens.direction == "HOLD"
    assert ens.p_up == 0.5
    assert ens.disagreement == 1.0  # Max disagreement when no signals


def test_combine_signals_all_hold():
    """Test that all HOLD signals produce HOLD direction."""
    signals = [
        AnalystSignal(
            name="Tech",
            direction="HOLD",
            p_up=0.5,
            confidence=0.8,
            weight=1.0,
        ),
        AnalystSignal(
            name="News",
            direction="HOLD",
            p_up=0.52,
            confidence=0.7,
            weight=1.0,
        ),
    ]

    ens = combine_signals(signals)

    assert ens.direction == "HOLD"


def test_combine_signals_weighted_averaging():
    """Test that analyst weights affect the final decision."""
    signals = [
        AnalystSignal(
            name="Tech",
            direction="BUY",
            p_up=0.8,
            confidence=0.9,
            weight=2.0,  # Higher weight
        ),
        AnalystSignal(
            name="News",
            direction="SELL",
            p_up=0.3,
            confidence=0.6,
            weight=1.0,  # Lower weight
        ),
    ]

    ens = combine_signals(signals)

    # Tech signal has higher weight, should bias towards BUY
    # (but might be HOLD due to conflict)
    assert ens.p_up > 0.5  # Should lean towards BUY


def test_apply_disagreement_penalty_low_disagreement():
    """Test that low disagreement doesn't penalize confidence."""
    base_conf = 0.8
    disagreement = 0.3  # Below threshold
    threshold = 0.5

    adjusted = apply_disagreement_penalty(base_conf, disagreement, threshold)

    assert adjusted == base_conf  # No penalty


def test_apply_disagreement_penalty_high_disagreement():
    """Test that high disagreement reduces confidence."""
    base_conf = 0.8
    disagreement = 0.7  # Above threshold
    threshold = 0.5

    adjusted = apply_disagreement_penalty(base_conf, disagreement, threshold)

    assert adjusted < base_conf  # Penalty applied
    assert adjusted > 0  # But not zero


def test_apply_disagreement_penalty_max_disagreement():
    """Test that maximum disagreement heavily penalizes confidence."""
    base_conf = 0.9
    disagreement = 1.0  # Maximum disagreement
    threshold = 0.5

    adjusted = apply_disagreement_penalty(base_conf, disagreement, threshold)

    assert adjusted < base_conf * 0.1  # Heavy penalty


def test_get_analyst_weights_no_history():
    """Test that no performance history returns empty dict."""
    weights = get_analyst_weights(None)

    assert weights == {}


def test_get_analyst_weights_with_history():
    """Test that performance history produces valid weights."""
    perf = {"Technical": 0.65, "News": 0.52, "Macro": 0.48}

    weights = get_analyst_weights(perf)

    assert "Technical" in weights
    assert "News" in weights
    assert "Macro" in weights

    # Better performers get higher weights
    assert weights["Technical"] > weights["News"]
    assert weights["News"] > weights["Macro"]


def test_get_analyst_weights_edge_cases():
    """Test edge cases for analyst weight calculation."""
    perf = {
        "Perfect": 1.0,   # 100% win rate
        "Good": 0.6,      # 60% win rate
        "Average": 0.5,   # 50% win rate (baseline)
        "Bad": 0.4,       # 40% win rate
        "Terrible": 0.0,  # 0% win rate
    }

    weights = get_analyst_weights(perf)

    # All weights should be positive (min 0.1)
    for w in weights.values():
        assert w >= 0.1

    # Higher win rate -> higher weight
    assert weights["Perfect"] > weights["Good"]
    assert weights["Good"] > weights["Average"]
    assert weights["Average"] > weights["Bad"]


def test_analyst_signal_dataclass():
    """Test AnalystSignal dataclass construction."""
    signal = AnalystSignal(
        name="TestAnalyst",
        direction="BUY",
        p_up=0.75,
        confidence=0.85,
        weight=1.5,
    )

    assert signal.name == "TestAnalyst"
    assert signal.direction == "BUY"
    assert signal.p_up == 0.75
    assert signal.confidence == 0.85
    assert signal.weight == 1.5


def test_ensemble_decision_explanation():
    """Test that ensemble decision includes human-readable explanation."""
    signals = [
        AnalystSignal(
            name="Tech",
            direction="BUY",
            p_up=0.7,
            confidence=0.9,
            weight=1.0,
        ),
    ]

    ens = combine_signals(signals)

    assert isinstance(ens.explanation, str)
    assert len(ens.explanation) > 0
    assert "Tech" in ens.explanation
    assert "BUY" in ens.explanation
