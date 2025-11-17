#!/usr/bin/env python
"""
Example: Integrating RL agent as 4th analyst in ensemble.

Demonstrates how to:
1. Load trained RL policy
2. Build RL state from market data
3. Get RL signal
4. Combine with other analysts in ensemble
"""

from datetime import date

from otonom_trader.data import get_session
from otonom_trader.patron import (
    AnalystSignal,
    combine_signals,
    RlAnalyst,
    RlAnalystConfig,
)
from otonom_trader.research.rl_state_builder import RlStateBuilder


def example_rl_in_ensemble():
    """
    Example: Use RL agent alongside other analysts.
    """
    print("=" * 60)
    print("RL Agent Ensemble Integration Example")
    print("=" * 60)

    # 1. Create RL analyst
    print("\n1. Loading RL analyst...")
    rl_analyst = RlAnalyst(
        RlAnalystConfig(
            model_path="models/rl_bc/policy_best.pt",
            enabled=True,
            weight=1.0,
            confidence_threshold=0.4,
        )
    )

    # Print info
    info = rl_analyst.get_info()
    print(f"   Model: {info['model_path']}")
    print(f"   State dim: {info['state_dim']}")
    print(f"   Val accuracy: {info['val_accuracy']:.3f}" if info['val_accuracy'] else "   Val accuracy: N/A")

    # 2. Build RL state from current market data
    print("\n2. Building RL state from market data...")

    with get_session() as session:
        state_builder = RlStateBuilder(session)

        # Build state for current date
        state = state_builder.build_state(
            symbol="BTC-USD",
            current_date=date.today(),
            current_position=0.0,
            portfolio_equity=10000.0,
        )

        print(f"   State built: {state.to_vector().shape[0]} features")

        # 3. Get RL signal
        print("\n3. Getting RL signal...")
        rl_signal = rl_analyst.infer(state)

        print(f"   RL Signal:")
        print(f"     Direction: {rl_signal.direction}")
        print(f"     P(up): {rl_signal.p_up:.3f}")
        print(f"     Confidence: {rl_signal.confidence:.3f}")
        print(f"     Weight: {rl_signal.weight:.3f}")

    # 4. Simulate other analysts
    print("\n4. Simulating other analyst signals...")

    # Technical analyst (example)
    tech_signal = AnalystSignal(
        name="Analist-1 (Technical)",
        direction="BUY",
        p_up=0.65,
        confidence=0.75,
        weight=1.0,
    )
    print(f"   Technical: {tech_signal.direction} (conf={tech_signal.confidence:.2f})")

    # News analyst (example)
    news_signal = AnalystSignal(
        name="Analist-2 (News)",
        direction="HOLD",
        p_up=0.52,
        confidence=0.60,
        weight=1.0,
    )
    print(f"   News: {news_signal.direction} (conf={news_signal.confidence:.2f})")

    # Risk/Regime analyst (example)
    risk_signal = AnalystSignal(
        name="Analist-3 (Risk)",
        direction="BUY",
        p_up=0.58,
        confidence=0.70,
        weight=1.0,
    )
    print(f"   Risk: {risk_signal.direction} (conf={risk_signal.confidence:.2f})")

    # 5. Combine signals
    print("\n5. Combining signals in ensemble...")

    ensemble_decision = combine_signals([
        tech_signal,
        news_signal,
        risk_signal,
        rl_signal,
    ])

    print(f"\n   Ensemble Decision:")
    print(f"     Direction: {ensemble_decision.direction}")
    print(f"     P(up): {ensemble_decision.p_up:.3f}")
    print(f"     Disagreement: {ensemble_decision.disagreement:.3f}")
    print(f"     Explanation: {ensemble_decision.explanation}")

    print("\n" + "=" * 60)
    print("Integration complete!")
    print("=" * 60)


def example_rl_only():
    """
    Example: Use RL agent standalone (without other analysts).
    """
    print("=" * 60)
    print("RL Agent Standalone Example")
    print("=" * 60)

    # Create RL analyst
    print("\n1. Creating RL analyst...")
    rl_analyst = RlAnalyst(
        RlAnalystConfig(
            model_path="models/rl_bc/policy_best.pt",
            enabled=True,
            weight=1.0,
        )
    )

    # Build state
    print("\n2. Building state...")
    with get_session() as session:
        state_builder = RlStateBuilder(session)

        state = state_builder.build_state(
            symbol="BTC-USD",
            current_date=date.today(),
            current_position=0.0,
            portfolio_equity=10000.0,
        )

        # Get signal
        print("\n3. Getting signal...")
        signal = rl_analyst.infer(state)

        print(f"\nRL Agent Signal:")
        print(f"  Action: {signal.direction}")
        print(f"  P(up): {signal.p_up:.3f}")
        print(f"  Confidence: {signal.confidence:.3f}")
        print(f"  Weight: {signal.weight:.3f}")

        # Decision logic
        if signal.direction == "BUY" and signal.confidence > 0.6:
            print("\n✓ STRONG BUY signal - Execute trade")
        elif signal.direction == "SELL" and signal.confidence > 0.6:
            print("\n✓ STRONG SELL signal - Execute trade")
        elif signal.direction == "HOLD":
            print("\n○ HOLD - No action")
        else:
            print("\n⚠ Weak signal - Consider waiting")

    print("\n" + "=" * 60)


def example_multiple_symbols():
    """
    Example: Get RL signals for multiple symbols.
    """
    print("=" * 60)
    print("RL Agent Multiple Symbols Example")
    print("=" * 60)

    # Create RL analyst
    rl_analyst = RlAnalyst(
        RlAnalystConfig(
            model_path="models/rl_bc/policy_best.pt",
            enabled=True,
        )
    )

    symbols = ["BTC-USD", "ETH-USD", "AAPL", "TSLA"]

    print(f"\nGetting signals for {len(symbols)} symbols...\n")

    with get_session() as session:
        state_builder = RlStateBuilder(session)

        for symbol in symbols:
            try:
                # Build state
                state = state_builder.build_state(
                    symbol=symbol,
                    current_date=date.today(),
                    current_position=0.0,
                    portfolio_equity=10000.0,
                )

                # Get signal
                signal = rl_analyst.infer(state)

                print(f"{symbol:10s}: {signal.direction:4s} "
                      f"(p_up={signal.p_up:.3f}, conf={signal.confidence:.3f})")

            except Exception as e:
                print(f"{symbol:10s}: ERROR - {e}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    print("\nChoose example:")
    print("1. RL in ensemble (with other analysts)")
    print("2. RL standalone")
    print("3. Multiple symbols")

    choice = input("\nEnter choice (1-3): ").strip()

    if choice == "1":
        example_rl_in_ensemble()
    elif choice == "2":
        example_rl_only()
    elif choice == "3":
        example_multiple_symbols()
    else:
        print("Invalid choice")
