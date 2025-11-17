#!/usr/bin/env python
"""
Example: Multi-Analyst Ensemble Decision Making

This script demonstrates how to use all 3 analysts together:
- Analist-1 (Regime): Technical + market regime + data quality
- Analist-2 (News): News sentiment analysis via LLM
- Analist-3 (Macro): Macroeconomic risk assessment

The ensemble combines their signals with confidence weighting
and disagreement tracking.

Usage:
    python scripts/example_multi_analyst_ensemble.py --symbol BTCUSDT
"""
import argparse
import sys
from datetime import date, datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from otonom_trader.data import get_session
from otonom_trader.patron import (
    build_regime_analyst_signal,
    build_news_analyst_signal_from_db,
    build_macro_risk_signal,
    combine_signals,
    get_regime_summary,
    get_recent_news_summary,
    get_macro_summary,
    format_macro_summary,
)


def run_multi_analyst_ensemble(
    symbol: str,
    analysis_date: date = None,
) -> None:
    """
    Run multi-analyst ensemble decision process.

    Args:
        symbol: Trading symbol (e.g., "BTCUSDT")
        analysis_date: Analysis date (default: today)
    """
    if analysis_date is None:
        analysis_date = date.today()

    print("=" * 80)
    print(f"MULTI-ANALYST ENSEMBLE DECISION")
    print("=" * 80)
    print(f"Symbol: {symbol}")
    print(f"Date: {analysis_date}")
    print("=" * 80)
    print()

    with next(get_session()) as session:
        # ========================================
        # Analist-1: Regime & Technical
        # ========================================
        print("📊 ANALIST-1: REGIME & TECHNICAL ANALYSIS")
        print("-" * 80)

        regime_signal = build_regime_analyst_signal(session, symbol, analysis_date)
        regime_summary = get_regime_summary(session, symbol, analysis_date)

        print(f"Signal: {regime_signal.direction}")
        print(f"P(up): {regime_signal.p_up:.2f}")
        print(f"Confidence: {regime_signal.confidence:.2f}")
        print()
        if regime_summary:
            print(regime_summary)
        else:
            print("No regime data available")
        print()

        # ========================================
        # Analist-2: News Sentiment
        # ========================================
        print("📰 ANALIST-2: NEWS SENTIMENT ANALYSIS")
        print("-" * 80)

        # Use datetime for news (last 24 hours)
        analysis_datetime = datetime.combine(analysis_date, datetime.min.time())
        news_signal = build_news_analyst_signal_from_db(
            session, symbol, analysis_datetime, window_hours=24
        )

        print(f"Signal: {news_signal.direction}")
        print(f"P(up): {news_signal.p_up:.2f}")
        print(f"Confidence: {news_signal.confidence:.2f}")
        print()

        news_summary = get_recent_news_summary(
            session, symbol, analysis_datetime, window_hours=24, max_articles=5
        )
        print(news_summary)
        print()

        # ========================================
        # Analist-3: Macro Risk
        # ========================================
        print("🌍 ANALIST-3: MACROECONOMIC RISK ASSESSMENT")
        print("-" * 80)

        macro_signal = build_macro_risk_signal(session, symbol, analysis_date)

        print(f"Signal: {macro_signal.direction}")
        print(f"P(up): {macro_signal.p_up:.2f}")
        print(f"Confidence: {macro_signal.confidence:.2f}")
        print()

        macro_summary = get_macro_summary(session, analysis_date)
        print(format_macro_summary(macro_summary))
        print()

        # ========================================
        # Ensemble Decision
        # ========================================
        print("=" * 80)
        print("🎯 ENSEMBLE DECISION")
        print("=" * 80)

        # Combine all signals
        all_signals = [regime_signal, news_signal, macro_signal]
        ensemble_decision = combine_signals(all_signals)

        print(f"Final Direction: {ensemble_decision.direction}")
        print(f"Ensemble P(up): {ensemble_decision.p_up:.2f}")
        print(f"Disagreement: {ensemble_decision.disagreement:.2f}")
        print()

        print("Individual Analyst Votes:")
        for signal in all_signals:
            print(f"  - {signal.name}: {signal.direction} "
                  f"(p_up={signal.p_up:.2f}, conf={signal.confidence:.2f})")
        print()

        print("Interpretation:")
        if ensemble_decision.disagreement > 0.5:
            print("  ⚠️  HIGH DISAGREEMENT: Analysts disagree significantly.")
            print("      Consider waiting for clearer consensus.")
        elif ensemble_decision.disagreement > 0.3:
            print("  ⚡ MODERATE DISAGREEMENT: Mixed signals.")
            print("      Proceed with caution.")
        else:
            print("  ✅ CONSENSUS: Analysts largely agree.")
            print(f"      Recommended action: {ensemble_decision.direction}")
        print()

        # Risk assessment
        avg_confidence = sum(s.confidence for s in all_signals) / len(all_signals)
        print(f"Average Confidence: {avg_confidence:.2f}")

        if avg_confidence < 0.4:
            print("  🔴 LOW CONFIDENCE: Data quality or signal strength is weak.")
        elif avg_confidence < 0.6:
            print("  🟡 MEDIUM CONFIDENCE: Reasonable signal quality.")
        else:
            print("  🟢 HIGH CONFIDENCE: Strong and reliable signals.")
        print()

        print("=" * 80)
        print(f"FINAL RECOMMENDATION: {ensemble_decision.direction}")
        print("=" * 80)


def main():
    """
    Main entry point for example script.
    """
    parser = argparse.ArgumentParser(
        description="Multi-analyst ensemble decision example"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTCUSDT",
        help="Trading symbol (default: BTCUSDT)",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Analysis date (YYYY-MM-DD, default: today)",
    )

    args = parser.parse_args()

    # Parse date
    if args.date:
        analysis_date = date.fromisoformat(args.date)
    else:
        analysis_date = date.today()

    # Run ensemble
    run_multi_analyst_ensemble(args.symbol, analysis_date)


if __name__ == "__main__":
    main()
