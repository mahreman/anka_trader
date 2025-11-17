"""
LLM Agent for Analist-2 (P2).

This module provides LLM-based fundamental analysis.
Supports multiple LLM backends: DeepSeek, Gemma, Qwen, OpenAI, etc.

P2 Features:
- LLM prompt construction from news + calendar + anomaly context
- Direction hint generation (BUY/SELL/HOLD)
- Probability estimation (p_up)
- Confidence scoring

Input: news items, macro events, anomaly context
Output: direction_hint, p_up, confidence, reasoning
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from datetime import date
from typing import List, Literal, Optional

logger = logging.getLogger(__name__)


Direction = Literal["BUY", "SELL", "HOLD"]


@dataclass
class LLMSignal:
    """
    LLM-generated trading signal.

    Attributes:
        direction: Trading direction hint
        p_up: Probability price will go up (0-1)
        confidence: Model confidence (0-1)
        reasoning: Human-readable explanation
    """
    direction: Direction
    p_up: float
    confidence: float
    reasoning: str


def _build_prompt(
    symbol: str,
    anomaly_context: str,
    news_summary: str,
    macro_summary: str,
    regime_context: str = "",
) -> str:
    """
    Build LLM prompt for fundamental analysis.

    Args:
        symbol: Asset symbol
        anomaly_context: Recent anomaly description
        news_summary: News summary (sentiment, headlines)
        macro_summary: Macro events summary (bias, key events)
        regime_context: Optional regime/DSI context

    Returns:
        Formatted prompt string

    Example:
        >>> prompt = _build_prompt(
        ...     symbol="BTC-USD",
        ...     anomaly_context="SPIKE_DOWN detected on 2025-01-15, zscore=-4.5",
        ...     news_summary="3-day sentiment: -0.3 (bearish). Key: 'SEC delays ETF'",
        ...     macro_summary="7-day macro bias: +0.2 (slightly bullish). Key: CPI beat.",
        ... )
    """
    prompt = f"""You are a trading analyst. Analyze the following context and provide a trading recommendation.

**Symbol:** {symbol}

**Recent Anomaly:**
{anomaly_context}

**News Summary (last 3 days):**
{news_summary}

**Macro Summary (last 7 days):**
{macro_summary}

**Regime Context:**
{regime_context if regime_context else "No regime data available"}

**Task:**
Based on the above context, provide:
1. **Direction:** BUY, SELL, or HOLD
2. **P_UP:** Probability that price will go up (0.0 to 1.0)
3. **Confidence:** Your confidence in this recommendation (0.0 to 1.0)
4. **Reasoning:** Brief explanation (2-3 sentences)

**Format your response EXACTLY as follows:**
DIRECTION: <BUY/SELL/HOLD>
P_UP: <0.0-1.0>
CONFIDENCE: <0.0-1.0>
REASONING: <your explanation>

**Example:**
DIRECTION: BUY
P_UP: 0.65
CONFIDENCE: 0.7
REASONING: Despite the recent crash, positive macro backdrop and oversold conditions suggest a mean reversion opportunity. News sentiment is improving.

Now provide your analysis:
"""
    return prompt


def _parse_llm_response(response: str) -> Optional[LLMSignal]:
    """
    Parse LLM response into structured LLMSignal.

    Args:
        response: Raw LLM response text

    Returns:
        LLMSignal object, or None if parsing fails

    Example:
        >>> response = '''
        ... DIRECTION: BUY
        ... P_UP: 0.65
        ... CONFIDENCE: 0.7
        ... REASONING: Good setup for mean reversion.
        ... '''
        >>> signal = _parse_llm_response(response)
        >>> signal.direction
        'BUY'
    """
    try:
        # Extract fields using regex
        direction_match = re.search(r"DIRECTION:\s*(BUY|SELL|HOLD)", response, re.IGNORECASE)
        p_up_match = re.search(r"P_UP:\s*(0?\.\d+|1\.0|0|1)", response, re.IGNORECASE)
        confidence_match = re.search(r"CONFIDENCE:\s*(0?\.\d+|1\.0|0|1)", response, re.IGNORECASE)
        reasoning_match = re.search(r"REASONING:\s*(.+?)(?:\n\n|\Z)", response, re.IGNORECASE | re.DOTALL)

        if not all([direction_match, p_up_match, confidence_match, reasoning_match]):
            logger.warning("Failed to parse LLM response - missing required fields")
            return None

        direction = direction_match.group(1).upper()
        p_up = float(p_up_match.group(1))
        confidence = float(confidence_match.group(1))
        reasoning = reasoning_match.group(1).strip()

        # Validate
        if direction not in ["BUY", "SELL", "HOLD"]:
            logger.warning(f"Invalid direction: {direction}")
            return None

        if not (0.0 <= p_up <= 1.0):
            logger.warning(f"Invalid p_up: {p_up}")
            return None

        if not (0.0 <= confidence <= 1.0):
            logger.warning(f"Invalid confidence: {confidence}")
            return None

        return LLMSignal(
            direction=direction,  # type: ignore
            p_up=p_up,
            confidence=confidence,
            reasoning=reasoning,
        )

    except Exception as e:
        logger.error(f"Error parsing LLM response: {e}")
        return None


def _call_openai(
    prompt: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.3,
) -> Optional[str]:
    """
    Call OpenAI API.

    Args:
        prompt: Prompt text
        model: Model name (default: gpt-4o-mini)
        temperature: Sampling temperature

    Returns:
        LLM response text, or None on error
    """
    try:
        import openai

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set, skipping OpenAI call")
            return None

        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )

        return response.choices[0].message.content

    except ImportError:
        logger.warning("openai package not installed")
        return None
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return None


def _call_deepseek(
    prompt: str,
    model: str = "deepseek-chat",
    temperature: float = 0.3,
) -> Optional[str]:
    """
    Call DeepSeek API (OpenAI-compatible).

    Args:
        prompt: Prompt text
        model: Model name (default: deepseek-chat)
        temperature: Sampling temperature

    Returns:
        LLM response text, or None on error
    """
    try:
        import openai

        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            logger.warning("DEEPSEEK_API_KEY not set, skipping DeepSeek call")
            return None

        client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
        )
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )

        return response.choices[0].message.content

    except ImportError:
        logger.warning("openai package not installed")
        return None
    except Exception as e:
        logger.error(f"DeepSeek API error: {e}")
        return None


def _call_local_llm(
    prompt: str,
    model: str = "gemma2:2b",
    temperature: float = 0.3,
) -> Optional[str]:
    """
    Call local LLM via Ollama.

    Supports Gemma, Qwen, LLaMA, etc.

    Args:
        prompt: Prompt text
        model: Model name (default: gemma2:2b)
        temperature: Sampling temperature

    Returns:
        LLM response text, or None on error
    """
    try:
        import requests

        # Ollama default endpoint
        url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")

        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "stream": False,
        }

        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()

        data = response.json()
        return data.get("response", None)

    except ImportError:
        logger.warning("requests package not installed")
        return None
    except Exception as e:
        logger.error(f"Local LLM error: {e}")
        return None


def get_llm_signal(
    symbol: str,
    anomaly_context: str,
    news_summary: str,
    macro_summary: str,
    regime_context: str = "",
    backend: str = "openai",
    model: Optional[str] = None,
) -> Optional[LLMSignal]:
    """
    Get LLM trading signal from context.

    Args:
        symbol: Asset symbol
        anomaly_context: Recent anomaly description
        news_summary: News summary
        macro_summary: Macro events summary
        regime_context: Optional regime/DSI context
        backend: LLM backend ("openai", "deepseek", "local")
        model: Optional model override

    Returns:
        LLMSignal object, or None on error

    Example:
        >>> signal = get_llm_signal(
        ...     symbol="BTC-USD",
        ...     anomaly_context="SPIKE_DOWN on 2025-01-15, zscore=-4.5",
        ...     news_summary="Bearish sentiment: -0.3",
        ...     macro_summary="Macro bias: +0.2",
        ...     backend="openai",
        ... )
        >>> if signal:
        ...     print(f"{signal.direction} (p_up={signal.p_up:.2f})")
    """
    logger.info(f"Calling LLM backend: {backend}")

    # Build prompt
    prompt = _build_prompt(
        symbol=symbol,
        anomaly_context=anomaly_context,
        news_summary=news_summary,
        macro_summary=macro_summary,
        regime_context=regime_context,
    )

    # Call LLM backend
    response = None

    if backend == "openai":
        response = _call_openai(prompt, model=model or "gpt-4o-mini")
    elif backend == "deepseek":
        response = _call_deepseek(prompt, model=model or "deepseek-chat")
    elif backend == "local":
        response = _call_local_llm(prompt, model=model or "gemma2:2b")
    else:
        logger.error(f"Unknown backend: {backend}")
        return None

    if not response:
        logger.warning("No response from LLM")
        return None

    # Parse response
    signal = _parse_llm_response(response)

    if signal:
        logger.info(f"LLM signal: {signal.direction} (p_up={signal.p_up:.2f}, conf={signal.confidence:.2f})")
    else:
        logger.warning("Failed to parse LLM response")

    return signal


def get_llm_signal_with_fallback(
    symbol: str,
    anomaly_context: str,
    news_summary: str,
    macro_summary: str,
    regime_context: str = "",
) -> LLMSignal:
    """
    Get LLM signal with fallback to simple heuristic.

    Tries backends in order: deepseek -> openai -> local -> heuristic

    Args:
        symbol: Asset symbol
        anomaly_context: Recent anomaly description
        news_summary: News summary
        macro_summary: Macro events summary
        regime_context: Optional regime/DSI context

    Returns:
        LLMSignal object (always returns something, even if heuristic)

    Example:
        >>> signal = get_llm_signal_with_fallback(
        ...     symbol="BTC-USD",
        ...     anomaly_context="SPIKE_DOWN",
        ...     news_summary="Bearish",
        ...     macro_summary="Bullish",
        ... )
        >>> signal.direction
        'HOLD'
    """
    # Try DeepSeek first (cheap and fast)
    signal = get_llm_signal(
        symbol, anomaly_context, news_summary, macro_summary, regime_context, backend="deepseek"
    )
    if signal:
        return signal

    # Try OpenAI
    signal = get_llm_signal(
        symbol, anomaly_context, news_summary, macro_summary, regime_context, backend="openai"
    )
    if signal:
        return signal

    # Try local LLM
    signal = get_llm_signal(
        symbol, anomaly_context, news_summary, macro_summary, regime_context, backend="local"
    )
    if signal:
        return signal

    # Fallback: simple heuristic
    logger.warning("All LLM backends failed, using simple heuristic")

    # Extract sentiment from summaries (very basic)
    news_bearish = "bearish" in news_summary.lower() or "negative" in news_summary.lower()
    news_bullish = "bullish" in news_summary.lower() or "positive" in news_summary.lower()
    macro_bearish = "bearish" in macro_summary.lower() or "negative" in macro_summary.lower()
    macro_bullish = "bullish" in macro_summary.lower() or "positive" in macro_summary.lower()

    score = 0
    if news_bullish:
        score += 1
    if news_bearish:
        score -= 1
    if macro_bullish:
        score += 1
    if macro_bearish:
        score -= 1

    if score > 0:
        direction: Direction = "BUY"
        p_up = 0.6
    elif score < 0:
        direction = "SELL"
        p_up = 0.4
    else:
        direction = "HOLD"
        p_up = 0.5

    return LLMSignal(
        direction=direction,
        p_up=p_up,
        confidence=0.3,  # Low confidence for heuristic
        reasoning="LLM unavailable - using simple sentiment heuristic",
    )
