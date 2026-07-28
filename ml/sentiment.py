"""Sentiment Analysis — léxico financiero mejorado + FinBERT opcional.

Reemplaza VADER (léxico genérico de redes sociales) con un léxico financiero
curado que entiende términos como "missed earnings", "guidance cut", "downgrade".

Si `transformers` está instalado y el modelo FinBERT está disponible, lo usa
para mayor precisión. Si no, usa el léxico financiero como fallback sin dependencias.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger("inversion_helper.ml.sentiment")

# ── Léxico financiero curado ───────────────────────────────────────
# Score: +1 (bullish), -1 (bearish), ponderado por contexto

BULLISH_WORDS = {
    # Resultados
    "beat": 1.2,
    "beats": 1.2,
    "surpass": 1.0,
    "exceed": 1.0,
    "exceeds": 1.0,
    "outperform": 1.3,
    "outperforms": 1.3,
    "top": 0.8,
    # Guías
    "raise": 0.9,
    "raises": 0.9,
    "raised": 0.9,
    "upgrade": 1.1,
    "upgrades": 1.1,
    "upgraded": 1.1,
    "raise guidance": 1.5,
    "raised guidance": 1.5,
    "strong guidance": 1.2,
    # Crecimiento
    "growth": 0.7,
    "grow": 0.6,
    "growing": 0.7,
    "surge": 1.3,
    "surges": 1.3,
    "soar": 1.2,
    "soars": 1.2,
    "jump": 0.8,
    "jumps": 0.8,
    "rally": 1.0,
    "rallies": 1.0,
    "gain": 0.7,
    "gains": 0.7,
    # Positivos generales
    "bullish": 1.5,
    "optimistic": 0.8,
    "record": 0.9,
    "high": 0.5,
    "strong": 0.8,
    "robust": 0.8,
    "profit": 0.7,
    "profitable": 0.9,
    "dividend": 0.5,
    "buyback": 0.8,
    "buybacks": 0.8,
    "acquire": 0.5,
    "acquisition": 0.5,
    "innovation": 0.5,
    "breakthrough": 1.0,
    # Analistas
    "overweight": 1.0,
    "buy": 0.8,
    "strong buy": 1.5,
    "accumulate": 0.6,
    "target raise": 1.2,
    "price target raise": 1.3,
}

BEARISH_WORDS = {
    # Resultados
    "miss": -1.2,
    "misses": -1.2,
    "missed": -1.2,
    "fall short": -1.3,
    "falls short": -1.3,
    "disappoint": -1.0,
    "disappoints": -1.0,
    "disappointing": -1.1,
    # Guías
    "cut": -0.8,
    "cuts": -0.8,
    "lower": -0.6,
    "lowers": -0.6,
    "downgrade": -1.2,
    "downgrades": -1.2,
    "downgraded": -1.2,
    "cut guidance": -1.5,
    "lower guidance": -1.5,
    "weak guidance": -1.2,
    "withdraw guidance": -1.8,
    # Caídas
    "decline": -0.7,
    "declines": -0.7,
    "drop": -0.8,
    "drops": -0.8,
    "plunge": -1.3,
    "plunges": -1.3,
    "tumble": -1.2,
    "tumbles": -1.2,
    "slide": -0.7,
    "slides": -0.7,
    "slump": -0.9,
    "slumps": -0.9,
    "fall": -0.6,
    "falls": -0.6,
    # Negativos generales
    "bearish": -1.5,
    "pessimistic": -0.8,
    "loss": -0.7,
    "losses": -0.8,
    "weak": -0.7,
    "feeble": -0.7,
    "concern": -0.6,
    "concerns": -0.7,
    "worried": -0.6,
    "risk": -0.4,
    "risks": -0.5,
    "lawsuit": -1.0,
    "investigation": -1.1,
    "probe": -0.9,
    "fraud": -1.5,
    "sec charges": -1.8,
    # Analistas
    "underweight": -1.0,
    "sell": -0.8,
    "strong sell": -1.5,
    "reduce": -0.7,
    "target cut": -1.2,
    "price target cut": -1.3,
    # Macro
    "recession": -1.0,
    "inflation": -0.5,
    "rate hike": -0.6,
    "fed tightening": -0.7,
}

# Intensificadores que amplifican la palabra siguiente
INTENSIFIERS = {"very": 1.5, "extremely": 2.0, "highly": 1.5, "significantly": 1.4, "sharply": 1.4, "strongly": 1.3}

# Negaciones que invierten la palabra siguiente
NEGATIONS = {"not", "no", "never", "neither", "nor", "without", "hardly", "barely"}


class FinancialLexiconAnalyzer:
    """Analizador de sentiment con léxico financiero curado."""

    def __init__(self):
        self.bullish = BULLISH_WORDS
        self.bearish = BEARISH_WORDS
        self.intensifiers = INTENSIFIERS
        self.negations = NEGATIONS

    def analyze(self, text: str) -> dict[str, Any]:
        """Analiza texto y retorna scores de sentiment financiero."""
        if not text or not text.strip():
            return {"compound": 0.0, "pos": 0.0, "neu": 1.0, "neg": 0.0, "label": "NEUTRAL"}

        # Normalizar texto
        text_lower = text.lower()
        # Tokenizar palabras individuales
        words = re.findall(r"\b[a-z]+\b", text_lower)

        total_score = 0.0
        matches = 0
        pos_scores: list[float] = []
        neg_scores: list[float] = []

        # Buscar bigramas primero en el texto completo
        found_spans: list[tuple[int, int]] = []  # (start_word_idx, end_word_idx)

        # Primera pasada: bigramas
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i + 1]}"
            if bigram in self.bullish:
                score = self.bullish[bigram]
                # Verificar intensificador/negación antes
                if i > 0:
                    if words[i - 1] in self.intensifiers:
                        score *= self.intensifiers[words[i - 1]]
                    if words[i - 1] in self.negations:
                        score = -score * 0.7
                total_score += score
                matches += 1
                if score > 0:
                    pos_scores.append(score)
                elif score < 0:
                    neg_scores.append(abs(score))
                found_spans.append((i, i + 2))
            elif bigram in self.bearish:
                score = self.bearish[bigram]
                if i > 0:
                    if words[i - 1] in self.intensifiers:
                        score *= self.intensifiers[words[i - 1]]
                    if words[i - 1] in self.negations:
                        score = -score * 0.7
                total_score += score
                matches += 1
                if score > 0:
                    pos_scores.append(score)
                elif score < 0:
                    neg_scores.append(abs(score))
                found_spans.append((i, i + 2))

        # Segunda pasada: unigrams (saltando palabras ya consumidas por bigramas)
        consumed = set()
        for start, end in found_spans:
            for j in range(start, end):
                consumed.add(j)

        for i, word in enumerate(words):
            if i in consumed:
                continue
            if i in consumed:
                continue

            multiplier = 1.0
            negate = False
            if i > 0 and words[i - 1] in self.intensifiers:
                multiplier *= self.intensifiers[words[i - 1]]
            if i > 0 and words[i - 1] in self.negations:
                negate = True

            score = 0.0
            matched = False
            if word in self.bullish:
                score = self.bullish[word] * multiplier
                matched = True
            elif word in self.bearish:
                score = self.bearish[word] * multiplier
                matched = True

            if matched:
                if negate:
                    score = -score * 0.7
                total_score += score
                matches += 1
                if score > 0:
                    pos_scores.append(score)
                elif score < 0:
                    neg_scores.append(abs(score))

        if matches == 0:
            return {"compound": 0.0, "pos": 0.0, "neu": 1.0, "neg": 0.0, "label": "NEUTRAL"}

        # Normalizar compound a [-1, 1] con tanh
        compound = float(np_tanh(total_score / max(matches, 1)))

        pos_sum = sum(pos_scores)
        neg_sum = sum(neg_scores)
        total_abs = pos_sum + neg_sum + 1e-8

        pos_ratio = pos_sum / total_abs
        neg_ratio = neg_sum / total_abs
        neu_ratio = 1.0 - pos_ratio - neg_ratio

        if compound >= 0.05:
            label = "ALCISTA"
        elif compound <= -0.05:
            label = "BAJISTA"
        else:
            label = "NEUTRAL"

        return {
            "compound": round(compound, 4),
            "pos": round(pos_ratio, 4),
            "neu": round(neu_ratio, 4),
            "neg": round(neg_ratio, 4),
            "label": label,
            "matches": matches,
        }


def np_tanh(x: float) -> float:
    import math

    return math.tanh(x)


# ── FinBERT opcional (lazy load) ───────────────────────────────────

_FINBERT_MODEL = None
_FINBERT_AVAILABLE = None


def _try_load_finbert():
    """Intenta cargar FinBERT si transformers está instalado."""
    global _FINBERT_MODEL, _FINBERT_AVAILABLE
    if _FINBERT_AVAILABLE is not None:
        return _FINBERT_AVAILABLE

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

        model_name = "ProsusAI/finBERT"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        _FINBERT_MODEL = pipeline("text-classification", model=model, tokenizer=tokenizer)
        _FINBERT_AVAILABLE = True
        logger.info("FinBERT loaded successfully")
        return True
    except Exception as exc:
        _FINBERT_AVAILABLE = False
        logger.info("FinBERT not available, using financial lexicon: %s", str(exc)[:80])
        return False


def _finbert_analyze(text: str) -> dict[str, Any] | None:
    """Analiza con FinBERT si está disponible."""
    if not _try_load_finbert():
        return None

    try:
        result = _FINBERT_MODEL(text[:512])[0]  # truncate to model max
        label = result["label"].lower()
        score = float(result["score"])

        # FinBERT labels: positive, negative, neutral
        if label == "positive":
            return {"compound": score, "pos": score, "neu": 0.0, "neg": 0.0, "label": "ALCISTA"}
        elif label == "negative":
            return {"compound": -score, "pos": 0.0, "neu": 0.0, "neg": score, "label": "BAJISTA"}
        else:
            return {"compound": 0.0, "pos": 0.0, "neu": score, "neg": 0.0, "label": "NEUTRAL"}
    except Exception:
        return None


# ── API pública (compatible con SentimentAnalyzer viejo) ──────────


class SentimentAnalyzer:
    """Analizador de sentiment financiero.

    Usa FinBERT si está disponible (requiere `pip install transformers`).
    Fallback a léxico financiero curado (sin dependencias).
    """

    def __init__(self):
        self.lexicon = FinancialLexiconAnalyzer()

    def analyze(self, text: str) -> dict:
        """Returns sentiment scores for a given text."""
        if not text:
            return {"compound": 0.0, "pos": 0.0, "neu": 1.0, "neg": 0.0, "label": "NEUTRAL"}

        # Intentar FinBERT primero
        finbert_result = _finbert_analyze(text)
        if finbert_result is not None:
            return finbert_result

        # Fallback: léxico financiero
        return self.lexicon.analyze(text)

    def analyze_news_batch(self, news_list: list) -> dict:
        """Analyzes a list of news items and adds sentiment data."""
        if not news_list:
            return {"news": [], "average_sentiment": 0.0, "global_label": "NEUTRAL"}

        total_compound = 0.0
        processed_news = []

        for item in news_list:
            headline = item.get("headline") or item.get("title") or ""
            summary = item.get("summary", "")
            text_to_analyze = f"{headline} {summary}".strip()

            sentiment = self.analyze(text_to_analyze)

            processed_item = item.copy()
            processed_item["title"] = headline
            processed_item["publisher"] = item.get("source") or item.get("publisher") or ""
            processed_item["link"] = item.get("url") or item.get("link") or ""
            processed_item["time"] = item.get("created_at") or item.get("time") or ""
            processed_item["sentiment_score"] = sentiment["compound"]
            processed_item["sentiment_label"] = sentiment["label"]
            processed_news.append(processed_item)

            total_compound += sentiment["compound"]

        avg = total_compound / len(news_list)

        if avg >= 0.05:
            global_label = "ALCISTA"
        elif avg <= -0.05:
            global_label = "BAJISTA"
        else:
            global_label = "NEUTRAL"

        return {"news": processed_news, "average_sentiment": avg, "global_label": global_label}

    def fetch_ticker_news_sentiment(self, ticker: str) -> dict:
        """Escanea los titulares más recientes del ticker vía yfinance y retorna el sentimiento promedio.

        Permite actuar como Sentinel para bloquear compras ante noticias altamente negativas (< -0.4).
        """
        try:
            import yfinance as yf

            tk = yf.Ticker(ticker)
            news = tk.news or []
            if not news:
                return {"average_sentiment": 0.0, "news_count": 0, "should_block_buy": False, "status": "NO_NEWS"}

            res = self.analyze_news_batch(news[:5])
            avg = res.get("average_sentiment", 0.0)

            return {
                "average_sentiment": float(avg),
                "news_count": len(news[:5]),
                "should_block_buy": float(avg) < -0.4,
                "status": "OK",
            }
        except Exception as e:
            logger.warning("Error buscando noticias para %s: %s", ticker, e)
            return {"average_sentiment": 0.0, "news_count": 0, "should_block_buy": False, "status": "ERROR"}
