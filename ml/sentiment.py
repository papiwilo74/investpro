from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class SentimentAnalyzer:
    """Analyzes text sentiment using VADER NLP."""

    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def analyze(self, text: str) -> dict:
        """
        Returns sentiment scores for a given text.
        compound score: -1.0 (most negative) to +1.0 (most positive)
        """
        if not text:
            return {"compound": 0.0, "pos": 0.0, "neu": 1.0, "neg": 0.0, "label": "NEUTRAL"}

        scores = self.analyzer.polarity_scores(text)
        compound = scores['compound']

        # Categorize
        if compound >= 0.05:
            label = "ALCISTA"
        elif compound <= -0.05:
            label = "BAJISTA"
        else:
            label = "NEUTRAL"

        scores['label'] = label
        return scores

    def analyze_news_batch(self, news_list: list) -> dict:
        """Analyzes a list of news items and adds sentiment data."""
        if not news_list:
            return {"news": [], "average_sentiment": 0.0, "global_label": "NEUTRAL"}

        total_compound = 0.0
        processed_news = []

        for item in news_list:
            headline = item.get('headline') or item.get('title') or ''
            summary = item.get('summary', '')
            text_to_analyze = f"{headline} {summary}".strip()

            sentiment = self.analyze(text_to_analyze)

            processed_item = item.copy()
            processed_item['sentiment_score'] = sentiment['compound']
            processed_item['sentiment_label'] = sentiment['label']
            processed_news.append(processed_item)

            total_compound += sentiment['compound']

        avg = total_compound / len(news_list)

        if avg >= 0.05:
            global_label = "ALCISTA"
        elif avg <= -0.05:
            global_label = "BAJISTA"
        else:
            global_label = "NEUTRAL"

        return {
            "news": processed_news,
            "average_sentiment": avg,
            "global_label": global_label
        }
