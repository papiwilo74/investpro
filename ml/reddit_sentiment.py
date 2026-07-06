"""
Análisis de Sentimiento Social desde Reddit (r/WallStreetBets y otros).

Escanea los posts más recientes sobre un ticker y calcula un score
de sentimiento social usando VADER NLP.
"""
from __future__ import annotations

import re
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class RedditSentimentAnalyzer:
    """Scraper liviano de Reddit para sentimiento de mercado."""

    SUBREDDITS = ["wallstreetbets", "stocks", "investing"]
    BASE_URL = "https://old.reddit.com/r/{sub}/search.json"

    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
        }

    @staticmethod
    def _clean_text(text: str) -> str:
        """Elimina emojis, URLs y caracteres especiales."""
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"[^\w\s.,!?'-]", "", text)
        return text.strip()

    def fetch_posts(self, ticker: str, limit: int = 10) -> list[dict]:
        """Busca posts en Reddit sobre un ticker."""
        all_posts = []

        for sub in self.SUBREDDITS:
            try:
                # Intentar búsqueda directa
                url = self.BASE_URL.format(sub=sub)
                params = {
                    "q": ticker,
                    "restrict_sr": "on",
                    "sort": "new",
                    "t": "week",
                    "limit": limit,
                    "type": "link",
                }
                resp = requests.get(
                    url, headers=self.headers, params=params, timeout=8
                )

                # Si falla la búsqueda, intentar con el endpoint de hot posts
                if resp.status_code != 200:
                    hot_url = f"https://old.reddit.com/r/{sub}/hot.json?limit=50"
                    resp = requests.get(hot_url, headers=self.headers, timeout=8)

                if resp.status_code != 200:
                    continue

                data = resp.json()
                children = data.get("data", {}).get("children", [])

                for child in children:
                    post = child.get("data", {})
                    title = post.get("title", "")
                    selftext = post.get("selftext", "")[:500]
                    score = post.get("score", 0)
                    num_comments = post.get("num_comments", 0)

                    # Filtrar: solo posts que mencionen el ticker
                    combined = f"{title} {selftext}".upper()
                    if ticker.upper() not in combined:
                        continue

                    all_posts.append({
                        "subreddit": sub,
                        "title": title,
                        "body": selftext,
                        "upvotes": score,
                        "comments": num_comments,
                    })
            except Exception as e:
                print(f"[Reddit] Error scraping r/{sub} para {ticker}: {e}")
                continue

        return all_posts

    def analyze_ticker(self, ticker: str, limit: int = 10) -> dict:
        """
        Analiza el sentimiento social de un ticker.

        Retorna:
            {
                "ticker": str,
                "posts_analyzed": int,
                "avg_sentiment": float,      # -1.0 a +1.0
                "label": str,                # EUFORIA | ALCISTA | NEUTRAL | BAJISTA | PANICO
                "top_posts": list[dict],
                "hype_score": float,          # 0.0 a 1.0 (basado en upvotes + comentarios)
            }
        """
        posts = self.fetch_posts(ticker, limit=limit)

        if not posts:
            return {
                "ticker": ticker,
                "posts_analyzed": 0,
                "avg_sentiment": 0.0,
                "label": "NEUTRAL",
                "top_posts": [],
                "hype_score": 0.0,
            }

        total_compound = 0.0
        total_engagement = 0
        analyzed = []

        for post in posts:
            text = self._clean_text(f"{post['title']} {post['body']}")
            scores = self.analyzer.polarity_scores(text)
            compound = scores["compound"]

            engagement = post["upvotes"] + post["comments"]
            total_engagement += engagement
            total_compound += compound

            analyzed.append({
                **post,
                "sentiment": compound,
                "engagement": engagement,
            })

        avg = total_compound / len(posts)

        # Clasificar con más granularidad para redes sociales
        if avg >= 0.35:
            label = "EUFORIA"     # 🚀🚀🚀 "TO THE MOON"
        elif avg >= 0.10:
            label = "ALCISTA"
        elif avg >= -0.10:
            label = "NEUTRAL"
        elif avg >= -0.35:
            label = "BAJISTA"
        else:
            label = "PANICO"      # 💀 FUD masivo

        # Hype score: cuánto engagement hay (normalizado)
        max_possible = limit * 1000  # rough normalizer
        hype_score = min(1.0, total_engagement / max_possible) if max_possible > 0 else 0.0

        # Ordenar por engagement
        analyzed.sort(key=lambda x: x["engagement"], reverse=True)

        return {
            "ticker": ticker,
            "posts_analyzed": len(posts),
            "avg_sentiment": round(avg, 4),
            "label": label,
            "top_posts": analyzed[:5],
            "hype_score": round(hype_score, 4),
        }
