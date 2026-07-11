from datetime import datetime

import yfinance as yf


class NewsFetcher:
    """Fetches latest news from Yahoo Finance for a given ticker."""

    @staticmethod
    def get_latest_news(ticker: str, limit: int = 10) -> list:
        try:
            stock = yf.Ticker(ticker)
            news = stock.news

            if not news:
                return []

            formatted_news = []
            for item in news[:limit]:
                # yfinance returns different formats depending on the version.
                # Handle nested 'content' format if present
                content = item.get('content', item)

                title = content.get('title', item.get('title', 'Sin título'))
                link = item.get('link', content.get('canonicalUrl', {}).get('url', '#'))

                provider = item.get('provider', content.get('provider', {}))
                publisher = provider.get('displayName', item.get('publisher', 'Yahoo Finance')) if isinstance(provider, dict) else item.get('publisher', 'Yahoo Finance')

                pub_time = content.get('pubDate', item.get('providerPublishTime', 0))

                if isinstance(pub_time, (int, float)) and pub_time > 0:
                    try:
                        date_obj = datetime.fromtimestamp(pub_time)
                        time_str = date_obj.strftime('%Y-%m-%d %H:%M')
                    except Exception:
                        time_str = "Reciente"
                elif isinstance(pub_time, str):
                    time_str = pub_time[:10] + " " + pub_time[11:16] # Extract YYYY-MM-DD HH:MM
                else:
                    time_str = "Reciente"

                formatted_news.append({
                    "title": title,
                    "publisher": publisher,
                    "link": link,
                    "time": time_str
                })

            return formatted_news

        except Exception as e:
            print(f"Error fetching news for {ticker}: {e}")
            return []
