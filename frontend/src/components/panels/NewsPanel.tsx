'use client';

import { useEffect, useState } from 'react';
import { useAppStore } from '@/store/appStore';
import { useApi } from '@/hooks/useApi';
import { Components } from '@/components/ui/Components';

export function NewsPanel() {
  const { ticker } = useAppStore();
  const api = useApi();
  const [news, setNews] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchNews = async () => {
      setLoading(true);
      try {
        const data = await api.getNews(ticker);
        setNews(data);
      } catch (e) {
        console.error(e);
      } finally {
        setLoading(false);
      }
    };
    fetchNews();
  }, [ticker, api]);

  if (loading) return <>{Components.skeleton('chart')}</>;

  if (!news?.news?.length) {
    return (
      <section id="panel-news" className="panel flex flex-col gap-6 animate-fade-in-up w-full">
        <div className="glass-card">
          <p className="text-slate-500 text-center font-medium">No se encontraron noticias recientes para {ticker}.</p>
        </div>
      </section>
    );
  }

  const globalColor = news.global_label === 'ALCISTA' ? 'text-emerald-500' : news.global_label === 'BAJISTA' ? 'text-rose-500' : 'text-slate-500';

  return (
    <section id="panel-news" className="panel flex flex-col gap-6 animate-fade-in-up w-full">
      <div className="glass border border-slate-200/60 dark:border-slate-800 rounded-2xl p-6 shadow-premium dark:shadow-none mb-2">
        <h3 className="text-sm font-bold text-slate-400 uppercase tracking-wider mb-2">Sentimiento Global (Noticias Recientes)</h3>
        <div className="flex items-center gap-3">
          <span className={`text-2xl font-extrabold ${globalColor}`}>{news.global_label}</span>
          <span className="text-xs font-bold bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 px-3 py-1 rounded-full border border-slate-200 dark:border-slate-700">
            Score: {news.average_sentiment.toFixed(2)}
          </span>
        </div>
        <p className="text-xs text-slate-500 mt-2">Basado en el análisis de sentimiento VADER NLP aplicado a los últimos titulares financieros.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {news.news.map((n: any, i: number) => <div key={n.link || i}>{Components.newsCard(n.title, n.publisher, n.link, n.time, n.sentiment_label)}</div>)}
      </div>
    </section>
  );
}
