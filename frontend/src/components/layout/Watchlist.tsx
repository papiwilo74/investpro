'use client';

import { useEffect, useState } from 'react';
import { useAppStore } from '@/store/appStore';
import { useApi } from '@/hooks/useApi';

export function Watchlist() {
  const { ticker, setTicker } = useAppStore();
  const api = useApi();
  const [watchlist, setWatchlist] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.getWatchlist().then(w => { setWatchlist(w); setLoading(false); }).catch(() => setLoading(false));
  }, [api]);

  if (loading) return <div className="grid grid-cols-2 gap-2"><div className="skeleton h-10 rounded-lg" /><div className="skeleton h-10 rounded-lg" /></div>;

  return (
    <div className="grid grid-cols-2 gap-2">
      {watchlist.map(t => (
        <button
          key={t}
          className={`watchlist-btn py-2 px-2 text-xs font-bold rounded-lg border border-transparent bg-slate-800 text-slate-200 hover:bg-slate-700 hover:border-blue-500 transition-all ${t === ticker ? 'active' : ''}`}
          data-ticker={t}
          onClick={() => setTicker(t)}
        >
          {t}
        </button>
      ))}
    </div>
  );
}
