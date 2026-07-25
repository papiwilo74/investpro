'use client';

import { useEffect, useState } from 'react';
import { useAppStore } from '@/store/appStore';
import { useApi } from '@/hooks/useApi';
import { Components } from '@/components/ui/Components';

export function AdvisorPanel() {
  const { ticker, period, interval } = useAppStore();
  const api = useApi();
  const [data, setData] = useState<{ verdict: string; color: string; advice: string; rsi: number; rsi_status: string; macd_status: string; ml_direction: string; ml_prob: number } | null>(null);
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(true);
  const [activeQuestion, setActiveQuestion] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      try {
        const res = await api.getAdvisor(ticker, period, interval);
        setData(res);
        setAnswers({});
        setActiveQuestion(null);
      } catch (e) {
        console.error(e);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, [ticker, period, interval, api]);

  const handleQuestion = async (qId: string) => {
    setActiveQuestion(qId);
    setAnswers(prev => ({ ...prev, [qId]: 'loading' }));
    try {
      const res = await api.getAdvisorQuestion(ticker, qId, period, interval);
      setAnswers(prev => ({ ...prev, [qId]: res.answer }));
    } catch {
      setAnswers(prev => ({ ...prev, [qId]: 'Error al obtener la respuesta.' }));
    }
  };

  if (loading) return <>{Components.skeleton('advisor')}</>;

  if (!data) return <div className="text-center text-slate-500 py-12">Error cargando asesor</div>;

  const rsi = data.rsi ?? 0;
  const mlProb = data.ml_prob ?? 0;
  const rsiColor = rsi > 70 ? '#ef4444' : rsi < 30 ? '#10b981' : '#3b82f6';
  const macdColor = data.macd_status === 'Impulso Alcista' ? '#10b981' : '#ef4444';
  const mlColor = data.ml_direction === 'ALCISTA' ? '#10b981' : '#ef4444';

  return (
    <section id="panel-advisor" className="panel active flex flex-col gap-6 animate-fade-in-up w-full">
      {Components.verdictCard(data.verdict, data.color, data.advice)}

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {Components.advisorStatCard('Fuerza del RSI (14)', rsi.toFixed(1), data.rsi_status, rsiColor)}
        {Components.advisorStatCard('Impulso MACD', data.macd_status, 'Basado en histograma diario', macdColor)}
        {Components.advisorStatCard('Predicción ML', data.ml_direction !== 'N/A' ? `${data.ml_direction} (${(mlProb * 100).toFixed(0)}%)` : 'Sin modelo', 'Previsión a 5 días hábiles', mlColor)}
      </div>

      <div className="glass-card">
        <h3 className="text-base font-bold text-slate-900 dark:text-slate-100 mb-4">Consulta al Asesor de Inversiones</h3>
        <div className="space-y-2">
          {[
            { id: '1', text: 'Niveles clave de soporte y resistencia' },
            { id: '2', text: 'Principales factores de riesgo' },
            { id: '3', text: 'Tendencia de largo plazo (SMA 200)' },
            { id: '4', text: 'Porcentaje recomendado de capital a invertir' },
          ].map(q => (
            <button
              key={q.id}
              onClick={() => handleQuestion(q.id)}
              className={`question-btn w-full text-left px-4 py-3 rounded-xl border bg-slate-50 dark:bg-slate-950 text-sm font-semibold text-slate-700 dark:text-slate-300 transition-all ${
                activeQuestion === q.id ? 'border-blue-400 text-blue-600 dark:text-blue-400' : 'border-slate-200 dark:border-slate-800 hover:border-blue-400 hover:text-blue-600 dark:hover:text-blue-400'
              }`}
            >
              {q.text}
            </button>
          ))}
        </div>
        {activeQuestion && answers[activeQuestion] && (
          <div id="advisor-answer-container" className="mt-4 animate-fade-in-up">
            {answers[activeQuestion] === 'loading' ? (
              <div className="skeleton h-4 w-full rounded mb-2"></div>
            ) : (
              <div className="bg-slate-50 dark:bg-slate-950 border-l-4 border-l-blue-500 rounded-xl p-5">
                <strong className="block text-sm text-blue-600 dark:text-blue-400 mb-2">Respuesta del Asesor:</strong>
                <p className="text-sm leading-relaxed text-slate-700 dark:text-slate-300 whitespace-pre-wrap">{answers[activeQuestion]}</p>
              </div>
            )}
          </div>
        )}
      </div>
    </section>
  );
}
