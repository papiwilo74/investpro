'use client';

import { useState, useEffect, useRef } from 'react';
import { useAppStore } from '@/store/appStore';
import { api } from '@/api/client';
import type { ChatMessage, LLMStatusResponse } from '@/types/api';

interface DisplayMessage extends ChatMessage {
  id: string;
  source?: string;
  timestamp?: string;
}

export function CopilotPanel() {
  const { ticker } = useAppStore();
  const [messages, setMessages] = useState<DisplayMessage[]>([
    {
      id: 'initial',
      role: 'assistant',
      content: `¡Hola! Soy **Axiom Copilot**, tu Asistente Cuantitativo personal potenciado por **Llama 3.1 8B** corriendo localmente en tu GPU **NVIDIA RTX 4060**.

Tengo acceso en tiempo real a tus balances de Alpaca, el estado de tus posiciones, los indicadores técnicos (RSI, Bollinger, MACD, SMA 200) y el régimen macroeconómico.

Puedes hacerme preguntas libres sobre el mercado, pedirme que analice un activo o usar las sugerencias rápidas aquí abajo.`,
      source: 'ollama:llama3.1:8b',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    },
  ]);

  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState<LLMStatusResponse | null>(null);
  const [contextMode, setContextMode] = useState<'ticker' | 'portfolio'>('portfolio');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Consultar estado de Ollama al cargar
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const res = await api.getLLMStatus();
        setStatus(res);
      } catch (err) {
        console.error('Error obteniendo estado LLM:', err);
      }
    };
    fetchStatus();
    const interval = setInterval(fetchStatus, 15000);
    return () => clearInterval(interval);
  }, []);

  // Auto-scroll al final del chat
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const handleSend = async (queryText?: string) => {
    const text = (queryText || input).trim();
    if (!text || loading) return;

    const userMsg: DisplayMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: text,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    const newHistory = [...messages, userMsg];
    setMessages(newHistory);
    setInput('');
    setLoading(true);

    try {
      const historyPayload: ChatMessage[] = newHistory.slice(-8).map(m => ({
        role: m.role,
        content: m.content,
      }));

      const activeTicker = contextMode === 'ticker' ? ticker : undefined;
      const res = await api.chatCopilot(text, activeTicker, historyPayload);

      const aiMsg: DisplayMessage = {
        id: `ai-${Date.now()}`,
        role: 'assistant',
        content: res.response,
        source: res.source,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };

      setMessages(prev => [...prev, aiMsg]);
    } catch (err: any) {
      const errorMsg: DisplayMessage = {
        id: `err-${Date.now()}`,
        role: 'assistant',
        content: `⚠️ **Error al consultar el asistente**: ${err.message || 'Verifica que Ollama esté activo en tu PC'}.`,
        source: 'error',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleClear = () => {
    setMessages([
      {
        id: 'welcome',
        role: 'assistant',
        content: 'Conversación reiniciada. ¿En qué te puedo ayudar ahora con tu portafolio o el análisis de mercado?',
        source: 'ollama:llama3.1:8b',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      },
    ]);
  };

  const quickPrompts = [
    { label: '📊 Salud del Portafolio', query: '¿Cómo está la salud general de mi portafolio hoy y qué nivel de liquidez tenemos?' },
    { label: '🔍 ¿Por qué no compramos?', query: '¿Por qué Axiom no ha abierto nuevas posiciones hoy? ¿Qué filtros no pasaron?' },
    { label: `🛡️ Riesgo en ${ticker}`, query: `¿Cuál es el riesgo actual y el análisis cuantitativo de mantener ${ticker}?` },
    { label: '⚖️ Regla 60/40', query: '¿Cómo está la asignación entre Cripto y Renta Variable respecto a la meta de 60% Cripto / 40% Acciones?' },
    { label: '📈 Régimen de Mercado', query: '¿Qué régimen de mercado tenemos en SPY y VIX, y qué precaución debemos tener?' },
  ];

  const isOnline = status?.available && !status?.fallback_active;

  return (
    <section className="panel active flex flex-col gap-6 animate-fade-in-up w-full max-w-5xl mx-auto">
      {/* Header del Copiloto */}
      <div className="glass-card flex flex-col md:flex-row items-start md:items-center justify-between gap-4 p-6">
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 rounded-2xl bg-gradient-to-tr from-blue-600 to-indigo-500 flex items-center justify-center text-white text-2xl shadow-lg shadow-blue-500/20">
            🤖
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-xl font-bold text-slate-900 dark:text-white">Axiom Copilot</h2>
              <span className="text-xs font-bold px-2 py-0.5 rounded-full bg-blue-100 dark:bg-blue-900/40 text-blue-600 dark:text-blue-400 border border-blue-200 dark:border-blue-800">
                Llama 3.1 8B
              </span>
            </div>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
              Asistente Cuantitativo en tiempo real • Inferencia local en NVIDIA RTX 4060
            </p>
          </div>
        </div>

        {/* Estado y Controles */}
        <div className="flex flex-wrap items-center gap-3">
          <div className={`flex items-center gap-2 px-3 py-1.5 rounded-xl border text-xs font-semibold ${
            isOnline
              ? 'bg-emerald-50 dark:bg-emerald-950/40 border-emerald-300 dark:border-emerald-800 text-emerald-700 dark:text-emerald-300'
              : 'bg-amber-50 dark:bg-amber-950/40 border-amber-300 dark:border-amber-800 text-amber-700 dark:text-amber-300'
          }`}>
            <span className={`w-2 h-2 rounded-full ${isOnline ? 'bg-emerald-500 animate-pulse' : 'bg-amber-500'}`} />
            {isOnline ? 'GPU RTX 4060 En Línea' : 'Modo Respaldo Algorítmico'}
          </div>

          <div className="flex bg-slate-200 dark:bg-slate-800 p-1 rounded-xl text-xs font-medium">
            <button
              onClick={() => setContextMode('portfolio')}
              className={`px-3 py-1 rounded-lg transition-all ${
                contextMode === 'portfolio' ? 'bg-white dark:bg-slate-700 text-slate-900 dark:text-white shadow-sm font-bold' : 'text-slate-500 hover:text-slate-900 dark:hover:text-white'
              }`}
            >
              Portafolio
            </button>
            <button
              onClick={() => setContextMode('ticker')}
              className={`px-3 py-1 rounded-lg transition-all ${
                contextMode === 'ticker' ? 'bg-white dark:bg-slate-700 text-slate-900 dark:text-white shadow-sm font-bold' : 'text-slate-500 hover:text-slate-900 dark:hover:text-white'
              }`}
            >
              {ticker}
            </button>
          </div>

          <button
            onClick={handleClear}
            className="px-3 py-1.5 text-xs font-semibold text-slate-500 hover:text-rose-600 dark:text-slate-400 dark:hover:text-rose-400 hover:bg-slate-100 dark:hover:bg-slate-800/60 rounded-xl transition-all border border-transparent hover:border-slate-200 dark:hover:border-slate-700"
            title="Reiniciar conversación"
          >
            🗑️ Limpiar
          </button>
        </div>
      </div>

      {/* Sugerencias Rápidas */}
      <div className="flex flex-wrap gap-2">
        {quickPrompts.map((p, idx) => (
          <button
            key={idx}
            onClick={() => handleSend(p.query)}
            disabled={loading}
            className="px-3 py-2 rounded-xl text-xs font-semibold bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-slate-700 dark:text-slate-300 hover:border-blue-500 hover:text-blue-600 dark:hover:text-blue-400 shadow-sm transition-all hover:scale-[1.02] active:scale-[0.98] disabled:opacity-50"
          >
            {p.label}
          </button>
        ))}
      </div>

      {/* Ventana de Mensajes */}
      <div className="glass-card flex flex-col p-4 md:p-6 min-h-[460px] max-h-[620px] overflow-y-auto space-y-4 rounded-2xl shadow-inner">
        {messages.map(m => (
          <div
            key={m.id}
            className={`flex flex-col max-w-[88%] md:max-w-[80%] ${
              m.role === 'user' ? 'self-end items-end' : 'self-start items-start'
            }`}
          >
            <div className="flex items-center gap-1.5 mb-1 px-1">
              <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
                {m.role === 'user' ? 'Tú' : 'Axiom Copilot'}
              </span>
              {m.source && (
                <span className="text-[9px] px-1.5 py-0.2 rounded bg-slate-200 dark:bg-slate-800 text-slate-500 dark:text-slate-400">
                  {m.source}
                </span>
              )}
              {m.timestamp && <span className="text-[10px] text-slate-400">{m.timestamp}</span>}
            </div>

            <div
              className={`p-4 rounded-2xl text-sm leading-relaxed whitespace-pre-wrap transition-all shadow-sm ${
                m.role === 'user'
                  ? 'bg-blue-600 text-white rounded-tr-none font-medium'
                  : 'bg-white dark:bg-slate-900 text-slate-800 dark:text-slate-100 border border-slate-200 dark:border-slate-800 rounded-tl-none'
              }`}
            >
              {m.content}
            </div>
          </div>
        ))}

        {loading && (
          <div className="self-start flex flex-col items-start max-w-[80%]">
            <div className="flex items-center gap-1.5 mb-1 px-1">
              <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400">Axiom Copilot</span>
              <span className="text-[9px] px-1.5 py-0.2 rounded bg-blue-100 dark:bg-blue-950 text-blue-500">
                Razonando en RTX 4060...
              </span>
            </div>
            <div className="p-4 rounded-2xl rounded-tl-none bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-blue-500 animate-bounce [animation-delay:-0.3s]"></span>
              <span className="w-2 h-2 rounded-full bg-blue-500 animate-bounce [animation-delay:-0.15s]"></span>
              <span className="w-2 h-2 rounded-full bg-blue-500 animate-bounce"></span>
              <span className="text-xs text-slate-400 ml-2">Analizando datos y generando respuesta...</span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input de Chat */}
      <div className="glass-card p-3 md:p-4 flex items-end gap-3 rounded-2xl">
        <textarea
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={`Hazle una pregunta al Copiloto sobre ${contextMode === 'ticker' ? ticker : 'tu portafolio'}... (Enter para enviar)`}
          rows={2}
          disabled={loading}
          className="flex-1 bg-transparent border-0 resize-none text-sm text-slate-900 dark:text-slate-100 placeholder-slate-400 focus:ring-0 focus:outline-none p-1"
        />

        <button
          onClick={() => handleSend()}
          disabled={!input.trim() || loading}
          className="px-5 py-3 rounded-xl font-bold text-sm bg-blue-600 hover:bg-blue-500 text-white shadow-md shadow-blue-500/20 disabled:opacity-40 disabled:cursor-not-allowed transition-all flex items-center gap-2 hover:scale-105 active:scale-95"
        >
          <span>Enviar</span>
          <svg className="w-4 h-4 transform rotate-45" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
          </svg>
        </button>
      </div>
    </section>
  );
}
