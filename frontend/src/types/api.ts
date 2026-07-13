// Tipos para las respuestas de la API

export interface Candle {
  time: number | string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

export interface IndicatorPoint {
  time: number | string;
  value: number;
}

export interface BBPoint {
  time: number | string;
  upper: number;
  middle: number;
  lower: number;
}

export interface MACDPoint {
  time: number | string;
  macd: number;
  signal: number;
  histogram: number;
}

export interface Indicators {
  sma_20: IndicatorPoint[];
  sma_50: IndicatorPoint[];
  sma_200: IndicatorPoint[];
  rsi: IndicatorPoint[];
  macd: MACDPoint[];
  bb: BBPoint[];
}

export interface LatestData {
  close: number;
  change?: number;
  change_pct: number;
  high?: number;
  low?: number;
  open?: number;
  volume: number;
}

export interface MarketDataResponse {
  ticker: string;
  period?: string;
  interval?: string;
  candles: Candle[];
  indicators: Indicators;
  latest: LatestData;
}

export interface Signal {
  action: 'BUY' | 'SELL' | 'HOLD';
  strength: number;
  reason: string;
}

export interface SignalsResponse {
  ticker: string;
  period?: string;
  interval?: string;
  signals: Signal[];
  composite_score: number;
}

export interface AdvisorResponse {
  ticker?: string;
  period?: string;
  interval?: string;
  verdict: string;
  color: string;
  advice: string;
  rsi: number;
  rsi_status: string;
  macd_status: string;
  ml_direction: string;
  ml_prob: number;
}

export interface AdvisorQuestionResponse {
  answer: string;
}

export interface BacktestMetrics {
  capital_final: number;
  retorno_total: number;
  retorno_anualizado: number;
  sharpe_ratio: number;
  max_drawdown: number;
  total_trades: number;
  win_rate: number;
  profit_factor: number;
  expectancy_pct?: number;
}

export interface BacktestTrade {
  entry_date: string;
  exit_date: string;
  side: string;
  entry_price: number;
  exit_price: number;
  shares: number;
  pnl: number;
  pnl_pct: number;
  commission: number;
  reason: string;
}

export interface BacktestResponse {
  ticker: string;
  period?: string;
  interval?: string;
  params: {
    initial_capital: number;
    commission_pct: number;
    slippage_pct: number;
  };
  metrics: BacktestMetrics;
  equity_curve: Array<{ time: number | string; value: number }>;
  trades: BacktestTrade[];
}

export interface PortfolioOptimizeRequest {
  tickers: string[];
  period: string;
  risk_free_rate: number;
}

export interface PortfolioResult {
  return: number;
  volatility: number;
  sharpe_ratio: number;
  weights: Record<string, number>;
}

export interface PortfolioOptimizeResponse {
  max_sharpe: PortfolioResult;
  min_volatility: PortfolioResult;
  equal_weight: PortfolioResult;
  frontier: Array<{ return: number; volatility: number }>;
}

export interface MLMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  train_size: number;
  test_size: number;
}

export interface MLBestParams {
  n_estimators: number;
  max_depth: number;
}

export interface MLPrediction {
  direction: 'ALCISTA' | 'BAJISTA' | 'N/A';
  probability: number;
  prediction_date: string;
  calibrated_prob?: number;
  raw_prob?: number;
  best_threshold?: number;
}

export interface MLStatusResponse {
  ticker: string;
  has_model: boolean;
  best_params: MLBestParams | null;
  optimized: boolean;
  prediction: MLPrediction;
  metrics: MLMetrics;
  feature_importances: Record<string, number>;
}

export interface MLTrainResponse {
  message?: string;
  metrics: MLMetrics;
  best_params: MLBestParams;
  optimized: boolean;
}

export interface MLSimulateResponse {
  metrics: {
    ml: BacktestMetrics;
    ta: BacktestMetrics;
    bh: BacktestMetrics;
  };
  equity_curves: {
    ml: Array<{ time: number | string; value: number }>;
    ta: Array<{ time: number | string; value: number }>;
    bh: Array<{ time: number | string; value: number }>;
  };
}

export interface NewsItem {
  title: string;
  publisher: string;
  link: string;
  time: string;
  sentiment_label: string;
  sentiment_score?: number;
  summary?: string;
}

export interface NewsResponse {
  ticker?: string;
  news: NewsItem[];
  global_label: string;
  average_sentiment: number;
}

export type WatchlistResponse = string[];

export interface ValidationMetrics {
  retorno_total: number;
  retorno_anualizado: number;
  sharpe_ratio: number;
  max_drawdown: number;
  dd_peak: string;
  dd_valley: string;
  total_trades: number;
  win_rate: number;
  profit_factor: number;
  capital_final: number;
  buy_hold_return: number;
  leverage: number;
}

export interface ValidationReport {
  ticker: string;
  period: string;
  total_data_years: number;
  verdict: string;
  overfit_flags: string[];
  walk_forward: Array<{
    window_idx: number;
    sharpe_is: number;
    sharpe_oos: number;
    overfit_ratio: number;
  }>;
  monte_carlo: {
    n_simulations: number;
    p5_return_pct: number;
    p50_return_pct: number;
    p95_return_pct: number;
    p50_max_drawdown_pct: number;
    prob_negative_return_pct: number;
    prob_sharpe_above_1_pct: number;
  };
  is_metrics: ValidationMetrics;
  oos_metrics: ValidationMetrics;
  html_report: string;
}

export interface GeneticJobLaunchResponse {
  job_id: string;
  status: string;
  message: string;
}

export interface GeneticJobStatus {
  job_id: string;
  status: 'running' | 'completed' | 'failed' | 'cancelled';
  elapsed_seconds?: number;
  created_at?: number;
  progress?: {
    current_gen?: number;
    total_gens?: number;
    pct?: number;
    gen_metrics?: {
      best_fitness: number;
      sharpe: number;
      retorno: number;
      max_drawdown: number;
      elapsed_s: number;
    };
  };
  result?: {
    best_params: Record<string, number>;
    best_fitness: number;
    best_sharpe: number;
    best_return: number;
    best_max_dd: number;
    generations: number;
    population_size: number;
  };
  error?: string;
}

export interface BrokerDashboardResponse {
  bot_status: {
    active: boolean;
    connected: boolean;
    strategy_mode?: string;
    mode: string;
    logs?: string[];
  };
  account: {
    equity: number;
    cash: number;
    buying_power: number;
    pnl_today: number;
    pnl_pct_today: number;
  };
  positions: Array<{
    symbol: string;
    qty: number;
    market_value: number;
    unrealized_pl: number;
    unrealized_plpc: number;
    avg_entry_price: number;
    current_price: number;
  }>;
  orders: Array<{
    id: string;
    symbol: string;
    qty: number;
    side: string;
    type: string;
    status: string;
    filled_avg_price: number;
    created_at: string;
  }>;
  market_breadth?: {
    level: string;
    reason?: string;
  };
  market_regime?: {
    regime: string;
    can_trade_long: boolean;
    reason: string;
  };
  advisor?: {
    active: boolean;
    last_decision: string;
    accuracy: number;
  };
  ml_models?: Array<{
    ticker: string;
    accuracy: number;
    age_hours: number;
  }>;
  risk?: {
    daily_pnl_pct: number;
    consecutive_losses: number;
    consecutive_loss_limit: number;
    circuit_breaker_active: boolean;
    circuit_breaker_remaining_min: number;
    kelly: {
      kelly_pct: number;
      half_kelly_pct: number;
      quarter_kelly_pct: number;
      win_rate: number;
      avg_win_pct: number;
      avg_loss_pct: number;
      odds_ratio: number;
      total_trades: number;
    };
    sector_exposures: Record<string, number>;
  };
  config?: Record<string, unknown>;
}

export interface BrokerAccountResponse {
  equity: number;
  cash: number;
  buying_power: number;
  pnl_today: number;
  pnl_pct_today: number;
}

export interface BrokerPosition {
  symbol: string;
  qty: number;
  market_value: number;
  unrealized_pl: number;
  unrealized_plpc: number;
  avg_entry_price: number;
  current_price: number;
}

export interface BrokerOrder {
  id: string;
  symbol: string;
  qty: number;
  side: string;
  type: string;
  status: string;
  filled_avg_price: number;
  created_at: string;
}

export interface BotStatusResponse {
  active: boolean;
  connected?: boolean;
  strategy_mode?: string;
  mode: string;
  last_scan?: string;
  logs?: string[];
}

export interface BotConfigResponse {
  strategy_mode: string;
  params: Record<string, unknown>;
  risk?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface MarketRegimeResponse {
  regime: string;
  can_trade_long: boolean;
  reason: string;
}

export interface AdvisorStatusResponse {
  active: boolean;
  last_decision: string;
  accuracy: number;
}

export interface MarketBreadthResponse {
  level: string;
  can_trade: boolean;
  reason: string;
  pct_above_sma50?: number;
  rsp_vs_spy_ratio?: number;
  rsp_vs_spy_trend?: string;
  qqq_vs_spy_ratio?: number;
  qqq_vs_spy_trend?: string;
  force_index_10d?: number;
  force_index_trend?: string;
  details?: string;
}

export interface KellyStatsResponse {
  kelly_pct: number;
  half_kelly_pct: number;
  quarter_kelly_pct: number;
  win_rate: number;
  avg_win_pct: number;
  avg_loss_pct: number;
  odds_ratio: number;
  total_trades: number;
}

export interface MLModelsStatusResponse {
  models: Array<{
    ticker: string;
    accuracy: number;
    age_hours: number;
    precision?: number;
  }>;
  note?: string;
}

export interface RiskStatusResponse {
  daily_pnl_pct: number;
  consecutive_losses: number;
  circuit_breaker_active: boolean;
  circuit_breaker_remaining_min: number;
  account_liquidated: boolean;
  portfolio_value: number;
  initial_portfolio_value: number;
  total_exposure_pct: number;
  sector_exposures: Record<string, number>;
  kelly: {
    kelly_pct: number;
    half_kelly_pct: number;
    quarter_kelly_pct: number;
    win_rate: number;
    avg_win_pct: number;
    avg_loss_pct: number;
    odds_ratio: number;
    total_trades: number;
  };
  performance: {
    total_trades: number;
    win_rate: number;
    avg_win_pct: number;
    avg_loss_pct: number;
    profit_factor: number;
    expectancy_pct: number;
    max_consecutive_losses: number;
    kelly_pct: number;
    odds_ratio: number;
  };
}
