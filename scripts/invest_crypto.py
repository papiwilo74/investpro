"""Inversión puntual en criptomoneda usando el MISMO pipeline de análisis del bot.

Replica el camino exacto de `TradingBot._scan_and_trade_crypto` (mismo
DataFetcher + indicadores + señales + TradingBrain + parámetros legacy),
para que la decisión que ves aquí sea idéntica a la que tomaría el bot en
Render. Es una forma rápida de comprobar "si el bot sabe analizar".

Uso:
  python -m scripts.invest_crypto                        # solo análisis (dry-run)
  python -m scripts.invest_crypto --execute              # compra si hay señal BUY
  python -m scripts.invest_crypto --symbol ETH/USD --amount 10000 --execute
"""

from __future__ import annotations

import argparse
import sys

if sys.stdout and sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from bot.strategy import TradingBrain
from bot.strategy_params import StrategyParams
from broker import create_crypto_client
from broker.crypto_client import DEFAULT_CRYPTO_WATCHLIST
from config import BROKER_CONFIG
from data.fetcher import DataFetcher
from indicators.signals import SignalGenerator
from indicators.technical import TechnicalIndicators


def build_params() -> StrategyParams:
    """Replica los parámetros legacy que usa el bot en modo daemon (main.py)."""
    return StrategyParams(
        buy_score_threshold=BROKER_CONFIG.buy_score_threshold,
        sell_score_threshold=BROKER_CONFIG.sell_score_threshold,
        stop_loss_pct=BROKER_CONFIG.stop_loss_pct,
        take_profit_pct=BROKER_CONFIG.take_profit_pct,
        max_position_size_pct=BROKER_CONFIG.max_position_size_pct,
        min_ml_buy_probability=BROKER_CONFIG.min_ml_buy_probability,
        use_intraday_scalp=False,
        use_session_filter=False,
        use_vwap_filter=False,
        use_neural_brain=False,
    )


def analyze_symbol(brain: TradingBrain, fetcher: DataFetcher, symbol: str, interval: str) -> dict | None:
    """Analiza un símbolo crypto con el mismo pipeline del bot."""
    ticker = symbol.replace("/", "")  # BTC/USD -> BTCUSD (el fetcher lo normaliza)
    df = fetcher.get_data(ticker, period="3mo", interval=interval)
    if df.empty:
        return None

    df = TechnicalIndicators.add_all(df, intraday=False)
    df = SignalGenerator.add_signal_columns(df)
    score = SignalGenerator.composite_score(df)
    last_close = float(df["close"].iloc[-1])

    ticker_regime = TradingBrain._infer_market_regime(df)
    weekly_trend = TradingBrain._infer_weekly_trend(df)

    decision = brain.decide(
        df=df,
        score=score,
        has_position=False,
        position_pnl_pct=0.0,
        ml_direction=None,
        ml_probability=None,
        sentiment_label=None,
        ticker=ticker,
        weekly_trend=weekly_trend,
        market_regime=ticker_regime,
        advisor_action=None,
    )

    last = df.iloc[-1]
    return {
        "symbol": symbol,
        "ticker": ticker,
        "close": last_close,
        "score": score,
        "rsi": float(last.get("rsi")) if last.get("rsi") is not None and last.get("rsi") == last.get("rsi") else None,
        "adx": float(last.get("adx")) if last.get("adx") is not None and last.get("adx") == last.get("adx") else None,
        "weekly_trend": weekly_trend,
        "regime": ticker_regime,
        "action": decision.action,
        "reason": decision.reason,
        "confidence": decision.confidence,
    }


def print_report(results: list[dict]) -> None:
    print("\n" + "=" * 64)
    print("  ANALISIS DEL BOT — CRIPTO (mismo pipeline que en Render)")
    print("=" * 64)
    for r in results:
        action = r["action"]
        print(f"\n  {r['symbol']}  (último: ${r['close']:,.2f})")
        print(f"    Score compuesto : {r['score']:+.2f}  (mínimo BUY: {BROKER_CONFIG.buy_score_threshold:+.2f})")
        print(f"    RSI             : {r['rsi'] if r['rsi'] is not None else 'N/A'}")
        print(f"    ADX             : {r['adx'] if r['adx'] is not None else 'N/A'}")
        print(f"    Tendencia 50d   : {r['weekly_trend']}")
        print(f"    Régimen         : {r['regime']}")
        print(f"    Decisión        : {action} (conf={r['confidence']:.2f})")
        print(f"    Razón           : {r['reason']}")
    print("=" * 64)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default=None, help="Símbolo crypto (ej: BTC/USD). Default: toda la watchlist crypto.")
    ap.add_argument("--amount", type=float, default=10_000.0, help="Monto USD a invertir (default: 10000)")
    ap.add_argument("--interval", default="1d", help="Intervalo de velas (default: 1d)")
    ap.add_argument("--execute", action="store_true", help="Ejecutar la compra si la señal es BUY (paper)")
    args = ap.parse_args()

    symbols = [args.symbol] if args.symbol else list(DEFAULT_CRYPTO_WATCHLIST)

    print("Conectando al broker crypto (paper)...")
    client = create_crypto_client(paper=True)
    acc = client.get_account_summary()
    equity = float(acc.get("equity", 0.0))
    buying_power = float(acc.get("buying_power", 0.0))
    print(f"  Equity: ${equity:,.2f} | Buying power: ${buying_power:,.2f}")

    fetcher = DataFetcher()
    brain = TradingBrain(build_params())

    results = []
    for symbol in symbols:
        print(f"\n[Analizando {symbol}]")
        try:
            r = analyze_symbol(brain, fetcher, symbol, args.interval)
        except Exception as exc:
            print(f"  [ERROR] {exc}")
            continue
        if r is None:
            print("  [ERROR] Sin datos")
            continue
        results.append(r)

    if not results:
        print("\nNo se pudo analizar ningún activo.")
        return 1

    print_report(results)

    buys = [
        r
        for r in results
        if r["action"] == "BUY" and r["confidence"] >= 0.5 and r["score"] >= BROKER_CONFIG.buy_score_threshold
    ]
    if not buys:
        print("\nNinguna cripto pasa los filtros BUY del bot → NO se invierte.")
        print("Esto es la señal de 'espera' del propio análisis: el bot no ve setup alcista.")
        return 0

    buys.sort(key=lambda r: r["confidence"], reverse=True)
    best = buys[0]

    amount = min(args.amount, buying_power)
    if amount <= 0:
        print("\nSin buying power disponible para operar.")
        return 1

    print(f"\nEl bot quiere COMPRAR {best['symbol']} (conf={best['confidence']:.2f}, score={best['score']:+.2f})")

    if not args.execute:
        print(f"[DRY-RUN] Invertiría ${amount:,.0f} en {best['symbol']} @ ${best['close']:,.2f}.")
        print("Para ejecutar la orden en papel:  python -m scripts.invest_crypto --execute")
        return 0

    qty = amount / best["close"]
    print(f"Ejecutando orden de mercado: BUY {qty:.6f} {best['symbol']} (~${amount:,.0f})...")
    result = client.place_market_order(best["symbol"], qty, "BUY")
    if result.get("status") == "success":
        print(
            f"[OK] Orden enviada: {result.get('side')} {result.get('qty')} {result.get('symbol')} (id={result.get('order_id')})"
        )
        return 0

    print(f"[ERROR] Orden rechazada: {result.get('msg')}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
