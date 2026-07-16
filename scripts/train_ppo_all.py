"""Entrena modelos PPO para todos los tickers activos del universo de trading.

Uso:
    python scripts/train_ppo_all.py
    python scripts/train_ppo_all.py --tickers AAPL,MSFT,NVDA
    python scripts/train_ppo_all.py --universe nasdaq10
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import NASDAQ_100_UNIVERSE, PROJECT_ROOT, WATCHLIST
from ml.rl_train import RLTrainer

TECH10 = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "NFLX", "AMD", "INTC"]
DEFAULT_TICKERS = list(dict.fromkeys(WATCHLIST + TECH10))


def main():
    parser = argparse.ArgumentParser(description="Entrena modelos PPO para todos los tickers")
    parser.add_argument("--tickers", type=str, default=None, help="Tickers separados por coma")
    parser.add_argument(
        "--universe",
        type=str,
        default=None,
        choices=["watchlist", "nasdaq10", "tech10", "nasdaq100"],
        help="Universo predefinido (default: watchlist + tech10)",
    )
    parser.add_argument("--period", type=str, default="5y", help="Periodo de datos (default: 5y)")
    parser.add_argument(
        "--skip-existing", action="store_true", default=True, help="Saltar tickers que ya tienen modelo (default: True)"
    )
    args = parser.parse_args()

    models_dir = PROJECT_ROOT / "ml" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    elif args.universe == "watchlist":
        tickers = WATCHLIST
    elif args.universe == "nasdaq10":
        tickers = list(dict.fromkeys(WATCHLIST + TECH10))
    elif args.universe == "nasdaq100":
        tickers = NASDAQ_100_UNIVERSE
    else:
        tickers = DEFAULT_TICKERS

    tickers = list(dict.fromkeys(tickers))

    trainer = RLTrainer()
    success = []
    skipped = []
    failed = []

    for ticker in tickers:
        model_path = models_dir / f"{ticker}_ppo_model.zip"
        if args.skip_existing and model_path.exists():
            print(f"[SKIP] {ticker} — modelo ya existe en {model_path}")
            skipped.append(ticker)
            continue

        print(f"\n{'=' * 60}")
        print(f"  ENTRENANDO PPO: {ticker}")
        print(f"{'=' * 60}")
        try:
            trainer.train(ticker, period=args.period)
            success.append(ticker)
            print(f"[OK] {ticker} entrenado exitosamente")
            time.sleep(1)
        except Exception as e:
            print(f"[FAIL] {ticker}: {e}")
            failed.append(ticker)
            time.sleep(2)

    print(f"\n{'=' * 60}")
    print("  RESUMEN")
    print(f"{'=' * 60}")
    print(f"  Exitosos : {len(success):>3} — {', '.join(success) if success else '—'}")
    print(f"  Saltados : {len(skipped):>3} — {', '.join(skipped) if skipped else '—'}")
    print(f"  Fallidos : {len(failed):>3} — {', '.join(failed) if failed else '—'}")
    print(f"  Total    : {len(tickers):>3}")
    print(f"\n  Modelos en: {models_dir}")


if __name__ == "__main__":
    main()
