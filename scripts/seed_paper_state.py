"""Seed the paper_state table with current bot balances.

Usage:
    uv run python scripts/seed_paper_state.py --cash 103400.0

    # Or from a running Render instance via API:
    uv run python scripts/seed_paper_state.py --from-api http://localhost:8000

If no args given, prints the current DB state for inspection.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db import SessionLocal, init_db
from db.models import PaperState


def seed(cash: float, initial_cash: float = 100000.0) -> None:
    init_db()
    data = {
        "cash": cash,
        "initial_cash": initial_cash,
        "order_counter": 0,
        "positions": [],
        "orders": [],
        "trades": [],
        "equity_history": [],
        "updated_at": datetime.now(UTC),
    }
    with SessionLocal() as session:
        row = session.get(PaperState, 1)
        if row is not None:
            print(f"PaperState already exists — cash=${row.cash:,.2f}")
            yn = input("Overwrite? (y/N): ")
            if yn.lower() != "y":
                print("Aborted.")
                return
        row = PaperState(id=1, **data)
        session.add(row)
        session.commit()
    print(f"PaperState seeded: cash=${cash:,.2f} initial_cash=${initial_cash:,.2f}")


def show() -> None:
    init_db()
    with SessionLocal() as session:
        row = session.get(PaperState, 1)
        if row is None:
            print("No PaperState found in DB.")
            return
        print(
            f"cash={row.cash}, initial_cash={row.initial_cash}, "
            f"positions={len(row.positions)}, orders={len(row.orders)}, "
            f"trades={len(row.trades)}, equity_samples={len(row.equity_history)}"
        )


def fetch_from_api(base_url: str) -> None:
    r = requests.get(f"{base_url.rstrip('/')}/api/broker/dashboard", timeout=10)
    r.raise_for_status()
    data = r.json()
    acc = data.get("account", {})
    cash = acc.get("cash", 100000.0)
    seed(cash=cash, initial_cash=100000.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed paper_state table")
    parser.add_argument("--cash", type=float, help="Cash balance to seed")
    parser.add_argument("--from-api", metavar="URL", help="Fetch cash from a running instance API")
    args = parser.parse_args()

    if args.from_api:
        fetch_from_api(args.from_api)
    elif args.cash is not None:
        seed(cash=args.cash)
    else:
        show()


if __name__ == "__main__":
    main()
