"""Reset circuit breaker and clean up stale state."""

import json
from pathlib import Path

# Reset circuit breaker
rp = Path("data/risk_state.json")
rd = json.loads(rp.read_text())
rd["circuit_breaker_until"] = None
rd["daily_pnl"] = []
rd["consecutive_losses"] = 0
rp.write_text(json.dumps(rd, indent=2))
print(f"Circuit breaker reseteado. Portfolio: ${rd['portfolio_value']:,.2f}")

# Clean kelly trades dedup
kp = Path("data/kelly_trades.json")
kd = json.loads(kp.read_text())
n = len(kd["trades"])
# Remove duplicates while preserving order
seen = set()
unique = []
for t in kd["trades"]:
    rounded = round(t, 4)
    if rounded not in seen:
        seen.add(rounded)
        unique.append(t)
kd["trades"] = unique[-200:]  # keep last 200
kp.write_text(json.dumps(kd, indent=2))
print(f"Kelly trades: {n} -> {len(kd['trades'])} (deduplicados, ultimos 200)")

# Fix genetic hall of fame
hp = Path("data/genetic_hall_of_fame.json")
try:
    json.loads(hp.read_text())
    print("HOF OK")
except json.JSONDecodeError:
    import re

    t = hp.read_text()
    t = re.sub(r":\s*-?Infinity", ": null", t)
    t = re.sub(r":\s*NaN", ": null", t)
    hp.write_text(t)
    print("HOF reparado (Infinity/NaN reemplazado)")
