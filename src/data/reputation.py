"""Stage 3: each provider's public reputation R from news tone (GDELT).

    python -m src.data.reputation --start 2026-10-12 --end 2026-11-11

For every provider this asks the GDELT DOC 2.0 API (free, no key) for the
average tone of news articles that mention it, per day, and for the number of
such articles. GDELT's tone runs roughly from -10 (very negative) to +10 (very
positive). A provider's tone is the mean of its daily tones over the window
(30 days by default, so a single news day does not decide it). R puts the
providers on one scale:

    R_p = (tone_p - mean tone over providers) / s

where s is the spread (standard deviation) of all daily tones in the window. So
R = +1 means one typical daily swing above the average provider. Writes
data/reputation/gdelt_daily.csv (date, provider, tone, articles) and
data/reputation/reputation.json (window, query, tone, days and R per provider).
GDELT allows about one request every few seconds, so requests are spaced and
retried.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path
from typing import Dict, Optional, Sequence

import pandas as pd

API = "https://api.gdeltproject.org/api/v2/doc/doc"
# Provider -> GDELT search term (the company name; product names are too ambiguous).
PROVIDERS = {"anthropic": "Anthropic", "openai": "OpenAI"}
PAUSE = 10.0  # seconds between requests


def fetch(query: str, mode: str, start: date, end: date, tries: int = 6) -> pd.Series:
    """One GDELT timeline (daily values) for ``query``; ``mode`` is timelinetone or timelinevolraw."""
    params = {
        "query": query, "mode": mode, "format": "json",
        "startdatetime": start.strftime("%Y%m%d000000"), "enddatetime": end.strftime("%Y%m%d235959"),
    }
    url = f"{API}?{urllib.parse.urlencode(params)}"
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                points = json.loads(response.read().decode("utf-8"))["timeline"][0]["data"]
            break
        except urllib.error.HTTPError as error:
            if error.code != 429 or attempt == tries - 1:
                raise
            time.sleep(PAUSE * (attempt + 2))  # asked too fast: wait longer each time
    days = pd.to_datetime([p["date"][:8] for p in points], format="%Y%m%d").date
    return pd.Series([float(p["value"]) for p in points], index=days)


def reputation(daily: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Mean tone, days with news and R per provider from the daily table."""
    tone = daily.groupby("provider")["tone"].mean()
    spread = float(daily["tone"].std())
    r = (tone - tone.mean()) / spread
    days = daily.groupby("provider")["tone"].size()
    return {p: {"tone": round(float(tone[p]), 4), "days": int(days[p]), "R": round(float(r[p]), 4)} for p in tone.index}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch each provider's news tone from GDELT and compute R.")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2026, 10, 12))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 11, 11))
    parser.add_argument("--out-dir", type=Path, default=Path("data/reputation"))
    args = parser.parse_args(argv)

    rows = []
    for provider, query in PROVIDERS.items():
        tone = fetch(query, "timelinetone", args.start, args.end)
        time.sleep(PAUSE)
        articles = fetch(query, "timelinevolraw", args.start, args.end)
        time.sleep(PAUSE)
        frame = pd.DataFrame({"tone": tone, "articles": articles.reindex(tone.index)})
        rows.append(frame.rename_axis("date").reset_index().assign(provider=provider))
        print(f"{provider}: {len(tone)} days, mean tone {tone.mean():+.3f}")
    daily = pd.concat(rows, ignore_index=True)[["date", "provider", "tone", "articles"]]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    daily.to_csv(args.out_dir / "gdelt_daily.csv", index=False)
    summary = {
        "source": "GDELT DOC 2.0 API, modes timelinetone and timelinevolraw",
        "window": [args.start.isoformat(), args.end.isoformat()],
        "query": PROVIDERS,
        "spread": round(float(daily["tone"].std()), 4),
        "providers": reputation(daily),
    }
    (args.out_dir / "reputation.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for provider, values in summary["providers"].items():
        print(f"{provider}: tone {values['tone']:+.3f} over {values['days']} days -> R = {values['R']:+.3f}")
    print(f"wrote {args.out_dir}/gdelt_daily.csv and reputation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
