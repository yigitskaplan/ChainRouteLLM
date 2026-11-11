"""Ask the live API models one batch of SuperGPQA questions and save every answer.

    python -m src.data.collect_live --batch data/supergpqa/batches/batch_1_90.csv

Every question goes to all five models with the same settings as the
10-question test run (effort "low", at most 4,000 answer tokens). Each model
answers in its own thread, one call at a time, so its latency is not distorted;
a call over the account's rate limit waits as long as the provider says and is
sent again, and the latency is that of the attempt that answered. GPT calls are
spaced to stay under the 3-requests-per-minute limit of a new OpenAI account,
which also allows only 50 requests per model per day, so a GPT batch can take
days. ``--models`` asks only some of the models. Every answer is
appended to data/live/supergpqa_<batch>.csv as soon as it arrives: correctness,
tokens, cost (USD), latency (seconds) and the call time (UTC). Failed calls are
saved with their error; running the command again asks only what is missing.
A provider is no longer called once its spend over all files in data/live/
reaches its budget. API keys are read from .env and never written anywhere.
"""
from __future__ import annotations

import argparse
import csv
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import anthropic
import openai
import pandas as pd
from dotenv import load_dotenv

# USD per 1M tokens (input, output), standard tier, Nov 2026.
MODELS = {
    "claude-haiku-5-5": ("anthropic", 0.10, 0.50),
    "claude-sonnet-5-5": ("anthropic", 2.00, 10.00),
    "claude-opus-5-5": ("anthropic", 4.00, 20.00),
    "gpt-6.1-terra": ("openai", 1.00, 5.00),
    "gpt-6.1-sol": ("openai", 2.00, 10.00),
}
BUDGET = {"anthropic": 4.70, "openai": 4.70}  # of the $5 on each account
# Seconds between the starts of two calls to the same model (new OpenAI accounts: 3 per minute).
GAP = {"anthropic": 0.0, "openai": 20.5}
EFFORT, MAX_TOKENS = "low", 4000
LETTERS = "ABCDEFGHIJ"
COLUMNS = ["question", "uuid", "difficulty", "discipline", "truth", "model", "seconds", "answer", "stop",
           "input_tokens", "output_tokens", "details", "choice", "correct", "cost", "error", "time_utc"]


def prompt(question: str, options: Sequence[str]) -> str:
    choices = "\n".join(f"{LETTERS[i]}) {text}" for i, text in enumerate(options))
    return (
        "Answer the following multiple choice question. There is only one correct answer. "
        "The last line of your response must be in the format 'Answer: X', where X is the letter of your choice.\n\n"
        f"{question}\n\n{choices}"
    )


def wait_seconds(error: Exception) -> float:
    """How long a rate-limit error says to wait ("try again in 20s", "28m48s", "1h2m3s")."""
    wait = re.search(r"try again in ([\d.hms]+)", str(error))
    units = {"h": 3600, "m": 60, "s": 1, "ms": 0.001}
    parts = re.findall(r"([\d.]+)(ms|h|m|s)", wait.group(1)) if wait else []
    return sum(float(n) * units[u] for n, u in parts) or 20.0


def ask(clients: dict, model: str, text: str) -> dict:
    """One call; the answer, its tokens and its latency."""
    start = time.perf_counter()
    if MODELS[model][0] == "anthropic":
        r = clients["anthropic"].messages.create(
            model=model, max_tokens=MAX_TOKENS, output_config={"effort": EFFORT},
            messages=[{"role": "user", "content": text}],
        )
        seconds = time.perf_counter() - start
        return dict(seconds=seconds, answer="".join(b.text for b in r.content if b.type == "text"), stop=r.stop_reason,
                    input_tokens=r.usage.input_tokens, output_tokens=r.usage.output_tokens,
                    details=str(r.usage.output_tokens_details))
    r = clients["openai"].responses.create(
        model=model, input=text, reasoning={"effort": EFFORT}, max_output_tokens=MAX_TOKENS, store=False
    )
    seconds = time.perf_counter() - start
    return dict(seconds=seconds, answer=r.output_text or "", stop=r.status, input_tokens=r.usage.input_tokens,
                output_tokens=r.usage.output_tokens, details=f"reasoning_tokens={r.usage.output_tokens_details.reasoning_tokens}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Ask the live models one batch of SuperGPQA questions.")
    parser.add_argument("--batch", required=True, type=Path, help="data/supergpqa/batches/batch_<n>_<size>.csv")
    parser.add_argument("--data-dir", type=Path, default=Path("data/supergpqa"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/live"))
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    args = parser.parse_args(argv)
    load_dotenv(".env")
    out = args.out_dir / f"supergpqa_{args.batch.stem}.csv"

    batch = pd.read_csv(args.batch)
    questions = pd.read_json(args.data_dir / "SuperGPQA-all.jsonl", lines=True).set_index("uuid").loc[batch.uuid]
    # Spend so far over every file in data/live/, and the answers this batch already has.
    saved = [pd.read_csv(path) for path in sorted(args.out_dir.glob("*.csv"))]
    spent = {provider: 0.0 for provider in BUDGET}
    for frame in saved:
        for model, cost in frame.groupby("model")["cost"].sum().items():
            spent[MODELS[model][0]] += cost
    done = set()
    if out.exists():
        old = pd.read_csv(out)
        done = set(zip(old.uuid[old.error.isna()], old.model[old.error.isna()]))
    print("spent so far: " + ", ".join(f"{p} ${s:.4f}" for p, s in spent.items()) + f"; {len(done)} answers already in {out}")

    clients = {"anthropic": anthropic.Anthropic(max_retries=0), "openai": openai.OpenAI(max_retries=0)}
    lock = threading.Lock()
    new_file = not out.exists()
    out.parent.mkdir(parents=True, exist_ok=True)
    handle = open(out, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=COLUMNS)
    if new_file:
        writer.writeheader()

    def run(model: str) -> None:
        provider, price_in, price_out = MODELS[model]
        last = 0.0
        for number, (uuid, row) in enumerate(questions.iterrows(), 1):
            if (uuid, model) in done:
                continue
            with lock:
                if spent[provider] >= BUDGET[provider]:
                    print(f"{model}: {provider} budget reached, stopping", flush=True)
                    return
            try:
                for attempt in range(100):
                    time.sleep(max(0.0, last + GAP[provider] - time.monotonic()))
                    last = time.monotonic()
                    called = datetime.now(timezone.utc).isoformat(timespec="seconds")
                    try:
                        result = ask(clients, model, prompt(row.question, row.options))
                        break
                    except (anthropic.RateLimitError, openai.RateLimitError) as error:
                        time.sleep(wait_seconds(error) + 1)
                else:
                    raise RuntimeError("rate limit: 100 attempts")
                found = re.findall(r"Answer:\s*\**\s*\(?([A-J])\b", result["answer"])
                result["choice"] = found[-1] if found else None
                result["correct"] = result["choice"] == row.answer_letter
                result["cost"] = (result["input_tokens"] * price_in + result["output_tokens"] * price_out) / 1e6
                result["error"] = None
            except Exception as error:  # keep going; a failed call is evidence too
                text = re.sub(r"org-[A-Za-z0-9]+", "org-<hidden>", str(error))  # keep account ids out of the file
                result = dict(error=f"{type(error).__name__}: {text[:200]}", correct=False, cost=0.0)
            record = dict(question=number, uuid=uuid, difficulty=row.difficulty, discipline=row.discipline,
                          truth=row.answer_letter, model=model, time_utc=called, **result)
            with lock:
                spent[provider] += record["cost"]
                writer.writerow(record)
                handle.flush()
                print(f"q{number:<3} {model:18s} " + (f"ERROR {record['error']}" if record["error"] else
                      f"{record['choice']} vs {row.answer_letter} {'right' if record['correct'] else 'WRONG'}  "
                      f"{record['seconds']:5.1f}s  ${record['cost']:.4f}"), flush=True)

    threads = [threading.Thread(target=run, args=(model,)) for model in args.models]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    handle.close()

    results = pd.read_csv(out)
    ok = results[results.error.isna()]
    summary = ok.groupby("model", sort=False).agg(answers=("correct", "size"), right=("correct", "sum"),
                                                  median_s=("seconds", "median"), cost=("cost", "sum"))
    summary["errors"] = results[results.error.notna()].groupby("model").size().reindex(summary.index, fill_value=0)
    print(summary.round(4).to_string())
    print("spent in total: " + ", ".join(f"{p} ${s:.4f}" for p, s in spent.items()) + f" -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
