"""LLM-as-judge evaluation for the generated briefs and alerts.

Scores each {context, response} record on three 1-5 metrics with the same
Gemini model the app uses: groundedness (is every claim supported by the
retrieved SOP context), coherence, and relevance. Prints per-metric averages.

Input is a JSON Lines file with one object per line:
    {"context": "<retrieved SOP text>", "query": "<optional>", "response": "<brief or alert>"}

Usage:
    python ml/evaluate_llm.py results.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from statistics import mean

from langchain_google_genai import ChatGoogleGenerativeAI

from predictivecare import config

METRICS = {
    "groundedness": (
        "Rate GROUNDEDNESS 1-5: is every claim in the response supported by the "
        "provided context? Judge support by the context only, not real-world truth."
    ),
    "coherence": (
        "Rate COHERENCE 1-5: is the response logically structured, fluent, and "
        "easy to follow, independent of factual accuracy?"
    ),
    "relevance": (
        "Rate RELEVANCE 1-5: does the response directly address the user's "
        "situation described in the context/query?"
    ),
}

_PROMPT = (
    "You are a strict evaluation assistant.\n{rubric}\n\n"
    "Context:\n{context}\n\nResponse:\n{response}\n\n"
    'Reply with ONLY a JSON object: {{"score": <1-5>, "reason": "<one sentence>"}}'
)


def _get_judge() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=config.GOOGLE_MODEL,
        google_api_key=config.require_google_api_key(),
        temperature=0.0,
    )


def _parse_score(raw: str) -> int:
    try:
        return int(json.loads(raw)["score"])
    except Exception:
        m = re.search(r'"?score"?\s*[:=]\s*([1-5])', raw)
        return int(m.group(1)) if m else 0


def judge_record(judge, context: str, response: str) -> dict[str, int]:
    scores: dict[str, int] = {}
    for name, rubric in METRICS.items():
        prompt = _PROMPT.format(rubric=rubric, context=context, response=response)
        scores[name] = _parse_score(judge.invoke(prompt).content)
    return scores


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="LLM-as-judge over generated briefs/alerts.")
    ap.add_argument("results", help="JSONL file with {context, response} records")
    args = ap.parse_args(argv)

    records = [json.loads(line) for line in open(args.results, encoding="utf-8") if line.strip()]
    judge = _get_judge()

    totals: dict[str, list[int]] = {m: [] for m in METRICS}
    for i, rec in enumerate(records, 1):
        s = judge_record(judge, rec.get("context", ""), rec.get("response", ""))
        for m, v in s.items():
            totals[m].append(v)
        print(f"[{i}/{len(records)}] " + "  ".join(f"{m[0].upper()}={v}" for m, v in s.items()))

    print("\n=== Averages (1-5) ===")
    for m, vals in totals.items():
        if vals:
            print(f"  {m:<13}: {mean(vals):.2f}")


if __name__ == "__main__":
    main()
