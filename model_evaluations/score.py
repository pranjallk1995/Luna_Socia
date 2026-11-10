"""Score old.jsonl against new.jsonl: Luna's texting style, her create_image calls, failures, time.

usage: python score.py <dir with old.jsonl and new.jsonl> [--scenarios scenarios.json]
Needs only the Python standard library. Writes scores.json and report.md to the same
directory. The checks follow the persona's rules; whether she sounds like Luna is still
for a person to judge, so read report.md's replies.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean, median

parser = argparse.ArgumentParser()
parser.add_argument("dir", type=Path)
parser.add_argument("--scenarios", type=Path, default=Path(__file__).with_name("scenarios.json"))
args = parser.parse_args()

SCENARIOS = {s["id"]: s for s in json.loads(args.scenarios.read_text())}
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿❤✨]")
# A sentence ends at . ! ? … or a run of emoji, followed by a space or the end.
SENTENCE_BREAK = re.compile(r"(?:[.!?…]|[\U0001F300-\U0001FAFF\u2600-\u27BF\u2764\u2728\uFE0F\u200D])+(?:\s+|$)")
LIST_LINE = re.compile(r"^\s*([-*•]|\d+[.)])\s", re.M)
SOUNDS_LIKE_AI = re.compile(
    r"\b(as an ai|language model|ai assistant|i'?m an ai|i am an ai|virtual assistant|how (can|may) i (help|assist))\b"
    r"|def \w+\(|```", re.I)
DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def sentences(text: str) -> int:
    return sum(bool(re.search(r"\w", piece)) for piece in SENTENCE_BREAK.split(text))


def arguments(call: dict) -> dict:
    value = (call.get("function") or {}).get("arguments") or {}
    return json.loads(value) if isinstance(value, str) else value


def check(row: dict) -> dict:
    scenario = SCENARIOS[row["id"]]
    if "crash" in row:
        return {"crash": row["crash"], "seconds": row["seconds"]}
    text = row["text"].strip()
    calls = [c for c in row["tool_calls"] if (c.get("function") or {}).get("name") == "create_image"]
    result = {
        "tool_right": bool(calls) == (scenario["tool"] == "create_image"),
        "empty": not text and not calls,
        "thinking_shown": "<think>" in text,
        "sounds_like_ai": bool(SOUNDS_LIKE_AI.search(text)),
        "list_or_essay": bool(LIST_LINE.search(text)) or sentences(text) > 3,
        "too_many_emoji": len(EMOJI.findall(text)) > 3,
        "chars": len(text),
        "seconds": row["seconds"],
    }
    if calls:
        call = arguments(calls[0])
        result["arguments_right"] = bool(str(call.get("description") or "").strip()) and all(
            bool(call.get(key)) == scenario[key] for key in ("include_luna", "use_their_photo") if key in scenario
        )
    if scenario.get("script") == "devanagari":
        result["language_right"] = bool(DEVANAGARI.search(text))
    if scenario.get("mentions"):
        result["on_topic"] = any(word in text.lower() for word in scenario["mentions"])
    return result


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


runs = {label: load(args.dir / f"{label}.jsonl") for label in ("old", "new")}
checked = {label: [{**row, "check": check(row)} for row in rows] for label, rows in runs.items()}

GOOD = ("tool_right", "arguments_right", "language_right", "on_topic")
BAD = ("empty", "thinking_shown", "sounds_like_ai", "list_or_essay", "too_many_emoji")


def summarize(rows: list[dict]) -> dict:
    ok = [r["check"] for r in rows if "crash" not in r["check"]]
    summary = {"model": rows[0]["model"], "replies": len(rows), "failures": len(rows) - len(ok)}
    for key in GOOD:
        values = [c[key] for c in ok if key in c]
        summary[key] = f"{sum(values)}/{len(values)}"
    for key in BAD:
        summary[key] = sum(c[key] for c in ok)
    summary["chars_median"] = median(c["chars"] for c in ok) if ok else None
    summary["mean_seconds"] = round(mean(r["check"]["seconds"] for r in rows), 1)
    return summary


summary = {label: summarize(rows) for label, rows in checked.items()}
misses = {label: Counter(r["id"] for r in rows if "crash" in r["check"] or not all(
    r["check"].get(k, True) for k in GOOD) or any(r["check"].get(k) for k in BAD)) for label, rows in checked.items()}
summary["scenarios_with_misses"] = {label: dict(count) for label, count in misses.items()}
(args.dir / "scores.json").write_text(json.dumps({"summary": summary, "replies": checked}, indent=1, ensure_ascii=False))

lines = ["# Socia chat evaluation", "", "```json", json.dumps(summary, indent=1, ensure_ascii=False), "```", ""]
for scenario_id, scenario in SCENARIOS.items():
    lines += [f"## {scenario_id} (expects {scenario['tool']})", "", f"> {scenario['messages'][-1]['content']}", ""]
    for label, rows in checked.items():
        for row in (r for r in rows if r["id"] == scenario_id):
            calls = [arguments(c) for c in row.get("tool_calls", [])]
            flags = [k for k in BAD if row["check"].get(k)] + [k for k in GOOD if row["check"].get(k) is False]
            text = row.get("crash") or row["text"].strip().replace("\n", " ")
            lines.append(f"- {label}: {text}" + (f" — create_image {json.dumps(calls[0], ensure_ascii=False)}" if calls else "")
                         + (f" **{', '.join(flags)}**" if flags else ""))
    lines.append("")
(args.dir / "report.md").write_text("\n".join(lines))
print(json.dumps(summary, indent=1, ensure_ascii=False))
