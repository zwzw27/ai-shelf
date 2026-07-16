"""
AI Shelf study — runner.
Asks the same questions to 4 AI assistants and saves every answer + citations.

HOW TO USE:
  1. Make sure .env has your 4 keys (see manual, module 3).
  2. Leave PILOT = True and run:  python runner.py   (quick 5-query test)
  3. If the test looks good, change PILOT to False below and run again (full study, ~4 hrs).
You can stop and re-run any time — finished work is skipped automatically.
"""

from dotenv import load_dotenv; load_dotenv()
import os, csv, json, time, datetime, requests

# PILOT comes from the environment: PILOT=false python runner.py  (defaults to pilot mode)
PILOT = os.environ.get("PILOT", "true").lower() == "true"

# ---- check keys before doing anything ----
NEEDED = ["OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY"]
missing = [k for k in NEEDED if not os.environ.get(k)]
if missing:
    raise SystemExit(f"Missing keys in .env: {missing}. Open .env and paste them in.")

# ---- the assistants ----

from openai import OpenAI
oa = OpenAI()

def ask_openai(q):
    # If this errors with "model not found", check the current mini model name at
    # platform.openai.com/docs/models and change the line below.
    r = oa.responses.create(model="gpt-5-mini",
        tools=[{"type": "web_search"}], input=q)
    cites = [a.url for item in r.output if item.type == "message"
             for c in item.content for a in (getattr(c, "annotations", None) or [])]
    return r.output_text, cites

from google import genai
from google.genai import types
g = genai.Client()

def ask_gemini(q):
    # If this errors with "model not found", check the current Flash name in AI Studio.
    r = g.models.generate_content(model="gemini-3-flash-preview",
        contents=q,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())]))
    md = r.candidates[0].grounding_metadata
    uris = [c.web.uri for c in (md.grounding_chunks or []) if c.web]
    real = []
    for u in uris:  # Google returns redirect links; follow them to the real site
        try:
            real.append(requests.head(u, allow_redirects=True, timeout=20).url)
        except Exception:
            real.append(u)
    return r.text, real

import anthropic
cl = anthropic.Anthropic()

def ask_claude(q):
    r = cl.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
        tools=[{"type": "web_search_20260209", "name": "web_search", "max_uses": 3}],
        messages=[{"role": "user", "content": q}])
    text = "".join(b.text for b in r.content if b.type == "text")
    cites = {c.url for b in r.content if b.type == "text"
             for c in (getattr(b, "citations", None) or [])}
    return text, sorted(cites)

ASSISTANTS = {"openai": ask_openai,
              "gemini": ask_gemini, "claude": ask_claude}

# ---- the loop ----
rows = list(csv.DictReader(open("basket.csv")))
RUNS = 3
if PILOT:
    rows, RUNS = rows[:5], 1
    print("PILOT MODE: 5 queries, 1 run. Change PILOT to False for the full study.\n")

# resume support: load what's already done
out = json.load(open("raw.json")) if os.path.exists("raw.json") else []
done = {(r["query"], r["assistant"], r["run"]) for r in out if "error" not in r}

total = len(rows) * len(ASSISTANTS) * RUNS
i = 0
for run in range(RUNS):
    for row in rows:
        for name, fn in ASSISTANTS.items():
            i += 1
            if (row["query"], name, run) in done:
                continue
            try:
                text, cites = fn(row["query"])
                out.append({**row, "assistant": name, "run": run,
                            "ts": datetime.datetime.utcnow().isoformat(),
                            "citations": cites, "answer": text})
                print(f"[{i}/{total}] {name:<10} ok  {len(cites)} citations  {row['query'][:40]}")
            except Exception as e:
                out.append({**row, "assistant": name, "run": run, "error": str(e)})
                print(f"[{i}/{total}] {name:<10} ERROR  {e}")
            time.sleep(2)  # polite pause so no provider rate-limits us
        json.dump(out, open("raw.json", "w"), indent=1)  # progress is saved constantly

ok = sum(1 for r in out if "error" not in r)
print(f"\nDone. {ok} good records, {len(out)-ok} errors, saved to raw.json")
