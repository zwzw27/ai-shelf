"""
AI Shelf study — classifier.
Reads raw.json, finds every cited website, and asks Claude to sort each
domain into one bucket (open_editorial, ugc_forum, video, vendor,
retailer, news_wire, other). Saves the result to labels.json.

Run:  python classify.py
"""
from dotenv import load_dotenv; load_dotenv()
from urllib.parse import urlparse
import json, re, anthropic

out = json.load(open("raw.json"))
domains = sorted({urlparse(u).netloc.removeprefix("www.")
                  for r in out for u in r.get("citations", []) if u})
print(f"{len(domains)} unique domains to classify")

TAXONOMY = """Labels: open_editorial | ugc_forum | video | vendor | retailer | news_wire | other
Rules: Amazon product pages = retailer. Publisher commerce arms
(e.g. Wirecutter) = open_editorial. Affiliate-first content farms = other."""

cl = anthropic.Anthropic()
labels = {}
for i in range(0, len(domains), 50):
    batch = domains[i:i+50]
    r = cl.messages.create(model="claude-haiku-4-5", max_tokens=2000,
        messages=[{"role": "user", "content":
            f"Classify each domain. {TAXONOMY}\nDomains: {batch}\n"
            "Reply with only JSON mapping domain to label."}])
    text = r.content[0].text
    text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.M)  # remove code fences if present
    labels.update(json.loads(text))
    print(f"  labeled {min(i+50, len(domains))}/{len(domains)}")

json.dump(labels, open("labels.json", "w"), indent=1)
print(f"Done. {len(labels)} domains labeled, saved to labels.json")
print("Next: hand-check a 10% sample (see manual, module 6, step 3).")
