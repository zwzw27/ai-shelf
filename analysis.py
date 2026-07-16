"""
AI Shelf study — analysis.
Reads raw.json + labels.json and prints the study's headline numbers.

Run:  python analysis.py
"""
import pandas as pd, json
from urllib.parse import urlparse

raw = json.load(open("raw.json")); labels = json.load(open("labels.json"))
df = (pd.DataFrame([r for r in raw if "citations" in r])
        .explode("citations").dropna(subset=["citations"]))
df["domain"] = df.citations.map(lambda u: urlparse(u).netloc.removeprefix("www."))
df["label"] = df.domain.map(labels)

print(f"\nTotal citations analysed: {len(df)}\n")

print("H1 — HEADLINE: open-web editorial share of all citations")
print(f"   {round((df.label == 'open_editorial').mean() * 100, 1)}%\n")

print("Scoreboard — open-web share by market and industry")
print(df.groupby(["market", "industry"]).label
        .apply(lambda s: round((s == "open_editorial").mean() * 100, 1)), "\n")

print("Source split — where citations come from overall")
print((df.label.value_counts(normalize=True) * 100).round(1), "\n")

print("Publisher leaderboard — most-cited domains")
print(df.domain.value_counts().head(10), "\n")

print("H6 — self-preferencing check: % of citations that are YouTube, per assistant")
print((df.assign(yt=df.domain.str.contains("youtube", na=False))
         .groupby("assistant").yt.mean() * 100).round(1))
