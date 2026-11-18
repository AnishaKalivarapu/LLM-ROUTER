"""Offline evaluation of the heuristic classifier on a small hand-labelled set.
No API keys needed.   python -m scripts.eval_classifier

This is a sanity check, not a benchmark: 45 queries labelled by hand. Replace
data/eval_queries.json with real traffic samples for meaningful numbers."""
import json
import pathlib
from collections import Counter

from app.classifiers.heuristic import HeuristicClassifier
from app.config import get_settings

LABELS = ["simple", "moderate", "complex"]
data = json.loads((pathlib.Path(__file__).parent.parent / "data" / "eval_queries.json").read_text())
h = HeuristicClassifier()
threshold = get_settings().confidence_threshold

confusion = Counter()
correct = confident = confident_correct = 0
for row in data:
    r = h.classify_sync(row["query"])
    pred = r.level.label
    confusion[(row["label"], pred)] += 1
    correct += pred == row["label"]
    if r.confidence >= threshold:
        confident += 1
        confident_correct += pred == row["label"]

n = len(data)
print(f"Heuristic accuracy:            {correct}/{n} = {correct / n:.0%}")
print(f"Handled w/o LLM (conf>={threshold}):  {confident}/{n} = {confident / n:.0%}  "
      f"(accuracy on those: {confident_correct / max(confident, 1):.0%})")
print(f"Would be sent to LLM in hybrid: {n - confident}/{n} = {(n - confident) / n:.0%}\n")
print("confusion (rows=true, cols=predicted)")
print(f"{'':10}" + "".join(f"{l:>10}" for l in LABELS))
for t in LABELS:
    print(f"{t:10}" + "".join(f"{confusion[(t, p)]:>10}" for p in LABELS))
print("\nmisclassified:")
for row in data:
    r = h.classify_sync(row["query"])
    if r.level.label != row["label"]:
        print(f"  [{row['label']} -> {r.level.label} @ {r.confidence}] {row['query'][:80]!r}")
