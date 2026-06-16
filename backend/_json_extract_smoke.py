"""Test the JSON extractor against the failure modes models actually produce."""
from backend.decision_engine import _extract_json_object as P

cases = [
    ("plain",
     '{"action":"ENTER_LONG","confidence":0.7}'),
    ("trailing prose (the Haiku failure mode)",
     '{"action":"HOLD","confidence":0.4}\n\nI recommend HOLD because the spread is wide.'),
    ("leading prose + json",
     'Here is my decision:\n\n{"action":"REDUCE","confidence":0.6}\n\nThe risk overlay is satisfied.'),
    ("markdown fence",
     '```json\n{"action":"CLOSE","confidence":0.5}\n```'),
    ("fence + trailing prose",
     '```json\n{"action":"ENTER_SHORT"}\n```\n\nNote: low confidence call.'),
    ("nested object",
     '{"action":"ENTER_LONG","components":{"alpha":0.3,"vol":0.02}}'),
    ("nested + trailing",
     '{"action":"ADD","components":{"alpha":0.3}}\n\nADD makes sense here.'),
    ("very long trailing prose",
     '{"action":"HOLD"}\n\n' + ("This is a long explanation. " * 50)),
]
for name, t in cases:
    try:
        result = P(t)
        action = result.get("action", "?")
        print(f"  {name:<42} -> action={action}  (keys: {sorted(result.keys())})")
    except Exception as e:
        print(f"  {name:<42} FAIL: {e}")
