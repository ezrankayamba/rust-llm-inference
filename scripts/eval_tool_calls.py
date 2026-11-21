"""Accuracy and latency of tool calls through a running rli-server, on short English, Swahili and
mixed utterances against a voice-app schema (navigate, fill_field, submit_form).

usage: python3 scripts/eval_tool_calls.py [--url http://127.0.0.1:8080] [--cases cases.json] [--think] [--save out.json]

A case is {"text": ..., "expect": [[name, {args}], ...], "optional": [[name, {args}], ...]}:
the reply must hold exactly the expected calls in any order, plus any of the optional ones.
Amounts and phone numbers are compared as digits only, other values case-insensitively. The
first request warms the server (it caches the system prompt and tools), then each case runs
once; latency is the whole HTTP request.
"""
import argparse
import json
import re
import statistics
import time
import urllib.request

SYSTEM = (
    "You operate a mobile loan app by calling tools. Pages: home, profile, settings, "
    "loan_application. The loan_application form has the fields amount, full_name and phone. "
    "The user may speak English, Swahili or both. Write amounts as digits "
    "(elfu = 1000, laki = 100000, milioni = 1000000) and phone numbers as digits only. "
    "If the request needs no action, answer briefly in the user's language."
)

TOOLS = [
    {"type": "function", "function": {
        "name": "navigate", "description": "Open a page of the app.",
        "parameters": {"type": "object", "properties": {
            "page": {"type": "string", "enum": ["home", "profile", "settings", "loan_application"]}},
            "required": ["page"], "additionalProperties": False},
        "strict": True}},
    {"type": "function", "function": {
        "name": "fill_field", "description": "Fill one field of a form.",
        "parameters": {"type": "object", "properties": {
            "form_id": {"type": "string", "enum": ["loan_application"]},
            "field_id": {"type": "string", "enum": ["amount", "full_name", "phone"]},
            "value": {"type": "string"}},
            "required": ["form_id", "field_id", "value"], "additionalProperties": False},
        "strict": True}},
    {"type": "function", "function": {
        "name": "submit_form", "description": "Submit a filled form.",
        "parameters": {"type": "object", "properties": {
            "form_id": {"type": "string", "enum": ["loan_application"]}},
            "required": ["form_id"], "additionalProperties": False},
        "strict": True}},
]

F = "loan_application"
OPEN = [["navigate", {"page": F}]]


def fill(field, value):
    return ["fill_field", {"form_id": F, "field_id": field, "value": value}]


CASES = [
    {"text": "Nifungulie fomu ya mkopo, jina ni Asha Juma, kiasi laki tano",
     "expect": OPEN + [fill("full_name", "Asha Juma"), fill("amount", "500000")]},
    {"text": "Open the loan application and put my phone as 0712345678",
     "expect": OPEN + [fill("phone", "0712345678")]},
    {"text": "Nenda kwenye ukurasa wa mwanzo", "expect": [["navigate", {"page": "home"}]]},
    {"text": "Take me to my profile", "expect": [["navigate", {"page": "profile"}]]},
    {"text": "Mipangilio tafadhali", "expect": [["navigate", {"page": "settings"}]]},
    {"text": "Jina langu ni John Mwangi", "expect": [fill("full_name", "John Mwangi")], "optional": OPEN},
    {"text": "Weka kiasi elfu hamsini", "expect": [fill("amount", "50000")], "optional": OPEN},
    {"text": "Change the amount to two million shillings", "expect": [fill("amount", "2000000")],
     "optional": OPEN},
    {"text": "Namba yangu ya simu ni 0754 111 222", "expect": [fill("phone", "0754111222")],
     "optional": OPEN},
    {"text": "Tuma fomu", "expect": [["submit_form", {"form_id": F}]]},
    {"text": "Nataka mkopo wa shilingi milioni moja, jina ni Fatuma Ali, then submit it",
     "expect": [fill("amount", "1000000"), fill("full_name", "Fatuma Ali"),
                ["submit_form", {"form_id": F}]], "optional": OPEN},
    {"text": "Habari, unaweza kunisaidia nini?", "expect": []},
]


def norm(call):
    name, args = call
    out = {}
    for k, v in args.items():
        v = str(v).strip()
        digits = args.get("field_id") in ("amount", "phone") and k == "value"
        out[k] = re.sub(r"\D", "", v) if digits else v.lower()
    return json.dumps([name, out], sort_keys=True)


def post(url, text, max_tokens, think=False):
    body = {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}],
            "tools": TOOLS, "tool_choice": "auto", "temperature": 0, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": think}}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t = time.perf_counter()
    with urllib.request.urlopen(req) as r:
        resp = json.load(r)
    return resp, time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--cases")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--think", action="store_true", help="let Qwen3 reason before answering")
    ap.add_argument("--save", help="write every reply's calls and text here, to compare runs")
    a = ap.parse_args()
    cases = json.load(open(a.cases)) if a.cases else CASES
    post(a.url, "hello", 8)
    right, secs, toks, saved = 0, [], [], []
    for c in cases:
        resp, dt = post(a.url, c["text"], a.max_tokens, a.think)
        msg = resp["choices"][0]["message"]
        got = [norm([t["function"]["name"], json.loads(t["function"]["arguments"])])
               for t in msg.get("tool_calls") or []]
        want = sorted(norm(x) for x in c["expect"])
        optional = {norm(x) for x in c.get("optional", [])}
        ok = sorted(g for g in got if g not in optional or g in want) == want
        right += ok
        secs.append(dt)
        toks.append(resp["usage"]["completion_tokens"])
        saved.append({"text": c["text"], "calls": [[t["function"]["name"], t["function"]["arguments"]]
                                                   for t in msg.get("tool_calls") or []],
                      "content": msg.get("content"), "ok": ok})
        print(f"{'ok  ' if ok else 'MISS'} {dt:5.2f}s {toks[-1]:3d} tok  {c['text']}")
        if not ok:
            print(f"       got {got or msg.get('content')!r}")
    if a.save:
        json.dump(saved, open(a.save, "w"), ensure_ascii=False, indent=1)
    print(f"\n{right}/{len(cases)} right; latency mean {statistics.mean(secs):.2f}s, "
          f"median {statistics.median(secs):.2f}s, max {max(secs):.2f}s; "
          f"reply tokens mean {statistics.mean(toks):.0f}")


if __name__ == "__main__":
    main()
