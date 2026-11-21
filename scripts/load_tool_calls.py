"""Latency of tools requests when several users talk to one rli-server at once.

usage: python3 scripts/load_tool_calls.py [--url http://127.0.0.1:8080] [--users 1,2,4,8]
       [--requests 12] [--cases scripts/tool_cases.json] [--digits]

For each user count, that many threads each send --requests requests in a row (one user
speaking one utterance after another), every thread starting at a different case, all with the
system prompt and tools of eval_tool_calls.py. Prints the median, 90th percentile and worst
latency over all requests, the requests per second, and how many prompt tokens a request had to
evaluate (the rest came from the KV cache the users share).
"""
import argparse
import json
import statistics
import threading
import time
import urllib.request

from eval_tool_calls import SYSTEM, TOOLS
from number_words import to_digits


def post(url, text):
    body = {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}],
            "tools": TOOLS, "tool_choice": "auto", "temperature": 0, "max_tokens": 512,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        resp = json.load(r)
    return time.perf_counter() - t, resp.get("timings", {}).get("prompt_n", 0)


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--users", default="1,2,4,8")
    ap.add_argument("--requests", type=int, default=12, help="requests per user")
    ap.add_argument("--cases", default="scripts/tool_cases.json")
    ap.add_argument("--digits", action="store_true")
    a = ap.parse_args()
    texts = [c["text"] for c in json.load(open(a.cases)) if c["expect"]]
    if a.digits:
        texts = [to_digits(t) for t in texts]
    post(a.url, "hello")
    print("users  requests  median     p90     max   req/s  prompt tokens evaluated (median, max)")
    for n in (int(x) for x in a.users.split(",")):
        lat, evald, lock = [], [], threading.Lock()

        def user(u):
            for i in range(a.requests):
                dt, pn = post(a.url, texts[(u * 7 + i) % len(texts)])
                with lock:
                    lat.append(dt)
                    evald.append(pn)

        threads = [threading.Thread(target=user, args=(u,)) for u in range(n)]
        t = time.perf_counter()
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        wall = time.perf_counter() - t
        print(f"{n:5d}  {len(lat):8d}  {statistics.median(lat):6.2f}s {pct(lat, 0.9):6.2f}s "
              f"{max(lat):6.2f}s  {len(lat) / wall:5.2f}  {statistics.median(evald):.0f}, {max(evald)}")


if __name__ == "__main__":
    main()
