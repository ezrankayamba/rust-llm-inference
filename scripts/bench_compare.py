#!/usr/bin/env python3
"""Compare this engine with llama.cpp on one machine and model, on the CPU, as a Markdown table.

usage: bench_compare.py --bin-dir <dir with rli and examples/bench> --llama-dir <dir with llama-bench
       and llama-completion> --model <model.gguf> [--threads 1,2] [--pp 256] [--tg 64] [--reps 3]

Throughput comes from each project's own benchmark. First token and memory come from fresh
processes with the same prompt, with the file already in the page cache: first token is the whole
run of a one-token generation, including loading and exit. Peak RSS counts mapped weights, which the
OS can evict, so Linux also reports the peak of anonymous memory, which it can't evict without swap.
"""

import argparse
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import threading
import time

TEXT = (
    "The history of computing is a story of steadily shrinking machines and growing ambitions. "
    "Early computers filled rooms and consumed enormous power, yet performed fewer operations per "
    "second than a modern wristwatch. Over the decades, transistors replaced vacuum tubes, "
    "integrated circuits replaced discrete components, and software grew from hand-written machine "
    "code into vast layered systems. "
)
PROMPT = TEXT * 3
CTX = 512
COLD_REPS = 3


def rss_unit():
    # ru_maxrss is in bytes on macOS and KiB on Linux.
    return 1 if sys.platform == "darwin" else 1024


def anon_kib(pid):
    try:
        with open(f"/proc/{pid}/status") as f:
            return int(re.search(r"RssAnon:\s+(\d+)", f.read()).group(1))
    except (OSError, AttributeError):
        return 0  # Not Linux, or the process just exited.


def measure(cmd):
    """Runs cmd and returns (wall seconds, peak RSS in MiB, peak anonymous MiB or None)."""
    start = time.perf_counter()
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    peak_anon = [0]
    done = threading.Event()

    def poll():
        while not done.wait(0.02):
            peak_anon[0] = max(peak_anon[0], anon_kib(p.pid))

    poller = threading.Thread(target=poll)
    poller.start()
    err = []
    drain = threading.Thread(target=lambda: err.append(p.stderr.read()))
    drain.start()
    _, status, usage = os.wait4(p.pid, 0)
    wall = time.perf_counter() - start
    p.returncode = os.waitstatus_to_exitcode(status)
    done.set()
    poller.join()
    drain.join()
    if p.returncode != 0:
        sys.exit(f"{' '.join(cmd)} failed:\n{err[0].decode(errors='replace')[-2000:]}")
    anon = peak_anon[0] / 1024 if sys.platform == "linux" else None
    return wall, usage.ru_maxrss * rss_unit() / 2**20, anon


def ours_throughput(a, t, extra=()):
    out = subprocess.run(
        [f"{a.bin_dir}/examples/bench", a.model, "--pp", str(a.pp), "--tg", str(a.tg),
         "--reps", str(a.reps), "--threads", str(t), *extra],
        check=True, capture_output=True, text=True,
    ).stdout
    pp = re.search(rf"pp{a.pp}:\s*([\d.]+)", out)
    tg = re.search(rf"tg{a.tg}:\s*([\d.]+)", out)
    return float(pp.group(1)), float(tg.group(1))


def llama_throughput(a, t):
    out = subprocess.run(
        [f"{a.llama_dir}/llama-bench", "-m", a.model, "-p", str(a.pp), "-n", str(a.tg),
         "-r", str(a.reps), "-t", str(t), "-ngl", "0", "-o", "json"],
        check=True, capture_output=True, text=True,
    ).stdout
    rows = json.loads(out)
    pp = next(r for r in rows if r["n_prompt"] == a.pp)
    tg = next(r for r in rows if r["n_gen"] == a.tg)
    return pp["avg_ts"], tg["avg_ts"]


def ours_run(a, t, n, extra=()):
    return [f"{a.bin_dir}/rli", "-m", a.model, "-p", PROMPT, "--raw", "-n", str(n), "-c", str(CTX),
            "--temp", "0", "--gpu-layers", "0", "-t", str(t), *extra]


def with_repack(f):
    return lambda *args: f(*args, extra=["--repack"])


def llama_run(a, t, n):
    return [f"{a.llama_dir}/llama-completion", "-m", a.model, "-p", PROMPT, "-n", str(n),
            "-c", str(CTX), "--temp", "0", "-ngl", "0", "-t", str(t), "--no-display-prompt"]


def cold(run, a, t):
    """Returns (median seconds to first token, peak RSS MiB, peak anonymous MiB) across runs."""
    first = [measure(run(a, t, 1))[0] for _ in range(COLD_REPS)]
    _, rss, anon = measure(run(a, t, a.tg))
    return statistics.median(first), rss, anon


def cpu_name():
    try:
        if sys.platform == "darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True).stdout.strip()
        with open("/proc/cpuinfo") as f:
            return re.search(r"model name\s*:\s*(.+)", f.read()).group(1)
    except (OSError, AttributeError):
        return platform.processor()


def fmt_anon(x):
    return "n/a" if x is None else f"{x:.0f}"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bin-dir", required=True)
    p.add_argument("--llama-dir", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--threads", default="1,2")
    p.add_argument("--pp", type=int, default=256)
    p.add_argument("--tg", type=int, default=64)
    p.add_argument("--reps", type=int, default=3)
    a = p.parse_args()

    print(f"### {os.path.basename(a.model)}\n")
    print(f"{cpu_name()}, {os.cpu_count()} logical CPUs. Prompt {a.pp} tokens, decode {a.tg}, "
          f"{a.reps} reps; first token and memory from a prompt of about 210 tokens.\n")
    print("| Threads | Engine | Prompt tok/s | Decode tok/s | First token s | Peak RSS MiB "
          "| Peak anon MiB |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for t in (int(x) for x in a.threads.split(",")):
        engines = [("rli", ours_throughput, ours_run),
                   ("llama.cpp", llama_throughput, llama_run)]
        if platform.machine() in ("x86_64", "AMD64"):  # the flag does nothing elsewhere
            engines.insert(1, ("rli --repack", with_repack(ours_throughput), with_repack(ours_run)))
        for name, tp, run in engines:
            pp, tg = tp(a, t)
            first, rss, anon = cold(run, a, t)
            print(f"| {t} | {name} | {pp:.1f} | {tg:.1f} | {first:.2f} | {rss:.0f} "
                  f"| {fmt_anon(anon)} |", flush=True)
    print()


if __name__ == "__main__":
    main()
