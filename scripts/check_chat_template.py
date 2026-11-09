"""Compare ChatTemplate::render with the model's own Jinja chat template.

usage: python scripts/check_chat_template.py <model.gguf>
needs: pip install gguf jinja2; builds and runs the `render_chat` example.

Assistant turns start with a space because that is how replies are generated, and how the engine
replays them. Differences mean the engine's format and the file's template disagree; some are known
and accepted (see the printed diff), so read them rather than treating this as pass/fail alone.
"""
import json
import os
import subprocess
import sys
import tempfile

from gguf import GGUFReader
from jinja2.sandbox import ImmutableSandboxedEnvironment


def msg(role, content):
    return {"role": role, "content": content}


CONVERSATIONS = [
    [msg("user", "Hello there")],
    [msg("user", "Hi"), msg("assistant", " Hello!"), msg("user", "What's 2+2?")],
    [msg("user", "a"), msg("assistant", " b"), msg("user", "c"), msg("assistant", " d"), msg("user", "e")],
    [msg("system", "Be brief."), msg("user", "Hi")],
]


def read_str(reader, key):
    field = reader.get_field(key)
    return None if field is None else field.contents()


def main(model):
    reader = GGUFReader(model)
    jinja = read_str(reader, "tokenizer.chat_template")
    if jinja is None:
        print("no chat template in file")
        return 0
    tokens = reader.get_field("tokenizer.ggml.tokens").contents()
    token = lambda key: tokens[reader.get_field(key).contents()] if reader.get_field(key) else ""
    bos, eos = token("tokenizer.ggml.bos_token_id"), token("tokenizer.ggml.eos_token_id")

    def raise_exception(text):
        raise ValueError(text)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = raise_exception
    template = env.from_string(jinja)

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(CONVERSATIONS, f)
    run = ["cargo", "run", "-q", "--release", "-p", "rli-models", "--example", "render_chat", "--", model, f.name]
    ours = json.loads(subprocess.run(run, check=True, capture_output=True, text=True).stdout)
    os.unlink(f.name)

    bad = 0
    for conversation, mine in zip(CONVERSATIONS, ours):
        try:
            ref = template.render(messages=conversation, bos_token=bos, eos_token=eos, add_generation_prompt=True)
        except ValueError as e:
            print(f"template rejects {[m['role'] for m in conversation]}: {e}")
            continue
        # The tokenizer adds BOS, not the renderer. Only strip it if the template wrote it: for
        # SmolLM2 the BOS token is <|im_start|>, which the template also uses as a marker.
        ref = ref.removeprefix(bos) if "bos_token" in jinja else ref
        if ref != mine:
            bad += 1
            print(f"DIFF {[m['role'] for m in conversation]}\n  template {ref!r}\n  ours     {mine!r}")
    print(f"{len(CONVERSATIONS) - bad}/{len(CONVERSATIONS)} match or are rejected by the template")
    return 1 if bad else 0


sys.exit(main(sys.argv[1]))
