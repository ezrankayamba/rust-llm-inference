"""Compare the engine's tokenizer with Hugging Face `tokenizers` on a tricky corpus.

usage: python scripts/check_tokenizer.py <model.gguf> <hf-repo-id>
       e.g. models/SmolLM2-135M-Instruct-Q8_0.gguf HuggingFaceTB/SmolLM2-135M-Instruct
needs: pip install tokenizers huggingface_hub; builds and runs the `tokenize` example.
"""
import json
import os
import subprocess
import sys
import tempfile

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

CORPUS = [
    "Hello world", "The quick brown fox jumps over the lazy dog.", "  leading and   multiple   spaces  ",
    "tabs\tand\r\nCRLF\n\n\nnewlines", "naïve café — 日本語のテキスト", "emoji 🎉🦀 ok!!",
    "I'm sure they'll say it's \"fine\", won't they?",
    "In 1990, 12345 people paid $3.14 each (≈ 1,000,000 total).", "def f(x):\n    return x**2  # square\n",
    "<html><body class=\"a\">Hi</body></html>", "<|im_start|>user\nHi<|im_end|>\n", "UPPER lower MiXeD",
    "a\n \nb", " ", "x" * 50, "Ελληνικά, русский, العربية, हिन्दी", "zero-width​joiner and  nbsp",
]

model, repo = sys.argv[1], sys.argv[2]
with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
    json.dump(CORPUS, f, ensure_ascii=False)
run = ["cargo", "run", "-q", "--release", "-p", "rli-models", "--example", "tokenize", "--", model, f.name]
ours = json.loads(subprocess.run(run, check=True, capture_output=True, text=True).stdout)
os.unlink(f.name)
ref_tok = Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json"))
bad = 0
for text, mine in zip(CORPUS, ours):
    ref = ref_tok.encode(text, add_special_tokens=False).ids
    if ref != mine:
        bad += 1
        print(f"MISMATCH {text!r}\n  hf   {ref}\n  ours {mine}")
print(f"{len(CORPUS) - bad}/{len(CORPUS)} match")
sys.exit(1 if bad else 0)
