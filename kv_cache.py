"""Task 1: KV-cache size of Qwen2.5-7B at 8k and 32k tokens.

Usage:
    python3 kv_cache.py                         # print the working
    python3 kv_cache.py --key 448MiB,1.75GiB    # also compare with the answer key (10% rule)
"""
import argparse
import re

# From https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json
CONFIG = {
    "num_hidden_layers": 28,
    "num_attention_heads": 28,
    "num_key_value_heads": 4,   # grouped-query attention: 7 query heads share each KV head
    "hidden_size": 3584,
    "torch_dtype": "bfloat16",  # 2 bytes per value
}
BYTES_PER_VALUE = 2
CONTEXTS = {"8k": 8 * 1024, "32k": 32 * 1024}
UNITS = {"B": 1, "KB": 1e3, "MB": 1e6, "GB": 1e9, "KIB": 2**10, "MIB": 2**20, "GIB": 2**30}


def bytes_per_token(cfg=CONFIG, bytes_per_value=BYTES_PER_VALUE):
    head_dim = cfg["hidden_size"] // cfg["num_attention_heads"]
    # 2 = one key vector and one value vector per KV head per layer
    return 2 * cfg["num_hidden_layers"] * cfg["num_key_value_heads"] * head_dim * bytes_per_value


def kv_cache_bytes(tokens, cfg=CONFIG):
    return tokens * bytes_per_token(cfg)


def parse_size(text):
    m = re.fullmatch(r"\s*([\d.]+)\s*([KMG]?i?B)\s*", text, re.IGNORECASE)
    if not m:
        raise ValueError(f"cannot parse size {text!r}; use e.g. 448MiB or 0.47GB")
    return float(m.group(1)) * UNITS[m.group(2).upper()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", help="answer key for 8k,32k, e.g. 448MiB,1.75GiB")
    args = ap.parse_args()

    c = CONFIG
    head_dim = c["hidden_size"] // c["num_attention_heads"]
    per_tok = bytes_per_token()
    print("Qwen2.5-7B, batch 1, bf16 cache")
    print(f"  head_dim = hidden_size / attention_heads = {c['hidden_size']} / {c['num_attention_heads']} = {head_dim}")
    print(f"  per token = 2 (K,V) x {c['num_hidden_layers']} layers x {c['num_key_value_heads']} KV heads"
          f" x {head_dim} dims x {BYTES_PER_VALUE} bytes = {per_tok:,} bytes ({per_tok / 2**10:.0f} KiB)")
    results = {}
    for name, n in CONTEXTS.items():
        b = kv_cache_bytes(n)
        results[name] = b
        print(f"  {name:>3} = {n:,} tokens x {per_tok:,} B = {b:,} B"
              f" = {b / 2**20:,.0f} MiB = {b / 2**30:.2f} GiB = {b / 1e9:.2f} GB")
    no_gqa = per_tok * c["num_attention_heads"] // c["num_key_value_heads"]
    print(f"  (without GQA, 28 KV heads: {no_gqa * CONTEXTS['32k'] / 2**30:.2f} GiB at 32k, 7x more)")

    if args.key:
        keys = [parse_size(k) for k in args.key.split(",")]
        if len(keys) != 2:
            ap.error("--key needs two sizes: 8k,32k")
        print("\nComparison with answer key (pass: within 10%)")
        ok = True
        for (name, mine), key in zip(results.items(), keys):
            diff = (mine - key) / key
            passed = abs(diff) <= 0.10
            ok &= passed
            print(f"  {name:>3}: mine {mine / 2**30:.3f} GiB, key {key / 2**30:.3f} GiB,"
                  f" diff {diff:+.1%} -> {'PASS' if passed else 'FAIL'}")
        print("  overall:", "PASS" if ok else "FAIL")


# self-check: 57,344 B/token is the widely quoted figure for Qwen2.5-7B
assert bytes_per_token() == 57_344
assert kv_cache_bytes(8192) == 448 * 2**20
assert abs(parse_size("1.75GiB") - kv_cache_bytes(32768)) < 1

if __name__ == "__main__":
    main()
