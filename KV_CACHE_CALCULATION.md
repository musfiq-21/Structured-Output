# Task 1: KV-cache size of Qwen2.5-7B (hand calculation)

---

## Given

From `Qwen/Qwen2.5-7B-Instruct/config.json`:

| Symbol | Meaning | Value |
|---|---|---|
| L | number of layers (`num_hidden_layers`) | 28 |
| H_q | query heads (`num_attention_heads`) | 28 |
| H_kv | key/value heads (`num_key_value_heads`) | 4 |
| d_model | hidden size (`hidden_size`) | 3584 |
| b | bytes per value (bf16) | 2 |
| T | context length (tokens) | 8,192 and 32,768 |

Assume batch size = 1. Count the KV cache only (not the weights).

---

## Formula

```
KV cache = 2 × L × H_kv × d_head × b × T
           ↑
           one K vector + one V vector per token
```

---

## Step 1: Head dimension

```
d_head = d_model / H_q
       = 3584 / 28
       = 128
```

---

## Step 2: Bytes per token

```
bytes/token = 2 × L  × H_kv × d_head × b
            = 2 × 28 × 4    × 128    × 2

  2 × 28     =     56
  56 × 4     =    224
  224 × 128  = 28,672
  28,672 × 2 = 57,344 B
```

```
57,344 B / 1024 = 56 KiB per token
```

---

## Step 3: 8k context (T = 8,192)

```
KV = 57,344 × 8,192
   = 469,762,048 B

in MiB:  469,762,048 / 1024 / 1024 = 448 MiB
in GB:   469,762,048 / 10^9        ≈ 0.47 GB
```

Check: 56 KiB × 8,192 = 458,752 KiB, and 458,752 / 1024 = **448 MiB** ✓

> **Answer (8k): 448 MiB ≈ 0.47 GB**

---

## Step 4: 32k context (T = 32,768)

```
KV = 57,344 × 32,768
   = 1,879,048,192 B

in MiB:  1,879,048,192 / 1024 / 1024 = 1,792 MiB
in GiB:  1,792 / 1024                = 1.75 GiB
in GB:   1,879,048,192 / 10^9        ≈ 1.88 GB
```

Check: 32k is 4 × 8k, so 4 × 448 MiB = **1,792 MiB = 1.75 GiB** ✓

> **Answer (32k): 1.75 GiB ≈ 1.88 GB**

---

## Step 5: Why GQA matters (comparison)

Without GQA, every query head would have its own K/V head (H_kv = 28 instead of 4):

```
ratio = 28 / 4 = 7×

 8k:   448 MiB × 7 = 3,136 MiB  ≈ 3.06 GiB
32k: 1,792 MiB × 7 = 12,544 MiB = 12.25 GiB
```

Grouped-query attention saves 7× memory.

---

## Step 6: Sanity check against a real measurement

Use the same formula on `qwen2.5:3b` (L = 36, H_kv = 2, d_head = 128, b = 2) at T = 4,096:

```
bytes/token = 2 × 36 × 2 × 128 × 2 = 36,864 B

KV = 36,864 × 4,096
   = 150,994,944 B
   = 150,994,944 / 1024 / 1024
   = 144 MiB
```

Ollama's server log for this model reported a KV cache of 144.00 MiB. The formula matches the measurement exactly ✓

---

## Summary

| Context | Bytes | MiB / GiB | GB |
|---|---|---|---|
| 8k (8,192) | 469,762,048 | 448 MiB | 0.47 |
| 32k (32,768) | 1,879,048,192 | 1.75 GiB | 1.88 |
| 32k without GQA | 13,153,337,344 | 12.25 GiB | 13.15 |

1 MiB = 1024² B and 1 GB = 10⁹ B. Batch size 1; model weights (about 15 GB in bf16) are extra.
