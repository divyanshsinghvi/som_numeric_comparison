import json, random, os, math
from statistics import mean
import numpy as np

random.seed(2025)

# OUT_PATH = "/mnt/d/code/open_source/som_numeric_comparison/data/gemma_numeric_ab_dataset.jsonl"
OUT_PATH = "gemma_numeric_ab_dataset.jsonl"
STRING_OUT_PATH = "gemma_string_ab_dataset.jsonl"
N_TOTAL = 2000
N_PER_BUCKET = N_TOTAL // 4  # 500 each


def numeric_truth(a: str, b: str, mode: str) -> str:
    if mode == 'yn':
        return "Yes" if float(a) > float(b) else "No"
    elif mode == 'tf':
        return "Truth" if float(a) > float(b) else "False"
    else:
        return "A" if float(a) > float(b) else "B"

def lex_compare(a: str, b: str, mode: str) -> str:
    if mode == 'yn':
        return "Yes" if a > b else "No"
    elif mode == 'tf':
        return "Truth" if a > b else "False"
    else:
        return "A" if a > b else "B"
    


def prompt_mcq(a: str, b: str) -> str:
    return f"Which is larger?\nA: {a}\nB: {b}\nAnswer with A or B only."

def prompt_simple(a: str, b: str) -> str:
    return f"Question: Is {a} > {b}? Answer:"


def make_equal_length_pair(low=0, high=9999):
    x = random.randint(low, high)
    digits = len(str(x))
    # Pick y with same number of digits, but different from x
    while True:
        y = random.randint(10**(digits-1), 10**digits - 1)
        if y != x:
            break
    return str(x), str(y)

def make_unequal_length_pair(low=0, high=9999):
    x = random.randint(low, high)
    digits = len(str(x))
    # Force y to have different number of digits
    possible_lengths = [d for d in range(1, len(str(high)) + 1) if d != digits]
    target_digits = random.choice(possible_lengths)
    y = random.randint(10**(target_digits-1), 10**target_digits - 1)

    if random.random() < 0.5:
        x, y = y, x
    return str(x), str(y)


def make_decimal_equal_len(int_low=0, int_high=99, frac_digits=2):
    ip = random.randint(int_low, int_high)
    f1 = random.randint(0, 10**frac_digits - 1)
    f2 = random.randint(0, 10**frac_digits - 1)
    while f2 == f1:
        f2 = random.randint(0, 10**frac_digits - 1)
    a = f"{ip}.{f1:0{frac_digits}d}"
    b = f"{ip}.{f2:0{frac_digits}d}"
    return a, b

def make_decimal_diff_len(int_low=0, int_high=99, len_options=(1,2,3)):
    ip = random.randint(int_low, int_high)
    l1, l2 = random.sample(len_options, 2)
    f1 = "".join(random.choice("0123456789") for _ in range(l1))
    f2 = "".join(random.choice("0123456789") for _ in range(l2))
    # avoid accidental equality
    while f2 == f1:
        f2 = "".join(random.choice("0123456789") for _ in range(l2))
    a = f"{ip}.{f1}"
    b = f"{ip}.{f2}"
    return a, b


def gen_bucket(kind: str, n: int):
    items = []
    for i in range(n):
        if kind == "integers_equal_len":
            a,b = make_equal_length_pair()
        elif kind == "integers_diff_len":
            a,b = make_unequal_length_pair()
        elif kind == "decimals_equal_len":
            a,b = make_decimal_equal_len(frac_digits=random.choice([2,3]))
        elif kind == "decimals_diff_len":
            a,b = make_decimal_diff_len(len_options=(1,2,3))
        else:
            raise ValueError("unknown kind")
        # ensure they aren't numerically equal
        while float(a) == float(b):
            if kind == "integers_equal":
                a,b = make_equal_length_pair()
            elif kind == "integers_diff_len":
                a,b = make_unequal_length_pair()
            elif kind == "decimals_equal_len":
                a,b = make_decimal_equal_len(frac_digits=random.choice([2,3]))
            else:
                a,b = make_decimal_diff_len(len_options=(1,2,3))

        item = {
            "id": f"{kind}-{i}",
            "pair_type": kind,
            "a": a, "b": b,
            "numeric_truth": numeric_truth(a,b, 'ab'),         # 'A' or 'B'
            "numeric_truth_yn": numeric_truth(a,b, 'yn'),         # 'A' or 'B'
            "numeric_truth_tf": numeric_truth(a,b,'tf'),         # 'A' or 'B'
            "lex_truth": lex_compare(a,b, 'ab'),               # 'A'/'B'/eq
            "lex_truth_yn": lex_compare(a,b,'yn'),               # 'A'/'B'/eq
            "lex_truth_tf": lex_compare(a,b, 'tf'),               # 'A'/'B'/eq
            "prompt_mcq": prompt_mcq(a,b),
            "prompt_simple": prompt_simple(a,b)
        }
        items.append(item)
    return items

# Generate
all_items = []
for kind in ["integers_equal_len", "integers_diff_len", "decimals_equal_len", "decimals_diff_len"]:
    all_items.extend(gen_bucket(kind, N_PER_BUCKET))

# Shuffle to interleave kinds
random.shuffle(all_items)

# Write JSONL
with open(OUT_PATH, "w", encoding="utf-8") as f:
    for it in all_items:
        f.write(json.dumps(it) + "\n")

# print(all_items[0])
# Quick stats
stats = {
    "total": len(all_items),
    "by_type": {"integers_equal_len":0, "integers_diff_len":0,"decimals_equal_len":0,"decimals_diff_len":0},
    "lex_vs_numeric_disagree": 0,
    "avg_len_a": mean(len(x["a"]) for x in all_items),
    "avg_len_b": mean(len(x["b"]) for x in all_items),
    "numeric_truth_yn": list(np.unique(np.array([x["numeric_truth_yn"] for x in all_items]), return_counts=True)),
    "numeric_truth_tf": list(np.unique(np.array([x["numeric_truth_tf"] for x in all_items]), return_counts=True)),
    "lex_truth_yn": list(np.unique(np.array([x["lex_truth_yn"] for x in all_items]), return_counts=True)),
    "lex_truth_tf": list(np.unique(np.array([x["lex_truth_tf"] for x in all_items]), return_counts=True)),
}
for it in all_items:
    stats["by_type"][it["pair_type"]] += 1
    if it["lex_truth"] != "eq" and it["lex_truth"] != it["numeric_truth"]:
        stats["lex_vs_numeric_disagree"] += 1

print(OUT_PATH), print(stats)

import string
def random_string(length, chars=string.ascii_lowercase):
    return ''.join(random.choice(chars) for _ in range(length))

def make_equal_length_str_pair(min_len=1, max_len=10):
    length = random.randint(min_len, max_len)
    a = random_string(length)
    b = random_string(length)
    while b == a:
        b = random_string(length)
    return a, b

def make_unequal_length_str_pair(min_len=1, max_len=10):
    a_len = random.randint(min_len, max_len)
    b_len = random.choice([l for l in range(min_len, max_len+1) if l != a_len])
    a = random_string(a_len)
    b = random_string(b_len)
    return a, b


def gen_string_bucket(kind: str, n: int):
    items = []
    for i in range(n):
        if kind == "string_equal_len":
            a,b = make_equal_length_str_pair()
        elif kind == "string_diff_len":
            a,b = make_unequal_length_str_pair()
        else:
            raise ValueError("unknown kind")
        # ensure they aren't numerically equal
        while a == b:
            if kind == "string_equal_len":
                a,b = make_equal_length_str_pair()
            elif kind == "string_diff_len":
                a,b = make_unequal_length_str_pair()

        item = {
            "id": f"{kind}-{i}",
            "pair_type": kind,
            "a": a, "b": b,
            "lex_truth": lex_compare(a,b, 'ab'),               # 'A'/'B'/eq
            "lex_truth_yn": lex_compare(a,b,'yn'),               # 'A'/'B'/eq
            "lex_truth_tf": lex_compare(a,b, 'tf'),               # 'A'/'B'/eq
            "prompt_mcq": prompt_mcq(a,b),
            "prompt_simple": prompt_simple(a,b)
        }
        items.append(item)
    return items



# Generate
all_items = []
for kind in ["string_equal_len", "string_diff_len"]:
    all_items.extend(gen_string_bucket(kind, 500))

# Shuffle to interleave kinds
random.shuffle(all_items)

# Write JSONL
with open(STRING_OUT_PATH, "w", encoding="utf-8") as f:
    for it in all_items:
        f.write(json.dumps(it) + "\n")