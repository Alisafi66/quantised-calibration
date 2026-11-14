"""Which property of a quantisation scheme destroys selective prediction?

The reported result compares static INT8 against weight-only block-128 INT4.
Those two arms differ in three ways at once: the bit width, whether activations
are quantised at all, and the granularity of the weight scales. The claim that
the scheme rather than the bit width drives the risk therefore rests on a
two-point comparison with three confounded variables, which an examiner can
reasonably decline to accept.

This adds three arms so that every comparison moves exactly one property:

    int8_static        8 bit   activations static, calibrated   per-channel
    int8_dynamic       8 bit   activations dynamic, per input   per-channel
    int8_weight_only   8 bit   activations in float             block 128
    int4_block128      4 bit   activations in float             block 128
    int4_block32       4 bit   activations in float             block  32

    static calibration    int8_dynamic     against int8_static
    activation quantising int8_weight_only against int8_dynamic
    bit width             int4_block128    against int8_weight_only
    weight granularity    int4_block32     against int4_block128

No fine-tuning happens here. The exported FP32 graph of every seed is already
on the machine that trained it, so this re-quantises that graph and scores the
test and probability-calibration splits again. It is CPU work: the GPU is not
used and need not be present.

Run:  python experiments/run_scheme_decomposition.py --variant long_tailed --seeds 0 1 2 3 4
      python experiments/run_scheme_decomposition.py --variant balanced --seeds 0 --arms int8_dynamic
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from qcal.data import read_jsonl  # noqa: E402
from qcal.onnx_quant import (EncodedReader, assert_quantised,  # noqa: E402
                             model_size_mb, quantise_int8,
                             quantise_int8_dynamic, quantise_weight_only,
                             run_session)

def say(*args) -> None:
    """Print and flush.

    The workstation runs this unattended for a couple of hours. Python buffers
    stdout whenever it is not a terminal, so without flushing a redirected run
    shows nothing until it finishes, and a stalled run is indistinguishable
    from a slow one.
    """
    print(*args, flush=True)


PROC = pathlib.Path("data/processed")
MODELS = pathlib.Path("models")
PROBS = pathlib.Path("results/probs")
OUT = PROBS / "scheme"

# What each arm changes, and how it is built. `published` marks the two arms
# already measured and stored under their original names, so they are not
# recomputed; they are named here because the contrasts refer to them.
ARMS: dict[str, dict] = {
    "int8_static": dict(
        bits=8, activations="static", grain="per-channel",
        route="static", published="int8"),
    "int8_dynamic": dict(
        bits=8, activations="dynamic", grain="per-channel",
        route="dynamic", build=lambda fp32, out, _: quantise_int8_dynamic(fp32, out)),
    "int8_weight_only": dict(
        bits=8, activations="none", grain="block-128",
        route="weight_only",
        build=lambda fp32, out, _: quantise_weight_only(fp32, out, bits=8,
                                                        block_size=128)),
    "int4_block128": dict(
        bits=4, activations="none", grain="block-128",
        route="weight_only", published="int4"),
    "int4_block32": dict(
        bits=4, activations="none", grain="block-32",
        route="weight_only",
        build=lambda fp32, out, _: quantise_weight_only(fp32, out, bits=4,
                                                        block_size=32)),
}

NEW_ARMS = [a for a, spec in ARMS.items() if "published" not in spec]

# Each contrast isolates one property. The named arm is compared against the
# reference arm, and only the listed property differs between them.
CONTRASTS = [
    ("static calibration", "int8_dynamic", "int8_static",
     "activation scales recomputed per input, against scales fixed from a "
     "calibration sample"),
    ("activation quantisation", "int8_weight_only", "int8_dynamic",
     "activations left in float, against activations quantised at run time"),
    ("bit width", "int4_block128", "int8_weight_only",
     "four bits against eight, both weight-only at block 128"),
    ("weight granularity", "int4_block32", "int4_block128",
     "block 32 against block 128, both weight-only at four bits"),
]


def tokeniser(cfg: dict):
    """The tokeniser alone, without loading any model weights.

    Scoring needs the same encoding the trainer used, not its parameters, and
    `max_length` is read from the same config entry, so the encoding is
    identical by construction rather than by agreement.
    """
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model"]["transformer_name"])
    max_length = cfg["model"]["max_length"]

    def encode(text: list[str]):
        enc = tok(text, truncation=True, padding="max_length",
                  max_length=max_length, return_tensors="np")
        return (enc["input_ids"].astype(np.int64),
                enc["attention_mask"].astype(np.int64))
    return encode


def softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


def published_gap(fp32: pathlib.Path, variant: str, seed: int,
                  ids: np.ndarray, mask: np.ndarray, label: np.ndarray,
                  n: int = 512) -> tuple[float | None, float]:
    """The largest probability difference between this graph and the published
    FP32 column, and this graph's accuracy. A gap of None means there is no
    published column to compare against."""
    stored = PROBS / f"{variant}_seed{seed}_test.npz"
    got = softmax(run_session(fp32, ids[:n], mask[:n]))
    acc = float((got.argmax(1) == label[:n]).mean())
    if not stored.exists():
        return None, acc
    want = np.load(stored)["fp32"][:n]
    return float(np.abs(want - got).max()), acc


def inventory(cfg: dict, variants: list[str], seeds: list[int], encode) -> None:
    """Which FP32 graphs are present, and which still match the published runs.

    Worth a few minutes before any long run. `models/` is not under version
    control, so a graph directory can hold an earlier training run with no
    record saying so, and a decomposition built on one would compare arms from
    a different model against the published baselines. That exact case was
    found on the author's laptop: a coherent export set at 0.899 accuracy
    sitting where the published 0.942 run was assumed to be.
    """
    test = read_jsonl(PROC / "test.jsonl", "test")
    ids, mask = encode(test.text[:512])
    say(f"  {'variant':14s} {'seed':>4s} {'graph':>9s} {'acc(512)':>9s} "
          f"{'gap vs published':>17s}  verdict")
    for variant in variants:
        for seed in seeds:
            fp32 = MODELS / f"{variant}_seed{seed}" / "fp32.onnx"
            if not fp32.exists():
                say(f"  {variant:14s} {seed:4d} {'-':>9s} {'-':>9s} "
                      f"{'-':>17s}  MISSING")
                continue
            gap, acc = published_gap(fp32, variant, seed, ids, mask,
                                     test.label)
            if gap is None:
                verdict, shown = "NO PUBLISHED COLUMN", "-"
            elif gap > 1e-4:
                verdict, shown = "STALE, DO NOT USE", f"{gap:.2e}"
            else:
                verdict, shown = "ok", f"{gap:.1e}"
            say(f"  {variant:14s} {seed:4d} "
                  f"{model_size_mb(fp32):8.1f}M {acc:9.4f} {shown:>17s}  "
                  f"{verdict}")


def check_graph_is_the_published_one(fp32: pathlib.Path, variant: str, seed: int,
                                    ids: np.ndarray, mask: np.ndarray,
                                    label: np.ndarray, n: int = 512) -> None:
    """Confirm this FP32 graph is the one that produced the published numbers.

    The new arms are compared against the stored FP32 and INT8 columns, which
    were written months earlier. If the graph on disk were not the graph that
    wrote them, every contrast below would be measured against the wrong
    baseline and nothing would say so. Re-scoring a few hundred rows settles it
    in seconds, and `models/` is not under version control, so there is no
    other record to check against.
    """
    gap, acc = published_gap(fp32, variant, seed, ids, mask, label, n)
    if gap is None:
        say(f"  no published probabilities for {variant} seed {seed}; "
              f"baseline unverified, graph accuracy {acc:.4f}")
        return
    if gap > 1e-4:
        raise SystemExit(
            f"{variant} seed {seed}: this fp32.onnx does not reproduce the "
            f"published probabilities (max difference {gap:.2e} over {n} rows). "
            f"The graph on disk is not the one behind the published results, so "
            f"the contrasts would use the wrong baseline. This graph scores "
            f"{acc:.4f} on those rows. Resolve it before scoring: run "
            f"--inventory to see which seeds are affected.")
    say(f"  fp32 graph matches the published probabilities "
          f"(max difference {gap:.1e} over {n} rows)")


def run_seed(cfg: dict, variant: str, seed: int, arms: list[str],
             encode, keep_graphs: bool) -> dict:
    graph_dir = MODELS / f"{variant}_seed{seed}"
    fp32 = graph_dir / "fp32.onnx"
    if not fp32.exists():
        raise SystemExit(
            f"no {fp32}. The FP32 graphs stay on the machine that trained "
            f"them, so run this there, or re-run "
            f"experiments/train_transformer.py --variant {variant} "
            f"--seeds {seed} to recreate it.")

    say(f"\n[{variant} seed {seed}] {model_size_mb(fp32):.1f} MB fp32")
    test = read_jsonl(PROC / "test.jsonl", "test")
    test_ids, test_mask = encode(test.text)
    check_graph_is_the_published_one(fp32, variant, seed, test_ids, test_mask,
                                     test.label)

    ptq = read_jsonl(PROC / "ptq_calib.jsonl", "ptq_calib")
    n_calib = cfg["quantisation"]["calibration_samples"]
    calib_ids, calib_mask = encode(ptq.text[:n_calib])

    graphs: dict[str, pathlib.Path] = {}
    sizes: dict[str, float] = {}
    for arm in arms:
        spec = ARMS[arm]
        out = graph_dir / f"{arm}.onnx"
        t0 = time.time()
        if arm == "int8_static":
            quantise_int8(fp32, out, EncodedReader(calib_ids, calib_mask))
        else:
            spec["build"](fp32, out, cfg)
        found = assert_quantised(out, spec["route"], f"{variant} seed {seed} {arm}")
        sizes[arm] = round(model_size_mb(out), 2)
        graphs[arm] = out
        say(f"  built {arm:17s} {sizes[arm]:6.1f} MB  "
              f"{(time.time() - t0) / 60:4.1f} min  {found}")

    records = {}
    for split in ("test", "prob_calib"):
        data = (test if split == "test"
                else read_jsonl(PROC / f"{split}.jsonl", split))
        ids, mask = ((test_ids, test_mask) if split == "test"
                     else encode(data.text))
        payload: dict[str, np.ndarray] = {"label": data.label}
        for arm, graph in graphs.items():
            t0 = time.time()
            payload[arm] = softmax(run_session(graph, ids, mask))
            say(f"  scored {arm:17s} {split:10s} "
                  f"{(time.time() - t0) / 60:4.1f} min")
        OUT.mkdir(parents=True, exist_ok=True)
        path = OUT / f"{variant}_seed{seed}_{split}.npz"
        np.savez_compressed(path, **payload)
        records[split] = str(path)
        say(f"  wrote {path.name}  ({path.stat().st_size / 1024:.0f} KB, "
              f"n={len(data):,})")

    if not keep_graphs:
        for arm, graph in graphs.items():
            graph.unlink(missing_ok=True)
        say(f"  removed {len(graphs)} quantised graph(s); pass --keep-graphs "
              f"to retain them for a device run")

    return dict(variant=variant, seed=seed, arms=arms, sizes_mb=sizes,
                probs=records)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--variant", default="long_tailed")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--arms", nargs="+", default=NEW_ARMS,
                    choices=list(ARMS), metavar="ARM",
                    help=f"default: {' '.join(NEW_ARMS)}")
    ap.add_argument("--inventory", action="store_true",
                    help="report which FP32 graphs are present and still match "
                         "the published runs, score nothing, and exit; run this "
                         "first on a machine you have not used for a while")
    ap.add_argument("--inventory-variants", nargs="+",
                    default=["balanced", "long_tailed", "lt_r3", "lt_r10",
                             "lt_r100"],
                    help="variants the inventory looks for")
    ap.add_argument("--keep-graphs", action="store_true",
                    help="keep the quantised graphs instead of deleting them; "
                         "about 130 MB each, and only RQ3 on the handset needs "
                         "them")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    seeds = args.seeds if args.seeds is not None else cfg["seed_list"]

    if args.inventory:
        say("FP32 graph inventory. A stale graph is an earlier training run "
              "left in place;\nmodels/ is not versioned, so nothing else "
              "records that.\n")
        inventory(cfg, args.inventory_variants, seeds, tokeniser(cfg))
        return

    say(f"variant {args.variant}  seeds {seeds}  arms {args.arms}")
    say("No fine-tuning: the exported FP32 graphs are re-quantised and "
          "re-scored on CPU.\n")

    encode = tokeniser(cfg)
    records, t0 = [], time.time()
    for seed in seeds:
        records.append(run_seed(cfg, args.variant, seed, args.arms, encode,
                                args.keep_graphs))

    MODELS.mkdir(parents=True, exist_ok=True)
    manifest = MODELS / f"manifest_scheme_{args.variant}.json"
    manifest.write_text(json.dumps(
        dict(variant=args.variant, arms={a: {k: v for k, v in ARMS[a].items()
                                             if k != "build"}
                                         for a in args.arms},
             contrasts=[dict(zip(("property", "arm", "reference", "note"), c))
                        for c in CONTRASTS],
             seeds=records), indent=2) + "\n")
    say(f"\n{len(records)} seed(s) in {(time.time() - t0) / 60:.1f} min")
    say(f"manifest: {manifest}")
    say(f"\nPush {OUT}/ and this manifest, then run "
          f"experiments/run_scheme_analysis.py on any machine.")


if __name__ == "__main__":
    main()
