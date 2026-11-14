# Round three: which property of the scheme costs the coverage

The reported result compares static INT8 against weight-only block-128 INT4
and concludes that the scheme rather than the bit width drives the risk. Those
two arms differ in three ways at once:

| | bit width | activations | weight scales |
|---|---|---|---|
| `int8_static` (published) | 8 | static, calibrated on the PTQ sample | per-channel |
| `int4_block128` (published) | 4 | left in float | block 128 |

So the claim rests on a two-point comparison with three confounded variables,
and an examiner can decline to accept it. This round adds three arms so that
every comparison moves exactly one property.

| arm | bit width | activations | weight scales |
|---|---|---|---|
| `int8_static` | 8 | static, calibrated | per-channel |
| `int8_dynamic` | 8 | dynamic, recomputed per input | per-channel |
| `int8_weight_only` | 8 | float | block 128 |
| `int4_block128` | 4 | float | block 128 |
| `int4_block32` | 4 | float | block 32 |

| property under test | arm | against | everything else held |
|---|---|---|---|
| static calibration | `int8_dynamic` | `int8_static` | bits, operators, weight grain |
| activation quantisation | `int8_weight_only` | `int8_dynamic` | bits |
| bit width | `int4_block128` | `int8_weight_only` | activations, block size |
| weight granularity | `int4_block32` | `int4_block128` | bits, activations |

**Nothing is fine-tuned.** Every seed's exported FP32 graph is already on the
machine that trained it, so this re-quantises that graph and scores the test
and probability-calibration splits again. It is CPU work throughout; the GPU is
not used. That is also why it has to run on the workstation: the graphs are
there, not in the repository.

---

## Run the inventory first

```powershell
cd $HOME\quantised-calibration
git pull
.venv\Scripts\activate
```

```powershell
python experiments\run_scheme_decomposition.py --inventory --seeds 0 1 2 3 4
```

About three minutes. For each variant and seed it reports whether `fp32.onnx`
is present and whether it still reproduces the published FP32 probabilities on
512 test rows.

This is not a formality. `models/` is not under version control, so a graph
directory can hold an earlier training run with nothing recording the fact.
That case is real: the author's laptop had `models/balanced_seed0` holding a
coherent export set at 0.8993 accuracy where the published 0.9420 run was
assumed to be, dated from the exploratory phase. Decomposing on it would have
compared new arms against published baselines from a different model, and every
contrast would have carried a four-point accuracy difference that has nothing
to do with quantisation.

Read the verdict column:

- **`ok`** means the graph reproduces the published probabilities and can be used.
- **`STALE, DO NOT USE`** means it is an earlier run. The runner refuses it.
- **`MISSING`** means there is no graph. If the October runs were cleaned up,
  `long_tailed` may be missing while last night's `lt_r3`, `lt_r10` and
  `lt_r100` are present. That is not a problem: the mechanism question does not
  need the exact published split, so decompose on `lt_r10` or `lt_r100`
  instead and say which split it was measured on. Only re-train if you want the
  decomposition at exactly 33:1.

---

## Prove the chain on one seed

```powershell
python experiments\run_scheme_decomposition.py --variant long_tailed --seeds 0 --arms int4_block32
```

One arm, one seed. It builds the graph, confirms the written graph really
contains quantised operators, scores both splits, writes one `.npz` and deletes
the graph again. Roughly ten minutes, and it prints its own per-step timings so
the full run can be estimated from it rather than guessed.

The guard on the written graph matters as much as the baseline check. ONNX
Runtime's `MatMulNBitsQuantizer` with `channel_wised_quantize` and the QDQ
format returns a graph of plain `MatMul` at full weight size and raises
nothing, so an unguarded arm can score the FP32 model and report it as
quantised. That failure produces a plausible number rather than an error, which
is why the written graph is read back and checked.

---

## The full run

```powershell
python experiments\run_scheme_decomposition.py --variant long_tailed --seeds 0 1 2 3 4
```

Three new arms, five seeds. The two published arms are not recomputed.

Expect roughly **sixteen to twenty minutes per seed**, so **about one and a
half to two hours**, almost all of it CPU inference on 13,600 rows per arm.
The first seed prints its real timings; trust those over this estimate.

Do not pipe it. Last night `2>&1 | Tee-Object` changed Python's stdout encoding
from the console's to cp1252 and a library's progress glyph then raised
`UnicodeEncodeError` mid-run. This script prints only ASCII and flushes every
line, so the console shows progress without a pipe. If you want a log anyway,
set `$env:PYTHONUTF8 = "1"` first.

The quantised graphs are deleted after scoring, about 130 MB each. Pass
`--keep-graphs` only if a graph is wanted for an on-device run.

### Send the results back

```powershell
git add results\probs\scheme models\manifest_scheme_*.json
```

```powershell
git commit -m "Probabilities for the quantisation-scheme decomposition, five seeds"
```

```powershell
git push
```

---

## Back on the laptop

```bash
cd ~/Developer/quantised-calibration
git pull
python experiments/run_scheme_analysis.py --variant long_tailed
```

It prints the five arms with their properties and coverage, then the four
single-property contrasts with paired within-seed t-tests on 4 df, then ECE for
every arm so the RQ1 point can be read off the same table. It runs on whatever
arms are present, so a partial run still reports what it has.

## What the outcomes would mean

- **`int8_dynamic` recovers the coverage and `int8_static` does not.** The cause
  is the fixed calibration, not eight-bit arithmetic. The practical
  recommendation becomes dynamic quantisation, and the mechanism is that a
  calibration sample cannot bound activation ranges it never saw.
- **`int8_weight_only` is benign but both activation arms are not.** The cause
  is activation quantisation as such, which connects directly to the
  activation-outlier literature that motivated LLM.int8.
- **`int8_weight_only` loses coverage too.** Then bit width and weights matter
  after all, and the `int4_block128` against `int8_weight_only` contrast says
  which, with the current claim needing revision.
- **`int4_block32` differs from `int4_block128`.** Granularity matters
  independently and the INT4 arm's benignity is partly a block-size artefact.

Any of the four is a publishable sharpening. The claim currently rests on a
confound, so there is no outcome here that leaves the thesis worse off.
