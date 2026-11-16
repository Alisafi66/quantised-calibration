"""Quantise an exported graph the way the deployment runtime will.

INT8 is static: the scale and zero point of every tensor are chosen by running
the PTQ calibration sample through the FP32 graph and observing the ranges.
That sample comes from training data and never from test, because a model
being prepared for deployment has not seen its test inputs.

INT4 is weight-only and block-wise, which is what ONNX Runtime offers and what
a 4-bit mobile deployment actually looks like.  It needs no GPU, so the CUDA
machine is required for fine-tuning alone.
"""
from __future__ import annotations

import logging
import os
import pathlib

import numpy as np
import onnxruntime as ort

# the INT4 quantiser logs a line per node, several hundred per model and
# several thousand across a seed sweep, which buries everything worth reading
logging.getLogger("onnxruntime.quantization").setLevel(logging.WARNING)
from onnxruntime.quantization import CalibrationDataReader, QuantType, quantize_static


class EncodedReader(CalibrationDataReader):
    """Feeds the PTQ calibration sample to the static quantiser, once."""

    def __init__(self, input_ids: np.ndarray, attention_mask: np.ndarray, batch_size: int = 8):
        self.batches = [
            {"input_ids": input_ids[i:i + batch_size].astype(np.int64),
             "attention_mask": attention_mask[i:i + batch_size].astype(np.int64)}
            for i in range(0, len(input_ids), batch_size)
        ]
        self.i = 0

    def get_next(self):
        if self.i >= len(self.batches):
            return None
        self.i += 1
        return self.batches[self.i - 1]

    def rewind(self):
        self.i = 0


def node_roles(graph: pathlib.Path) -> dict[str, list[str]]:
    """Split the matrix multiplications by the role they play.

    Three kinds behave differently under quantisation and must be separable:

    `weight`      a projection against a learned constant: the 36 query, key,
                  value, attention-output and two feed-forward matrices.
    `attention`   the 12 multiplications of one activation by another, Q.K^T
                  and attention.V, which have no constant operand. These carry
                  the activation outliers that motivated LLM.int8.
    `classifier`  the head, carrying both a weight and a bias.

    Static INT8 reaches all 48 of the first two; the weight-only quantisers
    reach only the 36. Comparing them therefore varies coverage as well as
    scheme unless the sets are controlled, which is what this makes possible.
    """
    import onnx
    g = onnx.load(str(graph), load_external_data=False).graph
    inits = {i.name for i in g.initializer}
    roles: dict[str, list[str]] = {"weight": [], "attention": [], "classifier": []}
    for n in g.node:
        if n.op_type not in ("MatMul", "Gemm"):
            continue
        n_const = sum(1 for i in n.input if i in inits)
        if n_const >= 2:
            roles["classifier"].append(n.name)
        elif n_const == 1:
            roles["weight"].append(n.name)
        else:
            roles["attention"].append(n.name)
    return roles


def quantise_int8(fp32: pathlib.Path, out: pathlib.Path, reader: CalibrationDataReader,
                  op_types: tuple[str, ...] = ("MatMul",),
                  nodes_to_exclude: list[str] | None = None) -> pathlib.Path:
    """Static INT8 on the matrix multiplications, calibrated on the PTQ sample.

    Restricting quantisation to MatMul is not a convenience, it is the only
    configuration that works.  Quantising every operator destroyed the model
    outright, 0.27 accuracy against 0.89 at full precision, and percentile
    clipping recovered accuracy only to 0.81 while driving ECE to 0.28.  The
    cause is the activation-outlier behaviour of transformer encoders that
    motivated LLM.int8 (Dettmers et al., 2022): a handful of extreme features
    set the scale for a whole tensor, and everything else collapses into a
    couple of levels.  LayerNorm and the attention softmax are left in higher
    precision for the same reason a mobile deployment would leave them there.
    """
    from onnxruntime.quantization.shape_inference import quant_pre_process
    prepped = out.with_suffix(".prep.onnx")
    quant_pre_process(str(fp32), str(prepped), skip_symbolic_shape=True)
    quantize_static(str(prepped), str(out), reader,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    per_channel=True, op_types_to_quantize=list(op_types),
                    nodes_to_exclude=list(nodes_to_exclude or []))
    prepped.unlink(missing_ok=True)
    return out


def quantise_weight_only(fp32: pathlib.Path, out: pathlib.Path, *,
                         bits: int = 4, block_size: int = 128,
                         symmetric: bool = False) -> pathlib.Path:
    """Weight-only block-wise quantisation at a chosen bit width.

    Activations are left in float throughout, so this isolates what the weights
    alone cost.  Runs on CPU; no CUDA involved.
    """
    import onnx
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
    q = MatMulNBitsQuantizer(onnx.load(str(fp32)), bits=bits,
                             block_size=block_size, is_symmetric=symmetric)
    q.process()
    out.parent.mkdir(parents=True, exist_ok=True)
    q.model.save_model_to_file(str(out), use_external_data_format=False)
    return out


def quantise_int4(fp32: pathlib.Path, out: pathlib.Path,
                  block_size: int = 128, symmetric: bool = False) -> pathlib.Path:
    """Weight-only block-wise INT4, the published arm."""
    return quantise_weight_only(fp32, out, bits=4, block_size=block_size,
                                symmetric=symmetric)


def quantise_int8_dynamic(fp32: pathlib.Path, out: pathlib.Path,
                          op_types: tuple[str, ...] = ("MatMul",),
                          nodes_to_exclude: list[str] | None = None,
                          exclude_classifier: bool = True) -> pathlib.Path:
    """Dynamic INT8: weights quantised ahead of time, activation scales chosen
    per input at run time rather than fixed from a calibration sample.

    This is the control for the static arm.  Bit width, operator coverage and
    weight granularity are the same; the only difference is where the
    activation scale comes from.  If the static arm loses coverage and this one
    does not, the cause is the fixed calibration rather than eight-bit
    arithmetic.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.shape_inference import quant_pre_process
    prepped = out.with_suffix(".prep.onnx")
    # Symbolic shape inference must run here, unlike in the static path.
    # `quantize_dynamic` re-infers shapes under a strict check, and without
    # ORT's symbolic pass the dynamic batch dimension is annotated
    # inconsistently, which fails with "Inferred shape and existing shape
    # differ in dimension 0: (768) vs (4)".
    quant_pre_process(str(fp32), str(prepped), skip_symbolic_shape=False)

    exclude = list(nodes_to_exclude or [])
    if exclude_classifier:
        # Pre-processing decomposes the classifier Gemm into MatMul plus Add,
        # so the node the quantiser sees is `node_linear_37_MatMul`, not
        # `node_linear_37`. Resolving the names against the pre-processed
        # graph keeps this arm on the same 36 projections as every other arm;
        # without it the head is quantised here and nowhere else, and the
        # contrast would vary coverage as well as scheme.
        import onnx
        heads = tuple(node_roles(fp32)["classifier"])
        g = onnx.load(str(prepped), load_external_data=False).graph
        found = [n.name for n in g.node if n.name.startswith(heads)]
        # The quantiser rewrites a Gemm into `<name>_MatMul` on its way to
        # MatMulInteger and matches the exclusion list against that new name,
        # so both forms are listed.
        exclude += found + [f"{n}_MatMul" for n in found]

    quantize_dynamic(str(prepped), str(out), weight_type=QuantType.QInt8,
                     per_channel=True, op_types_to_quantize=list(op_types),
                     nodes_to_exclude=exclude)
    prepped.unlink(missing_ok=True)
    return out


# The operator a working graph must contain, per route.  A quantiser that
# produces none of these has silently done nothing.
QUANTISED_OPS = {
    "static": ("QLinearMatMul", "MatMulInteger", "QuantizeLinear"),
    "dynamic": ("MatMulInteger", "DynamicQuantizeLinear"),
    "weight_only": ("MatMulNBits", "DequantizeLinear"),
}


def assert_quantised(path: pathlib.Path, route: str, label: str) -> dict[str, int]:
    """Read the written graph back and confirm it really is quantised.

    Not defensive padding.  `MatMulNBitsQuantizer` with `channel_wised_quantize`
    and the QDQ format returns a graph of plain `MatMul` at full weight size,
    raising nothing, so an unguarded arm can score the FP32 model and report it
    as quantised.  That failure produces a plausible number rather than an
    error, which is why it is checked rather than trusted.
    """
    import onnx
    counts: dict[str, int] = {}
    for node in onnx.load(str(path), load_external_data=False).graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    found = {op: counts[op] for op in QUANTISED_OPS[route] if op in counts}
    if not found:
        raise SystemExit(
            f"{label}: {path.name} contains no quantised operator. Expected one "
            f"of {QUANTISED_OPS[route]}, found {sorted(counts)}. The quantiser "
            f"returned a float graph without raising; do not score it.")
    return found


def run_session(path: pathlib.Path, input_ids: np.ndarray, attention_mask: np.ndarray,
                batch_size: int = 64) -> np.ndarray:
    """Logits from a graph, in batches, on CPU.

    Inference dominates the wall clock: fine-tuning a seed takes minutes on a
    GPU while scoring the test and calibration splits at three precisions takes
    longer, because ONNX Runtime runs on CPU and INT4 dequantises its weights
    on every matrix multiplication.  Threads are set to the core count, which
    is free.  The batch is not free: at sequence length 128 the GELU
    intermediate alone is batch x 128 x 3072 floats, so a large batch exhausts
    memory on some machines and not others.  Rather than pick a number that
    suits one laptop, the batch halves and retries whenever allocation fails.
    """
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = os.cpu_count() or 4
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.log_severity_level = 3
    sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])

    while batch_size >= 4:
        try:
            outs = []
            for i in range(0, len(input_ids), batch_size):
                outs.append(sess.run(None, {
                    "input_ids": input_ids[i:i + batch_size].astype(np.int64),
                    "attention_mask": attention_mask[i:i + batch_size].astype(np.int64),
                })[0])
            return np.concatenate(outs, axis=0)
        except Exception as exc:
            if "allocation" not in str(exc).lower() and "memory" not in str(exc).lower():
                raise
            batch_size //= 2
            print(f"    allocation failed, retrying at batch {batch_size}", flush=True)
    raise RuntimeError("inference failed even at batch size 4")


def model_size_mb(path: pathlib.Path) -> float:
    return path.stat().st_size / (1024 * 1024)
