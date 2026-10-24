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


def quantise_int8(fp32: pathlib.Path, out: pathlib.Path, reader: CalibrationDataReader,
                  op_types: tuple[str, ...] = ("MatMul",)) -> pathlib.Path:
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
                    per_channel=True, op_types_to_quantize=list(op_types))
    prepped.unlink(missing_ok=True)
    return out


def quantise_int4(fp32: pathlib.Path, out: pathlib.Path,
                  block_size: int = 128, symmetric: bool = False) -> pathlib.Path:
    """Weight-only block-wise INT4. Runs on CPU; no CUDA involved."""
    import onnx
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
    q = MatMulNBitsQuantizer(onnx.load(str(fp32)), bits=4,
                             block_size=block_size, is_symmetric=symmetric)
    q.process()
    out.parent.mkdir(parents=True, exist_ok=True)
    q.model.save_model_to_file(str(out), use_external_data_format=False)
    return out


def run_session(path: pathlib.Path, input_ids: np.ndarray, attention_mask: np.ndarray,
                batch_size: int = 256) -> np.ndarray:
    """Logits from a graph, in batches, on CPU.

    Inference dominates the wall clock here: fine-tuning a seed takes minutes
    on a GPU while scoring the test and calibration splits at three precisions
    takes longer, because ONNX Runtime runs on CPU and INT4 dequantises its
    weights on every matrix multiplication.  Threads are therefore set to the
    physical core count and the batch is large, which costs memory and buys
    most of the difference.
    """
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = os.cpu_count() or 4
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.log_severity_level = 3
    sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    outs = []
    for i in range(0, len(input_ids), batch_size):
        outs.append(sess.run(None, {
            "input_ids": input_ids[i:i + batch_size].astype(np.int64),
            "attention_mask": attention_mask[i:i + batch_size].astype(np.int64),
        })[0])
    return np.concatenate(outs, axis=0)


def model_size_mb(path: pathlib.Path) -> float:
    return path.stat().st_size / (1024 * 1024)
