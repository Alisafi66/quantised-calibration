"""The compact transformer, and quantisation through the deployment runtime.

The linear baseline quantises by simulation because it has no graph.  This one
does not need the pretence: it exports to ONNX and is quantised by ONNX Runtime
itself, INT8 statically against the PTQ calibration sample and INT4 by
weight-only block quantisation.  That matters beyond convenience.  ONNX Runtime
Mobile is what RQ3 measures on the handset, so quantising through the same
runtime means the desktop numbers and the device numbers describe one artefact
rather than two implementations that happen to share a name.

Fine-tuning wants a GPU.  Everything after it, including INT4, runs on CPU.
"""
from __future__ import annotations

import pathlib

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class CompactTransformer:
    """DistilBERT-class encoder fine-tuned for topic classification."""

    def __init__(self, model_name: str = "distilbert-base-uncased", n_classes: int = 4,
                 max_length: int = 128, seed: int = 0):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        torch.manual_seed(seed)
        self.seed = seed
        self.max_length = max_length
        self.model_name = model_name
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=n_classes)
        self.device = pick_device()

    def encode(self, text: list[str]):
        enc = self.tok(text, truncation=True, padding="max_length",
                       max_length=self.max_length, return_tensors="pt")
        return enc["input_ids"], enc["attention_mask"]

    def fit(self, text: list[str], label: np.ndarray, *, epochs: int = 2,
            batch_size: int = 32, lr: float = 5e-5, log_every: int = 200):
        ids, mask = self.encode(text)
        y = torch.as_tensor(label, dtype=torch.long)
        loader = DataLoader(TensorDataset(ids, mask, y), batch_size=batch_size,
                            shuffle=True, generator=torch.Generator().manual_seed(self.seed))
        self.model.to(self.device).train()
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr)
        for epoch in range(epochs):
            for step, (b_ids, b_mask, b_y) in enumerate(loader):
                opt.zero_grad()
                out = self.model(input_ids=b_ids.to(self.device),
                                 attention_mask=b_mask.to(self.device),
                                 labels=b_y.to(self.device))
                out.loss.backward()
                opt.step()
                if step % log_every == 0:
                    print(f"    epoch {epoch} step {step:>5}/{len(loader)}  "
                          f"loss {out.loss.item():.4f}", flush=True)
        self.model.eval()
        return self

    def export_onnx(self, path: pathlib.Path) -> pathlib.Path:
        """Write the FP32 graph that every quantised variant is derived from.

        Weights are kept inside the single file rather than in a sidecar.
        The quantisation tools copy the graph into a temporary directory and
        leave external data behind, and DistilBERT is far below the 2 GB
        protobuf limit, so there is nothing to gain by splitting it.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.to("cpu").eval()
        ids, mask = self.encode(["placeholder text for tracing"])
        torch.onnx.export(
            self.model, (ids, mask), str(path),
            input_names=["input_ids", "attention_mask"], output_names=["logits"],
            dynamic_axes={"input_ids": {0: "batch"}, "attention_mask": {0: "batch"},
                          "logits": {0: "batch"}},
            opset_version=18, external_data=False, dynamo=True)
        return path
