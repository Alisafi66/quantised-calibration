"""Download a dataset snapshot and record its SHA-256 in the config.

Reproducibility depends on knowing exactly which version of the data produced a
figure, so nothing is used until it is hashed.
"""
import argparse, hashlib, pathlib, yaml


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    print(f"dataset: {cfg['dataset']['name']} — download + hash not implemented yet")
