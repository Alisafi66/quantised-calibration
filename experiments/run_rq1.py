"""RQ1 — see README. Entry point; implementation to follow."""
import argparse, yaml


def main(cfg: dict) -> None:
    raise NotImplementedError("RQ1 experiment not implemented yet")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    main(yaml.safe_load(open(args.config)))
