from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from ml.inkloc.data.synthetic import SyntheticInteractionDataset
from ml.inkloc.models import LocTouchNetLite
from ml.inkloc.training.forward import model_forward
from ml.inkloc.training.losses import LocTouchLoss
from ml.inkloc.utils.config import load_config
from ml.inkloc.utils.seed import set_seed


def run_overfit(config_path: str, steps: int = 12, min_drop: float = 0.02) -> tuple[float, float]:
    cfg = load_config(config_path)
    set_seed(int(cfg.get("seed", 7)))
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device", "auto") != "cpu" else "cpu")
    model = LocTouchNetLite.from_config(cfg).to(device)
    loss_fn = LocTouchLoss.from_config(cfg)
    ds = SyntheticInteractionDataset(size=6, height=cfg["model"]["input_height"], width=cfg["model"]["input_width"], seed=91, noise=0.0)
    loader = DataLoader(ds, batch_size=6, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    batch = {k: v.to(device) for k, v in batch.items()}
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"].get("overfit_learning_rate", 0.01)), weight_decay=0.0)
    model.train()
    first = None
    last = None
    for _ in range(steps):
        outputs = model_forward(model, batch)
        loss = loss_fn(outputs, batch)["loss"]
        if first is None:
            first = float(loss.detach().cpu())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last = float(loss.detach().cpu())
    assert first is not None and last is not None
    if last > first - min_drop:
        raise RuntimeError(f"Tiny overfit loss did not decrease enough: before={first:.4f} after={last:.4f}")
    return first, last


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="ml/configs/loctouch_lite.yaml")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--min-drop", type=float, default=0.02)
    args = parser.parse_args()
    before, after = run_overfit(args.config, args.steps, args.min_drop)
    print(f"tiny_overfit before={before:.4f} after={after:.4f} drop={before - after:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
