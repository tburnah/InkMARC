import torch

from ml.inkloc.data.synthetic import make_synthetic_batch
from ml.inkloc.models import LocTouchNetLite
from ml.inkloc.training.losses import LocTouchLoss


def test_loss_is_finite_and_masks_background_xy():
    model = LocTouchNetLite(width_mult=0.35, hidden_dim=64)
    batch = make_synthetic_batch(batch_size=3)
    batch["has_xy"][0] = 0.0
    batch["xy"][0] = torch.tensor([99.0, -99.0])
    out = model(batch["image"])
    losses = LocTouchLoss()(out, batch)
    assert torch.isfinite(losses["loss"])
    assert torch.isfinite(losses["xy_loss"])

    batch2 = {k: v.clone() for k, v in batch.items()}
    batch2["xy"][0] = torch.tensor([-99.0, 99.0])
    losses2 = LocTouchLoss()(out, batch2)
    assert torch.allclose(losses["xy_loss"], losses2["xy_loss"])


def test_temporal_loss_path_is_finite():
    model = LocTouchNetLite(width_mult=0.35, hidden_dim=64, temporal_enabled=True, num_motion_maps=1, motion_hidden_dim=16)
    batch = make_synthetic_batch(batch_size=3)
    out = model(batch["image"], motion_map=batch["motion_map"])
    losses = LocTouchLoss()(out, batch)
    assert torch.isfinite(losses["loss"])
