import torch

from ml.inkloc.models import LocTouchNetLite


def test_model_shapes_and_probabilities():
    model = LocTouchNetLite(width_mult=0.35, hidden_dim=64).eval()
    image = torch.rand(2, 3, 320, 480)
    with torch.no_grad():
        out = model(image)
    assert out["xy"].shape == (2, 2)
    assert out["heatmap_logits"].shape == (2, 1, 40, 60)
    assert out["heatmap_prob"].shape == (2, 1, 40, 60)
    assert out["loc_conf"].shape == (2,)
    assert out["valid_logit"].shape == (2,)
    assert out["touch_logit"].shape == (2,)
    assert out["state_logits"].shape == (2, 3)
    assert out["state_probs"].shape == (2, 3)
    assert torch.all(out["xy"] >= 0.0) and torch.all(out["xy"] <= 1.0)
    assert torch.allclose(out["state_probs"].sum(dim=1), torch.ones(2), atol=1e-5)
    for value in out.values():
        assert torch.isfinite(value).all()


def test_temporal_motion_branch_and_touch_ablations():
    model = LocTouchNetLite(
        width_mult=0.35,
        hidden_dim=64,
        temporal_enabled=True,
        num_motion_maps=1,
        motion_hidden_dim=16,
    ).eval()
    image = torch.rand(2, 3, 320, 480)
    prev = torch.rand(2, 3, 320, 480)
    motion = torch.rand(2, 1, 320, 480)
    with torch.no_grad():
        out_rgb_motion = model(image, motion_map=motion, touch_ablation_mode="rgb_motion", ablate_global_token=True)
        out_no_motion = model(image, motion_map=motion, touch_ablation_mode="no_motion", ablate_global_token=True)
        out_motion_only = model(image, motion_map=motion, touch_ablation_mode="motion_only", ablate_global_token=True)
        out_prev = model(image, image_t_minus_1=prev)
    for out in [out_rgb_motion, out_no_motion, out_motion_only, out_prev]:
        assert out["xy"].shape == (2, 2)
        assert out["state_probs"].shape == (2, 3)
        assert torch.isfinite(out["touch_logit"]).all()
    assert not torch.allclose(out_rgb_motion["touch_logit"], out_no_motion["touch_logit"])
    assert not torch.allclose(out_no_motion["touch_logit"], out_motion_only["touch_logit"])
