import torch

from ml.inkloc.data.inkmarc import build_prev_index, compute_motion_map


def test_prev_index_does_not_cross_sequence_boundary():
    records = [
        {"file_id": "seq_a", "frame_idx": 0},
        {"file_id": "seq_a", "frame_idx": 1},
        {"file_id": "seq_b", "frame_idx": 0},
        {"file_id": "seq_b", "frame_idx": 2},
        {"file_id": "seq_b", "frame_idx": 3},
        {"file_id": "seq_a", "frame_idx": 2},
    ]
    prev = build_prev_index(records)
    assert prev == [None, 0, None, None, 3, 1]


def test_first_frame_zero_motion_like_case():
    image = torch.rand(3, 320, 480)
    motion_abs = compute_motion_map(image, image, "abs_gray_diff")
    motion_signed = compute_motion_map(image, image, "signed_gray_diff")
    motion_rgb = compute_motion_map(image, image, "rgb_diff")
    assert motion_abs.shape == (1, 320, 480)
    assert motion_signed.shape == (1, 320, 480)
    assert motion_rgb.shape == (3, 320, 480)
    assert torch.allclose(motion_abs, torch.zeros_like(motion_abs))
    assert torch.allclose(motion_signed, torch.zeros_like(motion_signed))
    assert torch.allclose(motion_rgb, torch.zeros_like(motion_rgb))
