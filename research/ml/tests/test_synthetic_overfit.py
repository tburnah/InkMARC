from ml.inkloc.training.overfit_tiny import run_overfit


def test_tiny_overfit_loss_decreases():
    before, after = run_overfit("ml/configs/loctouch_lite.yaml", steps=4, min_drop=0.001)
    assert after < before
