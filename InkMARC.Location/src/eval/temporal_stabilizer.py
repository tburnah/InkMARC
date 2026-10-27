import numpy as np

class XYStabilizer:
    def __init__(
        self,
        small_motion=0.002,
        medium_motion=0.01,
        alpha_slow=0.2,
        alpha_mid=0.5,
        alpha_fast=0.85,
        jump_gate=0.08,      # optional: treat as outlier if sudden jump exceeds this (normalized)
        reset_on_mask0=True,
    ):
        self.small_motion = small_motion
        self.medium_motion = medium_motion
        self.alpha_slow = alpha_slow
        self.alpha_mid = alpha_mid
        self.alpha_fast = alpha_fast
        self.jump_gate = jump_gate
        self.reset_on_mask0 = reset_on_mask0
        self.reset()

    def reset(self):
        self.prev = None

    def step(self, xy_pred, xy_mask=1.0):
        # xy_pred: np.array shape (2,)
        if self.reset_on_mask0 and xy_mask < 0.5:
            self.prev = None
            return xy_pred, 0.0

        if self.prev is None:
            self.prev = xy_pred.copy()
            return xy_pred, 1.0

        delta = float(np.linalg.norm(xy_pred - self.prev))

        # Optional outlier gate: ignore huge one-frame jumps
        if self.jump_gate is not None and delta > self.jump_gate:
            # keep previous (no update)
            return self.prev.copy(), 1.0

        # Adaptive EMA alpha (higher alpha = follow new value more)
        if delta < self.small_motion:
            alpha = self.alpha_slow
        elif delta < self.medium_motion:
            alpha = self.alpha_mid
        else:
            alpha = self.alpha_fast

        stable = (1.0 - alpha) * self.prev + alpha * xy_pred
        self.prev = stable
        return stable, 1.0
