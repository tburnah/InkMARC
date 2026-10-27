from __future__ import annotations

import numpy as np


class PlanarEkf:
    def __init__(self, process_accel_std_mps2: float = 0.75) -> None:
        self.process_accel_std_mps2 = float(process_accel_std_mps2)
        self.state = np.zeros(4, dtype=np.float64)
        self.covariance = np.eye(4, dtype=np.float64)

    def initialize(
        self,
        position_xy: np.ndarray,
        velocity_xy: np.ndarray,
        position_std_m: float = 0.10,
        velocity_std_mps: float = 0.25,
    ) -> None:
        self.state = np.asarray(
            [position_xy[0], position_xy[1], velocity_xy[0], velocity_xy[1]],
            dtype=np.float64,
        )
        position_variance = max(float(position_std_m), 1e-4) ** 2
        velocity_variance = max(float(velocity_std_mps), 1e-4) ** 2
        self.covariance = np.diag(
            [position_variance, position_variance, velocity_variance, velocity_variance]
        ).astype(np.float64)

    def predict(self, dt_s: float) -> None:
        dt = max(float(dt_s), 1e-6)
        transition = np.asarray(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        process_gain = np.asarray(
            [
                [0.5 * dt * dt, 0.0],
                [0.0, 0.5 * dt * dt],
                [dt, 0.0],
                [0.0, dt],
            ],
            dtype=np.float64,
        )
        accel_variance = self.process_accel_std_mps2 * self.process_accel_std_mps2
        process_noise = accel_variance * (process_gain @ process_gain.T)
        self.state = transition @ self.state
        self.covariance = transition @ self.covariance @ transition.T + process_noise

    def update_velocity(
        self,
        measured_velocity_xy: np.ndarray,
        measurement_std_xy: np.ndarray,
    ) -> None:
        measurement = np.asarray(measured_velocity_xy, dtype=np.float64).reshape(2)
        measurement_std = np.clip(
            np.asarray(measurement_std_xy, dtype=np.float64).reshape(2),
            a_min=1e-4,
            a_max=None,
        )
        observation = np.asarray(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        measurement_noise = np.diag(np.square(measurement_std)).astype(np.float64)
        innovation = measurement - observation @ self.state
        innovation_covariance = observation @ self.covariance @ observation.T + measurement_noise
        kalman_gain = self.covariance @ observation.T @ np.linalg.inv(innovation_covariance)
        self.state = self.state + kalman_gain @ innovation
        identity = np.eye(self.covariance.shape[0], dtype=np.float64)
        joseph = identity - kalman_gain @ observation
        self.covariance = (
            joseph @ self.covariance @ joseph.T + kalman_gain @ measurement_noise @ kalman_gain.T
        )
