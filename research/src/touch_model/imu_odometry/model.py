from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


def make_activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


def format_model_outputs(
    regression: torch.Tensor,
    motion_logits: torch.Tensor | None = None,
    log_variance: torch.Tensor | None = None,
    bias_correction: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | None]:
    return {
        "regression": regression,
        "motion_logits": motion_logits,
        "log_variance": log_variance,
        "bias_correction": bias_correction,
    }


class ResidualTemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.activation = make_activation(activation)
        self.dropout = nn.Dropout(dropout)
        self.proj = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        residual = self.proj(inputs)
        outputs = self.conv1(inputs)
        outputs = self.bn1(outputs)
        outputs = self.activation(outputs)
        outputs = self.dropout(outputs)
        outputs = self.conv2(outputs)
        outputs = self.bn2(outputs)
        outputs = outputs + residual
        outputs = self.activation(outputs)
        return self.dropout(outputs)


class ConvTemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            make_activation(activation),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            make_activation(activation),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        return self.block(inputs)


class TemporalCnnRegressor(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_dim: int,
        channels: Sequence[int] = (64, 128, 128),
        kernel_size: int = 5,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_residual: bool = True,
        motion_head_enabled: bool = False,
        hybrid_head_enabled: bool = False,
        hybrid_bias_correction_enabled: bool = False,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        in_channels = input_channels
        block_cls = ResidualTemporalBlock if use_residual else ConvTemporalBlock
        for out_channels in channels:
            blocks.append(
                block_cls(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    activation=activation,
                )
            )
            in_channels = out_channels
        self.encoder = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.shared_head = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            make_activation(activation),
            nn.Dropout(dropout),
        )
        self.regression_head = nn.Linear(hidden_dim, output_dim)
        self.motion_head = nn.Linear(hidden_dim, 1) if motion_head_enabled else None
        self.log_variance_head = nn.Linear(hidden_dim, output_dim) if hybrid_head_enabled else None
        self.bias_head = nn.Linear(hidden_dim, output_dim) if hybrid_bias_correction_enabled else None

    def forward(
        self,
        inputs: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        features = self.encoder(inputs)
        pooled = self.pool(features).squeeze(-1)
        shared = self.shared_head(pooled)
        motion_logits = self.motion_head(shared) if self.motion_head is not None else None
        log_variance = self.log_variance_head(shared) if self.log_variance_head is not None else None
        bias_correction = self.bias_head(shared) if self.bias_head is not None else None
        return format_model_outputs(
            self.regression_head(shared),
            motion_logits,
            log_variance=log_variance,
            bias_correction=bias_correction,
        )


class CnnGruRegressor(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_dim: int,
        channels: Sequence[int] = (64, 128),
        kernel_size: int = 5,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        activation: str = "gelu",
        motion_head_enabled: bool = False,
        hybrid_head_enabled: bool = False,
        hybrid_bias_correction_enabled: bool = False,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        layers: list[nn.Module] = []
        in_channels = input_channels
        for out_channels in channels:
            layers.extend(
                [
                    nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
                    nn.BatchNorm1d(out_channels),
                    make_activation(activation),
                    nn.Dropout(dropout),
                ]
            )
            in_channels = out_channels
        self.conv = nn.Sequential(*layers)
        self.gru = nn.GRU(
            input_size=in_channels,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
        )
        self.shared_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            make_activation(activation),
            nn.Dropout(dropout),
        )
        self.regression_head = nn.Linear(hidden_dim, output_dim)
        self.motion_head = nn.Linear(hidden_dim, 1) if motion_head_enabled else None
        self.log_variance_head = nn.Linear(hidden_dim, output_dim) if hybrid_head_enabled else None
        self.bias_head = nn.Linear(hidden_dim, output_dim) if hybrid_bias_correction_enabled else None

    def forward(
        self,
        inputs: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        features = self.conv(inputs)
        features = features.transpose(1, 2)
        if lengths is not None:
            packed = pack_padded_sequence(
                features,
                lengths.detach().cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            outputs, hidden = self.gru(packed)
            del outputs
        else:
            outputs, hidden = self.gru(features)
            del outputs
        shared = self.shared_head(hidden[-1])
        motion_logits = self.motion_head(shared) if self.motion_head is not None else None
        log_variance = self.log_variance_head(shared) if self.log_variance_head is not None else None
        bias_correction = self.bias_head(shared) if self.bias_head is not None else None
        return format_model_outputs(
            self.regression_head(shared),
            motion_logits,
            log_variance=log_variance,
            bias_correction=bias_correction,
        )


class BiLstmRegressor(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        activation: str = "gelu",
        num_layers: int = 2,
        motion_head_enabled: bool = False,
        hybrid_head_enabled: bool = False,
        hybrid_bias_correction_enabled: bool = False,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.shared_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            make_activation(activation),
            nn.Dropout(dropout),
        )
        self.regression_head = nn.Linear(hidden_dim, output_dim)
        self.motion_head = nn.Linear(hidden_dim, 1) if motion_head_enabled else None
        self.log_variance_head = nn.Linear(hidden_dim, output_dim) if hybrid_head_enabled else None
        self.bias_head = nn.Linear(hidden_dim, output_dim) if hybrid_bias_correction_enabled else None

    def forward(
        self,
        inputs: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        sequence = inputs.transpose(1, 2)
        if lengths is not None:
            packed = pack_padded_sequence(
                sequence,
                lengths.detach().cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            _, (hidden, _) = self.lstm(packed)
            pooled = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            outputs, _ = self.lstm(sequence)
            pooled = outputs.mean(dim=1)
        shared = self.shared_head(pooled)
        motion_logits = self.motion_head(shared) if self.motion_head is not None else None
        log_variance = self.log_variance_head(shared) if self.log_variance_head is not None else None
        bias_correction = self.bias_head(shared) if self.bias_head is not None else None
        return format_model_outputs(
            self.regression_head(shared),
            motion_logits,
            log_variance=log_variance,
            bias_correction=bias_correction,
        )


def build_model(
    model_type: str,
    input_channels: int,
    output_dim: int,
    conv_channels: Sequence[int],
    kernel_size: int,
    hidden_dim: int,
    dropout: float,
    activation: str,
    use_residual: bool,
    motion_head_enabled: bool = False,
    hybrid_head_enabled: bool = False,
    hybrid_bias_correction_enabled: bool = False,
) -> nn.Module:
    if model_type == "temporal_cnn":
        return TemporalCnnRegressor(
            input_channels=input_channels,
            output_dim=output_dim,
            channels=conv_channels,
            kernel_size=kernel_size,
            hidden_dim=hidden_dim,
            dropout=dropout,
            activation=activation,
            use_residual=use_residual,
            motion_head_enabled=motion_head_enabled,
            hybrid_head_enabled=hybrid_head_enabled,
            hybrid_bias_correction_enabled=hybrid_bias_correction_enabled,
        )
    if model_type == "cnn_gru":
        return CnnGruRegressor(
            input_channels=input_channels,
            output_dim=output_dim,
            channels=conv_channels,
            kernel_size=kernel_size,
            hidden_dim=hidden_dim,
            dropout=dropout,
            activation=activation,
            motion_head_enabled=motion_head_enabled,
            hybrid_head_enabled=hybrid_head_enabled,
            hybrid_bias_correction_enabled=hybrid_bias_correction_enabled,
        )
    if model_type == "bilstm":
        return BiLstmRegressor(
            input_channels=input_channels,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            activation=activation,
            num_layers=2,
            motion_head_enabled=motion_head_enabled,
            hybrid_head_enabled=hybrid_head_enabled,
            hybrid_bias_correction_enabled=hybrid_bias_correction_enabled,
        )
    raise ValueError(f"Unsupported model type: {model_type}")
