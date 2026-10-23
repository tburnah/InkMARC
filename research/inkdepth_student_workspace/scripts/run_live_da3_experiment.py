#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/touch_live_da3_small_original_stride4_stable_w5_p1000.yaml"


@dataclass(frozen=True)
class CommandSpec:
    name: str
    argv: list[str]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the live DA3-small touch/location experiment.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--output-dir", default="outputs/touch_live_da3_small_original_stride4_stable_w5_p1000")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--all-variants",
        action="store_true",
        help="Override data.variant_kind_filter=null so frame manifests can use all variants.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["all"],
        choices=["all", "train", "eval_val", "eval_test", "hysteresis_val", "hysteresis_test"],
    )
    args = parser.parse_args()

    output_dir = resolve_path(args.output_dir)
    commands = select_commands(build_commands(args, output_dir), args.stages)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "commands_live_da3.json"
    manifest_path.write_text(json.dumps([spec.__dict__ for spec in commands], indent=2), encoding="utf-8")
    print(f"wrote {manifest_path}")

    for spec in commands:
        print(f"\n[{spec.name}]")
        print(" ".join(shell_quote(part) for part in spec.argv))
        if not args.dry_run:
            subprocess.run(spec.argv, cwd=str(ROOT), check=True)


def build_commands(args: argparse.Namespace, output_dir: Path) -> list[CommandSpec]:
    py = str(args.python)
    config = str(resolve_path(args.config))
    checkpoint = output_dir / "best_touch.pt"
    val_eval = output_dir / "eval_val_temporal"
    test_eval = output_dir / "eval_test_temporal"
    val_predictions = val_eval / "temporal_touch_frame_predictions.csv"
    test_predictions = test_eval / "temporal_touch_frame_predictions.csv"
    common_overrides = [
        f"output_dir={output_dir}",
        f"training.epochs={int(args.epochs)}",
        f"training.batch_size={int(args.batch_size)}",
        f"training.num_workers={int(args.num_workers)}",
    ]
    eval_overrides = [
        f"output_dir={output_dir}",
        f"training.batch_size={int(args.batch_size)}",
        f"training.num_workers={int(args.num_workers)}",
    ]
    if args.manifest_path:
        manifest_path = resolve_path(args.manifest_path)
        common_overrides.append(f"data.manifest_path={manifest_path}")
        eval_overrides.append(f"data.manifest_path={manifest_path}")
    if args.all_variants:
        common_overrides.append("data.variant_kind_filter=null")
        eval_overrides.append("data.variant_kind_filter=null")
    return [
        CommandSpec(
            "train",
            [
                py,
                str(ROOT / "scripts/train_student.py"),
                "--config",
                config,
                "--override",
                *common_overrides,
            ],
        ),
        CommandSpec(
            "eval_val",
            [
                py,
                str(ROOT / "scripts/evaluate_temporal_touch.py"),
                "--config",
                config,
                "--checkpoint",
                str(checkpoint),
                "--split",
                "val",
                "--output-dir",
                str(val_eval),
                "--save-frame-predictions",
                "--stable-only",
                "--override",
                *eval_overrides,
            ],
        ),
        CommandSpec(
            "eval_test",
            [
                py,
                str(ROOT / "scripts/evaluate_temporal_touch.py"),
                "--config",
                config,
                "--checkpoint",
                str(checkpoint),
                "--split",
                "test",
                "--output-dir",
                str(test_eval),
                "--save-frame-predictions",
                "--stable-only",
                "--override",
                *eval_overrides,
            ],
        ),
        CommandSpec(
            "hysteresis_val",
            [
                py,
                str(ROOT / "scripts/evaluate_touch_hysteresis.py"),
                "--predictions",
                str(val_predictions),
                "--output-dir",
                str(output_dir / "hysteresis_val"),
            ],
        ),
        CommandSpec(
            "hysteresis_test",
            [
                py,
                str(ROOT / "scripts/evaluate_touch_hysteresis.py"),
                "--predictions",
                str(test_predictions),
                "--output-dir",
                str(output_dir / "hysteresis_test"),
            ],
        ),
    ]


def select_commands(commands: list[CommandSpec], stages: list[str]) -> list[CommandSpec]:
    if "all" in stages:
        return commands
    wanted = set(stages)
    return [command for command in commands if command.name in wanted]


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def shell_quote(value: str) -> str:
    if not value:
        return "''"
    if all(ch.isalnum() or ch in "@%_+=:,./-" for ch in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


if __name__ == "__main__":
    main()
