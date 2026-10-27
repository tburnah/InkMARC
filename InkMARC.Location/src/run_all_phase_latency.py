from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any


def _resolve_ckpt_fallback(save_dir: Path) -> Path | None:
    best = save_dir / "checkpoint_best.pt"
    if best.exists():
        return best

    epoch_ckpts = sorted(
        save_dir.glob("checkpoint_epoch*.pt"),
        key=lambda p: _extract_epoch_num(p.name),
    )
    if epoch_ckpts:
        return epoch_ckpts[-1]
    return None


def _select_save_dir(
    rel_dirs: List[str],
    preferred_root: Path,
    candidate_roots: List[Path],
) -> Path:
    all_candidates: List[Path] = []
    for rel_dir in rel_dirs:
        all_candidates.extend([preferred_root / rel_dir] + [r / rel_dir for r in candidate_roots])
    for c in all_candidates:
        if _resolve_ckpt_fallback(c) is not None:
            return c
    return all_candidates[0]


def _extract_epoch_num(name: str) -> int:
    m = re.search(r"checkpoint_epoch(\d+)\.pt$", name)
    if not m:
        return -1
    return int(m.group(1))


def _parse_latency(stdout: str) -> Dict[str, float] | None:
    line = None
    for ln in stdout.splitlines():
        if "latency per image (ms)" in ln:
            line = ln
    if line is None:
        return None

    vals: Dict[str, float] = {}
    for key in ["mean", "std", "p50", "p90", "p95", "min", "max"]:
        m = re.search(rf"{key}=([0-9]+(?:\.[0-9]+)?)", line)
        if m:
            vals[key] = float(m.group(1))
    return vals if vals else None


def _run_one(name: str, cmd: List[str], cwd: Path) -> Tuple[int, str]:
    print(f"\n=== {name} ===")
    print("$", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)

    # print tail for quick debugging context
    out_lines = (proc.stdout or "").splitlines()
    err_lines = (proc.stderr or "").splitlines()
    tail = out_lines[-20:] + err_lines[-20:]
    if tail:
        print("\n".join(tail))

    return proc.returncode, proc.stdout + "\n" + proc.stderr


def main() -> None:
    ap = argparse.ArgumentParser(description="Run eval-only latency checks for all phase scripts and summarize results.")
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--runs-root", default=str(Path.home() / "inkmarc" / "runs"))
    ap.add_argument("--eval-split", default="test", choices=["val", "test"])
    ap.add_argument("--eval-max-samples", type=int, default=100)
    ap.add_argument("--latency-runs", type=int, default=5)
    ap.add_argument("--latency-warmup", type=int, default=1)
    ap.add_argument("--device", default="")
    ap.add_argument("--num-workers", type=int, default=-1)
    ap.add_argument("--continue-on-error", action="store_true")
    ap.add_argument("--output", default="", help="Optional output txt/json path")
    args = ap.parse_args()

    src_dir = Path(__file__).resolve().parent
    runs_root = Path(args.runs_root).expanduser().resolve()
    repo_root = src_dir.parent
    alt_runs_roots = [
        runs_root,
        Path.home() / "runs",
        repo_root / "runs",
        src_dir / "runs",
    ]

    py = sys.executable

    phases: List[Dict[str, Any]] = [
        {
            "name": "phase0",
            "script": "train_phase0.py",
            "save_rels": ["phase0"],
            "flags": ["--eval-only"],
            "supports_eval_split": True,
            "supports_ckpt_arg": True,
        },
        {
            "name": "phase1",
            "script": "train_phase1.py",
            "save_rels": ["phase1"],
            "flags": ["--eval-test"],
            "supports_eval_split": True,
            "supports_ckpt_arg": True,
        },
        {
            "name": "phase1_v2_location",
            "script": "train_phase1_v2_location.py",
            "save_rels": ["phase1_v2_location"],
            "flags": ["--eval-only"],
            "supports_eval_split": True,
            "supports_ckpt_arg": True,
        },
        {
            "name": "phase1a",
            "script": "train_phase1a.py",
            "save_rels": ["phase1a_raw_plus_corners_xy"],
            "flags": ["--eval-only"],
            "supports_eval_split": True,
            "supports_ckpt_arg": True,
        },
        {
            "name": "phase2",
            "script": "train_phase2.py",
            "save_rels": ["phase2_heatmap_dsnt", "phase2_hm480x320_full", "phase2_hm240x160_full"],
            "flags": ["--eval-only"],
            "supports_eval_split": True,
            "supports_ckpt_arg": True,
        },
        {
            "name": "phase2a",
            "script": "train_phase2a.py",
            "save_rels": ["phase2a_heatmap_dsnt", "phase2_heatmap_dsnt"],
            "flags": ["--eval-only"],
            "supports_eval_split": True,
            "supports_ckpt_arg": True,
        },
        {
            "name": "phase2a_analyse",
            "script": "train_phase2a_analyse.py",
            "save_rels": ["phase2a_heatmap_dsnt", "phase2_heatmap_dsnt"],
            "flags": ["--eval-only"],
            "needs_resume": True,
            "supports_eval_split": False,
            "supports_ckpt_arg": False,
        },
        {
            "name": "phase2c",
            "script": "train_phase2c.py",
            "save_rels": ["phase2c_xy_history"],
            "flags": ["--eval-only"],
            "supports_eval_split": True,
            "supports_ckpt_arg": True,
        },
        {
            "name": "phase2d",
            "script": "train_phase2d.py",
            "save_rels": ["phase2d_xy_history", "phase2c_xy_history"],
            "flags": ["--eval-only"],
            "supports_eval_split": True,
            "supports_ckpt_arg": True,
        },
        {
            "name": "phase3",
            "script": "train_phase3.py",
            "save_rels": ["phase2a_heatmap_dsnt", "phase2_heatmap_dsnt"],
            "flags": [],
            "supports_eval_split": True,
            "supports_ckpt_arg": True,
        },
        {
            "name": "phase4_tsm",
            "script": "train_phase4_tsm.py",
            "save_rels": ["phase4_tsm_clip5_full_w12", "phase4_tsm"],
            "flags": ["--eval-only"],
            "supports_eval_split": True,
            "supports_ckpt_arg": True,
        },
        {
            "name": "phase5",
            "script": "train_phase5.py",
            "save_rels": ["phase5_transformer_short", "phase5_transformer"],
            "flags": ["--eval-only"],
            "supports_eval_split": True,
            "supports_ckpt_arg": True,
        },
        {
            "name": "phase6",
            "script": "train_phase6.py",
            "save_rels": ["phase6_xy_weight200_hm120", "phase2c_xy_history"],
            "flags": ["--eval-only"],
            "supports_eval_split": True,
            "supports_ckpt_arg": True,
        },
    ]

    results = []

    for ph in phases:
        save_dir = _select_save_dir(
            rel_dirs=ph["save_rels"],
            preferred_root=runs_root,
            candidate_roots=alt_runs_roots,
        )
        cmd = [
            py,
            str(src_dir / ph["script"]),
            "--dataset-root",
            args.dataset_root,
            "--save-dir",
            str(save_dir),
            "--eval-max-samples",
            str(args.eval_max_samples),
            "--latency-runs",
            str(args.latency_runs),
            "--latency-warmup",
            str(args.latency_warmup),
        ]
        if ph.get("supports_eval_split", True):
            cmd.extend(["--eval-split", args.eval_split])

        cmd.extend(ph["flags"])

        if args.device:
            cmd.extend(["--device", args.device])
        if args.num_workers >= 0:
            cmd.extend(["--num-workers", str(args.num_workers)])

        ckpt = _resolve_ckpt_fallback(save_dir)
        if ph.get("supports_ckpt_arg", False):
            if ckpt is None:
                msg = f"No checkpoint found for {ph['name']} in {save_dir}"
                if args.continue_on_error:
                    print(f"[skip] {msg}")
                    results.append({"name": ph["name"], "status": "missing_ckpt", "error": msg})
                    continue
                raise FileNotFoundError(msg)
            cmd.extend(["--ckpt", str(ckpt)])

        if ph.get("needs_resume"):
            if ckpt is None:
                msg = f"No checkpoint found for {ph['name']} in {save_dir}"
                if args.continue_on_error:
                    print(f"[skip] {msg}")
                    results.append({"name": ph["name"], "status": "missing_ckpt", "error": msg})
                    continue
                raise FileNotFoundError(msg)
            cmd.extend(["--resume", str(ckpt)])

        rc, out = _run_one(ph["name"], cmd, src_dir)
        lat = _parse_latency(out)

        if rc != 0:
            msg = f"{ph['name']} failed (exit {rc})"
            if args.continue_on_error:
                print(f"[warn] {msg}")
                results.append({"name": ph["name"], "status": "error", "error": msg})
                continue
            raise RuntimeError(msg)

        if lat is None:
            msg = f"{ph['name']} completed but no latency line found"
            if args.continue_on_error:
                print(f"[warn] {msg}")
                results.append({"name": ph["name"], "status": "no_latency", "error": msg})
                continue
            raise RuntimeError(msg)

        results.append({"name": ph["name"], "status": "ok", **lat})

    ok_results = [r for r in results if r.get("status") == "ok"]
    ok_results.sort(key=lambda r: r.get("mean", float("inf")))

    print("\n=== Latency Comparison (ms/image, lower is better) ===")
    for i, r in enumerate(ok_results, start=1):
        print(
            f"{i}. {r['name']}: mean={r['mean']:.3f}, p50={r['p50']:.3f}, "
            f"p90={r['p90']:.3f}, p95={r['p95']:.3f}, std={r['std']:.3f}"
        )

    bad = [r for r in results if r.get("status") != "ok"]
    if bad:
        print("\n=== Non-OK Runs ===")
        for r in bad:
            print(f"- {r['name']}: {r.get('status')} ({r.get('error', '')})")

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix.lower() == ".json":
            import json

            out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        else:
            lines = []
            lines.append("Latency Comparison (ms/image)")
            for i, r in enumerate(ok_results, start=1):
                lines.append(
                    f"{i}. {r['name']}: mean={r['mean']:.3f}, p50={r['p50']:.3f}, p90={r['p90']:.3f}, p95={r['p95']:.3f}, std={r['std']:.3f}"
                )
            if bad:
                lines.append("")
                lines.append("Non-OK Runs")
                for r in bad:
                    lines.append(f"- {r['name']}: {r.get('status')} ({r.get('error', '')})")
            out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nSaved summary to: {out_path}")


if __name__ == "__main__":
    main()
