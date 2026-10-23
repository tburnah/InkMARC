from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import torch
from torch.utils.data import DataLoader

from ml.inkloc.models import LocTouchNetLite
from ml.inkloc.training.forward import model_forward
from ml.inkloc.training.train import build_dataset
from ml.inkloc.utils.config import load_config


GUARDRAIL_HOVER_AS_TOUCH = 456
REFERENCE = {
    "fixed_f1": 0.660,
    "best_f1": 0.682,
    "state_acc": 0.733,
    "xy_px": 58.0,
}


@dataclass
class Candidate:
    name: str
    checkpoint: str
    config: str


def default_candidates() -> List[Candidate]:
    candidates = [
        Candidate("selected", "/home/t/tburnah/ml/artifacts/checkpoints/selected.pt", "/home/t/tburnah/ml/configs/loctouch_lite.yaml"),
        Candidate("C_touch_pos_weight", "/home/t/tburnah/ml/artifacts/experiments/C_touch_pos_weight/checkpoints/best_touch_f1.pt", "/home/t/tburnah/ml/artifacts/experiments/hardmine_configs/C_touch_pos_weight.yaml"),
        Candidate("F2", "/home/t/tburnah/ml/artifacts/experiments/F2_touch_as_hover_x2_hover_as_touch_x1_5/checkpoints/best_touch_f1.pt", "/home/t/tburnah/ml/artifacts/experiments/constrained_hover_touch_configs/F2_touch_as_hover_x2_hover_as_touch_x1_5.yaml"),
        Candidate("F3", "/home/t/tburnah/ml/artifacts/experiments/F3_touch_as_hover_x3_hover_as_touch_x1/checkpoints/best_touch_f1.pt", "/home/t/tburnah/ml/artifacts/experiments/constrained_hover_touch_configs/F3_touch_as_hover_x3_hover_as_touch_x1.yaml"),
        Candidate("F4", "/home/t/tburnah/ml/artifacts/experiments/F4_touch_as_hover_x3_hover_as_touch_x1_5/checkpoints/best_touch_f1.pt", "/home/t/tburnah/ml/artifacts/experiments/constrained_hover_touch_configs/F4_touch_as_hover_x3_hover_as_touch_x1_5.yaml"),
        Candidate("F5", "/home/t/tburnah/ml/artifacts/experiments/F5_touch_as_hover_x3_with_batch_cap_20_percent/checkpoints/best_touch_f1.pt", "/home/t/tburnah/ml/artifacts/experiments/constrained_hover_touch_configs/F5_touch_as_hover_x3_with_batch_cap_20_percent.yaml"),
    ]
    return [c for c in candidates if Path(c.checkpoint).exists()]


def parse_candidate(spec: str) -> Candidate:
    parts = spec.split("::")
    if len(parts) != 3:
        raise ValueError(f"candidate must be name::checkpoint::config, got: {spec}")
    name, checkpoint, config = parts
    return Candidate(name=name, checkpoint=checkpoint, config=config)


def binary_metrics(pred_touch: torch.Tensor, true_touch: torch.Tensor) -> Dict[str, float]:
    tp = int((pred_touch & true_touch).sum().item())
    fp = int((pred_touch & ~true_touch).sum().item())
    fn = int((~pred_touch & true_touch).sum().item())
    tn = int((~pred_touch & ~true_touch).sum().item())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


def summarize_rule(
    name: str,
    params: Dict[str, float],
    pred_state: torch.Tensor,
    true_state: torch.Tensor,
    xy_pred: torch.Tensor,
    xy_true: torch.Tensor,
    has_xy: torch.Tensor,
    participant_id: torch.Tensor,
    touch_prob: torch.Tensor,
) -> Dict:
    true_touch = true_state == 2
    pred_touch = pred_state == 2
    touch = binary_metrics(pred_touch, true_touch)
    confusion = torch.zeros(3, 3, dtype=torch.long)
    for t, p in zip(true_state, pred_state):
        confusion[int(t), int(p)] += 1
    xy_px = torch.linalg.vector_norm((xy_pred - xy_true) * torch.tensor([480.0, 320.0]), dim=1)

    def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
        return float(values[mask].mean().item()) if bool(mask.any()) else float("nan")

    by_participant = {}
    for pid in torch.unique(participant_id):
        mask = participant_id == pid
        if not bool(mask.any()):
            continue
        part_touch = binary_metrics(pred_touch[mask], true_touch[mask])
        by_participant[str(int(pid.item()))] = {
            "count": int(mask.sum().item()),
            "touch_f1": part_touch["f1"],
            "touch_precision": part_touch["precision"],
            "touch_recall": part_touch["recall"],
        }

    return {
        "rule": name,
        "params": params,
        "touch_at_rule": touch,
        "touch_as_hover": int(confusion[2, 1].item()),
        "hover_as_touch": int(confusion[1, 2].item()),
        "state_accuracy": float((pred_state == true_state).float().mean().item()),
        "xy_mean_pixel_error": masked_mean(xy_px, has_xy),
        "mean_touch_probability_true_touch": masked_mean(touch_prob, true_touch),
        "mean_touch_probability_non_touch": masked_mean(touch_prob, ~true_touch),
        "confusion_matrix": confusion.tolist(),
        "by_participant": by_participant,
    }


def fallback_state(p_bg: torch.Tensor, p_hover: torch.Tensor) -> torch.Tensor:
    return torch.where(p_hover >= p_bg, torch.ones_like(p_bg, dtype=torch.long), torch.zeros_like(p_bg, dtype=torch.long))


def rule_simple(data: Dict, threshold: float) -> torch.Tensor:
    touch = data["p_touch"] >= threshold
    base = fallback_state(data["p_bg"], data["p_hover"])
    return torch.where(touch, torch.full_like(base, 2), base)


def rule_state_aware(data: Dict, threshold: float, hover_max: float) -> torch.Tensor:
    touch = (data["p_touch"] >= threshold) & (data["p_hover"] <= hover_max)
    base = fallback_state(data["p_bg"], data["p_hover"])
    return torch.where(touch, torch.full_like(base, 2), base)


def rule_margin(data: Dict, margin: float) -> torch.Tensor:
    touch = (data["p_touch"] - data["p_hover"]) >= margin
    base = fallback_state(data["p_bg"], data["p_hover"])
    return torch.where(touch, torch.full_like(base, 2), base)


def rule_valid_gated(data: Dict, threshold: float, valid_min: float) -> torch.Tensor:
    touch = (data["p_touch"] >= threshold) & (data["p_valid"] >= valid_min)
    base = fallback_state(data["p_bg"], data["p_hover"])
    return torch.where(touch, torch.full_like(base, 2), base)


def rule_hysteresis(data: Dict, touch_on: float, touch_off: float) -> torch.Tensor:
    base = fallback_state(data["p_bg"], data["p_hover"])
    pred = base.clone()
    seq = data["sequence_id"]
    frame = data["frame_idx"]
    touch_prob = data["p_touch"]
    for seq_id in torch.unique(seq):
        mask = seq == seq_id
        order = torch.argsort(frame[mask])
        idx = torch.where(mask)[0][order]
        active = False
        for i in idx.tolist():
            prob = float(touch_prob[i].item())
            if active:
                if prob < touch_off:
                    active = False
            else:
                if prob >= touch_on:
                    active = True
            if active:
                pred[i] = 2
    return pred


def choose_better(a: Dict | None, b: Dict, constrained: bool = False) -> Dict:
    if a is None:
        return b
    a_touch = a["touch_at_rule"]["f1"]
    b_touch = b["touch_at_rule"]["f1"]
    if constrained:
        a_ok = a["hover_as_touch"] <= GUARDRAIL_HOVER_AS_TOUCH
        b_ok = b["hover_as_touch"] <= GUARDRAIL_HOVER_AS_TOUCH
        if a_ok != b_ok:
            return b if b_ok else a
    if b_touch != a_touch:
        return b if b_touch > a_touch else a
    if b["touch_as_hover"] != a["touch_as_hover"]:
        return b if b["touch_as_hover"] < a["touch_as_hover"] else a
    if b["state_accuracy"] != a["state_accuracy"]:
        return b if b["state_accuracy"] > a["state_accuracy"] else a
    return b if b["xy_mean_pixel_error"] < a["xy_mean_pixel_error"] else a


@torch.no_grad()
def collect_outputs(candidate: Candidate, device: torch.device, touch_ablation_mode: str = "rgb_motion") -> Dict:
    cfg = load_config(candidate.config)
    ckpt = torch.load(candidate.checkpoint, map_location=device, weights_only=False)
    model = LocTouchNetLite.from_config(cfg).to(device).eval()
    model.load_state_dict(ckpt["model"])
    val_ds = build_dataset(cfg, split="eval")
    loader = DataLoader(
        val_ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["training"].get("num_workers", 0)),
        pin_memory=device.type == "cuda",
        persistent_workers=int(cfg["training"].get("num_workers", 0)) > 0,
    )
    chunks = {k: [] for k in ["p_bg", "p_hover", "p_touch", "p_valid", "xy_pred", "xy_true", "has_xy", "state", "participant_id", "sequence_id", "frame_idx"]}
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        out = model_forward(model, batch, touch_ablation_mode=touch_ablation_mode)
        chunks["p_bg"].append(out["state_probs"][:, 0].detach().cpu())
        chunks["p_hover"].append(out["state_probs"][:, 1].detach().cpu())
        chunks["p_touch"].append(out["state_probs"][:, 2].detach().cpu())
        chunks["p_valid"].append(torch.sigmoid(out["valid_logit"]).detach().cpu())
        chunks["xy_pred"].append(out["xy"].detach().cpu())
        chunks["xy_true"].append(batch["xy"].detach().cpu())
        chunks["has_xy"].append(batch["has_xy"].detach().cpu().bool())
        chunks["state"].append(batch["state"].detach().cpu().long())
        chunks["participant_id"].append(batch["participant_id"].detach().cpu().long())
        chunks["sequence_id"].append(batch["sequence_id"].detach().cpu().long())
        chunks["frame_idx"].append(batch["frame_idx"].detach().cpu().long())
    return {k: torch.cat(v, dim=0) for k, v in chunks.items()}


def sweep_rules(data: Dict) -> Dict[str, Dict]:
    results = {}
    fixed = {}
    calibrated = {}

    fixed["A_simple"] = summarize_rule("A_simple", {"touch_threshold": 0.5}, rule_simple(data, 0.5), data["state"], data["xy_pred"], data["xy_true"], data["has_xy"], data["participant_id"], data["p_touch"])
    best = None
    for threshold in torch.linspace(0.01, 0.99, 99):
        report = summarize_rule("A_simple", {"touch_threshold": float(threshold.item())}, rule_simple(data, float(threshold.item())), data["state"], data["xy_pred"], data["xy_true"], data["has_xy"], data["participant_id"], data["p_touch"])
        best = choose_better(best, report, constrained=True)
    calibrated["A_simple"] = best

    fixed["B_state_aware"] = summarize_rule("B_state_aware", {"touch_threshold": 0.5, "hover_max": 0.5}, rule_state_aware(data, 0.5, 0.5), data["state"], data["xy_pred"], data["xy_true"], data["has_xy"], data["participant_id"], data["p_touch"])
    best = None
    grid = [round(x, 2) for x in torch.linspace(0.05, 0.95, 19).tolist()]
    for threshold in grid:
        for hover_max in grid:
            report = summarize_rule("B_state_aware", {"touch_threshold": threshold, "hover_max": hover_max}, rule_state_aware(data, threshold, hover_max), data["state"], data["xy_pred"], data["xy_true"], data["has_xy"], data["participant_id"], data["p_touch"])
            best = choose_better(best, report, constrained=True)
    calibrated["B_state_aware"] = best

    fixed["C_margin"] = summarize_rule("C_margin", {"margin": 0.0}, rule_margin(data, 0.0), data["state"], data["xy_pred"], data["xy_true"], data["has_xy"], data["participant_id"], data["p_touch"])
    best = None
    for margin in torch.linspace(0.0, 0.6, 31):
        report = summarize_rule("C_margin", {"margin": float(margin.item())}, rule_margin(data, float(margin.item())), data["state"], data["xy_pred"], data["xy_true"], data["has_xy"], data["participant_id"], data["p_touch"])
        best = choose_better(best, report, constrained=True)
    calibrated["C_margin"] = best

    fixed["D_valid_gated"] = summarize_rule("D_valid_gated", {"touch_threshold": 0.5, "valid_min": 0.5}, rule_valid_gated(data, 0.5, 0.5), data["state"], data["xy_pred"], data["xy_true"], data["has_xy"], data["participant_id"], data["p_touch"])
    best = None
    for threshold in grid:
        for valid_min in grid:
            report = summarize_rule("D_valid_gated", {"touch_threshold": threshold, "valid_min": valid_min}, rule_valid_gated(data, threshold, valid_min), data["state"], data["xy_pred"], data["xy_true"], data["has_xy"], data["participant_id"], data["p_touch"])
            best = choose_better(best, report, constrained=True)
    calibrated["D_valid_gated"] = best

    fixed["E_hysteresis"] = summarize_rule("E_hysteresis", {"touch_on": 0.5, "touch_off": 0.3}, rule_hysteresis(data, 0.5, 0.3), data["state"], data["xy_pred"], data["xy_true"], data["has_xy"], data["participant_id"], data["p_touch"])
    best = None
    for touch_on in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        for touch_off in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
            if touch_off >= touch_on:
                continue
            report = summarize_rule("E_hysteresis", {"touch_on": touch_on, "touch_off": touch_off}, rule_hysteresis(data, touch_on, touch_off), data["state"], data["xy_pred"], data["xy_true"], data["has_xy"], data["participant_id"], data["p_touch"])
            best = choose_better(best, report, constrained=True)
    calibrated["E_hysteresis"] = best

    results["fixed"] = fixed
    results["calibrated"] = calibrated
    return results


def selection_passes(report: Dict) -> bool:
    return report["hover_as_touch"] <= GUARDRAIL_HOVER_AS_TOUCH


def better_than_reference(report: Dict) -> bool:
    score = 0
    if report["touch_at_rule"]["f1"] > REFERENCE["fixed_f1"]:
        score += 1
    if report["touch_at_rule"]["f1"] > REFERENCE["best_f1"]:
        score += 1
    if report["state_accuracy"] > REFERENCE["state_acc"]:
        score += 1
    if report["xy_mean_pixel_error"] <= REFERENCE["xy_px"] + 10.0:
        score += 1
    return score >= 1


def best_global_choice(all_reports: Dict[str, Dict]) -> tuple[str, str, Dict] | tuple[None, None, None]:
    best = None
    best_name = None
    best_rule = None
    for model_name, reports in all_reports.items():
        for rule_name, report in reports["calibrated"].items():
            if not selection_passes(report):
                continue
            if not better_than_reference(report):
                continue
            key = (report["touch_at_rule"]["f1"], -report["touch_as_hover"], report["state_accuracy"], -report["xy_mean_pixel_error"])
            if best is None or key > (best["touch_at_rule"]["f1"], -best["touch_as_hover"], best["state_accuracy"], -best["xy_mean_pixel_error"]):
                best = report
                best_name = model_name
                best_rule = rule_name
    return best_name, best_rule, best


def build_android_config(model_name: str, checkpoint: str, rule_name: str, report: Dict) -> Dict:
    return {
        "model_name": model_name,
        "checkpoint": checkpoint,
        "rule": rule_name,
        "params": report["params"],
        "guardrail_hover_as_touch_max": GUARDRAIL_HOVER_AS_TOUCH,
        "touch_f1": report["touch_at_rule"]["f1"],
        "touch_precision": report["touch_at_rule"]["precision"],
        "touch_recall": report["touch_at_rule"]["recall"],
        "hover_as_touch": report["hover_as_touch"],
        "touch_as_hover": report["touch_as_hover"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="/home/t/tburnah/ml/artifacts/calibration/decision_sweep.json")
    parser.add_argument("--selected-thresholds-out", default="/home/t/tburnah/ml/artifacts/calibration/decision_selected_thresholds.json")
    parser.add_argument("--android-config-out", default="/home/t/tburnah/ml/artifacts/calibration/android_decision_config.json")
    parser.add_argument("--selected-checkpoint-out", default="/home/t/tburnah/ml/artifacts/checkpoints/decision_selected.pt")
    parser.add_argument("--candidate", action="append", default=[], help="name::checkpoint::config")
    parser.add_argument("--touch-ablation-mode", choices=["rgb_motion", "no_motion", "motion_only"], default="rgb_motion")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidates = [parse_candidate(spec) for spec in args.candidate] if args.candidate else default_candidates()
    outputs = {}
    for candidate in candidates:
        data = collect_outputs(candidate, device, touch_ablation_mode=args.touch_ablation_mode)
        outputs[candidate.name] = {
            "checkpoint": candidate.checkpoint,
            "config": candidate.config,
            "touch_ablation_mode": args.touch_ablation_mode,
            **sweep_rules(data),
        }

    best_name, best_rule, best_report = best_global_choice(outputs)
    payload = {
        "guardrail_hover_as_touch_max": GUARDRAIL_HOVER_AS_TOUCH,
        "reference": REFERENCE,
        "models": outputs,
        "selected_model": best_name,
        "selected_rule": best_rule,
        "selected_report": best_report,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if best_name is not None:
        chosen = next(c for c in candidates if c.name == best_name)
        threshold_payload = {
            "rule": best_rule,
            "params": best_report["params"],
            "touch_f1": best_report["touch_at_rule"]["f1"],
            "touch_precision": best_report["touch_at_rule"]["precision"],
            "touch_recall": best_report["touch_at_rule"]["recall"],
            "hover_as_touch": best_report["hover_as_touch"],
            "touch_as_hover": best_report["touch_as_hover"],
        }
        Path(args.selected_thresholds_out).write_text(json.dumps(threshold_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        Path(args.android_config_out).write_text(json.dumps(build_android_config(best_name, chosen.checkpoint, best_rule, best_report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if Path(chosen.checkpoint) != Path(args.selected_checkpoint_out):
            import shutil
            shutil.copyfile(chosen.checkpoint, args.selected_checkpoint_out)

    summary = {
        name: {
            rule: {
                "fixed_f1": reports["fixed"][rule]["touch_at_rule"]["f1"],
                "best_f1": reports["calibrated"][rule]["touch_at_rule"]["f1"],
                "hover_as_touch": reports["calibrated"][rule]["hover_as_touch"],
                "touch_as_hover": reports["calibrated"][rule]["touch_as_hover"],
                "state_acc": reports["calibrated"][rule]["state_accuracy"],
                "xy_px": reports["calibrated"][rule]["xy_mean_pixel_error"],
                "params": reports["calibrated"][rule]["params"],
            }
            for rule in reports["calibrated"]
        }
        for name, reports in outputs.items()
    }
    print(json.dumps({"selected_model": best_name, "selected_rule": best_rule, "summary": summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
