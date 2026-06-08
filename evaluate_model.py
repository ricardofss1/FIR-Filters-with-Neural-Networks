import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy import signal
from torch.utils.data import DataLoader, Subset, random_split

from dataset_generator import evaluate_candidate_against_request
from fir_utils import build_response_masks, compute_comparison_metrics, design_reference_filter, format_metric
from hybrid_dataset import ParamDataset
from hybrid_helpers import ADJUSTABLE_PARAM_NAMES, decode_outputs, synthesize_fir
from hybrid_predict import load_paramnet
from hybrid_train import weighted_smooth_l1


SIMILARITY_METRIC_NAMES = ("mae_passband", "mae_stopband", "correlation", "erle")
FILTER_VARIANTS = ("predicted", "target", "direct")
GROUP_NAMES = ("overall", "lowpass", "highpass")


def request_spec_from_row(row):
    return {
        "fc": float(row[0]),
        "trans": float(row[1]),
        "Rp": float(row[2]),
        "As": float(row[3]),
        "order": int(round(float(row[4]))),
        "type": "lowpass" if int(round(float(row[5]))) == 0 else "highpass",
    }


def compute_band_metrics(h_coeffs, request_spec, n_fft=2048):
    freqs, response = torch_signal_freqz(h_coeffs, n_fft=n_fft)
    response_db = 20 * np.log10(np.maximum(np.abs(response), 1e-10))
    passband_mask, stopband_mask = build_response_masks(
        freqs, request_spec["fc"], request_spec["trans"], request_spec["type"]
    )

    ripple = float("nan")
    attenuation = float("nan")
    if np.any(passband_mask):
        ripple = float(np.max(response_db[passband_mask]) - np.min(response_db[passband_mask]))
    if np.any(stopband_mask):
        attenuation = float(-np.max(response_db[stopband_mask]))

    meets_ripple = (not math.isnan(ripple)) and ripple <= request_spec["Rp"]
    meets_attenuation = (not math.isnan(attenuation)) and attenuation >= request_spec["As"]
    return ripple, attenuation, meets_ripple, meets_attenuation


def torch_signal_freqz(h_coeffs, n_fft=2048):
    return signal.freqz(h_coeffs, worN=n_fft, fs=1.0)


def safe_stats(values):
    if not values:
        return {"mean": float("nan"), "median": float("nan"), "p90": float("nan")}

    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "p90": float("nan")}

    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.quantile(arr, 0.9)),
    }


def init_group_bucket():
    return {
        "count": 0,
        "parameter_abs_error": {name: [] for name in ADJUSTABLE_PARAM_NAMES},
        "similarity_vs_target": {name: [] for name in SIMILARITY_METRIC_NAMES},
        "similarity_vs_direct": {name: [] for name in SIMILARITY_METRIC_NAMES},
        "band_metrics": {
            variant: {
                "ripple_db": [],
                "attenuation_db": [],
                "meets_ripple": 0,
                "meets_attenuation": 0,
                "meets_both": 0,
            }
            for variant in FILTER_VARIANTS
        },
        "score_against_request": {variant: [] for variant in FILTER_VARIANTS},
    }


def safe_rate(numerator, denominator):
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def summarize_group(bucket):
    count = bucket["count"]
    score_pred = np.asarray(bucket["score_against_request"]["predicted"], dtype=np.float64)
    score_target = np.asarray(bucket["score_against_request"]["target"], dtype=np.float64)
    score_direct = np.asarray(bucket["score_against_request"]["direct"], dtype=np.float64)

    summary = {
        "count": count,
        "parameter_mae": {
            name: safe_stats(values)["mean"]
            for name, values in bucket["parameter_abs_error"].items()
        },
        "similarity_vs_target": {
            metric: safe_stats(values)
            for metric, values in bucket["similarity_vs_target"].items()
        },
        "similarity_vs_direct": {
            metric: safe_stats(values)
            for metric, values in bucket["similarity_vs_direct"].items()
        },
        "band_metrics": {},
        "score_against_request": {
            variant: safe_stats(values)
            for variant, values in bucket["score_against_request"].items()
        },
    }

    for variant, variant_bucket in bucket["band_metrics"].items():
        summary["band_metrics"][variant] = {
            "ripple_db": safe_stats(variant_bucket["ripple_db"]),
            "attenuation_db": safe_stats(variant_bucket["attenuation_db"]),
            "meets_ripple_rate": safe_rate(variant_bucket["meets_ripple"], count),
            "meets_attenuation_rate": safe_rate(variant_bucket["meets_attenuation"], count),
            "meets_both_rate": safe_rate(variant_bucket["meets_both"], count),
        }

    if count > 0:
        summary["score_against_request"]["predicted_better_than_direct_rate"] = float(
            np.mean(score_pred < score_direct)
        )
        summary["score_against_request"]["predicted_better_than_direct_by_1_rate"] = float(
            np.mean((score_direct - score_pred) > 1.0)
        )
        summary["score_against_request"]["predicted_better_than_target_rate"] = float(
            np.mean(score_pred < score_target)
        )
        summary["score_against_request"]["predicted_close_to_target_rate"] = float(
            np.mean(np.abs(score_pred - score_target) < 0.5)
        )
    else:
        summary["score_against_request"]["predicted_better_than_direct_rate"] = float("nan")
        summary["score_against_request"]["predicted_better_than_direct_by_1_rate"] = float("nan")
        summary["score_against_request"]["predicted_better_than_target_rate"] = float("nan")
        summary["score_against_request"]["predicted_close_to_target_rate"] = float("nan")

    return summary


def make_json_safe(obj):
    if isinstance(obj, dict):
        return {key: make_json_safe(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(value) for value in obj]
    if isinstance(obj, tuple):
        return [make_json_safe(value) for value in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def direct_design_from_request(request_spec, method):
    try:
        return design_reference_filter(
            fc=request_spec["fc"],
            trans=request_spec["trans"],
            rp=request_spec["Rp"],
            attenuation_db=request_spec["As"],
            order=request_spec["order"],
            ftype=request_spec["type"],
            method=method,
            fs=1.0,
        )
    except Exception:
        return design_reference_filter(
            fc=request_spec["fc"],
            trans=request_spec["trans"],
            rp=request_spec["Rp"],
            attenuation_db=request_spec["As"],
            order=request_spec["order"],
            ftype=request_spec["type"],
            method="firwin",
            fs=1.0,
        )


def evaluate_checkpoint(
    checkpoint_path,
    dataset_path,
    method,
    batch_size,
    val_split,
    seed,
    n_fft,
    num_workers,
    max_samples=None,
):
    dataset = ParamDataset(dataset_path)
    total_count = len(dataset)
    val_count = int(total_count * val_split)
    test_count = val_count
    train_count = total_count - val_count - test_count
    _, _, test_ds = random_split(
        dataset,
        [train_count, val_count, test_count],
        generator=torch.Generator().manual_seed(seed),
    )

    if max_samples is not None:
        capped_indices = list(range(min(max_samples, len(test_ds))))
        test_ds = Subset(test_ds, capped_indices)

    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model, predictor_state, device = load_paramnet(checkpoint_path)

    groups = {name: init_group_bucket() for name in GROUP_NAMES}
    total_loss = 0.0
    total_samples = 0

    model.eval()
    with torch.no_grad():
        for batch in loader:
            x_data = batch["x"].to(device)
            y_data = batch["y"].to(device)
            prediction = model(x_data)
            loss = weighted_smooth_l1(prediction, y_data)

            batch_size_actual = x_data.size(0)
            total_loss += loss.item() * batch_size_actual
            total_samples += batch_size_actual

            specs = batch["spec"].cpu().numpy()
            target_specs = batch["target_spec"].cpu().numpy()
            target_coefs = batch["coef"].cpu().numpy()
            target_orders = batch["order"].cpu().numpy()

            pred_params = decode_outputs(
                prediction.detach().cpu().numpy(),
                predictor_state["target_scaler"],
                input_specs=specs,
            )
            pred_coefs = synthesize_fir(pred_params, method=method)

            for index in range(batch_size_actual):
                request_spec = request_spec_from_row(specs[index])
                request_type = request_spec["type"]
                group_keys = ("overall", request_type)
                target_taps = target_coefs[index][: int(target_orders[index])]
                pred_taps = pred_coefs[index]
                direct_taps = direct_design_from_request(request_spec, method)

                target_similarity = compute_comparison_metrics(
                    pred_taps,
                    target_taps,
                    request_spec["fc"],
                    request_spec["trans"],
                    request_type,
                    fs=1.0,
                    n_fft=n_fft,
                )
                direct_similarity = compute_comparison_metrics(
                    pred_taps,
                    direct_taps,
                    request_spec["fc"],
                    request_spec["trans"],
                    request_type,
                    fs=1.0,
                    n_fft=n_fft,
                )

                pred_band = compute_band_metrics(pred_taps, request_spec, n_fft=n_fft)
                target_band = compute_band_metrics(target_taps, request_spec, n_fft=n_fft)
                direct_band = compute_band_metrics(direct_taps, request_spec, n_fft=n_fft)

                pred_score = evaluate_candidate_against_request(pred_taps, request_spec, n_fft=n_fft)[0]
                target_score = evaluate_candidate_against_request(target_taps, request_spec, n_fft=n_fft)[0]
                direct_score = evaluate_candidate_against_request(direct_taps, request_spec, n_fft=n_fft)[0]

                for group_key in group_keys:
                    bucket = groups[group_key]
                    bucket["count"] += 1

                    for param_index, param_name in enumerate(ADJUSTABLE_PARAM_NAMES):
                        bucket["parameter_abs_error"][param_name].append(
                            abs(float(pred_params[param_name][index]) - float(target_specs[index, param_index]))
                        )

                    for metric_name, metric_value in target_similarity.items():
                        bucket["similarity_vs_target"][metric_name].append(metric_value)
                    for metric_name, metric_value in direct_similarity.items():
                        bucket["similarity_vs_direct"][metric_name].append(metric_value)

                    for variant_name, band_values in (
                        ("predicted", pred_band),
                        ("target", target_band),
                        ("direct", direct_band),
                    ):
                        ripple, attenuation, meets_ripple, meets_attenuation = band_values
                        band_bucket = bucket["band_metrics"][variant_name]
                        band_bucket["ripple_db"].append(ripple)
                        band_bucket["attenuation_db"].append(attenuation)
                        band_bucket["meets_ripple"] += int(meets_ripple)
                        band_bucket["meets_attenuation"] += int(meets_attenuation)
                        band_bucket["meets_both"] += int(meets_ripple and meets_attenuation)

                    bucket["score_against_request"]["predicted"].append(pred_score)
                    bucket["score_against_request"]["target"].append(target_score)
                    bucket["score_against_request"]["direct"].append(direct_score)

    report = {
        "metadata": {
            "checkpoint": str(Path(checkpoint_path).resolve()),
            "dataset": str(Path(dataset_path).resolve()),
            "method": method,
            "seed": seed,
            "val_split": val_split,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "n_fft": n_fft,
            "evaluated_test_samples": total_samples,
            "full_dataset_size": total_count,
            "train_split_size": train_count,
            "val_split_size": val_count,
            "test_split_size": test_count,
            "used_max_samples": max_samples,
        },
        "regression": {
            "weighted_smooth_l1_loss": float(total_loss / max(total_samples, 1)),
            "parameter_mae": summarize_group(groups["overall"])["parameter_mae"],
        },
        "overall": summarize_group(groups["overall"]),
        "by_type": {
            "lowpass": summarize_group(groups["lowpass"]),
            "highpass": summarize_group(groups["highpass"]),
        },
    }
    return report


def print_report(report):
    metadata = report["metadata"]
    overall = report["overall"]
    regression = report["regression"]

    print("=" * 72)
    print("MODEL EVALUATION REPORT")
    print("=" * 72)
    print(f"Checkpoint: {metadata['checkpoint']}")
    print(f"Dataset:    {metadata['dataset']}")
    print(
        f"Split sizes: train={metadata['train_split_size']}, val={metadata['val_split_size']}, "
        f"test={metadata['test_split_size']} | evaluated={metadata['evaluated_test_samples']}"
    )
    print(
        f"Method={metadata['method']} | n_fft={metadata['n_fft']} | batch_size={metadata['batch_size']} | seed={metadata['seed']}"
    )

    print("\nRegression")
    print(f"  weighted_smooth_l1_loss: {format_metric(regression['weighted_smooth_l1_loss'])}")
    for name, value in regression["parameter_mae"].items():
        print(f"  {name}_mae: {format_metric(value)}")

    print("\nSimilarity Vs Target")
    for metric_name, stats in overall["similarity_vs_target"].items():
        print(
            f"  {metric_name}: mean={format_metric(stats['mean'])}, "
            f"median={format_metric(stats['median'])}, p90={format_metric(stats['p90'])}"
        )

    print("\nSimilarity Vs Direct")
    for metric_name, stats in overall["similarity_vs_direct"].items():
        print(
            f"  {metric_name}: mean={format_metric(stats['mean'])}, "
            f"median={format_metric(stats['median'])}, p90={format_metric(stats['p90'])}"
        )

    print("\nBand Metrics")
    for variant_name, band_summary in overall["band_metrics"].items():
        print(
            f"  {variant_name}: ripple_mean={format_metric(band_summary['ripple_db']['mean'])} dB, "
            f"attenuation_mean={format_metric(band_summary['attenuation_db']['mean'])} dB, "
            f"meet_both_rate={format_metric(band_summary['meets_both_rate'])}"
        )

    print("\nScore Against Request")
    for variant_name in FILTER_VARIANTS:
        stats = overall["score_against_request"][variant_name]
        print(
            f"  {variant_name}: mean={format_metric(stats['mean'])}, "
            f"median={format_metric(stats['median'])}, p90={format_metric(stats['p90'])}"
        )
    print(
        f"  predicted_better_than_direct_rate: "
        f"{format_metric(overall['score_against_request']['predicted_better_than_direct_rate'])}"
    )
    print(
        f"  predicted_better_than_direct_by_1_rate: "
        f"{format_metric(overall['score_against_request']['predicted_better_than_direct_by_1_rate'])}"
    )
    print(
        f"  predicted_better_than_target_rate: "
        f"{format_metric(overall['score_against_request']['predicted_better_than_target_rate'])}"
    )
    print(
        f"  predicted_close_to_target_rate: "
        f"{format_metric(overall['score_against_request']['predicted_close_to_target_rate'])}"
    )

    print("\nBy Type")
    for filter_type in ("lowpass", "highpass"):
        group = report["by_type"][filter_type]
        print(
            f"  {filter_type}: count={group['count']}, "
            f"score_mean={format_metric(group['score_against_request']['predicted']['mean'])}, "
            f"meet_both_rate={format_metric(group['band_metrics']['predicted']['meets_both_rate'])}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint on the test split and generate a consolidated report."
    )
    parser.add_argument("--checkpoint", type=str, default="checkpoints_hybrid/best_paramnet.pth")
    parser.add_argument("--dataset", type=str, default="fir_dataset_adjusted_firwin_v2.npz")
    parser.add_argument("--method", type=str, choices=["remez", "firwin"], default="firwin")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-fft", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--json-out", type=str, default=None, help="Optional path to save the report as JSON.")
    args = parser.parse_args()

    start_time = time.time()
    report = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        dataset_path=args.dataset,
        method=args.method,
        batch_size=args.batch_size,
        val_split=args.val_split,
        seed=args.seed,
        n_fft=args.n_fft,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
    )
    report["metadata"]["elapsed_seconds"] = time.time() - start_time

    print_report(report)

    if args.json_out:
        output_path = Path(args.json_out)
        output_path.write_text(
            json.dumps(make_json_safe(report), indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        print(f"\nSaved JSON report to {output_path.resolve()}")


if __name__ == "__main__":
    main()
