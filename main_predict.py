import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from fir_utils import compute_comparison_metrics, design_reference_filter, export_coefficients, format_metric
from hybrid_predict import load_paramnet, predict_fir_from_specs


def plot_comprehensive_comparison(h_pred, h_design, specs, n_fft=2048):
    """Plot a side-by-side comparison between the hybrid and classical filters."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    freqs, pred_resp = signal.freqz(h_pred, worN=n_fft, fs=1.0)
    _, ref_resp = signal.freqz(h_design, worN=n_fft, fs=1.0)

    pred_db = 20 * np.log10(np.maximum(np.abs(pred_resp), 1e-10))
    ref_db = 20 * np.log10(np.maximum(np.abs(ref_resp), 1e-10))

    axes[0, 0].plot(freqs, ref_db, label="SciPy (direct design)", linewidth=2)
    axes[0, 0].plot(freqs, pred_db, "--", label="Hybrid prediction", alpha=0.8)
    axes[0, 0].set_title("Magnitude Response")
    axes[0, 0].set_xlabel("Normalized Frequency")
    axes[0, 0].set_ylabel("Magnitude (dB)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    pred_phase = np.unwrap(np.angle(pred_resp))
    ref_phase = np.unwrap(np.angle(ref_resp))
    axes[0, 1].plot(freqs, ref_phase, "--", label="SciPy (direct design)")
    axes[0, 1].plot(freqs, pred_phase, label="Hybrid prediction")
    axes[0, 1].set_title("Phase Response")
    axes[0, 1].set_xlabel("Normalized Frequency")
    axes[0, 1].set_ylabel("Phase (radians)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].stem(h_design, linefmt="C0-", markerfmt="C0o", basefmt="k-", label="SciPy")
    axes[1, 0].stem(h_pred, linefmt="C1--", markerfmt="C1x", basefmt="k-", label="Prediction")
    axes[1, 0].set_title("FIR Coefficients")
    axes[1, 0].set_xlabel("Coefficient Index")
    axes[1, 0].set_ylabel("Amplitude")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    error_db = pred_db - ref_db
    axes[1, 1].plot(freqs, error_db, "r-", label="Error")
    axes[1, 1].axhline(0, color="k", linestyle="--", alpha=0.5)
    axes[1, 1].set_title("Magnitude Error")
    axes[1, 1].set_xlabel("Normalized Frequency")
    axes[1, 1].set_ylabel("Error (dB)")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.suptitle(
        f"FIR {specs['type'].upper()} Comparison - fc={specs['fc']}, trans={specs['trans']}, "
        f"Rp={specs['Rp']} dB, As={specs['As']} dB, order={specs['order']}"
    )
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Hybrid FIR inference in normalized frequency.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_hybrid/best_paramnet.pth")
    parser.add_argument("--fc", type=float, default=0.25, help="Normalized cutoff frequency")
    parser.add_argument("--trans", type=float, default=0.05, help="Normalized transition width")
    parser.add_argument("--Rp", type=float, default=1.0, help="Passband ripple in dB")
    parser.add_argument("--As", type=float, default=60.0, help="Stopband attenuation in dB")
    parser.add_argument("--order", type=int, default=128, help="Requested number of taps")
    parser.add_argument("--type", type=str, choices=["lowpass", "highpass"], default="highpass")
    parser.add_argument("--method", type=str, choices=["remez", "firwin"], default="firwin")
    parser.add_argument("--export", type=str, choices=["txt", "python", "c", "matlab"])
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    if args.type == "lowpass" and args.fc + args.trans >= 0.5:
        raise ValueError("For lowpass, fc + trans must stay below Nyquist (0.5).")
    if args.type == "highpass" and args.fc - args.trans <= 0.0:
        raise ValueError("For highpass, fc - trans must stay above 0.0.")

    start_time = time.time()
    model, t_scaler, device = load_paramnet(args.checkpoint)

    type_val = 0 if args.type == "lowpass" else 1
    spec = np.array([[args.fc, args.trans, args.Rp, args.As, args.order, type_val]], dtype=np.float32)

    coefs_list, params = predict_fir_from_specs(spec, model, t_scaler, device=device, method=args.method)
    h_pred = coefs_list[0]

    h_design = design_reference_filter(
        fc=args.fc,
        trans=args.trans,
        rp=args.Rp,
        attenuation_db=args.As,
        order=args.order,
        ftype=args.type,
        method=args.method,
        fs=1.0,
    )
    metrics = compute_comparison_metrics(
        h_pred=h_pred,
        h_ref=h_design,
        fc=args.fc,
        trans=args.trans,
        ftype=args.type,
        fs=1.0,
    )
    inference_time = time.time() - start_time

    print("=" * 50)
    print(f"HYBRID FIR INFERENCE ({args.type.upper()})")
    print("=" * 50)
    print(
        f"fc={args.fc}, trans={args.trans}, Rp={args.Rp} dB, As={args.As} dB, "
        f"requested_order={args.order}, method={args.method}"
    )

    print("\nPredicted Parameters:")
    for key, values in params.items():
        print(f"  {key}: {values[0]:.6f}")

    print(f"\nPredicted taps: {len(h_pred)}")
    print(f"Reference taps: {len(h_design)}")

    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"  {key}: {format_metric(value)}")
    print(f"Inference time: {inference_time:.3f} s")

    if args.export:
        filename = f"fir_{args.type}_fc{args.fc}_ord{args.order}.{args.export}"
        export_coefficients(h_pred, filename, args.export)
        print(f"Exported coefficients to {filename} ({args.export})")

    if not args.no_plot:
        specs = {
            "fc": args.fc,
            "trans": args.trans,
            "Rp": args.Rp,
            "As": args.As,
            "order": args.order,
            "type": args.type,
        }
        plot_comprehensive_comparison(h_pred, h_design, specs)


if __name__ == "__main__":
    main()
