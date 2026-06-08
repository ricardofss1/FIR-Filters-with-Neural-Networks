import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from fir_utils import (
    compute_comparison_metrics,
    design_reference_filter,
    export_coefficients,
    format_metric,
    normalize_specs_from_hz,
)
from hybrid_predict import load_paramnet, predict_fir_from_specs


def plot_comprehensive_comparison(h_pred, h_design, specs, fs, n_fft=2048):
    """Plot the hybrid and classical filters using Hz on the x-axis."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    freqs, pred_resp = signal.freqz(h_pred, worN=n_fft, fs=fs)
    _, ref_resp = signal.freqz(h_design, worN=n_fft, fs=fs)

    pred_db = 20 * np.log10(np.maximum(np.abs(pred_resp), 1e-10))
    ref_db = 20 * np.log10(np.maximum(np.abs(ref_resp), 1e-10))

    axes[0, 0].plot(freqs, ref_db, label="SciPy (direct design)", linewidth=2)
    axes[0, 0].plot(freqs, pred_db, "--", label="Hybrid prediction", alpha=0.8)
    axes[0, 0].set_title("Magnitude Response")
    axes[0, 0].set_xlabel("Frequency (Hz)")
    axes[0, 0].set_ylabel("Magnitude (dB)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    pred_phase = np.unwrap(np.angle(pred_resp))
    ref_phase = np.unwrap(np.angle(ref_resp))
    axes[0, 1].plot(freqs, ref_phase, "--", label="SciPy")
    axes[0, 1].plot(freqs, pred_phase, label="Prediction")
    axes[0, 1].set_title("Phase Response")
    axes[0, 1].set_xlabel("Frequency (Hz)")
    axes[0, 1].set_ylabel("Phase (rad)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].stem(h_design, linefmt="C0-", markerfmt="C0o", basefmt="k-", label="SciPy")
    axes[1, 0].stem(h_pred, linefmt="C1--", markerfmt="C1x", basefmt="k-", label="Prediction")
    axes[1, 0].set_title("FIR Coefficients")
    axes[1, 0].set_xlabel("Index n")
    axes[1, 0].set_ylabel("Amplitude")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    error_db = pred_db - ref_db
    axes[1, 1].plot(freqs, error_db, "r-", label="Error")
    axes[1, 1].axhline(0, color="k", linestyle="--", alpha=0.5)
    axes[1, 1].set_title("Magnitude Error")
    axes[1, 1].set_xlabel("Frequency (Hz)")
    axes[1, 1].set_ylabel("Error (dB)")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.suptitle(
        f"FIR {specs['type'].upper()} Comparison (Hz): fc={specs['fc']} Hz, trans={specs['trans']} Hz, "
        f"Rp={specs['Rp']} dB, As={specs['As']} dB, order={specs['order']}"
    )
    plt.tight_layout()
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Hybrid FIR inference for real frequencies in Hz.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_hybrid/best_paramnet.pth")
    parser.add_argument("--fc", type=float, default=700.0, help="Cutoff frequency in Hz")
    parser.add_argument("--trans", type=float, default=200.0, help="Transition width in Hz")
    parser.add_argument("--Rp", type=float, default=1.0, help="Passband ripple in dB")
    parser.add_argument("--As", type=float, default=40.0, help="Stopband attenuation in dB")
    parser.add_argument("--order", type=int, default=128, help="Requested number of taps")
    parser.add_argument("--fs", type=float, default=16000.0, help="Sampling frequency in Hz")
    parser.add_argument("--type", type=str, choices=["lowpass", "highpass"], default="lowpass")
    parser.add_argument("--method", type=str, choices=["remez", "firwin"], default="firwin")
    parser.add_argument("--export", type=str, choices=["txt", "python", "c", "matlab"])
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--interactive", action="store_true", help="Prompt for parameters interactively")
    return parser.parse_args()


def maybe_collect_interactive_args(args):
    if not args.interactive:
        return args

    print("\n=== INTERACTIVE FIR DESIGN ===\n")
    args.type = input("Filter type [lowpass/highpass]: ") or args.type
    args.fs = float(input(f"Sampling frequency (Hz) [{args.fs}]: ") or args.fs)
    args.fc = float(input(f"Cutoff frequency (Hz) [{args.fc}]: ") or args.fc)
    args.trans = float(input(f"Transition width (Hz) [{args.trans}]: ") or args.trans)
    args.Rp = float(input(f"Ripple (dB) [{args.Rp}]: ") or args.Rp)
    args.As = float(input(f"Attenuation (dB) [{args.As}]: ") or args.As)
    args.order = int(input(f"Requested taps [{args.order}]: ") or args.order)
    args.method = input(f"Method [remez/firwin] [{args.method}]: ") or args.method
    export_choice = input("Export coefficients? [txt/python/c/matlab/none]: ") or "none"
    args.export = None if export_choice == "none" else export_choice
    return args


def main():
    args = maybe_collect_interactive_args(parse_args())

    if args.type == "lowpass" and args.fc + args.trans >= args.fs / 2:
        raise ValueError("For lowpass, fc + trans must stay below Nyquist.")
    if args.type == "highpass" and args.fc - args.trans <= 0.0:
        raise ValueError("For highpass, fc - trans must stay above 0 Hz.")

    start_time = time.time()
    model, t_scaler, device = load_paramnet(args.checkpoint)

    fc_norm, trans_norm = normalize_specs_from_hz(args.fc, args.trans, args.fs)
    ftype_val = 0 if args.type == "lowpass" else 1
    spec = np.array([[fc_norm, trans_norm, args.Rp, args.As, args.order, ftype_val]], dtype=np.float32)

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
        fs=args.fs,
    )
    metrics = compute_comparison_metrics(
        h_pred=h_pred,
        h_ref=h_design,
        fc=args.fc,
        trans=args.trans,
        ftype=args.type,
        fs=args.fs,
    )
    inference_time = time.time() - start_time

    print("=" * 50)
    print(f"HYBRID FIR INFERENCE IN HZ ({args.type.upper()})")
    print("=" * 50)
    print(f"fs={args.fs} Hz, fc={args.fc} Hz, trans={args.trans} Hz")
    print(f"Rp={args.Rp} dB, As={args.As} dB, requested_order={args.order}, method={args.method}")

    print("\nPredicted Parameters:")
    for key, values in params.items():
        print(f"  {key}: {values[0]:.6f}")

    print(f"\nPredicted taps: {len(h_pred)}")
    print(f"Reference taps: {len(h_design)}")

    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"  {key}: {format_metric(value)}")
    print(f"Inference time: {inference_time:.4f} s")

    if args.export:
        filename = f"fir_coeffs_fc{args.fc}hz_fs{args.fs}hz_order{args.order}.{args.export}"
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
        plot_comprehensive_comparison(h_pred, h_design, specs, args.fs)


if __name__ == "__main__":
    main()
