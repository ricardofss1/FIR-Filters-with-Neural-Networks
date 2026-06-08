import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from fir_utils import (
    build_response_masks,
    compute_comparison_metrics,
    design_reference_filter,
    export_coefficients,
    format_metric,
    normalize_specs_from_hz,
)
from hybrid_predict import load_paramnet, predict_fir_from_specs


def compute_band_metrics(h_coeffs, fs, fc, trans, ftype, n_fft=2048):
    """Measure ripple/attenuation on the appropriate bands for lowpass or highpass."""
    freqs, response = signal.freqz(h_coeffs, worN=n_fft, fs=fs)
    response_db = 20 * np.log10(np.maximum(np.abs(response), 1e-10))
    passband_mask, stopband_mask = build_response_masks(freqs, fc, trans, ftype)

    ripple = float("nan")
    attenuation = float("nan")
    if np.any(passband_mask):
        ripple = float(np.max(response_db[passband_mask]) - np.min(response_db[passband_mask]))
    if np.any(stopband_mask):
        attenuation = float(-np.max(response_db[stopband_mask]))
    return ripple, attenuation


def plot_comprehensive_comparison_real(h_pred, h_design, specs, fs, n_fft=2048):
    """Plot comparison charts for real-frequency FIR designs."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    freqs, pred_resp = signal.freqz(h_pred, worN=n_fft, fs=fs)
    _, ref_resp = signal.freqz(h_design, worN=n_fft, fs=fs)

    pred_db = 20 * np.log10(np.maximum(np.abs(pred_resp), 1e-10))
    ref_db = 20 * np.log10(np.maximum(np.abs(ref_resp), 1e-10))

    axes[0, 0].plot(freqs, ref_db, label="SciPy (direct design)", linewidth=2)
    axes[0, 0].plot(freqs, pred_db, "--", label="Hybrid prediction", alpha=0.8)
    axes[0, 0].axvline(specs["fc"], color="r", linestyle=":", alpha=0.7, label=f"Fc = {specs['fc']} Hz")
    if specs["type"] == "lowpass":
        axes[0, 0].axvline(specs["fc"] + specs["trans"], color="g", linestyle=":", alpha=0.7, label="Stopband edge")
    else:
        axes[0, 0].axvline(specs["fc"] - specs["trans"], color="g", linestyle=":", alpha=0.7, label="Stopband edge")
    axes[0, 0].set_title("Magnitude Response")
    axes[0, 0].set_xlabel("Frequency (Hz)")
    axes[0, 0].set_ylabel("Magnitude (dB)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim(0, fs / 2)

    axes[0, 1].plot(freqs, ref_db, label="SciPy", linewidth=2)
    axes[0, 1].plot(freqs, pred_db, "--", label="Prediction", alpha=0.8)
    zoom_left = max(specs["fc"] - 2 * specs["trans"], 0.0)
    zoom_right = min(specs["fc"] + 2 * specs["trans"], fs / 2)
    axes[0, 1].set_title("Transition Region")
    axes[0, 1].set_xlabel("Frequency (Hz)")
    axes[0, 1].set_ylabel("Magnitude (dB)")
    axes[0, 1].set_xlim(zoom_left, zoom_right)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].stem(h_design, linefmt="C0-", markerfmt="C0o", basefmt="k-", label="SciPy")
    axes[1, 0].stem(h_pred, linefmt="C1--", markerfmt="C1x", basefmt="k-", label="Prediction")
    axes[1, 0].set_title("FIR Coefficients")
    axes[1, 0].set_xlabel("Coefficient Index")
    axes[1, 0].set_ylabel("Amplitude")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    pred_phase = np.unwrap(np.angle(pred_resp))
    ref_phase = np.unwrap(np.angle(ref_resp))
    axes[1, 1].plot(freqs, pred_phase, label="Predicted")
    axes[1, 1].plot(freqs, ref_phase, label="Designed", linestyle="--")
    axes[1, 1].set_title("Phase Response")
    axes[1, 1].set_xlabel("Frequency (Hz)")
    axes[1, 1].set_ylabel("Phase (radians)")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.suptitle(
        f"FIR {specs['type'].upper()}: fc={specs['fc']} Hz, fs={fs} Hz, "
        f"Rp={specs['Rp']} dB, As={specs['As']} dB, order={specs['order']}"
    )
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Hybrid FIR inference for real implementations.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_hybrid/best_paramnet.pth")
    parser.add_argument("--fc", type=float, default=700.0, help="Cutoff frequency in Hz")
    parser.add_argument("--trans", type=float, default=200.0, help="Transition width in Hz")
    parser.add_argument("--Rp", type=float, default=1.0, help="Passband ripple in dB")
    parser.add_argument("--As", type=float, default=60.0, help="Stopband attenuation in dB")
    parser.add_argument("--order", type=int, default=128, help="Requested number of taps")
    parser.add_argument("--fs", type=float, default=44100.0, help="Sampling frequency in Hz")
    parser.add_argument("--type", type=str, choices=["lowpass", "highpass"], default="lowpass")
    parser.add_argument("--method", type=str, choices=["remez", "firwin"], default="firwin")
    parser.add_argument("--export", type=str, choices=["python", "c", "matlab"])
    parser.add_argument("--no-plot", action="store_true", help="Do not show plots")
    args = parser.parse_args()

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

    comparison = compute_comparison_metrics(
        h_pred=h_pred,
        h_ref=h_design,
        fc=args.fc,
        trans=args.trans,
        ftype=args.type,
        fs=args.fs,
    )
    pred_ripple, pred_attenuation = compute_band_metrics(
        h_pred, args.fs, args.fc, args.trans, args.type
    )
    ref_ripple, ref_attenuation = compute_band_metrics(
        h_design, args.fs, args.fc, args.trans, args.type
    )
    inference_time = time.time() - start_time

    print("=" * 60)
    print(f"REAL-WORLD HYBRID FIR DESIGN ({args.type.upper()})")
    print("=" * 60)
    print("\nSpecifications:")
    print(f"  Sampling frequency: {args.fs} Hz")
    print(f"  Cutoff frequency: {args.fc} Hz")
    print(f"  Transition width: {args.trans} Hz")
    print(f"  Ripple target: {args.Rp} dB")
    print(f"  Attenuation target: {args.As} dB")
    print(f"  Requested order: {args.order}")
    print(f"  Predicted taps: {len(h_pred)}")
    print(f"  Reference taps: {len(h_design)}")

    print("\nPredicted Parameters:")
    for key, values in params.items():
        print(f"  {key}: {values[0]:.6f}")

    print("\nPredicted Filter Performance:")
    print(f"  Ripple in passband: {format_metric(pred_ripple)} dB")
    print(f"  Attenuation in stopband: {format_metric(pred_attenuation)} dB")

    print("\nReference Filter Performance:")
    print(f"  Ripple in passband: {format_metric(ref_ripple)} dB")
    print(f"  Attenuation in stopband: {format_metric(ref_attenuation)} dB")

    print("\nComparison Metrics:")
    for key, value in comparison.items():
        print(f"  {key}: {format_metric(value)}")
    print(f"  Inference time: {inference_time:.3f} s")

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
        plot_comprehensive_comparison_real(h_pred, h_design, specs, args.fs)


if __name__ == "__main__":
    main()
