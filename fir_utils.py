import math
from typing import Dict, Tuple

import numpy as np
from scipy import signal


def compute_beta(attenuation_db: float) -> float:
    """Compute the Kaiser beta parameter from stopband attenuation."""
    if attenuation_db > 50:
        return 0.1102 * (attenuation_db - 8.7)
    if attenuation_db > 21:
        return 0.5842 * (attenuation_db - 21) ** 0.4 + 0.07886 * (attenuation_db - 21)
    return 0.0


def normalize_specs_from_hz(fc_hz: float, trans_hz: float, fs: float) -> Tuple[float, float]:
    """Convert Hz specifications to the normalized convention used in the dataset (fs=1.0)."""
    if fs <= 0:
        raise ValueError("Sampling frequency must be positive.")
    return fc_hz / fs, trans_hz / fs


def design_reference_filter(
    fc: float,
    trans: float,
    rp: float,
    attenuation_db: float,
    order: int,
    ftype: str = "lowpass",
    method: str = "firwin",
    fs: float = 1.0,
) -> np.ndarray:
    """Design a lowpass or highpass FIR filter using a classical SciPy method."""
    order = int(order)
    if ftype not in {"lowpass", "highpass"}:
        raise ValueError(f"Unsupported filter type: {ftype}")

    if method == "firwin":
        if ftype == "highpass" and order % 2 == 0:
            order += 1
        return signal.firwin(
            order,
            cutoff=fc,
            window=("kaiser", compute_beta(attenuation_db)),
            fs=fs,
            pass_zero=(ftype == "lowpass"),
        )

    if method == "remez":
        if ftype == "lowpass":
            bands = [0.0, fc, min(fc + trans, fs / 2), fs / 2]
            desired = [1.0, 0.0]
        else:
            bands = [0.0, max(fc - trans, 0.0), fc, fs / 2]
            desired = [0.0, 1.0]

        delta_p = 10 ** (-rp / 20)
        delta_s = 10 ** (-attenuation_db / 20)
        weights = [1 / max(delta_p, 1e-12), 1 / max(delta_s, 1e-12)]
        return signal.remez(order, bands, desired, weight=weights, fs=fs)

    raise ValueError(f"Unsupported design method: {method}")


def build_response_masks(
    freqs: np.ndarray,
    fc: float,
    trans: float,
    ftype: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create passband and stopband masks for lowpass or highpass filters."""
    if ftype == "lowpass":
        passband_mask = freqs <= fc
        stopband_mask = freqs >= (fc + trans)
    elif ftype == "highpass":
        passband_mask = freqs >= fc
        stopband_mask = freqs <= max(fc - trans, 0.0)
    else:
        raise ValueError(f"Unsupported filter type: {ftype}")
    return passband_mask, stopband_mask


def _safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def _safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def compute_comparison_metrics(
    h_pred: np.ndarray,
    h_ref: np.ndarray,
    fc: float,
    trans: float,
    ftype: str,
    fs: float = 1.0,
    n_fft: int = 2048,
) -> Dict[str, float]:
    """Compare predicted and reference filters in the appropriate pass/stop bands."""
    min_len = min(len(h_pred), len(h_ref))
    h_pred = h_pred[:min_len]
    h_ref = h_ref[:min_len]

    freqs, pred_resp = signal.freqz(h_pred, worN=n_fft, fs=fs)
    _, ref_resp = signal.freqz(h_ref, worN=n_fft, fs=fs)

    pred_db = 20 * np.log10(np.maximum(np.abs(pred_resp), 1e-10))
    ref_db = 20 * np.log10(np.maximum(np.abs(ref_resp), 1e-10))
    passband_mask, stopband_mask = build_response_masks(freqs, fc, trans, ftype)

    return {
        "mae_passband": _safe_mean(np.abs(pred_db[passband_mask] - ref_db[passband_mask])),
        "mae_stopband": _safe_mean(np.abs(pred_db[stopband_mask] - ref_db[stopband_mask])),
        "correlation": _safe_corrcoef(h_pred, h_ref),
        "erle": float(
            10
            * np.log10(
                np.sum(h_ref**2) / max(np.sum((h_pred - h_ref) ** 2), 1e-10)
            )
        ),
    }


def export_coefficients(h_coeffs: np.ndarray, filename: str, format_type: str = "txt") -> None:
    """Export FIR coefficients in a portable format."""
    if format_type == "txt":
        with open(filename, "w", encoding="utf-8") as file_obj:
            coeffs_str = [f"{coeff:.8f}" for coeff in h_coeffs]
            file_obj.write(",".join(coeffs_str))
    elif format_type == "python":
        np.savetxt(filename, h_coeffs, fmt="%.8f", delimiter=",")
    elif format_type == "c":
        with open(filename, "w", encoding="utf-8") as file_obj:
            file_obj.write("const float fir_coefficients[] = {\n")
            for index, coeff in enumerate(h_coeffs):
                suffix = ",\n" if index < len(h_coeffs) - 1 else "\n"
                file_obj.write(f"    {coeff:.8f}f{suffix}")
            file_obj.write("};\n")
    elif format_type == "matlab":
        with open(filename, "w", encoding="utf-8") as file_obj:
            file_obj.write("fir_coeffs = [\n")
            for coeff in h_coeffs:
                file_obj.write(f"    {coeff:.8f};\n")
            file_obj.write("];\n")
    else:
        raise ValueError(f"Unsupported export format: {format_type}")


def format_metric(value: float) -> str:
    """Format metrics while keeping NaN explicit in CLI output."""
    if math.isnan(value):
        return "nan"
    return f"{value:.3f}"
