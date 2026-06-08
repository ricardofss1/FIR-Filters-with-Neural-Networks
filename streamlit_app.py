import csv
import io
import warnings
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import signal
from streamlit.runtime.scriptrunner import get_script_run_ctx

from dataset_generator import evaluate_candidate_against_request
from fir_utils import build_response_masks, compute_comparison_metrics, design_reference_filter
from hybrid_predict import load_paramnet, predict_fir_from_specs


PRESETS = {
    "Custom": {
        "type": "lowpass",
        "fs": 16000.0,
        "fc": 700.0,
        "trans": 200.0,
        "Rp": 1.0,
        "As": 40.0,
        "order": 128,
    },
    "Voice Cleanup": {
        "type": "highpass",
        "fs": 44100.0,
        "fc": 80.0,
        "trans": 20.0,
        "Rp": 1.0,
        "As": 60.0,
        "order": 128,
    },
    "Sensor Smoothing": {
        "type": "lowpass",
        "fs": 1000.0,
        "fc": 40.0,
        "trans": 10.0,
        "Rp": 1.0,
        "As": 50.0,
        "order": 96,
    },
    "Anti-Alias Audio": {
        "type": "lowpass",
        "fs": 16000.0,
        "fc": 3200.0,
        "trans": 600.0,
        "Rp": 1.0,
        "As": 60.0,
        "order": 128,
    },
    "Noise Rejection": {
        "type": "highpass",
        "fs": 16000.0,
        "fc": 2000.0,
        "trans": 300.0,
        "Rp": 1.0,
        "As": 50.0,
        "order": 128,
    },
}

CURVE_OPTIONS = ["Previsto", "Direto", "Especificacao"]
PLOTLY_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "scrollZoom": True,
    "toImageButtonOptions": {
        "format": "png",
        "filename": "fir_plot",
        "scale": 2,
    },
}


def normalize_request_spec(filter_type, fs, fc_hz, trans_hz, rp, attenuation_db, order):
    return {
        "fc": fc_hz / fs,
        "trans": trans_hz / fs,
        "Rp": rp,
        "As": attenuation_db,
        "order": int(order),
        "type": filter_type,
    }


def compute_band_metrics(h_coeffs, request_spec_hz, n_fft=2048):
    freqs, response = signal.freqz(h_coeffs, worN=n_fft, fs=request_spec_hz["fs"])
    response_db = 20 * np.log10(np.maximum(np.abs(response), 1e-10))
    passband_mask, stopband_mask = build_response_masks(
        freqs, request_spec_hz["fc"], request_spec_hz["trans"], request_spec_hz["type"]
    )

    ripple = float("nan")
    attenuation = float("nan")
    if np.any(passband_mask):
        ripple = float(np.max(response_db[passband_mask]) - np.min(response_db[passband_mask]))
    if np.any(stopband_mask):
        attenuation = float(-np.max(response_db[stopband_mask]))

    meets_ripple = np.isfinite(ripple) and ripple <= request_spec_hz["Rp"]
    meets_attenuation = np.isfinite(attenuation) and attenuation >= request_spec_hz["As"]
    return {
        "ripple_db": ripple,
        "attenuation_db": attenuation,
        "meets_ripple": bool(meets_ripple),
        "meets_attenuation": bool(meets_attenuation),
        "meets_both": bool(meets_ripple and meets_attenuation),
    }


def serialize_coefficients(h_coeffs, format_type):
    if format_type == "txt":
        return ",".join(f"{value:.8f}" for value in h_coeffs)
    if format_type == "python":
        rows = ",\n    ".join(f"{value:.8f}" for value in h_coeffs)
        return f"fir_coeffs = [\n    {rows}\n]\n"
    if format_type == "c":
        rows = ",\n    ".join(f"{value:.8f}f" for value in h_coeffs)
        return f"const float fir_coefficients[] = {{\n    {rows}\n}};\n"
    if format_type == "matlab":
        rows = ";\n    ".join(f"{value:.8f}" for value in h_coeffs)
        return f"fir_coeffs = [\n    {rows}\n];\n"
    raise ValueError(f"Unsupported export format: {format_type}")


def is_streamlit_context():
    return get_script_run_ctx(suppress_warning=True) is not None


def safe_group_delay(h_coeffs, fs, n_fft):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        freqs, group_delay = signal.group_delay((h_coeffs, 1.0), w=n_fft, fs=fs)
    group_delay = np.asarray(group_delay, dtype=np.float64)
    group_delay[~np.isfinite(group_delay)] = np.nan
    return freqs, group_delay


def compute_spec_violation_db(magnitude_db, passband_mask, stopband_mask, rp_db, attenuation_db):
    violation = np.full_like(magnitude_db, np.nan, dtype=np.float64)
    pass_upper_db = rp_db / 2.0
    pass_lower_db = -rp_db / 2.0

    passband_error = np.maximum(magnitude_db - pass_upper_db, pass_lower_db - magnitude_db)
    stopband_error = magnitude_db + attenuation_db

    violation[passband_mask] = np.maximum(passband_error[passband_mask], 0.0)
    violation[stopband_mask] = np.maximum(stopband_error[stopband_mask], 0.0)
    return violation


def build_response_bundle(h_pred, h_direct, request_spec_hz, predicted_fc_hz, n_fft, unwrap_phase):
    freqs, pred_resp = signal.freqz(h_pred, worN=n_fft, fs=request_spec_hz["fs"])
    _, direct_resp = signal.freqz(h_direct, worN=n_fft, fs=request_spec_hz["fs"])

    pred_mag = np.abs(pred_resp)
    direct_mag = np.abs(direct_resp)
    pred_mag_db = 20 * np.log10(np.maximum(pred_mag, 1e-10))
    direct_mag_db = 20 * np.log10(np.maximum(direct_mag, 1e-10))

    pred_phase = np.angle(pred_resp)
    direct_phase = np.angle(direct_resp)
    if unwrap_phase:
        pred_phase = np.unwrap(pred_phase)
        direct_phase = np.unwrap(direct_phase)

    group_delay_freqs, pred_group_delay = safe_group_delay(h_pred, request_spec_hz["fs"], n_fft)
    _, direct_group_delay = safe_group_delay(h_direct, request_spec_hz["fs"], n_fft)

    passband_mask, stopband_mask = build_response_masks(
        freqs, request_spec_hz["fc"], request_spec_hz["trans"], request_spec_hz["type"]
    )
    transition_mask = ~(passband_mask | stopband_mask)

    ideal_mag = np.full_like(freqs, np.nan, dtype=np.float64)
    ideal_mag[passband_mask] = 1.0
    ideal_mag[stopband_mask] = 0.0

    pass_upper_linear = 10 ** (request_spec_hz["Rp"] / 40.0)
    pass_lower_linear = 10 ** (-request_spec_hz["Rp"] / 40.0)
    stop_upper_linear = 10 ** (-request_spec_hz["As"] / 20.0)

    pass_upper_db = np.full_like(freqs, np.nan, dtype=np.float64)
    pass_lower_db = np.full_like(freqs, np.nan, dtype=np.float64)
    stop_limit_db = np.full_like(freqs, np.nan, dtype=np.float64)
    pass_upper_db[passband_mask] = request_spec_hz["Rp"] / 2.0
    pass_lower_db[passband_mask] = -request_spec_hz["Rp"] / 2.0
    stop_limit_db[stopband_mask] = -request_spec_hz["As"]

    pass_upper_linear_curve = np.full_like(freqs, np.nan, dtype=np.float64)
    pass_lower_linear_curve = np.full_like(freqs, np.nan, dtype=np.float64)
    stop_limit_linear_curve = np.full_like(freqs, np.nan, dtype=np.float64)
    pass_upper_linear_curve[passband_mask] = pass_upper_linear
    pass_lower_linear_curve[passband_mask] = pass_lower_linear
    stop_limit_linear_curve[stopband_mask] = stop_upper_linear

    pred_spec_violation_db = compute_spec_violation_db(
        pred_mag_db,
        passband_mask,
        stopband_mask,
        request_spec_hz["Rp"],
        request_spec_hz["As"],
    )
    direct_spec_violation_db = compute_spec_violation_db(
        direct_mag_db,
        passband_mask,
        stopband_mask,
        request_spec_hz["Rp"],
        request_spec_hz["As"],
    )

    return {
        "freqs_hz": freqs,
        "pred_mag": pred_mag,
        "direct_mag": direct_mag,
        "pred_mag_db": pred_mag_db,
        "direct_mag_db": direct_mag_db,
        "pred_phase": pred_phase,
        "direct_phase": direct_phase,
        "group_delay_freqs_hz": group_delay_freqs,
        "pred_group_delay": pred_group_delay,
        "direct_group_delay": direct_group_delay,
        "ideal_mag": ideal_mag,
        "passband_mask": passband_mask,
        "stopband_mask": stopband_mask,
        "transition_mask": transition_mask,
        "pass_upper_linear_curve": pass_upper_linear_curve,
        "pass_lower_linear_curve": pass_lower_linear_curve,
        "stop_limit_linear_curve": stop_limit_linear_curve,
        "pass_upper_db": pass_upper_db,
        "pass_lower_db": pass_lower_db,
        "stop_limit_db": stop_limit_db,
        "pred_spec_violation_db": pred_spec_violation_db,
        "direct_spec_violation_db": direct_spec_violation_db,
        "predicted_fc_hz": predicted_fc_hz,
    }


def render_metric_table(pred_band, direct_band, comparison_metrics, pred_score, direct_score, bundle):
    max_pred_violation = float(np.nanmax(bundle["pred_spec_violation_db"]))
    max_direct_violation = float(np.nanmax(bundle["direct_spec_violation_db"]))
    return {
        "Ripple previsto (dB)": pred_band["ripple_db"],
        "Ripple direto (dB)": direct_band["ripple_db"],
        "Atenuacao prevista (dB)": pred_band["attenuation_db"],
        "Atenuacao direta (dB)": direct_band["attenuation_db"],
        "Score previsto": pred_score,
        "Score direto": direct_score,
        "Violacao maxima prevista (dB)": max_pred_violation,
        "Violacao maxima direta (dB)": max_direct_violation,
        "MAE passband": comparison_metrics["mae_passband"],
        "MAE stopband": comparison_metrics["mae_stopband"],
        "Correlacao": comparison_metrics["correlation"],
        "ERLE": comparison_metrics["erle"],
    }


def make_csv_bytes(columns):
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    headers = list(columns.keys())
    writer.writerow(headers)
    rows = zip(*[np.asarray(values).tolist() for values in columns.values()])
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def make_single_row_csv_bytes(row_mapping):
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    headers = list(row_mapping.keys())
    writer.writerow(headers)
    writer.writerow([row_mapping[key] for key in headers])
    return buffer.getvalue().encode("utf-8")


def format_metric_value(value):
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return "nan"
    return f"{float(value):.4f}" if isinstance(value, (int, float, np.floating, np.integer)) else str(value)


def render_status_with_help(container, message, variant, help_text, key):
    content_col, help_col = container.columns([0.9, 0.1], vertical_alignment="center")
    if variant == "success":
        content_col.success(message)
    elif variant == "warning":
        content_col.warning(message)
    else:
        content_col.info(message)

    with help_col.popover("?", help="Ajuda contextual", key=key):
        st.write(help_text)


def build_axis_range(use_custom_range, min_value, max_value):
    if not use_custom_range:
        return None
    if max_value <= min_value:
        return None
    return [float(min_value), float(max_value)]


def add_reference_markers(fig, request_spec_hz, predicted_fc_hz):
    if request_spec_hz["type"] == "lowpass":
        stop_edge_hz = request_spec_hz["fc"] + request_spec_hz["trans"]
    else:
        stop_edge_hz = max(request_spec_hz["fc"] - request_spec_hz["trans"], 0.0)

    fig.add_vline(
        x=request_spec_hz["fc"],
        line_dash="dot",
        line_color="#EF553B",
        annotation_text="Cutoff pedido",
        annotation_position="top left",
    )
    fig.add_vline(
        x=stop_edge_hz,
        line_dash="dot",
        line_color="#AB63FA",
        annotation_text="Borda stopband",
        annotation_position="top right",
    )
    fig.add_vline(
        x=predicted_fc_hz,
        line_dash="dash",
        line_color="#00CC96",
        annotation_text="Cutoff sugerido",
        annotation_position="bottom right",
    )


def base_plotly_layout(title, x_title, y_title, x_range=None, y_range=None):
    layout = {
        "title": title,
        "hovermode": "x unified",
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0},
        "margin": {"l": 60, "r": 20, "t": 70, "b": 50},
        "xaxis_title": x_title,
        "yaxis_title": y_title,
        "template": "plotly_white",
    }
    if x_range is not None:
        layout["xaxis_range"] = x_range
    if y_range is not None:
        layout["yaxis_range"] = y_range
    return layout


def add_response_traces(fig, bundle, selected_curves, scale_mode):
    if scale_mode == "linear":
        predicted_y = bundle["pred_mag"]
        direct_y = bundle["direct_mag"]
        pass_upper = bundle["pass_upper_linear_curve"]
        pass_lower = bundle["pass_lower_linear_curve"]
        stop_limit = bundle["stop_limit_linear_curve"]
        y_hover = ".6f"
        y_title = "Magnitude"
    else:
        predicted_y = bundle["pred_mag_db"]
        direct_y = bundle["direct_mag_db"]
        pass_upper = bundle["pass_upper_db"]
        pass_lower = bundle["pass_lower_db"]
        stop_limit = bundle["stop_limit_db"]
        y_hover = ".3f"
        y_title = "Magnitude (dB)"

    if "Previsto" in selected_curves:
        fig.add_trace(
            go.Scatter(
                x=bundle["freqs_hz"],
                y=predicted_y,
                mode="lines",
                name="Previsto",
                line={"width": 2.5, "color": "#1F77B4"},
                hovertemplate=f"Freq: %{{x:.2f}} Hz<br>Previsto: %{{y:{y_hover}}}<extra></extra>",
            )
        )
    if "Direto" in selected_curves:
        fig.add_trace(
            go.Scatter(
                x=bundle["freqs_hz"],
                y=direct_y,
                mode="lines",
                name="Direto",
                line={"width": 2.0, "dash": "dash", "color": "#FF7F0E"},
                hovertemplate=f"Freq: %{{x:.2f}} Hz<br>Direto: %{{y:{y_hover}}}<extra></extra>",
            )
        )
    if "Especificacao" in selected_curves:
        fig.add_trace(
            go.Scatter(
                x=bundle["freqs_hz"],
                y=pass_upper,
                mode="lines",
                name="Limite passband sup.",
                line={"width": 1.5, "dash": "dot", "color": "#2CA02C"},
                hovertemplate=f"Freq: %{{x:.2f}} Hz<br>Limite: %{{y:{y_hover}}}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=bundle["freqs_hz"],
                y=pass_lower,
                mode="lines",
                name="Limite passband inf.",
                line={"width": 1.5, "dash": "dot", "color": "#2CA02C"},
                hovertemplate=f"Freq: %{{x:.2f}} Hz<br>Limite: %{{y:{y_hover}}}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=bundle["freqs_hz"],
                y=stop_limit,
                mode="lines",
                name="Limite stopband",
                line={"width": 1.5, "dash": "dot", "color": "#D62728"},
                hovertemplate=f"Freq: %{{x:.2f}} Hz<br>Limite: %{{y:{y_hover}}}<extra></extra>",
            )
        )
    return y_title


def build_magnitude_figure(bundle, request_spec_hz, selected_curves, scale_mode, x_range=None, y_range=None):
    figure = go.Figure()
    y_title = add_response_traces(figure, bundle, selected_curves, scale_mode)
    add_reference_markers(figure, request_spec_hz, bundle["predicted_fc_hz"])
    figure.update_layout(
        **base_plotly_layout(
            title=f"Resposta em magnitude ({scale_mode})",
            x_title="Frequencia (Hz)",
            y_title=y_title,
            x_range=x_range,
            y_range=y_range,
        )
    )
    return figure


def build_phase_figure(bundle, request_spec_hz, selected_curves, x_range=None):
    figure = go.Figure()
    if "Previsto" in selected_curves:
        figure.add_trace(
            go.Scatter(
                x=bundle["freqs_hz"],
                y=bundle["pred_phase"],
                mode="lines",
                name="Fase prevista",
                line={"width": 2.2, "color": "#1F77B4"},
                hovertemplate="Freq: %{x:.2f} Hz<br>Fase: %{y:.4f} rad<extra></extra>",
            )
        )
    if "Direto" in selected_curves:
        figure.add_trace(
            go.Scatter(
                x=bundle["freqs_hz"],
                y=bundle["direct_phase"],
                mode="lines",
                name="Fase direta",
                line={"width": 2.0, "dash": "dash", "color": "#FF7F0E"},
                hovertemplate="Freq: %{x:.2f} Hz<br>Fase: %{y:.4f} rad<extra></extra>",
            )
        )
    add_reference_markers(figure, request_spec_hz, bundle["predicted_fc_hz"])
    figure.update_layout(
        **base_plotly_layout(
            title="Resposta em fase",
            x_title="Frequencia (Hz)",
            y_title="Fase (rad)",
            x_range=x_range,
        )
    )
    return figure


def build_group_delay_figure(bundle, selected_curves, x_range=None):
    figure = go.Figure()
    if "Previsto" in selected_curves:
        figure.add_trace(
            go.Scatter(
                x=bundle["group_delay_freqs_hz"],
                y=bundle["pred_group_delay"],
                mode="lines",
                name="Atraso previsto",
                line={"width": 2.2, "color": "#1F77B4"},
                hovertemplate="Freq: %{x:.2f} Hz<br>Atraso: %{y:.4f} amostras<extra></extra>",
            )
        )
    if "Direto" in selected_curves:
        figure.add_trace(
            go.Scatter(
                x=bundle["group_delay_freqs_hz"],
                y=bundle["direct_group_delay"],
                mode="lines",
                name="Atraso direto",
                line={"width": 2.0, "dash": "dash", "color": "#FF7F0E"},
                hovertemplate="Freq: %{x:.2f} Hz<br>Atraso: %{y:.4f} amostras<extra></extra>",
            )
        )
    figure.update_layout(
        **base_plotly_layout(
            title="Atraso de grupo",
            x_title="Frequencia (Hz)",
            y_title="Atraso (amostras)",
            x_range=x_range,
        )
    )
    return figure


def build_impulse_figure(h_pred, h_direct, selected_curves):
    min_len = min(len(h_pred), len(h_direct))
    delta = h_pred[:min_len] - h_direct[:min_len]

    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        subplot_titles=("Coeficientes / resposta ao impulso", "Erro coeficiente a coeficiente"),
        vertical_spacing=0.12,
    )

    if "Previsto" in selected_curves:
        figure.add_trace(
            go.Scatter(
                x=np.arange(len(h_pred)),
                y=h_pred,
                mode="lines+markers",
                name="Previsto",
                line={"width": 1.8, "color": "#1F77B4"},
                marker={"size": 5},
                hovertemplate="Indice: %{x}<br>Valor: %{y:.8f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    if "Direto" in selected_curves:
        figure.add_trace(
            go.Scatter(
                x=np.arange(len(h_direct)),
                y=h_direct,
                mode="lines+markers",
                name="Direto",
                line={"width": 1.8, "dash": "dash", "color": "#FF7F0E"},
                marker={"size": 5},
                hovertemplate="Indice: %{x}<br>Valor: %{y:.8f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    figure.add_trace(
        go.Scatter(
            x=np.arange(min_len),
            y=delta,
            mode="lines",
            name="Delta",
            line={"width": 1.8, "color": "#D62728"},
            hovertemplate="Indice: %{x}<br>Delta: %{y:.8f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    figure.update_xaxes(title_text="Indice do coeficiente", row=1, col=1)
    figure.update_xaxes(title_text="Indice do coeficiente", row=2, col=1)
    figure.update_yaxes(title_text="Amplitude", row=1, col=1)
    figure.update_yaxes(title_text="Erro", row=2, col=1)
    figure.update_layout(
        height=720,
        hovermode="x unified",
        template="plotly_white",
        showlegend=True,
        margin={"l": 60, "r": 20, "t": 70, "b": 50},
    )
    return figure


def build_spec_error_figure(bundle, selected_curves, x_range=None):
    figure = go.Figure()
    if "Previsto" in selected_curves:
        figure.add_trace(
            go.Scatter(
                x=bundle["freqs_hz"],
                y=bundle["pred_spec_violation_db"],
                mode="lines",
                name="Violacao prevista",
                line={"width": 2.2, "color": "#1F77B4"},
                hovertemplate="Freq: %{x:.2f} Hz<br>Violacao: %{y:.4f} dB<extra></extra>",
            )
        )
    if "Direto" in selected_curves:
        figure.add_trace(
            go.Scatter(
                x=bundle["freqs_hz"],
                y=bundle["direct_spec_violation_db"],
                mode="lines",
                name="Violacao direta",
                line={"width": 2.0, "dash": "dash", "color": "#FF7F0E"},
                hovertemplate="Freq: %{x:.2f} Hz<br>Violacao: %{y:.4f} dB<extra></extra>",
            )
        )
    figure.add_hline(y=0.0, line_dash="dot", line_color="#2CA02C")
    figure.update_layout(
        **base_plotly_layout(
            title="Erro entre especificacao e resposta obtida",
            x_title="Frequencia (Hz)",
            y_title="Violacao da especificacao (dB)",
            x_range=x_range,
        )
    )
    return figure


def build_export_datasets(result, bundle, metric_table):
    h_pred = result["h_pred"]
    h_direct = result["h_direct"]
    min_len = min(len(h_pred), len(h_direct))

    response_csv = make_csv_bytes(
        {
            "freq_hz": bundle["freqs_hz"],
            "pred_mag": bundle["pred_mag"],
            "pred_mag_db": bundle["pred_mag_db"],
            "pred_phase_rad": bundle["pred_phase"],
            "direct_mag": bundle["direct_mag"],
            "direct_mag_db": bundle["direct_mag_db"],
            "direct_phase_rad": bundle["direct_phase"],
            "spec_pass_upper_db": bundle["pass_upper_db"],
            "spec_pass_lower_db": bundle["pass_lower_db"],
            "spec_stop_limit_db": bundle["stop_limit_db"],
            "pred_spec_violation_db": bundle["pred_spec_violation_db"],
            "direct_spec_violation_db": bundle["direct_spec_violation_db"],
        }
    )
    group_delay_csv = make_csv_bytes(
        {
            "freq_hz": bundle["group_delay_freqs_hz"],
            "pred_group_delay_samples": bundle["pred_group_delay"],
            "direct_group_delay_samples": bundle["direct_group_delay"],
        }
    )
    impulse_csv = make_csv_bytes(
        {
            "index": np.arange(min_len),
            "pred_coeff": h_pred[:min_len],
            "direct_coeff": h_direct[:min_len],
            "delta": h_pred[:min_len] - h_direct[:min_len],
        }
    )
    metrics_csv = make_single_row_csv_bytes(metric_table)
    return {
        "Resposta em frequencia (CSV)": {
            "filename": "fir_frequency_response.csv",
            "bytes": response_csv,
        },
        "Atraso de grupo (CSV)": {
            "filename": "fir_group_delay.csv",
            "bytes": group_delay_csv,
        },
        "Coeficientes / impulso (CSV)": {
            "filename": "fir_impulse_response.csv",
            "bytes": impulse_csv,
        },
        "Resumo de metricas (CSV)": {
            "filename": "fir_metrics_summary.csv",
            "bytes": metrics_csv,
        },
    }


def figure_download_payload(figure, export_format):
    if export_format == "html":
        return figure.to_html(include_plotlyjs="cdn", full_html=True).encode("utf-8"), "text/html"
    if export_format == "png":
        return figure.to_image(format="png", scale=2), "image/png"
    if export_format == "svg":
        return figure.to_image(format="svg"), "image/svg+xml"
    raise ValueError(f"Unsupported export format: {export_format}")


def render_visual_controls(fs):
    with st.sidebar.expander("Visualizacao", expanded=True):
        primary_scale = st.radio(
            "Escala do grafico principal",
            ["dB", "linear"],
            index=0,
            horizontal=True,
        )
        frequency_points = st.select_slider(
            "Resolucao de frequencia",
            options=[512, 1024, 2048, 4096, 8192, 16384],
            value=4096,
        )
        selected_curves = st.multiselect(
            "Curvas visiveis",
            CURVE_OPTIONS,
            default=["Previsto", "Direto", "Especificacao"],
        )
        unwrap_phase = st.checkbox("Desenrolar fase", value=True)
        show_group_delay = st.checkbox("Mostrar atraso de grupo", value=True)
        db_floor = st.slider("Piso do grafico em dB", min_value=-180, max_value=-20, value=-120, step=5)

        custom_x_limits = st.checkbox("Definir limites de frequencia", value=False)
        if custom_x_limits:
            x_min = st.number_input("Frequencia minima (Hz)", min_value=0.0, value=0.0, step=10.0, format="%.2f")
            x_max = st.number_input(
                "Frequencia maxima (Hz)",
                min_value=0.0,
                value=float(fs / 2.0),
                step=10.0,
                format="%.2f",
            )
        else:
            x_min = 0.0
            x_max = float(fs / 2.0)

        custom_y_limits = st.checkbox("Definir limites verticais do grafico principal", value=False)
        if custom_y_limits:
            if primary_scale == "dB":
                y_min_default = float(db_floor)
                y_max_default = 5.0
            else:
                y_min_default = 0.0
                y_max_default = 1.2
            y_min = st.number_input("Y minimo", value=y_min_default, step=0.1, format="%.4f")
            y_max = st.number_input("Y maximo", value=y_max_default, step=0.1, format="%.4f")
        else:
            y_min = 0.0
            y_max = 0.0

    return {
        "primary_scale": primary_scale,
        "frequency_points": int(frequency_points),
        "selected_curves": selected_curves,
        "unwrap_phase": unwrap_phase,
        "show_group_delay": show_group_delay,
        "db_floor": float(db_floor),
        "x_range": build_axis_range(custom_x_limits, x_min, x_max),
        "y_range": build_axis_range(custom_y_limits, y_min, y_max),
    }


@st.cache_resource(show_spinner=False)
def load_model_cached(checkpoint_path):
    return load_paramnet(checkpoint_path)


def run_fir_prediction(checkpoint_path, filter_type, fs, fc, trans, rp, attenuation_db, order, method):
    model, predictor_state, device = load_model_cached(checkpoint_path)
    type_value = 0 if filter_type == "lowpass" else 1
    spec = np.array([[fc / fs, trans / fs, rp, attenuation_db, order, type_value]], dtype=np.float32)
    coefs_list, params = predict_fir_from_specs(spec, model, predictor_state, device=device, method=method)
    h_pred = coefs_list[0]
    h_direct = design_reference_filter(
        fc=fc,
        trans=trans,
        rp=rp,
        attenuation_db=attenuation_db,
        order=order,
        ftype=filter_type,
        method=method,
        fs=fs,
    )

    predicted_fc_hz = float(params["fc"][0] * fs)
    predicted_trans_hz = float(params["trans"][0] * fs)
    predicted_rp = float(params["Rp"][0])
    predicted_attenuation = float(params["As"][0])
    predicted_order = int(params["order"][0])

    request_spec_hz = {
        "fs": fs,
        "fc": fc,
        "trans": trans,
        "Rp": rp,
        "As": attenuation_db,
        "order": int(order),
        "type": filter_type,
    }
    request_spec_norm = normalize_request_spec(filter_type, fs, fc, trans, rp, attenuation_db, order)

    comparison_metrics = compute_comparison_metrics(
        h_pred,
        h_direct,
        fc=fc,
        trans=trans,
        ftype=filter_type,
        fs=fs,
    )
    pred_band = compute_band_metrics(h_pred, request_spec_hz)
    direct_band = compute_band_metrics(h_direct, request_spec_hz)
    pred_score = evaluate_candidate_against_request(h_pred, request_spec_norm)[0]
    direct_score = evaluate_candidate_against_request(h_direct, request_spec_norm)[0]

    return {
        "method": method,
        "request_spec_hz": request_spec_hz,
        "request_spec_norm": request_spec_norm,
        "comparison_metrics": comparison_metrics,
        "pred_band": pred_band,
        "direct_band": direct_band,
        "pred_score": pred_score,
        "direct_score": direct_score,
        "predicted_fc_hz": predicted_fc_hz,
        "predicted_trans_hz": predicted_trans_hz,
        "predicted_rp": predicted_rp,
        "predicted_attenuation": predicted_attenuation,
        "predicted_order": predicted_order,
        "h_pred": h_pred,
        "h_direct": h_direct,
    }


def main():
    st.set_page_config(page_title="FIR Filter Assistant", layout="wide")
    st.title("FIR Filter Assistant")
    st.caption(
        "Projete filtros FIR em Hz, compare o resultado assistido pela rede com o design direto do SciPy "
        "e explore os resultados com graficos interativos em Plotly."
    )

    with st.sidebar:
        st.header("Configuracao")
        preset_name = st.selectbox("Preset", list(PRESETS.keys()), index=0)
        method = st.selectbox("Metodo de sintese", ["firwin", "remez"], index=0)
        checkpoint_path = st.text_input(
            "Checkpoint",
            value="checkpoints_hybrid/best_paramnet.pth",
            help="Caminho do modelo treinado.",
        )
        st.info(
            "Preencha as especificacoes, gere o filtro e compare o resultado previsto com o design direto.",
        )

    defaults = PRESETS[preset_name]

    with st.form("fir_form"):
        st.subheader("Especificacao do filtro")
        col1, col2, col3 = st.columns(3)
        with col1:
            filter_type = st.selectbox(
                "Tipo de filtro",
                ["lowpass", "highpass"],
                index=0 if defaults["type"] == "lowpass" else 1,
            )
            fs = st.number_input(
                "Frequencia de amostragem (Hz)",
                min_value=1.0,
                value=float(defaults["fs"]),
                step=100.0,
                format="%.2f",
            )
            fc = st.number_input(
                "Frequencia de corte (Hz)",
                min_value=0.1,
                value=float(defaults["fc"]),
                step=10.0,
                format="%.2f",
            )
        with col2:
            trans = st.number_input(
                "Largura da transicao (Hz)",
                min_value=0.1,
                value=float(defaults["trans"]),
                step=10.0,
                format="%.2f",
            )
            rp = st.number_input(
                "Ripple maximo (dB)",
                min_value=0.01,
                value=float(defaults["Rp"]),
                step=0.1,
                format="%.2f",
            )
            attenuation_db = st.number_input(
                "Atenuacao minima (dB)",
                min_value=20.0,
                value=float(defaults["As"]),
                step=1.0,
                format="%.2f",
            )
        with col3:
            order = st.number_input(
                "Numero de taps pedido",
                min_value=8,
                max_value=256,
                value=int(defaults["order"]),
                step=1,
            )
            st.markdown(
                "Use um preset para comecar rapido e depois refine as especificacoes com os controles de visualizacao."
            )

        submitted = st.form_submit_button("Projetar filtro")

    if submitted:
        if not Path(checkpoint_path).exists():
            st.error(f"Checkpoint nao encontrado: {checkpoint_path}")
            return
        if filter_type == "lowpass" and fc + trans >= fs / 2:
            st.error("Para lowpass, `fc + trans` precisa ficar abaixo de Nyquist (`fs / 2`).")
            return
        if filter_type == "highpass" and fc - trans <= 0.0:
            st.error("Para highpass, `fc - trans` precisa ficar acima de 0 Hz.")
            return

        with st.spinner("Carregando modelo e projetando filtro..."):
            st.session_state["fir_result"] = run_fir_prediction(
                checkpoint_path=checkpoint_path,
                filter_type=filter_type,
                fs=fs,
                fc=fc,
                trans=trans,
                rp=rp,
                attenuation_db=attenuation_db,
                order=order,
                method=method,
            )

    result = st.session_state.get("fir_result")
    if result is None:
        st.info("Preencha os parametros e clique em `Projetar filtro` para gerar o filtro.")
        return

    visual_controls = render_visual_controls(result["request_spec_hz"]["fs"])
    bundle = build_response_bundle(
        result["h_pred"],
        result["h_direct"],
        result["request_spec_hz"],
        result["predicted_fc_hz"],
        n_fft=visual_controls["frequency_points"],
        unwrap_phase=visual_controls["unwrap_phase"],
    )

    primary_y_range = visual_controls["y_range"]
    db_plot_range = [visual_controls["db_floor"], 5.0]
    if visual_controls["primary_scale"] == "dB" and primary_y_range is not None:
        db_plot_range = primary_y_range

    magnitude_primary_figure = build_magnitude_figure(
        bundle,
        result["request_spec_hz"],
        visual_controls["selected_curves"],
        visual_controls["primary_scale"],
        x_range=visual_controls["x_range"],
        y_range=primary_y_range,
    )
    magnitude_db_figure = build_magnitude_figure(
        bundle,
        result["request_spec_hz"],
        visual_controls["selected_curves"],
        "dB",
        x_range=visual_controls["x_range"],
        y_range=db_plot_range,
    )
    phase_figure = build_phase_figure(
        bundle,
        result["request_spec_hz"],
        visual_controls["selected_curves"],
        x_range=visual_controls["x_range"],
    )
    impulse_figure = build_impulse_figure(
        result["h_pred"],
        result["h_direct"],
        visual_controls["selected_curves"],
    )
    spec_error_figure = build_spec_error_figure(
        bundle,
        visual_controls["selected_curves"],
        x_range=visual_controls["x_range"],
    )
    group_delay_figure = build_group_delay_figure(
        bundle,
        visual_controls["selected_curves"],
        x_range=visual_controls["x_range"],
    )

    metric_table = render_metric_table(
        result["pred_band"],
        result["direct_band"],
        result["comparison_metrics"],
        result["pred_score"],
        result["direct_score"],
        bundle,
    )

    result_tab, magnitude_tab, phase_tab, impulse_tab, export_tab, explain_tab = st.tabs(
        ["Resumo", "Magnitude", "Fase e atraso", "Impulso e erro", "Exportar", "Explicacao"]
    )

    with result_tab:
        top_metrics = st.columns(4)
        top_metrics[0].metric(
            "Score do pedido",
            f"{result['pred_score']:.2f}",
            delta=f"{result['direct_score'] - result['pred_score']:.2f} melhor que direto",
            help=(
                "Metrica interna do app para medir o quanto o filtro atende o pedido. "
                "Ela combina violacao de ripple, violacao de atenuacao, vies de ganho "
                "na banda util e excesso de taps. Menor e melhor; zero e o ideal."
            ),
        )
        top_metrics[1].metric(
            "Ripple previsto",
            f"{result['pred_band']['ripple_db']:.2f} dB",
            delta=f"{result['pred_band']['ripple_db'] - result['direct_band']['ripple_db']:.2f} dB",
            help=(
                "Variacao de ganho dentro da passband do filtro previsto. "
                "Compare este valor com o ripple maximo pedido (Rp). Menor costuma ser melhor."
            ),
        )
        top_metrics[2].metric(
            "Atenuacao prevista",
            f"{result['pred_band']['attenuation_db']:.2f} dB",
            delta=f"{result['pred_band']['attenuation_db'] - result['direct_band']['attenuation_db']:.2f} dB",
            help=(
                "Nivel de rejeicao obtido na stopband do filtro previsto. "
                "Compare este valor com a atenuacao minima pedida (As). Maior costuma ser melhor."
            ),
        )
        top_metrics[3].metric(
            "Violacao maxima",
            f"{np.nanmax(bundle['pred_spec_violation_db']):.2f} dB",
            delta=f"{np.nanmax(bundle['direct_spec_violation_db']) - np.nanmax(bundle['pred_spec_violation_db']):.2f} dB",
            help=(
                "Maior distancia entre a resposta do filtro previsto e a mascara de especificacao. "
                "Se estiver em zero, o filtro nao viola a mascara nas bandas avaliadas."
            ),
        )

        adjust_metrics = st.columns(4)
        adjust_metrics[0].metric(
            "Cutoff sugerido",
            f"{result['predicted_fc_hz']:.2f} Hz",
            delta=f"{result['predicted_fc_hz'] - result['request_spec_hz']['fc']:.2f} Hz",
            help=(
                "Frequencia de corte que o modelo sugeriu antes da sintese final do FIR. "
                "Ela pode diferir do valor pedido para tentar melhorar o atendimento da especificacao."
            ),
        )
        adjust_metrics[1].metric(
            "Transicao sugerida",
            f"{result['predicted_trans_hz']:.2f} Hz",
            delta=f"{result['predicted_trans_hz'] - result['request_spec_hz']['trans']:.2f} Hz",
            help=(
                "Largura de transicao sugerida pelo modelo. "
                "Uma transicao mais larga costuma facilitar atenuacao, enquanto uma mais estreita exige mais do filtro."
            ),
        )
        adjust_metrics[2].metric(
            "Atenuacao sugerida",
            f"{result['predicted_attenuation']:.2f} dB",
            delta=f"{result['predicted_attenuation'] - result['request_spec_hz']['As']:.2f} dB",
            help=(
                "Valor de atenuacao que o modelo sugeriu como parametro de projeto "
                "antes da sintese. Nao e a atenuacao medida final; essa aparece nas metricas acima."
            ),
        )
        adjust_metrics[3].metric(
            "Taps previstos",
            len(result["h_pred"]),
            delta=len(result["h_pred"]) - int(result["request_spec_hz"]["order"]),
            help=(
                "Numero final de coeficientes FIR do filtro previsto. "
                "Mais taps normalmente significam mais flexibilidade, porem maior custo computacional."
            ),
        )

        status_cols = st.columns(2)
        if result["pred_band"]["meets_both"]:
            render_status_with_help(
                status_cols[0],
                "O filtro previsto atende ripple e atenuacao pedidos.",
                "success",
                (
                    "O filtro previsto ficou dentro dos limites de ripple e de atenuacao "
                    "avaliados a partir da especificacao informada."
                ),
                key="predicted_status_help",
            )
        else:
            render_status_with_help(
                status_cols[0],
                "O filtro previsto ainda nao atende completamente a especificacao.",
                "warning",
                (
                    "Isso significa que o filtro previsto falhou em pelo menos um dos dois criterios principais: "
                    "ripple maximo na passband ou atenuacao minima na stopband. "
                    "Ainda assim, ele pode ficar melhor que o design direto em parte dos objetivos."
                ),
                key="predicted_status_help",
            )
        if result["direct_band"]["meets_both"]:
            render_status_with_help(
                status_cols[1],
                "O design direto tambem atende a especificacao.",
                "success",
                (
                    "O filtro projetado diretamente pelo metodo classico tambem ficou dentro "
                    "dos limites de ripple e atenuacao do pedido."
                ),
                key="direct_status_help",
            )
        else:
            render_status_with_help(
                status_cols[1],
                "O design direto nao atende completamente a especificacao.",
                "warning",
                (
                    "Isso significa que o projeto direto do SciPy falhou em pelo menos um dos dois criterios "
                    "principais: ripple maximo na passband ou atenuacao minima na stopband. "
                    "Esse aviso ajuda a comparar se a sugestao do modelo trouxe alguma melhora pratica."
                ),
                key="direct_status_help",
            )

        st.markdown("### Visao geral interativa")
        st.plotly_chart(
            magnitude_primary_figure,
            use_container_width=True,
            config=PLOTLY_CONFIG,
            key="summary_magnitude_primary_chart",
        )

        st.markdown("### Resumo numerico")
        st.dataframe(
            {
                "Metrica": list(metric_table.keys()),
                "Valor": [format_metric_value(value) for value in metric_table.values()],
            },
            use_container_width=True,
            hide_index=True,
        )

    with magnitude_tab:
        st.markdown("### Grafico principal")
        st.plotly_chart(
            magnitude_primary_figure,
            use_container_width=True,
            config=PLOTLY_CONFIG,
            key="magnitude_primary_chart",
        )
        st.markdown("### Magnitude em dB")
        st.plotly_chart(
            magnitude_db_figure,
            use_container_width=True,
            config=PLOTLY_CONFIG,
            key="magnitude_db_chart",
        )

    with phase_tab:
        st.markdown("### Fase")
        st.plotly_chart(
            phase_figure,
            use_container_width=True,
            config=PLOTLY_CONFIG,
            key="phase_chart",
        )
        if visual_controls["show_group_delay"]:
            st.markdown("### Atraso de grupo")
            st.plotly_chart(
                group_delay_figure,
                use_container_width=True,
                config=PLOTLY_CONFIG,
                key="group_delay_chart",
            )
        else:
            st.info("Ative `Mostrar atraso de grupo` na barra lateral para exibir esse grafico.")

    with impulse_tab:
        st.markdown("### Coeficientes FIR / resposta ao impulso")
        st.plotly_chart(
            impulse_figure,
            use_container_width=True,
            config=PLOTLY_CONFIG,
            key="impulse_chart",
        )
        st.markdown("### Erro entre especificacao e resposta obtida")
        st.plotly_chart(
            spec_error_figure,
            use_container_width=True,
            config=PLOTLY_CONFIG,
            key="spec_error_chart",
        )

    with export_tab:
        st.markdown("### Exportar coeficientes do FIR previsto")
        export_formats = {
            "txt": "Lista simples separada por virgulas",
            "python": "Array Python",
            "c": "Array C",
            "matlab": "Array MATLAB",
        }
        copy_format = st.selectbox(
            "Formato para copiar os coeficientes",
            list(export_formats.keys()),
            format_func=lambda value: value.upper(),
            key="copy_coeff_format",
        )
        copy_content = serialize_coefficients(result["h_pred"], copy_format)
        st.text_area(
            "Coeficientes para copiar",
            value=copy_content,
            height=220,
            key="copy_coefficients_text",
            help="Selecione o formato desejado, copie o conteudo e use no seu codigo ou ferramenta.",
        )
        for format_name, description in export_formats.items():
            content = serialize_coefficients(result["h_pred"], format_name)
            st.download_button(
                label=f"Baixar coeficientes em {format_name.upper()}",
                data=content,
                file_name=(
                    f"fir_{result['request_spec_hz']['type']}_{int(result['request_spec_hz']['fc'])}hz_"
                    f"{format_name}.{'txt' if format_name == 'txt' else format_name}"
                ),
                mime="text/plain",
                key=f"download_coeff_{format_name}",
                help=description,
            )

        st.markdown("### Exportar dados em CSV")
        export_datasets = build_export_datasets(result, bundle, metric_table)
        dataset_name = st.selectbox("Conjunto de dados", list(export_datasets.keys()))
        selected_dataset = export_datasets[dataset_name]
        st.download_button(
            label=f"Baixar {dataset_name}",
            data=selected_dataset["bytes"],
            file_name=selected_dataset["filename"],
            mime="text/csv",
            key="download_csv_dataset",
        )

        st.markdown("### Exportar figuras")
        export_figures = {
            "Magnitude principal": magnitude_primary_figure,
            "Magnitude em dB": magnitude_db_figure,
            "Fase": phase_figure,
            "Impulso / coeficientes": impulse_figure,
            "Erro de especificacao": spec_error_figure,
        }
        if visual_controls["show_group_delay"]:
            export_figures["Atraso de grupo"] = group_delay_figure

        figure_name = st.selectbox("Figura", list(export_figures.keys()))
        selected_figure = export_figures[figure_name]
        export_base = figure_name.lower().replace(" ", "_").replace("/", "_")

        html_bytes, html_mime = figure_download_payload(selected_figure, "html")
        st.download_button(
            label="Baixar figura em HTML",
            data=html_bytes,
            file_name=f"{export_base}.html",
            mime=html_mime,
            key="download_figure_html",
        )

        try:
            png_bytes, png_mime = figure_download_payload(selected_figure, "png")
            st.download_button(
                label="Baixar figura em PNG",
                data=png_bytes,
                file_name=f"{export_base}.png",
                mime=png_mime,
                key="download_figure_png",
            )
            svg_bytes, svg_mime = figure_download_payload(selected_figure, "svg")
            st.download_button(
                label="Baixar figura em SVG",
                data=svg_bytes,
                file_name=f"{export_base}.svg",
                mime=svg_mime,
                key="download_figure_svg",
            )
        except Exception as exc:
            st.warning(
                "A exportacao em PNG/SVG depende do Kaleido. "
                f"Erro atual: {exc}"
            )

    with explain_tab:
        st.markdown(
            """
### O que este app faz

Este app recebe uma especificacao de filtro FIR, gera um filtro previsto pelo modelo
e compara esse resultado com um design direto feito pelo metodo classico selecionado.
Assim voce consegue ver rapidamente se a sugestao prevista melhorou o atendimento da especificacao.

### Como usar

1. Escolha um preset ou preencha manualmente os parametros do filtro.
2. Clique em `Projetar filtro`.
3. Veja no `Resumo` se o filtro previsto melhorou score, ripple e atenuacao.
4. Explore os graficos nas abas para analisar magnitude, fase, atraso e impulso.
5. Na aba `Exportar`, copie ou baixe os coeficientes e os dados.

### Especificacoes do filtro

- `Tipo de filtro`: define se o filtro deixa passar baixas frequencias (`lowpass`) ou altas frequencias (`highpass`).
- `Frequencia de amostragem (fs)`: referencia em Hz usada para interpretar todas as frequencias do projeto.
- `Frequencia de corte (fc)`: ponto principal onde o filtro comeca a mudar de comportamento.
- `Largura da transicao`: faixa entre a banda util e a banda de rejeicao.
- `Ripple maximo (Rp)`: variacao tolerada na banda util, em dB.
- `Atenuacao minima (As)`: rejeicao minima esperada na banda que deve ser suprimida, em dB.
- `Numero de taps`: quantidade alvo de coeficientes FIR. Mais taps costumam dar mais liberdade ao filtro, com custo computacional maior.
- `Metodo de sintese`: tecnica classica usada para gerar o FIR final a partir dos parametros.

### Como o score e calculado

O `Score do pedido` resume o quanto o filtro atende a especificacao. Menor e melhor.
O valor combina quatro partes:

```text
score = 2 * violacao_de_ripple
      + 1 * violacao_de_atenuacao
      + 0.1 * vies_de_ganho_na_passband
      + penalidade_por_excesso_de_taps
```

- `Violacao de ripple`: so aparece se o ripple medido passar do limite `Rp`.
- `Violacao de atenuacao`: so aparece se a atenuacao medida ficar abaixo de `As`.
- `Vies de ganho`: mede quanto a passband se afastou, em media, do ganho esperado.
- `Penalidade por taps`: aparece se o filtro final usar mais taps do que o numero pedido.

Se o score for `0`, o filtro nao violou esses criterios nas bandas avaliadas.

### Como interpretar os graficos

- `Magnitude`: mostra o comportamento geral do filtro. Compare a curva prevista com a direta e com os limites da especificacao.
- `Magnitude em dB`: facilita enxergar ripple na passband e atenuacao na stopband.
- `Fase`: mostra como a fase varia com a frequencia.
- `Atraso de grupo`: ajuda a verificar consistencia temporal e comportamento de fase aproximadamente linear.
- `Coeficientes FIR / resposta ao impulso`: mostra os taps do filtro e a diferenca entre previsto e direto.
- `Erro entre especificacao e resposta obtida`: destaca em quais frequencias a resposta ainda viola a mascara do pedido.
            """
        )


if __name__ == "__main__":
    if not is_streamlit_context():
        print("Execute este app com:")
        print("python -m streamlit run streamlit_app.py")
        raise SystemExit(1)
    main()
