import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal

# ---- Espaço natural dos parâmetros ----
# spec = [fc, trans, Rp(dB), As(dB), order, type]
# - order = número de taps (L)
# - type  = 0 (lowpass), 1 (highpass)
PARAM_BOUNDS = {
    "fc":   (0.02, 0.45),
    "trans":(0.005, 0.12),
    "Rp":   (0.01, 1.0),    # dB
    "As":   (30.0, 100.0),  # dB
    "order":(8, 256),       # número de taps (L)
    "type": (0, 1),         # lowpass=0, highpass=1
}

def encode_targets(specs_np, logspace=("trans","Rp","As","order")):
    """
    Converte params naturais -> espaço de treino (log10 para faixas largas).
    Retorna array float32 shape (N,6).
    """
    specs = specs_np.copy().astype(np.float32)
    cols = ["fc","trans","Rp","As","order","type"]
    X = np.zeros_like(specs, dtype=np.float32)

    for i, name in enumerate(cols):
        v = specs[:, i]
        if name in logspace:
            v = np.log10(v)
        X[:, i] = v
    return X

def compute_standard_scaler(X):
    mean = X.mean(axis=0).astype(np.float32)
    std  = X.std(axis=0).astype(np.float32)
    std[std == 0] = 1.0
    return (mean, std)

def standardize(X, scaler):
    mean, std = scaler
    return (X - mean) / std

def destandardize(Z, scaler):
    mean, std = scaler
    return Z * std + mean

def decode_outputs(pred_z, scaler, logspace=("trans","Rp","As","order")):
    """
    Rede -> espaço padronizado -> (despadroniza) -> (anti-log10) -> recortes para bounds.
    Retorna dict com array shape (B,) para cada parâmetro.
    """
    cols = ["fc","trans","Rp","As","order","type"]
    z = destandardize(pred_z, scaler)
    out = {}
    for i, name in enumerate(cols):
        v = z[:, i]
        if name in logspace:
            v = (10.0 ** v)
        v = v.clip(PARAM_BOUNDS[name][0], PARAM_BOUNDS[name][1])
        out[name] = v

    # ordem = número de taps (inteiro dentro dos limites)
    out["order"] = np.rint(out["order"]).astype(np.int32)
    out["order"] = np.clip(out["order"],
                           PARAM_BOUNDS["order"][0],
                           PARAM_BOUNDS["order"][1])

    # type = discreto (0=lowpass, 1=highpass)
    out["type"] = np.rint(out["type"]).astype(np.int32)
    out["type"] = np.clip(out["type"],
                          PARAM_BOUNDS["type"][0],
                          PARAM_BOUNDS["type"][1])
    return out

def synthesize_fir(params_dict, method="remez"):
    """
    Gera coeficientes FIR a partir dos parâmetros.
    
    params_dict: dict com vetores 'fc', 'trans', 'order', 'type' etc. (todos shape (B,))
    - 'order' = número de taps (L)
    - 'fc' e 'trans' normalizados (0..0.5)
    - 'type' = 0 (lowpass), 1 (highpass)

    Retorna:
        lista de arrays numpy (coeficientes FIR), um por amostra.
    """
    B = len(params_dict["fc"])
    coefs_list = []

    for b in range(B):
        fc = float(params_dict["fc"][b])
        tr = float(params_dict["trans"][b])
        L  = int(params_dict["order"][b])
        ftype = int(params_dict["type"][b])

        # Garante limites válidos
        L = int(np.clip(L, PARAM_BOUNDS["order"][0], PARAM_BOUNDS["order"][1]))

        try:
            if method == "remez":
                if ftype == 0:  # lowpass
                    bands   = [0.0, fc, fc + tr, 0.5]
                    desired = [1.0, 0.0]
                else:  # highpass
                    bands   = [0.0, max(fc-tr, 0.0), fc, 0.5]
                    desired = [0.0, 1.0]
                h = signal.remez(L, bands, desired, fs=1.0)

            else:  # firwin
                fc_eff = min(max(fc, 0.01), 0.49)
                pass_zero = True if ftype == 0 else False
                h = signal.firwin(L, fc_eff, window="hamming", fs=1.0, pass_zero=pass_zero)

        except Exception:
            # Fallback robusto
            fc_eff = min(max(fc, 0.01), 0.49)
            pass_zero = True if ftype == 0 else False
            if ftype == 1:
                if L % 2 == 0:
                    L += 1  # força ímpar para evitar erro do firwin
                    h = signal.firwin(L, fc_eff, window="hamming", fs=1.0, pass_zero=pass_zero)

        # Garantia final: h tem L taps
        if len(h) != L:
            h = np.resize(h, L)

        coefs_list.append(h.astype(np.float32))

    return coefs_list
