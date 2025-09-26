import torch
import numpy as np
from hybrid_model import ParamNet
from hybrid_helpers import decode_outputs, synthesize_fir

def load_paramnet(ckpt_path, hidden=(256,256,128), dropout=0.1, device=None):
    """
    Carrega modelo treinado e scaler.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # modelo com 6 entradas/saídas
    model = ParamNet(in_dim=6, hidden=hidden, dropout=dropout).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    mean = ckpt["t_scaler_mean"]
    std  = ckpt["t_scaler_std"]
    t_scaler = (mean, std)

    return model, t_scaler, device

@torch.no_grad()
def predict_fir_from_specs(specs_np, model, t_scaler, device=None, method="remez"):
    """
    Predição híbrida: especificações -> parâmetros previstos -> FIR sintetizado.

    specs_np: array (B,6) no espaço natural:
        [fc, trans, Rp, As, order, type]
        - fc/trans normalizados (0..0.5)
        - type = 0 (lowpass) ou 1 (highpass)

    Retorna:
        coefs_list: lista de arrays (coeficientes FIR)
        params: dict com arrays decodificados
    """
    device = device or next(model.parameters()).device

    x = torch.from_numpy(specs_np.astype(np.float32)).to(device)
    z_pred = model(x).cpu().numpy()

    params = decode_outputs(z_pred, t_scaler)
    coefs_list = synthesize_fir(params, method=method)

    return coefs_list, params
