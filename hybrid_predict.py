import numpy as np
import torch

from hybrid_helpers import decode_outputs, encode_inputs, standardize, synthesize_fir
from hybrid_model import LegacyParamNet, ParamNet, SplitHeadParamNet


def load_paramnet(ckpt_path, hidden=None, dropout=None, device=None):
    """
    Load a trained ParamNet checkpoint and its scalers.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    state_dict = ckpt["model_state"]
    legacy_format = any(key.startswith("net.") for key in state_dict.keys())
    split_head_format = ckpt.get("architecture") == "split_head_v1" or "param_head.bias" in state_dict
    legacy_out_dim = None
    if legacy_format:
        bias_candidates = []
        for key, value in state_dict.items():
            if key.startswith("net.") and key.endswith(".bias"):
                try:
                    layer_index = int(key.split(".")[1])
                except (IndexError, ValueError):
                    continue
                bias_candidates.append((layer_index, value.shape[0]))
        if bias_candidates:
            legacy_out_dim = max(bias_candidates, key=lambda item: item[0])[1]
    split_head_out_dim = None
    if split_head_format and "param_head.bias" in state_dict and "order_head.bias" in state_dict:
        split_head_out_dim = state_dict["param_head.bias"].shape[0] + state_dict["order_head.bias"].shape[0]
    out_dim = int(
        ckpt.get(
            "out_dim",
            split_head_out_dim
            if split_head_out_dim is not None
            else state_dict["head.bias"].shape[0]
            if "head.bias" in state_dict
            else legacy_out_dim
            if legacy_out_dim is not None
            else 6,
        )
    )
    hidden = tuple(hidden or ckpt.get("hidden", (256, 256, 128)))
    default_dropout = ckpt.get("dropout", 0.1)
    dropout = float(default_dropout if dropout is None else dropout)
    in_dim = int(ckpt.get("in_dim", 6))
    residual_to_input = bool(ckpt.get("residual_to_input", False))
    residual_order_to_input = bool(ckpt.get("residual_order_to_input", True))
    order_head_width = ckpt.get("order_head_width")

    if legacy_format:
        model = LegacyParamNet(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden=hidden,
            dropout=dropout,
        ).to(device)
    elif split_head_format:
        model = SplitHeadParamNet(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden=hidden,
            dropout=dropout,
            residual_to_input=residual_to_input,
            residual_order_to_input=residual_order_to_input,
            order_head_width=order_head_width,
        ).to(device)
    else:
        model = ParamNet(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden=hidden,
            dropout=dropout,
            residual_to_input=residual_to_input,
        ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    predictor_state = {
        "input_scaler": None,
        "target_scaler": None,
        "out_dim": out_dim,
    }

    if "input_scaler_mean" in ckpt and "input_scaler_std" in ckpt:
        predictor_state["input_scaler"] = (
            ckpt["input_scaler_mean"],
            ckpt["input_scaler_std"],
        )

    if "target_scaler_mean" in ckpt and "target_scaler_std" in ckpt:
        predictor_state["target_scaler"] = (
            ckpt["target_scaler_mean"],
            ckpt["target_scaler_std"],
        )
    else:
        predictor_state["target_scaler"] = (
            ckpt["t_scaler_mean"],
            ckpt["t_scaler_std"],
        )

    return model, predictor_state, device


@torch.no_grad()
def predict_fir_from_specs(specs_np, model, predictor_state, device=None, method="remez"):
    """
    Hybrid prediction: specs -> predicted parameters -> synthesized FIR.

    specs_np must follow the natural input format used by the dataset:
        [fc, trans, Rp, As, order, type]
    """
    device = device or next(model.parameters()).device
    specs_np = specs_np.astype(np.float32)

    encoded_inputs = encode_inputs(specs_np)
    input_scaler = predictor_state.get("input_scaler")
    model_inputs = standardize(encoded_inputs, input_scaler) if input_scaler is not None else specs_np

    x_data = torch.from_numpy(model_inputs.astype(np.float32)).to(device)
    z_pred = model(x_data).cpu().numpy()

    target_scaler = predictor_state["target_scaler"]
    params = decode_outputs(z_pred, target_scaler, input_specs=specs_np)
    coefs_list = synthesize_fir(params, method=method)
    return coefs_list, params
