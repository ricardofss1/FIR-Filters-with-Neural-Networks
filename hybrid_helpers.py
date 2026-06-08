import numpy as np

from fir_utils import design_reference_filter

ALL_PARAM_NAMES = ("fc", "trans", "Rp", "As", "order", "type")
ADJUSTABLE_PARAM_NAMES = ("fc", "trans", "Rp", "As", "order")
CONTINUOUS_PARAM_NAMES = ("fc", "trans", "Rp", "As")
ORDER_PARAM_NAME = "order"
LOGSPACE_PARAMS = {"trans", "Rp", "As", "order"}

# spec = [fc, trans, Rp(dB), As(dB), order, type]
# - order = numero de taps (L)
# - type  = 0 (lowpass), 1 (highpass)
PARAM_BOUNDS = {
    "fc": (0.005, 0.45),
    "trans": (0.002, 0.12),
    "Rp": (0.01, 1.0),
    "As": (30.0, 100.0),
    "order": (8, 256),
    "type": (0, 1),
}


def encode_specs(specs_np, cols=ALL_PARAM_NAMES, logspace=LOGSPACE_PARAMS):
    """
    Converte specs naturais para o espaco de treino.
    """
    specs = specs_np.astype(np.float32)
    encoded = np.zeros((specs.shape[0], len(cols)), dtype=np.float32)
    name_to_index = {name: index for index, name in enumerate(ALL_PARAM_NAMES)}

    for out_index, name in enumerate(cols):
        values = specs[:, name_to_index[name]]
        if name in logspace:
            values = np.log10(values)
        encoded[:, out_index] = values
    return encoded


def encode_inputs(specs_np):
    return encode_specs(specs_np, cols=ALL_PARAM_NAMES)


def encode_targets(specs_np):
    return encode_specs(specs_np, cols=ADJUSTABLE_PARAM_NAMES)


def compute_standard_scaler(x_data):
    mean = x_data.mean(axis=0).astype(np.float32)
    std = x_data.std(axis=0).astype(np.float32)
    std[std == 0] = 1.0
    return mean, std


def standardize(x_data, scaler):
    mean, std = scaler
    return (x_data - mean) / std


def destandardize(z_data, scaler):
    mean, std = scaler
    return z_data * std + mean


def _decode_columns(decoded_values, cols):
    outputs = {}
    for index, name in enumerate(cols):
        values = decoded_values[:, index]
        if name in LOGSPACE_PARAMS:
            values = 10.0 ** values
        outputs[name] = values.clip(PARAM_BOUNDS[name][0], PARAM_BOUNDS[name][1])

    outputs["order"] = np.rint(outputs["order"]).astype(np.int32)
    outputs["order"] = np.clip(
        outputs["order"], PARAM_BOUNDS["order"][0], PARAM_BOUNDS["order"][1]
    )
    return outputs


def decode_outputs(pred_z, scaler, input_specs=None):
    """
    Rede -> espaco padronizado -> despadroniza -> anti-log -> recorte para limites.

    Suporta checkpoints antigos (6 saidas) e novos (5 saidas, com type preservado).
    """
    cols = ADJUSTABLE_PARAM_NAMES if pred_z.shape[1] == 5 else ALL_PARAM_NAMES
    decoded = destandardize(pred_z, scaler)
    outputs = _decode_columns(decoded, cols)

    if "type" not in outputs:
        if input_specs is None:
            raise ValueError("input_specs is required when the model does not predict filter type.")
        outputs["type"] = np.rint(input_specs[:, 5]).astype(np.int32)
    else:
        outputs["type"] = np.rint(outputs["type"]).astype(np.int32)

    outputs["type"] = np.clip(
        outputs["type"], PARAM_BOUNDS["type"][0], PARAM_BOUNDS["type"][1]
    )
    return outputs


def synthesize_fir(params_dict, method="remez"):
    """
    Gera coeficientes FIR a partir dos parametros previstos.
    """
    batch_size = len(params_dict["fc"])
    coefs_list = []

    for batch_index in range(batch_size):
        fc = float(params_dict["fc"][batch_index])
        trans = float(params_dict["trans"][batch_index])
        order = int(params_dict["order"][batch_index])
        ftype = int(params_dict["type"][batch_index])

        order = int(np.clip(order, PARAM_BOUNDS["order"][0], PARAM_BOUNDS["order"][1]))
        fc_eff = min(max(fc, PARAM_BOUNDS["fc"][0]), 0.49)
        ftype_name = "lowpass" if ftype == 0 else "highpass"

        try:
            taps = design_reference_filter(
                fc=fc_eff,
                trans=trans,
                rp=float(params_dict["Rp"][batch_index]),
                attenuation_db=float(params_dict["As"][batch_index]),
                order=order,
                ftype=ftype_name,
                method=method,
                fs=1.0,
            )
        except Exception:
            taps = design_reference_filter(
                fc=fc_eff,
                trans=trans,
                rp=float(params_dict["Rp"][batch_index]),
                attenuation_db=float(params_dict["As"][batch_index]),
                order=order,
                ftype=ftype_name,
                method="firwin",
                fs=1.0,
            )

        coefs_list.append(taps.astype(np.float32))

    return coefs_list
