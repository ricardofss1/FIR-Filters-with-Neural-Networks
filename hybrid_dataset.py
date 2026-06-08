import numpy as np
import torch
from torch.utils.data import Dataset

from hybrid_helpers import compute_standard_scaler, encode_inputs, encode_targets, standardize


class ParamDataset(Dataset):
    """
    Le um dataset .npz com:
    - specs: especificacoes de entrada do usuario
    - targets: especificacoes ajustadas desejadas (opcional; fallback para specs)
    - coefs: coeficientes FIR correspondentes ao alvo
    - orders: numero real de taps dos coeficientes
    - difficulty: peso relativo por amostra para enfatizar casos mais dificeis
    """

    def __init__(self, npz_path, input_scaler=None, target_scaler=None):
        data = np.load(npz_path)
        self.specs = data["specs"].astype(np.float32)
        self.target_specs = data["targets"].astype(np.float32) if "targets" in data else self.specs.copy()
        self.coefs = data["coefs"].astype(np.float32)
        self.orders = data["orders"].astype(np.int32)

        difficulty_raw = data["difficulty"].astype(np.float32) if "difficulty" in data else None
        if difficulty_raw is None:
            difficulty_raw = np.ones(self.specs.shape[0], dtype=np.float32)

        difficulty_raw = np.where(np.isfinite(difficulty_raw), difficulty_raw, 1.0).astype(np.float32)
        difficulty_raw = np.clip(difficulty_raw, 0.25, None)
        difficulty_mean = float(np.mean(difficulty_raw)) if difficulty_raw.size > 0 else 1.0
        if difficulty_mean <= 0.0:
            difficulty_mean = 1.0
        self.difficulty = np.clip(difficulty_raw / difficulty_mean, 0.5, 4.0).astype(np.float32)

        self.inputs_raw = encode_inputs(self.specs)
        self.targets_raw = encode_targets(self.target_specs)

        if input_scaler is None:
            self.x_scaler = compute_standard_scaler(self.inputs_raw)
        else:
            self.x_scaler = input_scaler

        if target_scaler is None:
            self.t_scaler = compute_standard_scaler(self.targets_raw)
        else:
            self.t_scaler = target_scaler

        self.inputs = standardize(self.inputs_raw, self.x_scaler)
        self.targets = standardize(self.targets_raw, self.t_scaler)

    def __len__(self):
        return self.specs.shape[0]

    def __getitem__(self, idx):
        return {
            "x": torch.from_numpy(self.inputs[idx]),
            "y": torch.from_numpy(self.targets[idx]),
            "spec": torch.from_numpy(self.specs[idx]),
            "target_spec": torch.from_numpy(self.target_specs[idx]),
            "coef": torch.from_numpy(self.coefs[idx]),
            "order": torch.tensor(self.orders[idx], dtype=torch.int32),
            "difficulty": torch.tensor(self.difficulty[idx], dtype=torch.float32),
        }

    def get_input_scaler(self):
        return self.x_scaler

    def get_target_scaler(self):
        return self.t_scaler

    @property
    def input_dim(self):
        return self.inputs.shape[1]

    @property
    def target_dim(self):
        return self.targets.shape[1]
