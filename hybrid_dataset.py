# hybrid_dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset
from hybrid_helpers import encode_targets, compute_standard_scaler, standardize

class ParamDataset(Dataset):
    """
    Lê fir_dataset.npz (specs, coefs, orders).
    Entrada (X): specs naturais (fc, trans, Rp, As, order, type)
    Alvo (y): specs transformadas (log10 em faixas largas) e padronizadas.
    """
    def __init__(self, npz_path, target_scaler=None):
        data = np.load(npz_path)
        self.specs  = data["specs"].astype(np.float32)  # (N,6)
        self.coefs  = data["coefs"].astype(np.float32)  # (N, Nmax)
        self.orders = data["orders"].astype(np.int32)   # (N,)

        # aplica encode para (N,6) com log nos campos apropriados
        self.targets_raw = encode_targets(self.specs)   

        if target_scaler is None:
            self.t_scaler = compute_standard_scaler(self.targets_raw)
        else:
            self.t_scaler = target_scaler

        self.targets = standardize(self.targets_raw, self.t_scaler)

    def __len__(self):
        return self.specs.shape[0]

    def __getitem__(self, idx):
        return {
            "x": torch.from_numpy(self.specs[idx]),     # natural
            "y": torch.from_numpy(self.targets[idx]),   # padronizado
        }

    def get_target_scaler(self):
        return self.t_scaler
