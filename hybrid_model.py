# hybrid_model.py
import torch.nn as nn
import torch
from typing import Tuple

class ParamNet(nn.Module):
    """
    MLP que mapeia specs naturais (fc, trans, Rp, As, order, type)
    -> parâmetros no espaço padronizado.
    
    Saídas: 6 valores padronizados (z-score):
        [fc, trans, Rp, As, order, type]
    
    A loss é MSE no espaço padronizado.
    """
    def __init__(self, in_dim=6, hidden=(256,256,128), dropout=0.1):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 6)]  # agora prevemos também "type"
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # retorna z-score dos 6 parâmetros
