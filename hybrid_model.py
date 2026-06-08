import torch
import torch.nn as nn


class LegacyParamNet(nn.Module):
    """
    Arquitetura original baseada em MLP sequencial simples.
    """

    def __init__(self, in_dim=6, out_dim=6, hidden=(256, 256, 128), dropout=0.1):
        super().__init__()
        layers = []
        prev = in_dim
        for width in hidden:
            layers.extend([nn.Linear(prev, width), nn.ReLU(inplace=True)])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = width
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ParamNet(nn.Module):
    """
    MLP residual para mapear specs padronizadas -> parametros ajustados padronizados.

    Entradas:
        [fc, trans, Rp, As, order, type]

    Saidas padronizadas por padrao:
        [fc, trans, Rp, As, order]

    O tipo do filtro permanece vindo da entrada e nao precisa ser previsto.
    """

    def __init__(
        self,
        in_dim=6,
        out_dim=5,
        hidden=(256, 256, 128),
        dropout=0.05,
        residual_to_input=True,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.residual_to_input = residual_to_input

        layers = []
        prev = in_dim
        for width in hidden:
            layers.extend(
                [
                    nn.Linear(prev, width),
                    nn.LayerNorm(width),
                    nn.GELU(),
                ]
            )
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = width

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(x)
        output = self.head(hidden)
        if self.residual_to_input:
            output = output + x[:, : self.out_dim]
        return output


class SplitHeadParamNet(nn.Module):
    """
    MLP com backbone compartilhado e cabeca dedicada para order.

    Saida final:
        [fc, trans, Rp, As, order]

    Mantem o formato de saida concatenado para preservar compatibilidade
    com o restante do pipeline, mas separa internamente order em uma head
    propria para facilitar o aprendizado do componente discreto.
    """

    def __init__(
        self,
        in_dim=6,
        out_dim=5,
        hidden=(256, 256, 128),
        dropout=0.05,
        residual_to_input=True,
        residual_order_to_input=True,
        order_head_width=None,
    ):
        super().__init__()
        if out_dim < 2:
            raise ValueError("SplitHeadParamNet requires out_dim >= 2.")

        self.out_dim = out_dim
        self.continuous_out_dim = out_dim - 1
        self.residual_to_input = residual_to_input
        self.residual_order_to_input = residual_order_to_input

        layers = []
        prev = in_dim
        for width in hidden:
            layers.extend(
                [
                    nn.Linear(prev, width),
                    nn.LayerNorm(width),
                    nn.GELU(),
                ]
            )
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = width

        self.backbone = nn.Sequential(*layers)
        self.param_head = nn.Linear(prev, self.continuous_out_dim)

        order_head_width = int(order_head_width or max(64, prev // 2))
        order_layers = [
            nn.Linear(prev, order_head_width),
            nn.LayerNorm(order_head_width),
            nn.GELU(),
        ]
        if dropout > 0:
            order_layers.append(nn.Dropout(dropout))
        self.order_tower = nn.Sequential(*order_layers)
        self.order_head = nn.Linear(order_head_width, 1)
        self.order_head_width = order_head_width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(x)

        continuous_output = self.param_head(hidden)
        order_output = self.order_head(self.order_tower(hidden))

        if self.residual_to_input:
            continuous_output = continuous_output + x[:, : self.continuous_out_dim]
        if self.residual_order_to_input:
            order_output = order_output + x[:, self.continuous_out_dim : self.continuous_out_dim + 1]

        return torch.cat([continuous_output, order_output], dim=1)
