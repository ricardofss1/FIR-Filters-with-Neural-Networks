import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from hybrid_dataset import ParamDataset
from hybrid_helpers import ADJUSTABLE_PARAM_NAMES, CONTINUOUS_PARAM_NAMES, decode_outputs
from hybrid_model import SplitHeadParamNet


CONTINUOUS_LOSS_WEIGHTS = torch.tensor([2.0, 1.5, 1.0, 1.0], dtype=torch.float32)
ORDER_LOSS_WEIGHT = 2.5


def split_prediction_targets(prediction, target):
    continuous_pred = prediction[:, : len(CONTINUOUS_PARAM_NAMES)]
    order_pred = prediction[:, len(CONTINUOUS_PARAM_NAMES) :]
    continuous_target = target[:, : len(CONTINUOUS_PARAM_NAMES)]
    order_target = target[:, len(CONTINUOUS_PARAM_NAMES) :]
    return continuous_pred, order_pred, continuous_target, order_target


def weighted_smooth_l1(prediction, target, sample_weight=None, return_parts=False):
    continuous_pred, order_pred, continuous_target, order_target = split_prediction_targets(
        prediction, target
    )

    continuous_weights = CONTINUOUS_LOSS_WEIGHTS.to(prediction.device)
    continuous_loss = F.smooth_l1_loss(
        continuous_pred,
        continuous_target,
        reduction="none",
        beta=0.25,
    )
    continuous_per_sample = (continuous_loss * continuous_weights).mean(dim=1)

    order_loss = F.smooth_l1_loss(
        order_pred,
        order_target,
        reduction="none",
        beta=0.15,
    ).squeeze(1)
    order_per_sample = ORDER_LOSS_WEIGHT * order_loss

    total_per_sample = continuous_per_sample + order_per_sample

    if sample_weight is not None:
        sample_weight = sample_weight.to(prediction.device).float()
        sample_weight = sample_weight / sample_weight.mean().clamp_min(1e-6)
        total_loss = (total_per_sample * sample_weight).mean()
    else:
        total_loss = total_per_sample.mean()

    if not return_parts:
        return total_loss

    return total_loss, {
        "continuous": float(continuous_per_sample.mean().item()),
        "order": float(order_per_sample.mean().item()),
        "total": float(total_per_sample.mean().item()),
    }


@torch.no_grad()
def eval_param_regression(model, loader, device, predictor_state):
    """
    Avaliacao do loss padronizado e do erro medio no espaco natural.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    total_abs_error = np.zeros(len(ADJUSTABLE_PARAM_NAMES), dtype=np.float64)

    for batch in loader:
        x_data = batch["x"].to(device)
        y_data = batch["y"].to(device)
        prediction = model(x_data)
        loss = weighted_smooth_l1(prediction, y_data)

        total_loss += loss.item() * x_data.size(0)
        total_samples += x_data.size(0)

        pred_params = decode_outputs(
            prediction.cpu().numpy(),
            predictor_state["target_scaler"],
            input_specs=batch["spec"].cpu().numpy(),
        )
        target_specs = batch["target_spec"].cpu().numpy()
        for index, name in enumerate(ADJUSTABLE_PARAM_NAMES):
            total_abs_error[index] += np.abs(pred_params[name] - target_specs[:, index]).sum()

    mean_loss = total_loss / max(total_samples, 1)
    mean_abs_error = total_abs_error / max(total_samples, 1)
    natural_mae = {
        name: float(value) for name, value in zip(ADJUSTABLE_PARAM_NAMES, mean_abs_error)
    }
    return mean_loss, natural_mae


def train_param_regression(
    npz_path,
    out_dir="checkpoints_hybrid",
    batch_size=128,
    epochs=50,
    lr=2e-3,
    val_split=0.15,
    seed=0,
    hidden=(256, 256, 128),
    dropout=0.05,
    max_grad_norm=5.0,
    num_workers=0,
    device=None,
):
    """
    Treina uma ParamNet com cabeca dedicada para order para regressao dos
    5 parametros ajustaveis: fc, trans, Rp, As e order.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)

    full = ParamDataset(npz_path)
    total_count = len(full)
    val_count = int(total_count * val_split)
    test_count = val_count
    train_count = total_count - val_count - test_count
    train_ds, val_ds, test_ds = random_split(
        full,
        [train_count, val_count, test_count],
        generator=torch.Generator().manual_seed(seed),
    )

    predictor_state = {
        "input_scaler": full.get_input_scaler(),
        "target_scaler": full.get_target_scaler(),
    }

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    model = SplitHeadParamNet(
        in_dim=full.input_dim,
        out_dim=full.target_dim,
        hidden=hidden,
        dropout=dropout,
        residual_to_input=True,
        residual_order_to_input=True,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )

    best_val = float("inf")
    best_path = os.path.join(out_dir, "best_paramnet.pth")

    print(
        f"[Hybrid] Train ParamNet: N_train={train_count}, N_val={val_count}, N_test={test_count}"
    )
    for epoch in range(1, epochs + 1):
        model.train()
        start_time = time.time()
        running_loss = 0.0
        running_base_loss = 0.0
        running_continuous_loss = 0.0
        running_order_loss = 0.0
        seen = 0

        for batch in train_loader:
            x_data = batch["x"].to(device)
            y_data = batch["y"].to(device)
            difficulty = batch["difficulty"].to(device)

            prediction = model(x_data)
            loss, loss_parts = weighted_smooth_l1(
                prediction,
                y_data,
                sample_weight=difficulty,
                return_parts=True,
            )

            optimizer.zero_grad()
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            batch_size_actual = x_data.size(0)
            running_loss += loss.item() * batch_size_actual
            running_base_loss += loss_parts["total"] * batch_size_actual
            running_continuous_loss += loss_parts["continuous"] * batch_size_actual
            running_order_loss += loss_parts["order"] * batch_size_actual
            seen += batch_size_actual

        train_loss = running_loss / max(seen, 1)
        train_base_loss = running_base_loss / max(seen, 1)
        train_continuous_loss = running_continuous_loss / max(seen, 1)
        train_order_loss = running_order_loss / max(seen, 1)
        val_loss, val_nat_mae = eval_param_regression(
            model, val_loader, device, predictor_state
        )
        scheduler.step(val_loss)
        elapsed = time.time() - start_time

        mae_summary = ", ".join(
            f"{name}={value:.4f}" for name, value in val_nat_mae.items()
        )
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | train_unweighted={train_base_loss:.6f} | "
            f"train_cont={train_continuous_loss:.6f} | train_order={train_order_loss:.6f} | "
            f"val_loss={val_loss:.6f} | val_mae[{mae_summary}] | dt={elapsed:.1f}s"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "architecture": "split_head_v1",
                    "model_state": model.state_dict(),
                    "input_scaler_mean": predictor_state["input_scaler"][0],
                    "input_scaler_std": predictor_state["input_scaler"][1],
                    "target_scaler_mean": predictor_state["target_scaler"][0],
                    "target_scaler_std": predictor_state["target_scaler"][1],
                    "hidden": tuple(hidden),
                    "dropout": float(dropout),
                    "in_dim": full.input_dim,
                    "out_dim": full.target_dim,
                    "continuous_out_dim": len(CONTINUOUS_PARAM_NAMES),
                    "order_head_width": model.order_head_width,
                    "residual_to_input": True,
                    "residual_order_to_input": True,
                    "difficulty_weighting": True,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_mae": val_nat_mae,
                },
                best_path,
            )
            print("  -> saved best_paramnet.pth")

    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_loss, test_nat_mae = eval_param_regression(
        model, test_loader, device, predictor_state
    )
    mae_summary = ", ".join(f"{name}={value:.4f}" for name, value in test_nat_mae.items())
    print(f"[Hybrid] Test loss={test_loss:.6f} | test_mae[{mae_summary}]")
    return model, predictor_state
