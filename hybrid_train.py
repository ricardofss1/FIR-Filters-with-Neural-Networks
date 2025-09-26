import os, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader

from hybrid_dataset import ParamDataset
from hybrid_helpers import decode_outputs, synthesize_fir
from hybrid_helpers import compute_standard_scaler
from hybrid_model import ParamNet

@torch.no_grad()
def eval_param_regression(model, loader, device, t_scaler):
    """
    Avaliação de MSE no espaço padronizado (z-score).
    """
    model.eval()
    mse = nn.MSELoss(reduction="mean")
    total = 0.0
    count = 0
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        z_pred = model(x)
        loss = mse(z_pred, y)
        total += loss.item() * x.size(0)
        count += x.size(0)
    return total / max(count, 1)

def train_param_regression(npz_path,
                           out_dir="checkpoints_hybrid",
                           batch_size=128,
                           epochs=50,
                           lr=2e-3,
                           val_split=0.15,
                           seed=0,
                           hidden=(256,256,128),
                           dropout=0.1,
                           max_grad_norm=5.0,
                           device=None):
    """
    Treina ParamNet para regressão de parâmetros FIR.
    Agora com 6 saídas: fc, trans, Rp, As, order, type.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)

    # Dataset & split
    full = ParamDataset(npz_path)
    N = len(full)
    n_val = int(N*val_split); n_test = n_val; n_train = N - n_val - n_test
    train_ds, val_ds, test_ds = random_split(full, [n_train, n_val, n_test],
                                             generator=torch.Generator().manual_seed(seed))
    # Reutilizamos scaler salvo no dataset
    t_scaler = full.get_target_scaler()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2)

    # Modelo atualizado para 6 entradas/saídas
    model = ParamNet(in_dim=6, hidden=hidden, dropout=dropout).to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)

    criterion = nn.MSELoss(reduction="mean")
    best_val = float("inf"); best_path = os.path.join(out_dir, "best_paramnet.pth")

    print(f"[Hybrid] Train ParamNet: N_train={n_train}, N_val={n_val}, N_test={n_test}")
    for ep in range(1, epochs+1):
        model.train()
        t0 = time.time()
        total = 0.0; seen = 0
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            z = model(x)
            loss = criterion(z, y)

            opt.zero_grad()
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()

            total += loss.item()*x.size(0); seen += x.size(0)

        tr = total/max(seen,1)
        va = eval_param_regression(model, val_loader, device, t_scaler)
        sched.step(va)
        dt = time.time()-t0
        print(f"Epoch {ep:03d} | train_zMSE={tr:.6f} | val_zMSE={va:.6f} | dt={dt:.1f}s")

        if va < best_val:
            best_val = va
            torch.save({
                "model_state": model.state_dict(),
                "t_scaler_mean": t_scaler[0],
                "t_scaler_std":  t_scaler[1],
                "epoch": ep,
                "val_zMSE": va
            }, best_path)
            print("  -> saved best_paramnet.pth")

    # Teste final
    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    test_zmse = eval_param_regression(model, test_loader, device, t_scaler)
    print(f"[Hybrid] Test zMSE={test_zmse:.6f}")
    return model, (ckpt["t_scaler_mean"], ckpt["t_scaler_std"])
