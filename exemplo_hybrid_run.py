# exemplo_hybrid_run.py
import numpy as np
from hybrid_train import train_param_regression
from hybrid_predict import load_paramnet, predict_fir_from_specs

# 1) Treinar a ParamNet
# model, t_scaler = train_param_regression("fir_dataset.npz", epochs=30)

# 2) Carregar o melhor modelo
model, t_scaler, device = load_paramnet("checkpoints_hybrid/best_paramnet.pth")

# 3) Fazer um pedido de lowpass (mesmo formato do dataset)
#    [fc, trans, Rp(dB), As(dB), order]
spec = np.array([[0.15, 0.03, 0.2, 60.0, 128]], dtype=np.float32)

coefs_list, params = predict_fir_from_specs(spec, model, t_scaler, device=device, method="remez")
h = coefs_list[0]
print("Parâmetros previstos (naturais):", {k: float(v[0]) if v.shape==() or len(v)==1 else v[0] for k,v in params.items()})
print("Número de taps gerado:", len(h))
