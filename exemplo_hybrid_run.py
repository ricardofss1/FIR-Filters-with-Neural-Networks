import numpy as np

from hybrid_predict import load_paramnet, predict_fir_from_specs

# 1) Carregar o melhor modelo
model, t_scaler, device = load_paramnet("checkpoints_hybrid/best_paramnet.pth")

# 2) Fazer um pedido no mesmo formato do dataset:
#    [fc, trans, Rp(dB), As(dB), order, type]
#    type = 0 para lowpass, 1 para highpass
spec = np.array([[0.15, 0.03, 0.2, 60.0, 128.0, 0.0]], dtype=np.float32)

coefs_list, params = predict_fir_from_specs(spec, model, t_scaler, device=device, method="remez")
h = coefs_list[0]

print("Parametros previstos:")
for key, values in params.items():
    print(f"  {key}: {values[0]}")
print("Numero de taps gerado:", len(h))
