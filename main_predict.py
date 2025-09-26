# main_predict.py (versão com suporte a lowpass/highpass)
import argparse
import numpy as np
import matplotlib.pyplot as plt
from hybrid_predict import load_paramnet, predict_fir_from_specs
from scipy import signal
import time

def compute_beta(As):
    """Calcula parâmetro beta para janela de Kaiser"""
    if As > 50:
        return 0.1102 * (As - 8.7)
    elif As > 21:
        return 0.5842 * (As - 21)**0.4 + 0.07886 * (As - 21)
    else:
        return 0.0

def design_filter(fc, trans, Rp, As, order, ftype="lowpass", method="firwin"):
    """Projeto de filtro FIR usando métodos clássicos (lowpass/highpass)."""
    try:
        if method == "firwin":
            # Ajuste: highpass precisa de número ímpar de taps
            if ftype == "highpass" and order % 2 == 0:
                order += 1  # força ímpar

            taps = signal.firwin(
                order,
                cutoff=fc,
                window=("kaiser", compute_beta(As)),
                fs=1.0,
                pass_zero=(ftype == "lowpass")
            )

        elif method == "remez":
            if ftype == "lowpass":
                bands   = [0, fc, fc + trans, 0.5]
                desired = [1, 0]
            else:  # highpass
                bands   = [0, fc - trans, fc, 0.5]
                desired = [0, 1]

            delta_p = 10**(-Rp/20)
            delta_s = 10**(-As/20)
            weights = [1/delta_p, 1/delta_s]

            taps = signal.remez(order, bands, desired, weight=weights, fs=1.0)

        return taps

    except Exception as e:
        print(f"Erro no design do filtro: {e}")
        return None

def compute_metrics(h_pred, h_design, fc, trans, n_fft=2048):
    """Calcula métricas quantitativas de similaridade"""
    min_len = min(len(h_pred), len(h_design))
    h_pred = h_pred[:min_len]
    h_design = h_design[:min_len]
    
    w, H_pred = signal.freqz(h_pred, worN=n_fft, fs=1.0)
    _, H_des  = signal.freqz(h_design, worN=n_fft, fs=1.0)

    H_pred_db = 20 * np.log10(np.maximum(np.abs(H_pred), 1e-10))
    H_des_db  = 20 * np.log10(np.maximum(np.abs(H_des), 1e-10))
    
    passband_mask = w <= fc
    stopband_mask = w >= (fc + trans)
    
    mae_passband = np.mean(np.abs(H_pred_db[passband_mask] - H_des_db[passband_mask]))
    mae_stopband = np.mean(np.abs(H_pred_db[stopband_mask] - H_des_db[stopband_mask]))
    
    corr = np.corrcoef(h_pred, h_design)[0, 1]
    erle = 10 * np.log10(np.sum(h_design**2) / np.sum((h_pred - h_design)**2 + 1e-10))

    return {
        'mae_passband': mae_passband,
        'mae_stopband': mae_stopband,
        'correlation': corr,
        'erle': erle,
    }

def plot_comprehensive_comparison(h_pred, h_design, specs, n_fft=2048):
    """Plota comparação abrangente entre filtros"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    w, H_pred = signal.freqz(h_pred, worN=n_fft, fs=1.0)
    _, H_des  = signal.freqz(h_design, worN=n_fft, fs=1.0)
    
    H_pred_db = 20 * np.log10(np.maximum(np.abs(H_pred), 1e-10))
    H_des_db  = 20 * np.log10(np.maximum(np.abs(H_des), 1e-10))
    
    axes[0,0].plot(w, H_des_db, label="Scipy (design direto)", linewidth=2)
    axes[0,0].plot(w, H_pred_db, "--", label="Predição híbrida", alpha=0.8)
    axes[0,0].set_title("Resposta em Magnitude")
    axes[0,0].set_xlabel("Frequência Normalizada")
    axes[0,0].set_ylabel("Magnitude (dB)")
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    phase_pred = np.unwrap(np.angle(H_pred))
    phase_des  = np.unwrap(np.angle(H_des))
    axes[0,1].plot(w, phase_des, "--", label="Scipy (design direto)")
    axes[0,1].plot(w, phase_pred, label="Predição híbrida")
    axes[0,1].set_title("Resposta em Fase")
    axes[0,1].set_xlabel("Frequência Normalizada")
    axes[0,1].set_ylabel("Fase (radianos)")
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    axes[1,0].stem(h_design, linefmt="C0-", markerfmt="C0o", basefmt="k-", label="Scipy")
    axes[1,0].stem(h_pred, linefmt="C1--", markerfmt="C1x", basefmt="k-", label="Predição")
    axes[1,0].set_title("Coeficientes FIR")
    axes[1,0].set_xlabel("Índice do Coeficiente")
    axes[1,0].set_ylabel("Amplitude")
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    error_db = H_pred_db - H_des_db
    axes[1,1].plot(w, error_db, 'r-', label="Erro")
    axes[1,1].axhline(0, color='k', linestyle='--', alpha=0.5)
    axes[1,1].set_title("Erro de Magnitude")
    axes[1,1].set_xlabel("Frequência Normalizada")
    axes[1,1].set_ylabel("Erro (dB)")
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].legend()
    
    plt.suptitle(f"Comparação FIR {specs['type'].upper()} - fc={specs['fc']}, "
                 f"trans={specs['trans']}, Rp={specs['Rp']} dB, As={specs['As']} dB, ordem={specs['order']}")
    plt.tight_layout()
    plt.show()

def export_coefficients(h_coeffs, filename, format_type='txt'):
    """Exporta coeficientes FIR para uso real"""
    if format_type == 'txt':
        with open(filename, 'w') as f:
            coeffs_str = [f"{coeff:.8f}" for coeff in h_coeffs]
            f.write(','.join(coeffs_str))
    elif format_type == 'python':
        np.savetxt(filename, h_coeffs, fmt='%.8f', delimiter=',')
    elif format_type == 'c':
        with open(filename, 'w') as f:
            f.write("const float fir_coefficients[] = {\n")
            for i, coeff in enumerate(h_coeffs):
                f.write(f"    {coeff:.8f}f")
                if i < len(h_coeffs) - 1:
                    f.write(",\n")
            f.write("\n};\n")
    elif format_type == 'matlab':
        with open(filename, 'w') as f:
            f.write("fir_coeffs = [\n")
            for coeff in h_coeffs:
                f.write(f"    {coeff:.8f};\n")
            f.write("];\n")
    print(f"Coeficientes exportados para {filename} ({format_type})")

def main():
    parser = argparse.ArgumentParser(description="Inferência híbrida de filtros FIR")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_hybrid/best_paramnet.pth")
    parser.add_argument("--fc", type=float, default=0.25, help="Frequência de corte normalizada")
    parser.add_argument("--trans", type=float, default=0.05, help="Largura da transição")
    parser.add_argument("--Rp", type=float, default=1.0, help="Ripple em dB")
    parser.add_argument("--As", type=float, default=60.0, help="Atenuação em dB")
    parser.add_argument("--order", type=int, default=128, help="Número de taps")
    parser.add_argument("--type", type=str, choices=["lowpass", "highpass"], default="highpass",
                        help="Tipo de filtro")
    parser.add_argument("--method", type=str, choices=["remez", "firwin"], default="firwin")
    parser.add_argument("--export", type=str, choices=['txt', 'python', 'c', 'matlab'])
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    
    start_time = time.time()
    model, t_scaler, device = load_paramnet(args.checkpoint)
    
    # Monta spec para a rede (inclui type codificado em int: 0=lowpass, 1=highpass)
    type_val = 0 if args.type == "lowpass" else 1
    spec = np.array([[args.fc, args.trans, args.Rp, args.As, args.order, type_val]], dtype=np.float32)
    
    coefs_list, params = predict_fir_from_specs(spec, model, t_scaler, device=device, method=args.method)
    h_pred = coefs_list[0]
    
    h_design = design_filter(args.fc, args.trans, args.Rp, args.As, args.order, ftype=args.type, method=args.method)
    if h_design is None:
        print("Erro no design do filtro de referência")
        return
    
    metrics = compute_metrics(h_pred, h_design, args.fc, args.trans)
    inference_time = time.time() - start_time
    
    print("="*50)
    print(f"INFERÊNCIA HÍBRIDA FIR ({args.type.upper()})")
    print("="*50)
    print(f"fc={args.fc}, trans={args.trans}, Rp={args.Rp} dB, As={args.As} dB, ordem={args.order}, método={args.method}")
    
    print("\nParâmetros Previstos:")
    for k, v in params.items():
        print(f"  {k}: {v[0]:.6f}")

    print(f"{len(h_design)}")

    print("\nMétricas:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}")
    print(f"Tempo de inferência: {inference_time:.3f} s")
    
    if args.export:
        filename = f"fir_{args.type}_fc{args.fc}_ord{args.order}.{args.export}"
        export_coefficients(h_pred, filename, args.export)
    
    if not args.no_plot:
        specs = {'fc': args.fc, 'trans': args.trans, 'Rp': args.Rp, 'As': args.As, 'order': args.order, 'type': args.type}
        plot_comprehensive_comparison(h_pred, h_design, specs)

if __name__ == "__main__":
    main()
