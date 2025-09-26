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

def design_filter(fc_hz, trans_hz, Rp, As, order, fs, ftype="lowpass", method="firwin"):
    """Projeto de filtro FIR usando métodos clássicos em Hz"""
    try:
        if method == "firwin":
            # 🔧 Ajuste: highpass precisa de número ímpar
            if ftype == "highpass" and order % 2 == 0:
                order += 1
            taps = signal.firwin(
                order,
                cutoff=fc_hz,
                window=("kaiser", compute_beta(As)),
                fs=fs,
                pass_zero=(ftype == "lowpass")
            )
        elif method == "remez":
            if ftype == "lowpass":
                bands   = [0, fc_hz, fc_hz + trans_hz, fs/2]
                desired = [1, 0]
            else:  # highpass
                bands   = [0, fc_hz - trans_hz, fc_hz, fs/2]
                desired = [0, 1]
            delta_p = 10**(-Rp/20)
            delta_s = 10**(-As/20)
            weights = [1/delta_p, 1/delta_s]
            taps = signal.remez(order, bands, desired, weight=weights, fs=fs)
        return taps
    except Exception as e:
        print(f"Erro no design do filtro: {e}")
        return None

def compute_metrics(h_pred, h_design, fs, fc_hz, trans_hz, n_fft=2048):
    """Calcula métricas quantitativas de similaridade"""
    min_len = min(len(h_pred), len(h_design))
    h_pred = h_pred[:min_len]
    h_design = h_design[:min_len]
    
    w, H_pred = signal.freqz(h_pred, worN=n_fft, fs=fs)
    _, H_des = signal.freqz(h_design, worN=n_fft, fs=fs)

    H_pred_db = 20 * np.log10(np.maximum(np.abs(H_pred), 1e-10))
    H_des_db = 20 * np.log10(np.maximum(np.abs(H_des), 1e-10))

    passband_mask = w <= fc_hz
    stopband_mask = w >= (fc_hz + trans_hz)

    mae_passband = np.mean(np.abs(H_pred_db[passband_mask] - H_des_db[passband_mask]))
    mae_stopband = np.mean(np.abs(H_pred_db[stopband_mask] - H_des_db[stopband_mask]))

    corr = np.corrcoef(h_pred, h_design)[0, 1]
    erle = 10 * np.log10(np.sum(h_design**2) / np.sum((h_pred - h_design)**2 + 1e-10))

    return {
        'mae_passband': mae_passband,
        'mae_stopband': mae_stopband,
        'correlation': corr,
        'erle': erle
    }

def plot_comprehensive_comparison(h_pred, h_design, specs, fs, n_fft=2048):
    """Plota comparação abrangente entre filtros (em Hz)"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    w, H_pred = signal.freqz(h_pred, worN=n_fft, fs=fs)
    _, H_des = signal.freqz(h_design, worN=n_fft, fs=fs)
    
    H_pred_db = 20 * np.log10(np.maximum(np.abs(H_pred), 1e-10))
    H_des_db = 20 * np.log10(np.maximum(np.abs(H_des), 1e-10))
    
    # 1. Magnitude
    axes[0,0].plot(w, H_des_db, label="Scipy (design direto)", linewidth=2)
    axes[0,0].plot(w, H_pred_db, "--", label="Predição híbrida", alpha=0.8)
    axes[0,0].set_title("Resposta em Magnitude")
    axes[0,0].set_xlabel("Frequência (Hz)")
    axes[0,0].set_ylabel("Magnitude (dB)")
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)

    # 2. Fase
    phase_pred = np.unwrap(np.angle(H_pred))
    phase_des = np.unwrap(np.angle(H_des))
    axes[0,1].plot(w, phase_des, "--", label="Scipy")
    axes[0,1].plot(w, phase_pred, label="Predição")
    axes[0,1].set_title("Resposta em Fase")
    axes[0,1].set_xlabel("Frequência (Hz)")
    axes[0,1].set_ylabel("Fase (rad)")
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)

    # 3. Coeficientes
    axes[1,0].stem(h_design, linefmt="C0-", markerfmt="C0o", basefmt="k-", label="Scipy")
    axes[1,0].stem(h_pred, linefmt="C1--", markerfmt="C1x", basefmt="k-", label="Predição")
    axes[1,0].set_title("Coeficientes FIR")
    axes[1,0].set_xlabel("Índice n")
    axes[1,0].set_ylabel("Amplitude")
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)

    # 4. Erro em dB
    error_db = H_pred_db - H_des_db
    axes[1,1].plot(w, error_db, 'r-', label="Erro")
    axes[1,1].axhline(0, color='k', linestyle='--', alpha=0.5)
    axes[1,1].set_title("Erro de Magnitude")
    axes[1,1].set_xlabel("Frequência (Hz)")
    axes[1,1].set_ylabel("Erro (dB)")
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].legend()

    plt.suptitle(f"Comparação FIR (Hz): fc={specs['fc']} Hz, trans={specs['trans']} Hz, "
                 f"Rp={specs['Rp']} dB, As={specs['As']} dB, ordem={specs['order']}")
    plt.tight_layout()
    plt.show()

def export_coefficients(h_coeffs, filename, format_type='txt'):
    """Exporta coeficientes para uso em implementações reais"""
    if format_type == 'txt':
        # Formato lista: 0.0,0.0,0.1,...,0.1
        with open(filename, 'w') as f:
            coeffs_str = [f"{coeff:.8f}" for coeff in h_coeffs]
            f.write(','.join(coeffs_str))
        print(f"Coeficientes exportados para {filename} (formato lista)")
    
    elif format_type == 'python':
        np.savetxt(filename, h_coeffs, fmt='%.8f', delimiter=',')
        print(f"Coeficientes exportados para {filename} (Python array)")
    
    elif format_type == 'c':
        with open(filename, 'w') as f:
            f.write("const float fir_coefficients[] = {\n")
            for i, coeff in enumerate(h_coeffs):
                f.write(f"    {coeff:.8f}f")
                if i < len(h_coeffs) - 1:
                    f.write(",\n")
            f.write("\n};\n")
        print(f"Coeficientes exportados para {filename} (C array)")
    
    elif format_type == 'matlab':
        with open(filename, 'w') as f:
            f.write(f"fir_coeffs = [\n")
            for coeff in h_coeffs:
                f.write(f"    {coeff:.8f};\n")
            f.write("];\n")
        print(f"Coeficientes exportados para {filename} (MATLAB array)")
        
def main():
    """
    parser = argparse.ArgumentParser(description="Inferência híbrida de filtros FIR (em Hz)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_hybrid/best_paramnet.pth")
    parser.add_argument("--fs", type=float, help="Taxa de amostragem (Hz)")
    parser.add_argument("--fc", type=float, help="Frequência de corte (Hz)")
    parser.add_argument("--trans", type=float, help="Largura de transição (Hz)")
    parser.add_argument("--Rp", type=float, help="Ripple (dB)")
    parser.add_argument("--As", type=float, help="Atenuação (dB)")
    parser.add_argument("--order", type=int, help="Número de taps")
    parser.add_argument("--type", type=str, choices=["lowpass", "highpass"], default="lowpass",
                    help="Tipo do filtro")
    parser.add_argument("--method", type=str, choices=["remez", "firwin"], help="Método de síntese")
    parser.add_argument("--export", type=str, choices=['txt', 'python', 'c', 'matlab'])
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--interactive", action="store_true", help="Entrar em modo interativo (solicitar parâmetros)")
    args = parser.parse_args()
    """
    parser = argparse.ArgumentParser(description="Inferência híbrida de filtros FIR para implementação real")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_hybrid/best_paramnet.pth",
                        help="Checkpoint do modelo treinado")
    parser.add_argument("--fc", type=float, default=700.0, help="Frequência de corte em Hz")
    parser.add_argument("--trans", type=float, default=200.0, help="Largura da transição em Hz")
    parser.add_argument("--Rp", type=float, default=1.0, help="Ripple em dB")
    parser.add_argument("--As", type=float, default=40.0, help="Atenuação em dB")
    parser.add_argument("--order", type=int, default=128, help="Ordem do filtro")
    parser.add_argument("--fs", type=float, default=16000.0, help="Frequência de amostragem em Hz")
    parser.add_argument("--type", type=str, choices=["lowpass", "highpass"], default="lowpass",
                    help="Tipo do filtro")
    parser.add_argument("--method", type=str, choices=["remez", "firwin"], default="firwin", help="Método de síntese")
    parser.add_argument("--export", type=str, choices=['txt', 'python', 'c', 'matlab'])
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--interactive", action="store_true", help="Entrar em modo interativo (solicitar parâmetros)")
    args = parser.parse_args()
    """
    # 🔹 Se o usuário escolheu modo interativo
    if not args.interactive:
        print("\n=== MODO INTERATIVO: Projeto de Filtro FIR ===\n")
        ftype = input("Tipo de filtro [lowpass/highpass]: ") or "lowpass"
        args.type = ftype
        args.fs = float(input("Taxa de amostragem (Hz): ") or 8000)
        args.fc = float(input("Frequência de corte (Hz): ") or 700)
        args.trans = float(input("Largura da transição (Hz): ") or 200)
        args.Rp = float(input("Ripple (dB): ") or 1.0)
        args.As = float(input("Atenuação (dB): ") or 60.0)
        args.order = int(input("Número de taps: ") or 128)
        args.method = input("Método [remez/firwin]: ") or "firwin"
        exp = input("Exportar coeficientes? [txt/python/c/matlab/none]: ") or "none"
        args.export = None if exp == "none" else exp
    """
    start_time = time.time()
    model, t_scaler, device = load_paramnet(args.checkpoint)

    # Normalizar especificações para entrada da rede (0..0.5)
    fc_norm = args.fc / args.fs
    trans_norm = args.trans / args.fs
    ftype_val = 0 if args.type == "lowpass" else 1
    spec = np.array([[fc_norm, trans_norm, args.Rp, args.As, args.order, ftype_val]], dtype=np.float32)


    coefs_list, params = predict_fir_from_specs(spec, model, t_scaler, device=device, method=args.method)
    h_pred = coefs_list[0]

    h_design = design_filter(args.fc, args.trans, args.Rp, args.As,
                         args.order, fs=args.fs, method=args.method, ftype=args.type)
    if h_design is None:
        print("Erro no design do filtro de referência")
        return

    metrics = compute_metrics(h_pred, h_design, args.fs, args.fc, args.trans)
    inference_time = time.time() - start_time

    print("="*50)
    print("INFERÊNCIA HÍBRIDA DE FILTROS FIR (Hz)")
    print("="*50)
    print(f"fs={args.fs} Hz, fc={args.fc} Hz, trans={args.trans} Hz")
    print(f"Rp={args.Rp} dB, As={args.As} dB, order={args.order}")
    print(f"\nMétricas:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}")
    print(f"Tempo de inferência: {inference_time:.4f} s")

    # Exportação de coeficientes
    if args.export:
        filename = f"fir_coeffs_fc{args.fc}hz_fs{args.fs}hz_order{args.order}.{args.export}"
        export_coefficients(h_pred, filename, args.export)
    
    # Visualização (CORRIGIDO: adicionado args.fs)
    if not args.no_plot:
        specs = {
            'fc': args.fc, 'trans': args.trans,
            'Rp': args.Rp, 'As': args.As, 'order': args.order
        }
        plot_comprehensive_comparison(h_pred, h_design, specs, args.fs)

if __name__ == "__main__":
    main()