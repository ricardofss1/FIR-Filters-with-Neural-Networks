# main_predict_real.py
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

def design_filter_real(fc_hz, trans_hz, Rp, As, order, fs, method="firwin"):
    """Projeto de filtro FIR para frequências reais em Hz"""
    try:
        # Converte para frequências normalizadas
        fc_norm = fc_hz / (fs / 2)
        trans_norm = trans_hz / (fs / 2)
        
        if method == "firwin":
            taps = signal.firwin(
                order + 1,
                cutoff=fc_hz,  # Em Hz para firwin com fs especificado
                window=("kaiser", compute_beta(As)),
                fs=fs,
                pass_zero=True
            )
        elif method == "remez":
            bands = [0, fc_hz, fc_hz + trans_hz, fs/2]  # Em Hz
            desired = [1, 0]
            delta_p = 10**(-Rp/20)
            delta_s = 10**(-As/20)
            weights = [1/delta_p, 1/delta_s]
            taps = signal.remez(order + 1, bands, desired, weight=weights, fs=fs)
        return taps
    except Exception as e:
        print(f"Erro no design do filtro: {e}")
        return None

def compute_metrics_real(h_pred, h_design, fs, fc, trans, n_fft=2048):
    """Calcula métricas para filtros com frequências reais"""
    min_len = min(len(h_pred), len(h_design))
    h_pred = h_pred[:min_len]
    h_design = h_design[:min_len]
    
    # Resposta em frequência
    w, H_pred = signal.freqz(h_pred, worN=n_fft, fs=fs)
    _, H_des = signal.freqz(h_design, worN=n_fft, fs=fs)

    # Frequências em Hz para máscaras
    frequencies = w * (fs / (2 * np.pi))  # Convert rad/sample to Hz
    
    H_pred_db = 20 * np.log10(np.maximum(np.abs(H_pred), 1e-10))
    H_des_db = 20 * np.log10(np.maximum(np.abs(H_des), 1e-10))
    
    # Máscaras para regiões (usando frequências em Hz)
    passband_mask = frequencies <= fc
    stopband_mask = frequencies >= (fc + trans)
    transition_mask = (frequencies > fc) & (frequencies < (fc + trans))
    
    mae_passband = np.mean(np.abs(H_pred_db[passband_mask] - H_des_db[passband_mask]))
    mae_stopband = np.mean(np.abs(H_pred_db[stopband_mask] - H_des_db[stopband_mask]))
    
    # Ripple máximo na passband
    ripple_pred = np.max(H_pred_db[passband_mask]) - np.min(H_pred_db[passband_mask])
    ripple_des = np.max(H_des_db[passband_mask]) - np.min(H_des_db[passband_mask])
    
    # Atenuação mínima na stopband
    attenuation_pred = -np.min(H_pred_db[stopband_mask])
    attenuation_des = -np.min(H_des_db[stopband_mask])
    
    corr = np.corrcoef(h_pred, h_design)[0, 1]
    erle = 10 * np.log10(np.sum(h_design**2) / np.sum((h_pred - h_design)**2 + 1e-10))
    
    return {
        'mae_passband': mae_passband,
        'mae_stopband': mae_stopband,
        'correlation': corr,
        'erle': erle,
        'ripple_pred': ripple_pred,
        'ripple_des': ripple_des,
        'attenuation_pred': attenuation_pred,
        'attenuation_des': attenuation_des
    }

def plot_comprehensive_comparison_real(h_pred, h_design, specs, fs, n_fft=2048):
    """Plota comparação para filtros com frequências reais"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Resposta em magnitude
    w, H_pred = signal.freqz(h_pred, worN=n_fft, fs=fs)
    _, H_des = signal.freqz(h_design, worN=n_fft, fs=fs)
    
    frequencies = w  # Já em Hz devido ao parâmetro fs
    
    H_pred_db = 20 * np.log10(np.maximum(np.abs(H_pred), 1e-10))
    H_des_db = 20 * np.log10(np.maximum(np.abs(H_des), 1e-10))
    
    axes[0,0].plot(frequencies, H_des_db, label="Scipy (design direto)", linewidth=2)
    axes[0,0].plot(frequencies, H_pred_db, "--", label="Predição híbrida", alpha=0.8)
    axes[0,0].axvline(specs['fc'], color='r', linestyle=':', alpha=0.7, label=f"Fc = {specs['fc']} Hz")
    axes[0,0].axvline(specs['fc'] + specs['trans'], color='g', linestyle=':', alpha=0.7, label=f"Transição")
    axes[0,0].set_title("Resposta em Magnitude")
    axes[0,0].set_xlabel("Frequência (Hz)")
    axes[0,0].set_ylabel("Magnitude (dB)")
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].set_xlim(0, fs/2)
    
    # Zoom na região de transição
    axes[0,1].plot(frequencies, H_des_db, label="Scipy", linewidth=2)
    axes[0,1].plot(frequencies, H_pred_db, "--", label="Predição", alpha=0.8)
    axes[0,1].axvline(specs['fc'], color='r', linestyle=':', alpha=0.7)
    axes[0,1].axvline(specs['fc'] + specs['trans'], color='g', linestyle=':', alpha=0.7)
    axes[0,1].set_title("Zoom: Região de Transição")
    axes[0,1].set_xlabel("Frequência (Hz)")
    axes[0,1].set_ylabel("Magnitude (dB)")
    axes[0,1].set_xlim(specs['fc'] - specs['trans'], specs['fc'] + 2*specs['trans'])
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].legend()
    
    # Coeficientes FIR
    axes[1,0].stem(h_design, linefmt="C0-", markerfmt="C0o", basefmt="k-", label="Scipy")
    axes[1,0].stem(h_pred, linefmt="C1--", markerfmt="C1x", basefmt="k-", label="Predição")
    axes[1,0].set_title("Coeficientes FIR")
    axes[1,0].set_xlabel("Índice do Coeficiente")
    axes[1,0].set_ylabel("Amplitude")
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Resposta de fase
    phase_pred = np.unwrap(np.angle(H_pred))
    phase_des = np.unwrap(np.angle(H_des))
    axes[1,1].plot(frequencies, phase_pred, label="Predicted")
    axes[1,1].plot(frequencies, phase_des, label="Designed", linestyle="--")
    axes[1,1].set_title("Resposta em Fase")
    axes[1,1].set_xlabel("Frequência (Hz)")
    axes[1,1].set_ylabel("Fase (radianos)")
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].legend()
    
    plt.suptitle(f"Filtro Passa-Baixa: fc={specs['fc']}Hz, fs={fs}Hz, "
                f"Rp={specs['Rp']}dB, As={specs['As']}dB, ordem={specs['order']}")
    plt.tight_layout()
    plt.show()

def export_coefficients(h_coeffs, filename, format_type='python'):
    """Exporta coeficientes para uso em implementações reais"""
    if format_type == 'python':
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
    parser = argparse.ArgumentParser(description="Inferência híbrida de filtros FIR para implementação real")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_hybrid/best_paramnet.pth",
                        help="Checkpoint do modelo treinado")
    parser.add_argument("--fc", type=float, default=700.0, help="Frequência de corte em Hz")
    parser.add_argument("--trans", type=float, default=200.0, help="Largura da transição em Hz")
    parser.add_argument("--Rp", type=float, default=1.0, help="Ripple em dB")
    parser.add_argument("--As", type=float, default=60.0, help="Atenuação em dB")
    parser.add_argument("--order", type=int, default=128, help="Ordem do filtro")
    parser.add_argument("--fs", type=float, default=44100.0, help="Frequência de amostragem em Hz")
    parser.add_argument("--method", type=str, choices=["remez", "firwin"], default="firwin",
                        help="Método de síntese")
    parser.add_argument("--export", type=str, choices=['python', 'c', 'matlab'], 
                        help="Exportar coeficientes no formato especificado")
    parser.add_argument("--no-plot", action="store_true", help="Não mostrar plots")
    
    args = parser.parse_args()
    
    # Verifica se as especificações são realizáveis
    if args.fc + args.trans >= args.fs/2:
        print(f"ERRO: Frequência de corte + transição ({args.fc + args.trans}Hz) "
              f"deve ser menor que Nyquist ({args.fs/2}Hz)")
        return
    
    start_time = time.time()
    
    # Carrega modelo (precisa ser adaptado para frequências reais)
    model, t_scaler, device = load_paramnet(args.checkpoint)
    
    # Converte specs para normalizadas para a rede neural
    fc_norm = args.fc / (args.fs / 2)
    trans_norm = args.trans / (args.fs / 2)
    
    spec = np.array([[fc_norm, trans_norm, args.Rp, args.As, args.order]], dtype=np.float32)
    coefs_list, params = predict_fir_from_specs(spec, model, t_scaler, device=device, method=args.method)
    h_pred = coefs_list[0]
    
    # Design clássico com frequências reais
    h_design = design_filter_real(args.fc, args.trans, args.Rp, args.As, args.order, args.fs, args.method)
    
    if h_design is None:
        return
    
    # Métricas
    metrics = compute_metrics_real(h_pred, h_design, args.fs, args.fc, args.trans)
    inference_time = time.time() - start_time
    
    # Resultados
    print("="*60)
    print("PROJETO DE FILTRO FIR PARA IMPLEMENTAÇÃO REAL")
    print("="*60)
    print(f"\nEspecificações:")
    print(f"  Frequência de amostragem: {args.fs} Hz")
    print(f"  Frequência de corte: {args.fc} Hz")
    print(f"  Largura de transição: {args.trans} Hz")
    print(f"  Ripple máximo: {args.Rp} dB")
    print(f"  Atenuação mínima: {args.As} dB")
    print(f"  Ordem do filtro: {args.order}")
    print(f"  Número de taps: {len(h_pred)}")
    
    print(f"\nPerformance do Filtro Predito:")
    print(f"  Ripple na passband: {metrics['ripple_pred']:.2f} dB")
    print(f"  Atenuação na stopband: {metrics['attenuation_pred']:.1f} dB")
    
    print(f"\nMétricas de Comparação:")
    print(f"  MAE Passband: {metrics['mae_passband']:.3f} dB")
    print(f"  MAE Stopband: {metrics['mae_stopband']:.3f} dB")
    print(f"  Correlação: {metrics['correlation']:.3f}")
    print(f"  Tempo de Inferência: {inference_time:.3f} s")
    
    # Exportação de coeficientes
    if args.export:
        filename = f"fir_coeffs_fc{args.fc}hz_fs{args.fs}hz_order{args.order}.{args.export}"
        export_coefficients(h_pred, filename, args.export)
    
    # Visualização
    if not args.no_plot:
        specs = {
            'fc': args.fc, 'trans': args.trans,
            'Rp': args.Rp, 'As': args.As, 'order': args.order
        }
        plot_comprehensive_comparison_real(h_pred, h_design, specs, args.fs)

if __name__ == "__main__":
    main()