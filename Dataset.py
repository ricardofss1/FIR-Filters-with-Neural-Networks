import os
import numpy as np  
from scipy import signal 

def sample_spec(): 
    """Gera especificações válidas para filtros lowpass ou highpass (normalizadas, fs=1)."""
    fcut = np.random.uniform(0.02, 0.45)  # cutoff 
    trans = 10**np.random.uniform(np.log10(0.005), np.log10(0.12))  # largura da transição (log-uniforme)
    Rp = np.random.uniform(0.01, 1.0)  # ripple dB 
    As = 10**np.random.uniform(np.log10(30), np.log10(100))  # stopband dB 
    order = np.random.randint(8, 129)  # número de taps (L)
    ftype = np.random.choice(["lowpass", "highpass"])
    
    return {'type': ftype, 'fc': fcut, 'trans': trans, 'Rp': Rp, 'As': As, 'order': order}

def is_valid_spec(spec, Nmax=256):
    """Aplica restrições básicas para specs ruins."""
    fc, trans, Rp, As, order = spec['fc'], spec['trans'], spec['Rp'], spec['As'], spec['order']
    
    if fc + trans >= 0.5: 
        return False
    if trans < 0.005: 
        return False
    if order > Nmax: 
        return False
    if As < 20: 
        return False
    if Rp > 2.0: 
        return False
    return True

def design_fir(spec, method='remez'):
    L = spec['order']  # número de taps
    fc = spec['fc']
    trans = spec['trans']
    ftype = spec['type']

    try:
        if method == 'remez':
            if ftype == "lowpass":
                bands = [0, fc, fc+trans, 0.5]
                desired = [1, 0]
            else:  # highpass
                bands = [0, fc-trans, fc, 0.5]
                desired = [0, 1]
            h = signal.remez(L, bands, desired, fs=1.0)
        else:
            pass_zero = True if ftype == "lowpass" else False
            h = signal.firwin(L, fc, window='hamming', fs=1.0, pass_zero=pass_zero)
    except Exception:
        # fallback robusto
        pass_zero = True if ftype == "lowpass" else False
        h = signal.firwin(L, fc, window='hamming', fs=1.0, pass_zero=pass_zero)
    return h

def generate_dataset(n_samples=50000, Nmax=256, out_fname='fir_dataset.npz'):
    specs = []
    coefs = np.zeros((n_samples, Nmax), dtype=np.float32)
    orders = np.zeros(n_samples, dtype=np.int32)

    i = 0
    attempts = 0
    while i < n_samples:
        attempts += 1
        s = sample_spec()
        if not is_valid_spec(s, Nmax=Nmax):
            continue
        
        try:
            h = design_fir(s, method='remez')
        except Exception:
            continue
        
        L = len(h)
        if L > Nmax:
            continue
        
        coefs[i, :L] = h
        # salvar specs: [fc, trans, Rp, As, order, type]
        type_flag = 0 if s['type'] == "lowpass" else 1
        specs.append([s['fc'], s['trans'], s['Rp'], s['As'], L, type_flag])
        orders[i] = L
        i += 1

    specs = np.array(specs, dtype=np.float32)
    full_path = os.path.abspath(out_fname)
    np.savez_compressed(out_fname, specs=specs, coefs=coefs, orders=orders)
    print(f"Salvo em: {full_path}")
    print(f"Taxa de descarte: {(attempts-n_samples)/attempts:.2%}")

if __name__ == '__main__':
    generate_dataset(n_samples=50000, Nmax=256, out_fname='fir_dataset.npz')
