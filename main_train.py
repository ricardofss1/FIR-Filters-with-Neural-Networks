# main_train.py
import argparse
from hybrid_train import train_param_regression

def main():
    parser = argparse.ArgumentParser(description="Treinar rede híbrida ParamNet para FIR (lowpass e highpass)")
    parser.add_argument("--dataset", type=str, default="fir_dataset.npz",
                        help="Caminho do dataset .npz (contendo specs, coefs, orders)")
    parser.add_argument("--out_dir", type=str, default="checkpoints_hybrid",
                        help="Diretório para salvar checkpoints")
    parser.add_argument("--epochs", type=int, default=50, help="Número de épocas")
    parser.add_argument("--batch_size", type=int, default=128, help="Tamanho do batch")
    parser.add_argument("--lr", type=float, default=2e-3, help="Taxa de aprendizado")
    parser.add_argument("--val_split", type=float, default=0.15, help="Proporção de validação")
    parser.add_argument("--seed", type=int, default=0, help="Semente de aleatoriedade")
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 256, 128],
                        help="Tamanhos das camadas escondidas")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")
    parser.add_argument("--device", type=str, default=None,
                        help="Dispositivo para treino: 'cpu' ou 'cuda' (auto por padrão)")

    args = parser.parse_args()

    train_param_regression(
        npz_path=args.dataset,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        val_split=args.val_split,
        seed=args.seed,
        hidden=tuple(args.hidden),
        dropout=args.dropout,
        device=args.device
    )

if __name__ == "__main__":
    main()
