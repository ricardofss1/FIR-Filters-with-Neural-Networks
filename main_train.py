import argparse

from hybrid_train import train_param_regression


def main():
    parser = argparse.ArgumentParser(
        description="Train the hybrid ParamNet model for lowpass and highpass FIR design."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="fir_dataset_adjusted_firwin_v2.npz",
        help="Path to the dataset .npz file (specs, coefs, orders).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="checkpoints_hybrid",
        help="Directory used to store checkpoints.",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate.")
    parser.add_argument("--val_split", type=float, default=0.15, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--hidden",
        type=int,
        nargs="+",
        default=[256, 256, 128],
        help="Hidden layer sizes.",
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Training device: cpu or cuda (auto-detected by default).",
    )
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
        num_workers=args.num_workers,
        device=args.device,
    )


if __name__ == "__main__":
    main()
