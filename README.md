# FIR Filters with Neural Networks

This project explores a hybrid workflow for FIR filter design:

1. The user provides filter specifications.
2. A neural network predicts adjusted parameters.
3. SciPy synthesizes the final FIR coefficients from those predicted parameters.

That combination keeps the model useful as a learned assistant while still relying on a classical DSP method to produce valid coefficients.

## What is in the repository

- `dataset_generator.py`: generates synthetic lowpass and highpass FIR requests, optional adjusted targets, and coefficients.
- `hybrid_dataset.py`: loads the `.npz` dataset for PyTorch.
- `hybrid_model.py`: defines `ParamNet`.
- `hybrid_train.py`: training loop and checkpoint saving.
- `hybrid_predict.py`: loads the model and performs hybrid inference.
- `hybrid_helpers.py`: parameter encoding/decoding and FIR synthesis glue.
- `fir_utils.py`: shared FIR design, metrics, export helpers, and Hz normalization.
- `main_train.py`: command-line training entry point.
- `main_predict.py`: normalized-frequency inference and comparison.
- `main_predict_.py`: inference in Hz, with optional interactive mode.
- `predict_real_fir.py`: real-world oriented inference with ripple/attenuation reporting.
- `evaluate_model.py`: consolidated checkpoint evaluation on the full test split with JSON report output.
- `streamlit_app.py`: accessible web interface for designing, comparing, and exporting FIR filters.
- `exemplo_hybrid_run.py`: minimal example using the current 6-feature request format.

## Reproducible setup

Do not rely on the committed `myenv/` directory. It points to a machine-specific Python path and is not portable.

Create a fresh environment instead:

```bash
python -m venv .venv
```

Activate it on PowerShell:

```bash
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Dataset generation

Generate a broad-coverage dataset with the same request format expected by the current model:

```bash
python dataset_generator.py --n-samples 50000 --nmax 256 --out fir_dataset_adjusted_firwin_v2.npz --method firwin --seed 0 --search-candidates 8 --profile broad
```

Generate a hard-focused companion dataset that emphasizes narrow transitions, high attenuation, tighter order budgets, and edge-frequency cases:

```bash
python dataset_generator.py --n-samples 15000 --nmax 256 --out fir_dataset_hard_focused_firwin_v1.npz --method firwin --seed 1 --search-candidates 24 --profile hard
```

Useful options:

- `--min-order` / `--max-order`: control the tap range sampled into the dataset.
- `--min-fc` / `--max-fc`: control the normalized cutoff range.
- `--min-trans` / `--max-trans`: control the normalized transition-width range.
- `--method`: choose the primary synthesis method used during dataset generation.
- `--search-candidates`: search for adjusted target parameters that better satisfy the requested specs.
- `--profile`: choose between `broad` coverage and the `hard` / `hard-focused` profile.
- `--seed`: make the sampled dataset reproducible.

The repository now includes `fir_dataset_adjusted_firwin_v2.npz`, which is the current recommended training set:

- it uses `firwin`, matching the default inference path
- it contains large-scale adjusted targets generated with `--search-candidates`
- it includes extra low-frequency coverage for real-frequency cases such as `700 Hz @ 44.1 kHz`
- the generator now performs a two-phase search: broad exploration followed by local refinement
- the generator also stores per-sample difficulty weights for the training loop

## Training

Train `ParamNet` from a dataset file:

```bash
python main_train.py --dataset fir_dataset_adjusted_firwin_v2.npz --epochs 50 --batch_size 128 --num_workers 0
```

Difficulty weighting is applied automatically when the dataset contains the `difficulty` field.

The best checkpoint is saved to:

```bash
checkpoints_hybrid/best_paramnet.pth
```

The checkpoint now stores input/target scalers and model architecture metadata needed to reload it consistently.

## Inference

### Normalized frequency mode

Use normalized frequencies in the same convention as the dataset (`fs = 1.0`, Nyquist at `0.5`):

```bash
python main_predict.py --fc 0.25 --trans 0.05 --Rp 1 --As 60 --order 128 --type lowpass --method firwin
```

### Hz mode

Use real frequencies directly:

```bash
python main_predict_.py --fs 16000 --fc 700 --trans 200 --Rp 1 --As 40 --order 128 --type highpass
```

Interactive mode:

```bash
python main_predict_.py --interactive
```

### Real-world reporting mode

This script reports both comparison metrics and band metrics such as ripple and stopband attenuation:

```bash
python predict_real_fir.py --fs 44100 --fc 700 --trans 200 --Rp 1 --As 60 --order 128 --type lowpass
```

## Streamlit interface

To open the accessible web interface locally:

```bash
python -m streamlit run streamlit_app.py
```

The app includes:

- presets for common use cases
- input in Hz instead of normalized frequency
- interactive Plotly charts with zoom, pan, hover, and clickable legends
- comparison between predicted and direct SciPy design
- tabs for magnitude, phase, group delay, impulse response, and specification error
- visualization controls for scale, axis limits, visible curves, and frequency resolution
- figure export in `HTML`, `PNG`, and `SVG`
- CSV export for frequency response, group delay, impulse response, and metrics
- coefficient export in `txt`, `Python`, `C`, and `MATLAB`
- simple explanations of ripple, attenuation, and the internal score

## Full evaluation

To evaluate a checkpoint on the entire test split and generate a consolidated report:

```bash
python evaluate_model.py --json-out evaluation_report.json
```

The script reports:

- regression loss and parameter MAE on the test split
- similarity versus the dataset target filter
- similarity versus the direct design from the request
- ripple/attenuation statistics for predicted, target, and direct filters
- request-satisfaction score statistics and improvement over direct design

## Metrics

The comparison scripts now align their masks with the filter type:

- `lowpass`: passband up to `fc`, stopband after `fc + trans`
- `highpass`: stopband before `fc - trans`, passband from `fc`

Reported metrics include:

- `mae_passband`
- `mae_stopband`
- `correlation`
- `erle`

`predict_real_fir.py` also reports ripple and stopband attenuation for the predicted and reference filters.

## Notes

- The model input format is `[fc, trans, Rp, As, order, type]`.
- The model predicts the 5 adjustable parameters `[fc, trans, Rp, As, order]` and preserves `type` from the request.
- Internally, `order` now uses a dedicated head while the other continuous parameters share a separate regression head.
- `type = 0` means `lowpass`, `type = 1` means `highpass`.
- `fir_dataset_adjusted_firwin_v2.npz` extends the supported normalized range down to `fc=0.005` and `trans=0.002`.
- Highpass designs created with `firwin` may use one extra tap when an odd length is required by the underlying FIR type.
