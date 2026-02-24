# 5-field MHD OmniFluids

OmniFluids architecture (spectral conv + MoE) adapted for 5-field Landau-fluid MHD.

## Quick Start

```bash
# Training
bash run.sh train 0 mhd5_v1

# Inference from checkpoint
bash run.sh inference 0 model/mhd5_v1/best.pt
```

## Architecture

- **SpectralConv2d_MHD**: DST (x, Dirichlet BC) + FFT (y, periodic BC)
- **MoE attention**: physics params → dynamic operator mixing
- **5 independent output heads**: each predicts `output_dim` frames per field
- **Residual + linear time interpolation**: ensures initial condition consistency

## Key Files

| File | Description |
|------|-------------|
| `model.py` | OmniFluids2D network definition |
| `psm_loss.py` | Physics loss via mhd_sim's `compute_rhs` |
| `train.py` | Training loop + autoregressive evaluation |
| `main.py` | Entry point with argument parsing |
| `tools.py` | Data loading + utilities |
| `run.sh` | Shell script for train/inference |

## Key Arguments

```
--Nx 512 --Ny 256          Grid resolution (must match data)
--modes_x 128 --modes_y 128  Spectral modes
--width 80                   Hidden channel width
--n_layers 12                Number of spectral conv layers
--K 4                        Number of MoE operators
--output_dim 10              Multi-frame prediction count
--rollout_dt 1.0             Time span per forward pass
--time_integrator crank_nicolson
--batch_size 4               (adjust for GPU memory)
--lr 0.002
--num_iterations 20000
```

## Data

Training data from `mhd_sim` 5-field MHD simulation:
- Path: `data/qruio_data/5field_mhd_batch/data/5field_mhd_dataset.pt`
- Format: dict with keys `n, U, vpar, psi, Ti`, each `(B, T, Nx, Ny)`
- Time window: `time_start=250, time_end=300` (aligned with mhd_sim baseline)

## Output Structure

```
log/log_{exp_name}/          Training logs (CSV)
model/{exp_name}/best.pt     Best checkpoint (by eval rel L2)
model/{exp_name}/latest.pt   Latest checkpoint
results/{exp_name}/           Evaluation results (JSON)
```
