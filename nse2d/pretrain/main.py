"""Entry point for 5-field MHD OmniFluids training and inference."""

import sys
import os
import hashlib
import argparse
from datetime import datetime

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

DATA_ROOT = '/zhangtao/project2026/OmniFluids/nse2d/data/qruio_data'
DEFAULT_DATA = os.path.join(DATA_ROOT, '5field_mhd_batch/data/5field_mhd_dataset.pt')
DEFAULT_EVAL_DATA = os.path.join(DATA_ROOT, '5field_mhd_batch_test/data/5field_mhd_dataset.pt')
DEFAULT_GRF_TEST = '/zhangtao/project2026/OmniFluids/nse2d/data/grf_testset/grf_testset_B10_T50_dt1.0_fromdata_radial_dealiased_seed1000.pt'


class TeeOutput:
    """Write to both file and terminal simultaneously."""
    def __init__(self, filepath_or_file, terminal=None, mode='w'):
        if isinstance(filepath_or_file, str):
            self.file = open(filepath_or_file, mode)
            self._owns_file = True
        else:
            self.file = filepath_or_file
            self._owns_file = False
        self.terminal = terminal if terminal is not None else sys.stdout

    def write(self, msg):
        self.terminal.write(msg)
        self.file.write(msg)

    def flush(self):
        self.terminal.flush()
        self.file.flush()
    
    def close(self):
        if self._owns_file:
            self.file.close()


def generate_run_hash(cfg):
    """Generate 8-char hash from key hyperparameters + timestamp."""
    key_params = (
        cfg.seed, cfg.K, cfg.modes_x, cfg.modes_y, cfg.width,
        cfg.n_layers, cfg.output_dim, cfg.rollout_dt, cfg.lr,
        cfg.batch_size, cfg.time_integrator, datetime.now().isoformat()
    )
    return hashlib.md5(str(key_params).encode()).hexdigest()[:8]


def make_run_tag(cfg):
    """Build run tag: {hash}-{date}-K{K}-mx{modes_x}-w{width}-L{layers}-od{output_dim}"""
    cfg.run_hash = generate_run_hash(cfg)
    ts = datetime.now().strftime('%m_%d_%H_%M_%S')
    cfg.run_tag = (f'{cfg.run_hash}-{ts}'
                   f'-K{cfg.K}-mx{cfg.modes_x}-w{cfg.width}'
                   f'-L{cfg.n_layers}-od{cfg.output_dim}')
    return cfg.run_tag


def make_run_dir(cfg):
    """Create per-run directory tree under results/{exp_name}/{run_tag}/."""
    base = os.path.join('results', cfg.exp_name, cfg.run_tag)
    cfg.run_dir = base
    for sub in ['log', 'model', 'vis', 'inference']:
        os.makedirs(os.path.join(base, sub), exist_ok=True)
    return base


def run_train(cfg):
    """Build model and run training."""
    from model import OmniFluids2D
    from train import train, HAS_ACCELERATE
    from tools import setup_seed, param_count

    # Check if using multi-GPU via Accelerate
    use_accelerate = getattr(cfg, 'use_accelerate', False) and HAS_ACCELERATE
    accelerator = None
    is_main_process = True
    
    if use_accelerate:
        from accelerate import Accelerator
        accelerator = Accelerator()
        is_main_process = accelerator.is_main_process
        # Different seed per process for GRF diversity
        cfg.seed = cfg.seed + accelerator.process_index
    
    setup_seed(cfg.seed)
    
    # Only main process handles logging and dir creation
    if is_main_process:
        run_tag = make_run_tag(cfg)
        make_run_dir(cfg)

        logfile = os.path.join(cfg.run_dir, 'log', f'log-{run_tag}.txt')
        # Redirect both stdout and stderr to the same log file
        log_handle = open(logfile, 'w')
        sys.stdout = TeeOutput(log_handle, terminal=sys.stdout)
        sys.stderr = TeeOutput(log_handle, terminal=sys.stderr)

        print('=' * 60)
        print(f'5-field MHD OmniFluids Training  [{run_tag}]')
        print('=' * 60)
        for k, v in sorted(vars(cfg).items()):
            print(f'  {k}: {v}')
        print('=' * 60)
        sys.stdout.flush()
    else:
        # Non-main processes only need run_tag and run_dir path, NOT create dirs
        run_tag = make_run_tag(cfg)
        cfg.run_dir = os.path.join('results', cfg.exp_name, run_tag)
    
    # Sync before model creation
    if accelerator is not None:
        accelerator.wait_for_everyone()

    net = OmniFluids2D(
        Nx=cfg.Nx, Ny=cfg.Ny, K=cfg.K, T=cfg.temperature,
        modes_x=cfg.modes_x, modes_y=cfg.modes_y,
        width=cfg.width, output_dim=cfg.output_dim,
        n_fields=5, n_params=cfg.n_params,
        n_layers=cfg.n_layers, factor=cfg.factor,
        n_ff_layers=cfg.n_ff_layers, layer_norm=cfg.layer_norm)
    
    if is_main_process:
        param_count(net)
        sys.stdout.flush()

    # Pass accelerator to train function
    train(cfg, net, accelerator=accelerator)

    if is_main_process:
        print(f'\nFinished at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        sys.stdout.flush()


def run_inference(cfg):
    """Load checkpoint and run evaluation."""
    import torch
    import json
    from model import OmniFluids2D
    from train import evaluate
    from psm_loss import build_mhd_instance

    assert cfg.checkpoint is not None, 'Must provide --checkpoint for inference'

    ckpt = torch.load(cfg.checkpoint, map_location=cfg.device, weights_only=False)
    saved_cfg = ckpt.get('config', {})
    
    # Check for architecture parameter conflicts between checkpoint and CLI
    # Note: 'temperature' maps to 'T' parameter in OmniFluids2D constructor
    arch_params = ['Nx', 'Ny', 'K', 'temperature', 'modes_x', 'modes_y', 'width', 'output_dim',
                   'n_params', 'n_layers', 'factor', 'n_ff_layers', 'layer_norm']
    conflicts = []
    missing_in_ckpt = []
    for param in arch_params:
        saved_val = saved_cfg.get(param)
        cli_val = getattr(cfg, param, None)
        if saved_val is None and cli_val is not None:
            missing_in_ckpt.append(f'  {param}: using CLI default={cli_val}')
        elif saved_val is not None and cli_val is not None and saved_val != cli_val:
            conflicts.append(f'  {param}: checkpoint={saved_val}, CLI={cli_val}')
    
    if conflicts or missing_in_ckpt:
        print('=' * 60)
        if conflicts:
            print('WARNING: Architecture parameter conflicts detected!')
            print('Using checkpoint values (ignoring CLI):')
            for c in conflicts:
                print(c)
        if missing_in_ckpt:
            print('NOTE: Some parameters missing from checkpoint (using CLI defaults):')
            for m in missing_in_ckpt:
                print(m)
        print('=' * 60)

    # Restore ALL architecture parameters from checkpoint to ensure model structure matches
    net = OmniFluids2D(
        Nx=saved_cfg.get('Nx', cfg.Nx),
        Ny=saved_cfg.get('Ny', cfg.Ny),
        K=saved_cfg.get('K', cfg.K),
        T=saved_cfg.get('temperature', cfg.temperature),
        modes_x=saved_cfg.get('modes_x', cfg.modes_x),
        modes_y=saved_cfg.get('modes_y', cfg.modes_y),
        width=saved_cfg.get('width', cfg.width),
        output_dim=saved_cfg.get('output_dim', cfg.output_dim),
        n_fields=5,
        n_params=saved_cfg.get('n_params', cfg.n_params),
        n_layers=saved_cfg.get('n_layers', cfg.n_layers),
        factor=saved_cfg.get('factor', cfg.factor),
        n_ff_layers=saved_cfg.get('n_ff_layers', cfg.n_ff_layers),
        layer_norm=saved_cfg.get('layer_norm', cfg.layer_norm))
    net.load_state_dict(ckpt['model_state_dict'])
    net = net.to(cfg.device)

    Nx = saved_cfg.get('Nx', cfg.Nx)
    Ny = saved_cfg.get('Ny', cfg.Ny)
    mhd = build_mhd_instance(device=cfg.device, Nx=Nx, Ny=Ny)

    # Determine run_dir: if checkpoint is in new layout (.../model/best-xxx.pt),
    # reuse the same per-run folder; otherwise create a sibling inference folder.
    ckpt_model_dir = os.path.dirname(os.path.abspath(cfg.checkpoint))
    ckpt_run_dir = os.path.dirname(ckpt_model_dir)
    if os.path.basename(ckpt_model_dir) == 'model':
        cfg.run_dir = ckpt_run_dir
    else:
        cfg.run_dir = os.path.join('results', cfg.exp_name, 'inference_run')
    for sub in ['inference', 'vis']:
        os.makedirs(os.path.join(cfg.run_dir, sub), exist_ok=True)

    print(f'Loaded checkpoint from {cfg.checkpoint} (step {ckpt.get("step", "?")})')
    print(f'Evaluating on test set: {cfg.eval_data_path}')
    print(f'Saving results to: {cfg.run_dir}/')
    results = evaluate(cfg, net, cfg.eval_data_path, mhd,
                       n_rollout_steps=cfg.eval_rollout_steps,
                       save_plots=True, step_tag='inference')

    out_path = os.path.join(cfg.run_dir, 'inference', 'inference_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Results saved to {out_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='5-field MHD OmniFluids')

    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'inference'])
    parser.add_argument('--checkpoint', type=str, default=None)

    parser.add_argument('--data_path', type=str, default=DEFAULT_DATA,
                        help='Training data path')
    parser.add_argument('--eval_data_path', type=str, default=DEFAULT_EVAL_DATA,
                        help='Evaluation/test data path (separate from training)')
    parser.add_argument('--eval_grf_data_path', type=str, default=DEFAULT_GRF_TEST,
                        help='GRF test set path for evaluation (default: grf_testset.pt)')
    parser.add_argument('--time_start', type=float, default=250.0)
    parser.add_argument('--time_end', type=float, default=300.0)
    parser.add_argument('--dt_data', type=float, default=1.0)

    parser.add_argument('--Nx', type=int, default=512)
    parser.add_argument('--Ny', type=int, default=256)

    parser.add_argument('--modes_x', type=int, default=128)
    parser.add_argument('--modes_y', type=int, default=128)
    parser.add_argument('--width', type=int, default=80)
    parser.add_argument('--n_layers', type=int, default=12)
    parser.add_argument('--K', type=int, default=4)
    parser.add_argument('--temperature', type=float, default=10.0)
    parser.add_argument('--output_dim', type=int, default=10)
    parser.add_argument('--n_params', type=int, default=8)
    parser.add_argument('--factor', type=int, default=4)
    parser.add_argument('--n_ff_layers', type=int, default=2)
    parser.add_argument('--layer_norm', action='store_true', default=True)

    parser.add_argument('--rollout_dt', type=float, default=0.1,
                        help='Model inference dt (= mhd_sim delta_t)')
    parser.add_argument('--time_integrator', type=str, default='crank_nicolson',
                        choices=['euler', 'crank_nicolson'])
    parser.add_argument('--dealias_input', type=int, default=1,
                        help='Dealias input before model forward (default=1, prevents aliasing)')
    parser.add_argument('--dealias_rhs', type=int, default=0,
                        help='Dealias inside RHS function (default=0, redundant if dealias_input=1)')
    parser.add_argument('--input_noise_scale', type=float, default=0.001,
                        help='Scale of additive Gaussian noise on training input')
    parser.add_argument('--mae_weight', type=float, default=0.0,
                        help='Weight for MAE loss term (0=off, e.g. 0.1)')
    parser.add_argument('--supervised_loss_weight', type=float, default=0.0,
                        help='Weight for supervised loss on real data (0=off, e.g. 1.0)')
    parser.add_argument('--supervised_mse_weight', type=float, default=1.0,
                        help='MSE weight within supervised loss (default=1.0)')
    parser.add_argument('--supervised_mae_weight', type=float, default=0.0,
                        help='MAE weight within supervised loss (default=0.0)')
    parser.add_argument('--supervised_n_substeps', type=int, default=1,
                        help='Number of autoregressive model calls in supervised training. '
                             'Set to round(dt_data/rollout_dt) to match evaluation (e.g., 10). '
                             'Default=1 means model predicts rollout_dt, target is dt_data apart.')
    parser.add_argument('--supervised_use_interpolation', type=int, default=0,
                        help='1=for supervised_n_substeps>1, supervise each substep against '
                             'a linearly interpolated target between x_t and x_{t+1}')
    parser.add_argument('--supervised_pair_interp_steps', type=int, default=1,
                        help='If >1, sample one interpolated one-step pseudo-pair from each '
                             'real (x_t, x_{t+1}) pair, split into this many equal segments')
    parser.add_argument('--physics_loss_weight', type=float, default=1.0,
                        help='Weight for PDE physics loss (default=1.0, set 0 to disable)')

    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_iterations', type=int, default=20000)
    parser.add_argument('--max_epochs', type=int, default=10000)
    parser.add_argument('--grad_clip', type=float, default=None)

    parser.add_argument('--data_mode', type=str, default='offline',
                        choices=['offline', 'online', 'staged', 'alternating'],
                        help='Data source: offline=mhd_sim data, online=GRF random, '
                             'staged=online then offline, alternating=cycle both')
    parser.add_argument('--online_warmup_steps', type=int, default=5000,
                        help='Steps of online training before switching (staged mode)')
    parser.add_argument('--alternate_online_steps', type=int, default=500,
                        help='Online steps per cycle (alternating mode)')
    parser.add_argument('--alternate_offline_steps', type=int, default=500,
                        help='Offline steps per cycle (alternating mode)')
    parser.add_argument('--grf_alpha', type=float, default=None,
                        help='GRF spectral decay exponent (None=per-field defaults)')
    parser.add_argument('--grf_tau', type=float, default=None,
                        help='GRF inverse correlation length (None=per-field defaults)')
    parser.add_argument('--grf_scale_from_data', type=int, default=1,
                        help='1=derive GRF field_scales from data stats, 0=use defaults')
    parser.add_argument('--grf_use_radial_mask', type=int, default=1,
                        help='1=use radial mask for GRF (default), 0=disable (full Dirichlet)')

    parser.add_argument('--is_overfitting_test', type=int, default=0,
                        help='1=overfitting test: use single trajectory for train and eval')
    parser.add_argument('--overfitting_traj_idx', type=int, default=0,
                        help='Trajectory index to use for overfitting test (default=0)')
    parser.add_argument('--is_grf_overfitting_test', type=int, default=0,
                        help='1=use fixed GRF random data for overfitting test')
    parser.add_argument('--grf_overfitting_seed', type=int, default=42,
                        help='Seed for generating fixed GRF data (default=42)')

    # Learnable GRF: make alpha/tau trainable via PDE loss gradient
    parser.add_argument('--learnable_grf', type=int, default=0,
                        help='1=enable learnable GRF (alpha/tau are trainable parameters)')
    parser.add_argument('--learnable_grf_start_step', type=int, default=20000,
                        help='Step to start learning GRF parameters (default=20000)')
    parser.add_argument('--learnable_grf_lr_ratio', type=float, default=0.01,
                        help='GRF optimizer lr = model_lr * this ratio (default=0.01)')
    parser.add_argument('--learnable_grf_alpha_min', type=float, default=1.0,
                        help='Minimum clamp value for alpha (default=1.0)')
    parser.add_argument('--learnable_grf_alpha_max', type=float, default=6.0,
                        help='Maximum clamp value for alpha (default=6.0)')
    parser.add_argument('--learnable_grf_tau_min', type=float, default=0.5,
                        help='Minimum clamp value for tau (default=0.5)')
    parser.add_argument('--learnable_grf_tau_max', type=float, default=20.0,
                        help='Maximum clamp value for tau (default=20.0)')
    parser.add_argument('--learnable_grf_reg_weight', type=float, default=0.0,
                        help='L2 regularization weight to prevent drift from initial values (default=0)')
    parser.add_argument('--learnable_grf_accum_steps', type=int, default=1,
                        help='Gradient accumulation steps for GRF optimizer (default=1)')
    parser.add_argument('--learnable_grf_log_every', type=int, default=1000,
                        help='Log GRF parameters every N steps (default=1000)')

    # Self-training mode: use training model itself to generate data
    parser.add_argument('--self_training_start_step', type=int, default=0,
                        help='Step to start using model-evolved data (0=disabled). '
                             'Before this step, uses raw GRF. After, uses model-evolved GRF.')
    parser.add_argument('--self_training_update_every', type=int, default=5000,
                        help='Update data generator model weights every N steps (0=never update). '
                             'Only applies to self-training mode.')
    parser.add_argument('--self_training_rollout_steps', type=int, default=10,
                        help='Number of model inference steps to evolve GRF in self-training mode')

    # External pretrained model mode: use a fixed pretrained model to generate data
    parser.add_argument('--pretrained_model_path', type=str, default=None,
                        help='Path to external pretrained model for data generation. '
                             'If set, uses this fixed model instead of self-training mode. '
                             'The pretrained model will evolve GRF before feeding to training.')
    parser.add_argument('--pretrained_rollout_steps', type=int, default=10,
                        help='Number of inference steps for external pretrained model to evolve GRF')

    # Mixed integrator (Euler/CN) parameters
    parser.add_argument('--use_mixed_integrator', type=int, default=0,
                        help='Enable mixed Euler/CN integrator (0=off, 1=on). '
                             'When enabled, computes both Euler and CN losses and combines them.')
    parser.add_argument('--euler_weight_init', type=float, default=1.0,
                        help='Initial Euler weight (1.0 = pure Euler, 0.0 = pure CN)')
    parser.add_argument('--euler_weight_min', type=float, default=0.0,
                        help='Minimum Euler weight after decay (0.0 = pure CN)')
    parser.add_argument('--euler_anneal_start', type=int, default=0,
                        help='Step to start annealing Euler weight (before this, use euler_weight_init)')
    parser.add_argument('--euler_half_life', type=int, default=10000,
                        help='Half-life for Euler weight exponential decay (steps)')

    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--eval_every', type=int, default=500)
    parser.add_argument('--eval_rollout_steps', type=int, default=10)
    parser.add_argument('--checkpoint_every', type=int, default=5000,
                        help='Save intermediate checkpoint every N steps (0=disabled, only save best/latest)')

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='mhd5_omnifluids_v1')
    parser.add_argument('--use_accelerate', type=int, default=0,
                        help='Use Accelerate for multi-GPU training (0=off, 1=on)')

    cfg = parser.parse_args()

    if cfg.mode == 'train':
        run_train(cfg)
    else:
        run_inference(cfg)
