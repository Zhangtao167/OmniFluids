"""Entry point for 5-field MHD OmniFluids training and inference."""

import sys
import os
import hashlib
import argparse
from datetime import datetime

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

DATA_ROOT = '/zhangtao/project2026/OmniFluids/nse2d/data/qruio_data'
DEFAULT_DATA = os.path.join(DATA_ROOT, '5field_mhd_batch/data/5field_mhd_dataset.pt')


class TeeOutput:
    """Write to both file and terminal simultaneously."""
    def __init__(self, filepath):
        self.file = open(filepath, 'w')
        self.terminal = sys.stdout

    def write(self, msg):
        self.terminal.write(msg)
        self.file.write(msg)

    def flush(self):
        self.terminal.flush()
        self.file.flush()


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


def make_dirs(cfg):
    """Create output directories."""
    for d in [f'log/log_{cfg.exp_name}',
              f'model/{cfg.exp_name}',
              f'results/{cfg.exp_name}']:
        os.makedirs(d, exist_ok=True)


def run_train(cfg):
    """Build model and run training."""
    from model import OmniFluids2D
    from train import train
    from tools import setup_seed, param_count

    setup_seed(cfg.seed)
    run_tag = make_run_tag(cfg)
    make_dirs(cfg)

    logfile = f'log/log_{cfg.exp_name}/log-{run_tag}.csv'
    sys.stdout = TeeOutput(logfile)

    print('=' * 60)
    print(f'5-field MHD OmniFluids Training  [{run_tag}]')
    print('=' * 60)
    for k, v in sorted(vars(cfg).items()):
        print(f'  {k}: {v}')
    print('=' * 60)
    sys.stdout.flush()

    net = OmniFluids2D(
        Nx=cfg.Nx, Ny=cfg.Ny, K=cfg.K, T=cfg.temperature,
        modes_x=cfg.modes_x, modes_y=cfg.modes_y,
        width=cfg.width, output_dim=cfg.output_dim,
        n_fields=5, n_params=cfg.n_params,
        n_layers=cfg.n_layers, factor=cfg.factor,
        n_ff_layers=cfg.n_ff_layers, layer_norm=cfg.layer_norm)
    param_count(net)
    sys.stdout.flush()

    train(cfg, net)

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
        n_layers=saved_cfg.get('n_layers', cfg.n_layers))
    net.load_state_dict(ckpt['model_state_dict'])
    net = net.to(cfg.device)

    Nx = saved_cfg.get('Nx', cfg.Nx)
    Ny = saved_cfg.get('Ny', cfg.Ny)
    mhd = build_mhd_instance(device=cfg.device, Nx=Nx, Ny=Ny)

    print(f'Loaded checkpoint from {cfg.checkpoint} (step {ckpt.get("step", "?")})')
    results = evaluate(cfg, net, cfg.data_path, mhd,
                       n_rollout_steps=cfg.eval_rollout_steps,
                       save_plots=True, step_tag='inference')

    os.makedirs(f'results/{cfg.exp_name}', exist_ok=True)
    out_path = f'results/{cfg.exp_name}/inference_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Results saved to {out_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='5-field MHD OmniFluids')

    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'inference'])
    parser.add_argument('--checkpoint', type=str, default=None)

    parser.add_argument('--data_path', type=str, default=DEFAULT_DATA)
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
    parser.add_argument('--input_noise_scale', type=float, default=0.001,
                        help='Scale of additive Gaussian noise on training input')

    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_iterations', type=int, default=20000)
    parser.add_argument('--max_epochs', type=int, default=10000)
    parser.add_argument('--grad_clip', type=float, default=None)

    parser.add_argument('--log_every', type=int, default=100)
    parser.add_argument('--eval_every', type=int, default=500)
    parser.add_argument('--eval_rollout_steps', type=int, default=10)

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='mhd5_omnifluids_v1')

    cfg = parser.parse_args()

    if cfg.mode == 'train':
        run_train(cfg)
    else:
        run_inference(cfg)
