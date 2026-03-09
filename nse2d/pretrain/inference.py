#!/usr/bin/env python
"""Inference script for 5-field MHD OmniFluids model.

Evaluate a trained model on test datasets and save results.
Results are saved to a subfolder under the checkpoint directory.

Usage:
    # Evaluate on both MHD and GRF test sets (default)
    python inference.py --checkpoint /path/to/best.pt
    
    # Evaluate on specific test set only
    python inference.py --checkpoint /path/to/best.pt --eval_data_path /path/to/data.pt
"""

import os
import sys
import json
import argparse
import hashlib
from datetime import datetime
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import OmniFluids2D
from tools import load_mhd5_trajectories, compute_metrics_and_visualize

FIELD_NAMES = ['n', 'U', 'vpar', 'psi', 'Ti']

# Default test set paths
DEFAULT_MHD_TEST = '/zhangtao/project2026/OmniFluids/nse2d/data/qruio_data/5field_mhd_batch_test/data/5field_mhd_dataset.pt'
DEFAULT_GRF_TEST = '/zhangtao/project2026/OmniFluids/nse2d/data/grf_testset/grf_testset.pt'


def load_model(ckpt_path, device='cuda:0'):
    """Load trained model from checkpoint."""
    print(f'Loading checkpoint: {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_cfg = ckpt.get('config', {})
    
    print(f'  Step: {ckpt.get("step", "?")}')
    print(f'  Best loss: {ckpt.get("best_loss", "?")}')
    
    net = OmniFluids2D(
        Nx=saved_cfg.get('Nx', 512),
        Ny=saved_cfg.get('Ny', 256),
        K=saved_cfg.get('K', 4),
        T=saved_cfg.get('temperature', 10.0),
        modes_x=saved_cfg.get('modes_x', 128),
        modes_y=saved_cfg.get('modes_y', 128),
        width=saved_cfg.get('width', 80),
        output_dim=saved_cfg.get('output_dim', 10),
        n_fields=5,
        n_params=saved_cfg.get('n_params', 8),
        n_layers=saved_cfg.get('n_layers', 12),
        factor=saved_cfg.get('factor', 4),
        n_ff_layers=saved_cfg.get('n_ff_layers', 2),
        layer_norm=saved_cfg.get('layer_norm', True))
    
    net.load_state_dict(ckpt['model_state_dict'])
    net = net.to(device)
    net.eval()
    
    n_params = sum(p.numel() for p in net.parameters()) / 1e6
    print(f'  Model loaded: {n_params:.2f}M params')
    return net, saved_cfg, ckpt


def evaluate_on_testset(net, eval_data_path, n_rollout_steps, device='cuda:0',
                        time_start=250.0, time_end=None, dt_data=1.0,
                        model_dt=0.1, save_dir='./inference_results',
                        sample_idx=0, save_plots=True):
    """Evaluate model on test dataset.
    
    Args:
        net: OmniFluids2D model
        eval_data_path: Path to test dataset (.pt file)
        n_rollout_steps: Number of data steps to rollout (not NFE)
        device: Device to run on
        time_start: Start time for evaluation
        time_end: End time for evaluation (None = auto based on n_rollout_steps)
        dt_data: Time step in test data
        model_dt: Model's time step per NFE
        save_dir: Directory to save results
        sample_idx: Which sample to visualize
        save_plots: Whether to save visualization plots
    
    Returns:
        metrics: dict with evaluation results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Auto-compute time_end if not specified
    if time_end is None:
        time_end = time_start + n_rollout_steps * dt_data
    
    print(f'\n=== Loading Test Data ===')
    print(f'  Path: {eval_data_path}')
    trajectories, meta = load_mhd5_trajectories(
        eval_data_path, time_start=time_start, time_end=time_end,
        dt_data=dt_data, single_trajectory=False)
    
    gt_traj = trajectories.permute(0, 1, 3, 4, 2)  # (B, T, Nx, Ny, 5)
    B, T_gt, Nx, Ny, n_fields = gt_traj.shape
    print(f'  GT trajectory shape: {gt_traj.shape} (B={B}, T={T_gt} frames)')
    
    # Compute alignment parameters
    steps_per_gt_frame = max(1, int(round(dt_data / model_dt)))
    n_gt_steps = min(n_rollout_steps + 1, T_gt)
    total_nfe = (n_gt_steps - 1) * steps_per_gt_frame
    
    print(f'\n=== Model Rollout ===')
    print(f'  Model dt: {model_dt}s, Data dt: {dt_data}s')
    print(f'  Steps per GT frame: {steps_per_gt_frame}')
    print(f'  GT steps: {n_gt_steps}, Total NFE: {total_nfe}')
    
    x_0_batch = gt_traj[:, 0].to(device)
    current = x_0_batch
    aligned_frames = [x_0_batch.cpu()]
    frame_counter = 1
    
    with torch.no_grad():
        for s in range(1, total_nfe + 1):
            out = net(current, inference=True)
            current = out[..., -1]
            if s % steps_per_gt_frame == 0 and frame_counter < n_gt_steps:
                aligned_frames.append(current.cpu())
                frame_counter += 1
            if s % 10 == 0 or s == total_nfe:
                print(f'    NFE {s}/{total_nfe} (collected {frame_counter}/{n_gt_steps} aligned frames)')
    
    pred_aligned = torch.stack(aligned_frames, dim=0).permute(1, 0, 2, 3, 4).numpy()
    gt_aligned = gt_traj[:, :n_gt_steps].numpy()
    
    print(f'\n=== Computing Metrics ===')
    print(f'  GT shape: {gt_aligned.shape}')
    print(f'  Pred shape: {pred_aligned.shape}')
    
    # Compute metrics
    metric_steps = [1, 3, 5, 10] if n_gt_steps > 10 else list(range(1, n_gt_steps))
    plot_steps = [0, 1, 3, 5, 10] if n_gt_steps > 10 else list(range(min(6, n_gt_steps)))
    
    metrics = compute_metrics_and_visualize(
        gt_traj=gt_aligned,
        pred_traj=pred_aligned,
        metric_step_list=metric_steps,
        plot_step_list=plot_steps,
        visualize=save_plots,
        save_dir=save_dir,
        time_start=time_start,
        dt_data=dt_data,
        sample_idx=sample_idx
    )
    
    return metrics


def make_inference_dir(ckpt_path, args):
    """Create inference output directory based on checkpoint path and inference params.
    
    Returns:
        save_dir: Path to output directory
        run_id: Unique identifier for this inference run
    """
    ckpt_dir = os.path.dirname(os.path.abspath(ckpt_path))
    if os.path.basename(ckpt_dir) == 'model':
        run_dir = os.path.dirname(ckpt_dir)
    else:
        run_dir = ckpt_dir
    
    # Create unique run ID from inference params
    param_str = f'{args.eval_data_path}_{args.n_rollout_steps}_{args.time_start}'
    run_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
    timestamp = datetime.now().strftime('%m_%d_%H_%M_%S')
    
    # Use eval data name as tag
    eval_name = os.path.basename(os.path.dirname(os.path.dirname(args.eval_data_path)))
    if not eval_name or eval_name == 'data':
        eval_name = os.path.basename(os.path.dirname(args.eval_data_path))
    if not eval_name:
        eval_name = 'eval'
    
    run_id = f'infer_{eval_name}_{run_hash}_{timestamp}'
    save_dir = os.path.join(run_dir, 'inference', run_id)
    os.makedirs(save_dir, exist_ok=True)
    
    return save_dir, run_id


def save_inference_config(save_dir, args, metrics, ckpt_info):
    """Save inference configuration and results."""
    config = {
        'checkpoint': args.checkpoint,
        'eval_data_path': args.eval_data_path,
        'n_rollout_steps': args.n_rollout_steps,
        'time_start': args.time_start,
        'time_end': args.time_end,
        'dt_data': args.dt_data,
        'device': args.device,
        'sample_idx': args.sample_idx,
        'ckpt_step': ckpt_info.get('step', None),
        'ckpt_best_loss': ckpt_info.get('best_loss', None),
        'timestamp': datetime.now().isoformat(),
    }
    
    results = {
        'config': config,
        'metrics': metrics,
    }
    
    config_path = os.path.join(save_dir, 'inference_config.json')
    with open(config_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nInference config saved to: {config_path}')
    
    return results


def run_single_eval(args, eval_data_path, net, saved_cfg, ckpt):
    """Run evaluation on a single test set."""
    device = args.device
    model_dt = saved_cfg.get('rollout_dt', 0.1)
    
    # Create a temporary args-like object for make_inference_dir
    class TempArgs:
        pass
    temp_args = TempArgs()
    temp_args.eval_data_path = eval_data_path
    temp_args.n_rollout_steps = args.n_rollout_steps
    temp_args.time_start = args.time_start
    
    save_dir, run_id = make_inference_dir(args.checkpoint, temp_args)
    print(f'\n=== Inference Run: {run_id} ===')
    print(f'  Output directory: {save_dir}')
    
    metrics = evaluate_on_testset(
        net=net,
        eval_data_path=eval_data_path,
        n_rollout_steps=args.n_rollout_steps,
        device=device,
        time_start=args.time_start,
        time_end=args.time_end,
        dt_data=args.dt_data,
        model_dt=model_dt,
        save_dir=save_dir,
        sample_idx=args.sample_idx,
        save_plots=args.save_plots
    )
    
    # Save config
    temp_args.checkpoint = args.checkpoint
    temp_args.time_end = args.time_end
    temp_args.dt_data = args.dt_data
    temp_args.device = args.device
    temp_args.sample_idx = args.sample_idx
    ckpt_info = {'step': ckpt.get('step'), 'best_loss': ckpt.get('best_loss')}
    save_inference_config(save_dir, temp_args, metrics, ckpt_info)
    
    return metrics, save_dir


def main(args):
    device = args.device
    
    # Load model once
    net, saved_cfg, ckpt = load_model(args.checkpoint, device=device)
    model_dt = saved_cfg.get('rollout_dt', 0.1)
    print(f'  Model rollout_dt: {model_dt}s')
    
    all_results = {}
    
    # Determine which test sets to evaluate
    if args.eval_data_path:
        # Single test set specified
        eval_paths = [('custom', args.eval_data_path)]
    else:
        # Default: evaluate on both MHD and GRF test sets
        eval_paths = []
        if os.path.exists(DEFAULT_MHD_TEST):
            eval_paths.append(('mhd', DEFAULT_MHD_TEST))
        else:
            print(f'WARNING: MHD test set not found: {DEFAULT_MHD_TEST}')
        if os.path.exists(DEFAULT_GRF_TEST):
            eval_paths.append(('grf', DEFAULT_GRF_TEST))
        else:
            print(f'WARNING: GRF test set not found: {DEFAULT_GRF_TEST}')
    
    # Run evaluations
    for tag, eval_path in eval_paths:
        print('\n' + '=' * 70)
        print(f'  Evaluating on: {tag.upper()} test set')
        print(f'  Path: {eval_path}')
        print('=' * 70)
        
        metrics, save_dir = run_single_eval(args, eval_path, net, saved_cfg, ckpt)
        all_results[tag] = {
            'metrics': metrics,
            'save_dir': save_dir,
        }
    
    # Print summary
    print('\n' + '=' * 70)
    print('All Evaluations Complete!')
    print('=' * 70)
    for tag, result in all_results.items():
        mean_l2 = result['metrics'].get('mean_rel_l2', 'N/A')
        print(f'  [{tag.upper()}] Mean Relative L2: {mean_l2}')
        print(f'         Results: {result["save_dir"]}')
    
    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='5-field MHD OmniFluids Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Evaluate on both MHD and GRF test sets (default)
  python inference.py --checkpoint /path/to/best.pt
  
  # Evaluate on specific test set only
  python inference.py --checkpoint /path/to/best.pt --eval_data_path /path/to/data.pt
  
  # Specify GPU
  python inference.py --checkpoint /path/to/best.pt --device cuda:1
''')
    
    # Required: checkpoint path
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pt file)')
    
    # Optional: specific test set (if not provided, evaluates on both MHD and GRF)
    parser.add_argument('--eval_data_path', type=str, default=None,
                        help='Path to evaluation dataset. If not provided, evaluates on both '
                             f'MHD ({DEFAULT_MHD_TEST}) and GRF ({DEFAULT_GRF_TEST}) test sets.')
    
    # Evaluation settings
    parser.add_argument('--n_rollout_steps', type=int, default=10,
                        help='Number of data steps to rollout (default: 10)')
    parser.add_argument('--time_start', type=float, default=250.0,
                        help='Start time for evaluation (seconds)')
    parser.add_argument('--time_end', type=float, default=None,
                        help='End time for evaluation (None=auto)')
    parser.add_argument('--dt_data', type=float, default=1.0,
                        help='Time step in test data (seconds)')
    
    # Output settings
    parser.add_argument('--sample_idx', type=int, default=0,
                        help='Which sample to visualize (default: 0)')
    parser.add_argument('--save_plots', type=int, default=1,
                        help='Save visualization plots (1=yes, 0=no)')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to run on')
    
    args = parser.parse_args()
    main(args)
