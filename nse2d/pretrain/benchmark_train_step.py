#!/usr/bin/env python
"""Benchmark script to profile training step timing breakdown.

Usage:
    # Baseline (fp32)
    CUDA_VISIBLE_DEVICES=0 python benchmark_train_step.py
    
    # With AMP (bf16)
    CUDA_VISIBLE_DEVICES=0 python benchmark_train_step.py --amp bf16
    
    # With AMP (fp16)
    CUDA_VISIBLE_DEVICES=0 python benchmark_train_step.py --amp fp16

Measures:
    1. GRF data generation
    2. Model forward pass
    3. Physics loss computation (with RHS breakdown)
    4. Backward pass
    5. Optimizer step
"""

import sys
import time
import argparse
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler

# Compatibility wrapper for autocast
def amp_autocast(dtype):
    """Create autocast context manager compatible with different PyTorch versions."""
    if hasattr(torch, 'autocast'):
        # PyTorch 2.0+
        return torch.autocast('cuda', dtype=dtype)
    else:
        # Older PyTorch
        return torch.cuda.amp.autocast(dtype=dtype)

sys.path.insert(0, '/zhangtao/project2026/OmniFluids/nse2d/pretrain')
sys.path.insert(0, '/zhangtao/project2026/mhd_sim')

from model import OmniFluids2D
from tools import MHD5FieldGRF
from psm_loss import build_mhd_instance, make_mhd5_rhs_fn


class Timer:
    """Context manager for timing code blocks."""
    def __init__(self, name, sync_cuda=True):
        self.name = name
        self.sync_cuda = sync_cuda
        self.elapsed_ms = 0
        
    def __enter__(self):
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed_ms = (time.perf_counter() - self.start) * 1000


def compute_physics_loss_timed(pred_traj, x_0, rhs_fn, rollout_dt, output_dim, time_integrator='crank_nicolson'):
    """Physics loss with internal timing for RHS calls."""
    import torch.nn.functional as F
    
    dt = rollout_dt / output_dim
    full_traj = torch.cat([x_0.unsqueeze(-1), pred_traj], dim=-1)
    
    total_mse = torch.tensor(0.0, device=x_0.device, dtype=x_0.dtype)
    rhs_time_ms = 0.0
    n_rhs_calls = 0
    
    for t in range(output_dim):
        state_t = full_traj[..., t]
        state_tp1 = full_traj[..., t + 1]
        time_diff = (state_tp1 - state_t) / dt
        
        if time_integrator == 'euler':
            with Timer('rhs', sync_cuda=True) as timer:
                target = rhs_fn(state_t)
            rhs_time_ms += timer.elapsed_ms
            n_rhs_calls += 1
        elif time_integrator == 'crank_nicolson':
            with Timer('rhs1', sync_cuda=True) as timer1:
                rhs_t = rhs_fn(state_t)
            with Timer('rhs2', sync_cuda=True) as timer2:
                rhs_tp1 = rhs_fn(state_tp1)
            target = (rhs_t + rhs_tp1) / 2.0
            rhs_time_ms += timer1.elapsed_ms + timer2.elapsed_ms
            n_rhs_calls += 2
        
        target = target.detach()
        total_mse = total_mse + F.mse_loss(time_diff, target)
    
    loss = total_mse / output_dim
    return loss, rhs_time_ms, n_rhs_calls


def benchmark(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # AMP setup
    use_amp = args.amp is not None and args.amp != 'no'
    amp_dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16}.get(args.amp, None)
    scaler = GradScaler() if args.amp == 'fp16' else None  # bf16 doesn't need scaler
    
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Iterations: {args.n_iters}")
    print(f"Output dim: {args.output_dim}")
    print(f"Time integrator: {args.time_integrator}")
    print(f"AMP: {args.amp} (dtype={amp_dtype})")
    print(f"torch.compile: {args.compile}")
    print(f"TF32: {args.tf32}")
    print("-" * 60)
    
    # TF32 setup (must be before any CUDA operations)
    if args.tf32:
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.allow_tf32 = True
        print("TF32 enabled: matmul precision='high', cudnn TF32=True")
    
    # Build components
    print("Building model...")
    net = OmniFluids2D(
        Nx=512, Ny=256, K=4, T=10.0,
        modes_x=128, modes_y=128, width=80,
        output_dim=args.output_dim, n_fields=5, n_params=8,
        n_layers=12, factor=4, n_ff_layers=2, layer_norm=True
    ).to(device)
    print(f"  Model params: {sum(p.numel() for p in net.parameters())/1e6:.2f}M")
    
    print("Building GRF generator...")
    grf = MHD5FieldGRF(Nx=512, Ny=256, device=device)
    
    print("Building MHD instance...")
    mhd = build_mhd_instance(device=device, Nx=512, Ny=256)
    # Model output dtype depends on AMP
    model_out_dtype = amp_dtype if use_amp else torch.float32
    rhs_fn = make_mhd5_rhs_fn(mhd, model_dtype=torch.float32)  # RHS always fp32 input (casted from amp)
    
    optimizer = optim.Adam(net.parameters(), lr=0.002)
    
    # torch.compile (if enabled)
    if args.compile:
        print("Compiling model with torch.compile...")
        compile_start = time.perf_counter()
        net = torch.compile(net)
        print(f"  torch.compile setup done in {(time.perf_counter()-compile_start)*1000:.0f}ms (actual compile happens on first forward)")
    
    # Warmup
    print("\nWarmup (2 iters)...")
    for _ in range(2):
        x_0 = grf(args.batch_size)
        if use_amp:
            with amp_autocast(amp_dtype):
                pred = net(x_0)
                # Physics loss needs fp32 for RHS, so cast back
                pred_fp32 = pred.float()
                x_0_fp32 = x_0.float()
            loss, _, _ = compute_physics_loss_timed(
                pred_fp32, x_0_fp32, rhs_fn, 0.1, args.output_dim, args.time_integrator)
        else:
            pred = net(x_0)
            loss, _, _ = compute_physics_loss_timed(
                pred, x_0, rhs_fn, 0.1, args.output_dim, args.time_integrator)
        
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    
    # Benchmark
    print(f"\nBenchmarking {args.n_iters} iterations...")
    
    timings = {
        'grf': [],
        'forward': [],
        'physics_loss_total': [],
        'rhs_calls': [],
        'backward': [],
        'optimizer': [],
        'total': [],
    }
    
    for i in range(args.n_iters):
        iter_start = time.perf_counter()
        
        # 1. GRF generation
        with Timer('grf') as t_grf:
            x_0 = grf(args.batch_size)
        timings['grf'].append(t_grf.elapsed_ms)
        
        # 2. Model forward (with AMP if enabled)
        net.train()
        if use_amp:
            with Timer('forward') as t_fwd:
                with amp_autocast(amp_dtype):
                    pred_traj = net(x_0)
                # Cast to fp32 for physics loss
                pred_traj_fp32 = pred_traj.float()
                x_0_fp32 = x_0.float()
            timings['forward'].append(t_fwd.elapsed_ms)
            
            # 3. Physics loss (always fp32 for RHS)
            with Timer('physics_loss') as t_phys:
                loss, rhs_ms, n_rhs = compute_physics_loss_timed(
                    pred_traj_fp32, x_0_fp32, rhs_fn, 0.1, args.output_dim, args.time_integrator)
            timings['physics_loss_total'].append(t_phys.elapsed_ms)
            timings['rhs_calls'].append(rhs_ms)
            
            # 4. Backward (with scaler for fp16)
            with Timer('backward') as t_bwd:
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            timings['backward'].append(t_bwd.elapsed_ms)
            
            # 5. Optimizer step
            with Timer('optimizer') as t_opt:
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            timings['optimizer'].append(t_opt.elapsed_ms)
        else:
            with Timer('forward') as t_fwd:
                pred_traj = net(x_0)
            timings['forward'].append(t_fwd.elapsed_ms)
            
            # 3. Physics loss (with RHS breakdown)
            with Timer('physics_loss') as t_phys:
                loss, rhs_ms, n_rhs = compute_physics_loss_timed(
                    pred_traj, x_0, rhs_fn, 0.1, args.output_dim, args.time_integrator)
            timings['physics_loss_total'].append(t_phys.elapsed_ms)
            timings['rhs_calls'].append(rhs_ms)
            
            # 4. Backward
            with Timer('backward') as t_bwd:
                loss.backward()
            timings['backward'].append(t_bwd.elapsed_ms)
            
            # 5. Optimizer step
            with Timer('optimizer') as t_opt:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            timings['optimizer'].append(t_opt.elapsed_ms)
        
        torch.cuda.synchronize()
        iter_total = (time.perf_counter() - iter_start) * 1000
        timings['total'].append(iter_total)
        
        if (i + 1) % max(1, args.n_iters // 5) == 0:
            print(f"  iter {i+1}/{args.n_iters}: total={iter_total:.0f}ms")
    
    # Summary
    print("\n" + "=" * 60)
    print("TIMING BREAKDOWN (ms per iteration)")
    print("=" * 60)
    
    for name, values in timings.items():
        avg = sum(values) / len(values)
        std = (sum((v - avg)**2 for v in values) / len(values)) ** 0.5
        pct = avg / (sum(timings['total']) / len(timings['total'])) * 100
        print(f"  {name:20s}: {avg:8.1f} ± {std:6.1f} ms  ({pct:5.1f}%)")
    
    print("\n" + "-" * 60)
    print(f"RHS calls per iter: {n_rhs} (time_integrator={args.time_integrator})")
    print(f"Avg RHS time: {sum(timings['rhs_calls'])/len(timings['rhs_calls'])/n_rhs:.1f} ms per call")
    print(f"Total iter time: {sum(timings['total'])/len(timings['total']):.0f} ms")
    print(f"Throughput: {1000 / (sum(timings['total'])/len(timings['total'])):.2f} it/s")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--n_iters', type=int, default=10)
    parser.add_argument('--output_dim', type=int, default=10)
    parser.add_argument('--time_integrator', type=str, default='crank_nicolson',
                        choices=['euler', 'crank_nicolson'])
    parser.add_argument('--amp', type=str, default=None,
                        choices=[None, 'no', 'fp16', 'bf16'],
                        help='Mixed precision: no (fp32), fp16, or bf16')
    parser.add_argument('--compile', action='store_true',
                        help='Enable torch.compile for operator fusion')
    parser.add_argument('--tf32', action='store_true',
                        help='Enable TF32 matmul precision (torch.set_float32_matmul_precision high)')
    args = parser.parse_args()
    benchmark(args)
