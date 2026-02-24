import torch
import numpy as np
import random
import math
import os
import json
import sys
from kse import *
from sampler import Init_generation, HW_Init_generation
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     os.environ['PYTHONHASHSEED'] = str(seed) 
     torch.backends.cudnn.deterministic = True
     torch.backends.cudnn.benchmark = False
     torch.backends.cudnn.enabled = True




def generate_hw_data(cfg):
    device = cfg.device
    s = cfg.s
    sub = cfg.sub
    dt = cfg.dt
    T = cfg.T
    record_steps = int(cfg.record_ratio * T)
    mode = cfg.mode

    if mode == 'train':
        setup_seed(0)
        cfg.N = 2
    if mode == 'test':
        setup_seed(1)
        cfg.N = 10
    if mode == 'val':
        setup_seed(2)
        cfg.N = 2
    N = cfg.N
    alpha_range = cfg.alpha_range
    kappa_range = cfg.kappa_range
    lognu_range = cfg.lognu_range
    data_save_path = f'./dataset'
    if not os.path.exists(data_save_path):
        os.makedirs(data_save_path)

    HW_GRF = HW_Init_generation(s, device=device)
    bsize = min(100, N)
    c = 0
    u = torch.zeros(N, s // sub, s // sub, record_steps + 1, 2)
    param_record = torch.zeros(N, 3)

    max_retries = 10
    total_retries = 0
    
    for j in range(N // bsize):
        valid_samples = 0
        batch_sol = []
        batch_param = []
        
        while valid_samples < bsize:
            retries = 0
            sample_valid = False
            
            while not sample_valid and retries < max_retries:
                z0, n0 = HW_GRF(1)
                param = torch.zeros(1, 3, device=device)
                param[:, 0] = (alpha_range[1] - alpha_range[0]) * torch.rand(1, device=device) + alpha_range[0]
                param[:, 1] = (kappa_range[1] - kappa_range[0]) * torch.rand(1, device=device) + kappa_range[0]
                param[:, 2] = (lognu_range[1] - lognu_range[0]) * torch.rand(1, device=device) + lognu_range[0]
                
                sol = hw_2d_rk4(z0, n0, T, param, dt=dt, record_steps=record_steps, device=device)
                sol = sol[:, ::sub, ::sub, :, :].to('cpu')
                
                if torch.isnan(sol).any() or torch.isinf(sol).any():
                    retries += 1
                    total_retries += 1
                    print(f"  ⚠ Sample {c+valid_samples} invalid (NaN/Inf detected), retry {retries}/{max_retries}")
                else:
                    max_val = sol.abs().max().item()
                    if max_val > 1e6:
                        retries += 1
                        total_retries += 1
                        print(f"  ⚠ Sample {c+valid_samples} invalid (max={max_val:.2e}), retry {retries}/{max_retries}")
                    else:
                        sample_valid = True
                        batch_sol.append(sol)
                        batch_param.append(param.cpu())
                        print(f"  ✓ Sample {c+valid_samples} valid: α={param[0,0]:.3f}, κ={param[0,1]:.3f}, "
                              f"logν={param[0,2]:.3f}, max|ζ|={sol[0,:,:,:,0].abs().max():.3f}")
            
            if not sample_valid:
                print(f"  ✗ Failed to generate valid sample after {max_retries} retries, skipping")
                continue
            
            valid_samples += 1
        
        batch_sol = torch.cat(batch_sol, dim=0)
        batch_param = torch.cat(batch_param, dim=0)
        u[c:(c + bsize), ...] = batch_sol
        param_record[c:(c + bsize), ...] = batch_param
        c += bsize
        print(f"Batch {j+1}/{N//bsize} complete: {c}/{N} samples, total retries: {total_retries}")
        print(f"  max|ζ|={u[:c,:,:,:,0].abs().max():.4f}, max|n|={u[:c,:,:,:,1].abs().max():.4f}")
    
    print(f"\n{'='*60}")
    print(f"Data generation complete: {N} valid samples, {total_retries} total retries")
    print(f"{'='*60}\n")
    torch.save(u, f'{data_save_path}/{cfg.name}')
    torch.save(param_record, f'{data_save_path}/param_{cfg.name}')