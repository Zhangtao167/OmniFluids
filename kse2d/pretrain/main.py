from model import OmniFluids2D
from train import train
from tools import setup_seed, param_flops

import json
import sys
import copy
import hashlib
from datetime import datetime
import random
import argparse
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
plt.rcParams["animation.html"] = "jshtml"


def generate_run_hash(cfg):
    """根据配置生成 8 位唯一 hash code"""
    key_params = (
        cfg.seed, cfg.K, cfg.modes, cfg.width, cfg.n_layers, cfg.output_dim,
        cfg.size, cfg.lr, cfg.rollout_DT, cfg.train_loss_type,
        cfg.operator_discretization, datetime.now().isoformat()
    )
    hash_str = hashlib.md5(str(key_params).encode()).hexdigest()[:8]
    return hash_str


def main(cfg):
    if not os.path.exists(f'log'):
        os.mkdir(f'log')
    if not os.path.exists(f'log/log_{cfg.file_name}'):
        os.mkdir(f'log/log_{cfg.file_name}')
    if not os.path.exists(f'model'):
        os.mkdir(f'model')
    setup_seed(cfg.seed)
    
    # 生成 8 位 hash code
    cfg.run_hash = generate_run_hash(cfg)
    
    dateTimeObj = datetime.now()
    timestring = f'{dateTimeObj.date().month}_{dateTimeObj.date().day}_{dateTimeObj.time().hour}_{dateTimeObj.time().minute}_{dateTimeObj.time().second}'
    cfg.model_name = f'{cfg.run_hash}-w{cfg.width}-m{cfg.modes}-l{cfg.n_layers}-K{cfg.K}'
    logfile = f'log/log_{cfg.file_name}/log-{cfg.run_hash}-{timestring}.csv'
    sys.stdout = open(logfile, 'w')

    print('--------args----------')
    for k in list(vars(cfg).keys()):
        print('%s: %s' % (k, vars(cfg)[k]))
    print('--------args----------\n')
    net = OmniFluids2D(s=cfg.size, K=cfg.K, modes=cfg.modes, width=cfg.width,
                       output_dim=cfg.output_dim, n_layers=cfg.n_layers,
                       n_fields=2, n_params=5, f_nu_hidden=cfg.f_nu_hidden)
    param_flops(net)
    sys.stdout.flush()
    train(cfg, net)

    final_time = datetime.now()
    print(f'FINAL_TIME_{final_time.date().month}_{final_time.date().day}_{final_time.time().hour}_{final_time.time().minute}_{final_time.time().second}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='OmniFluids HW pre-training')

    # ---- 数据 (mhd_sim 格式) ----
    parser.add_argument('--train_data', type=str, required=True,
            help='mhd_sim hw_dataset.pt for training')
    parser.add_argument('--test_data', type=str, required=True,
            help='mhd_sim hw_dataset.pt for testing')
    parser.add_argument('--time_start', type=int, default=200,
            help='training window start frame (inclusive)')
    parser.add_argument('--time_end', type=int, default=249,
            help='training window end frame (inclusive)')

    # ---- 通用 ----
    parser.add_argument('--file_name', type=str, default='search_param',
            help='log sub-directory name')
    parser.add_argument('--device', type=str, default='cuda:0',
            help='device')
    parser.add_argument('--seed', type=int, default=0)

    # ---- 模型 ----
    parser.add_argument('--modes', type=int, default=32)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=8)
    parser.add_argument('--output_dim', type=int, default=40)
    parser.add_argument('--size', type=int, default=128,
            help='spatial resolution for neural operator')
    parser.add_argument('--K', type=int, default=4,
            help='number of mixture-of-operators kernels')
    parser.add_argument('--f_nu_hidden', type=int, default=128,
            help='hidden dim for MoE weight generator (default 128 for backward compat)')

    # ---- 训练 ----
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--val_size', type=int, default=10)
    parser.add_argument('-loss_mode', type=str, default='cn',
            help='PSM time discretization: cn or mid')
    parser.add_argument('--rollout_DT', type=float, default=0.1,
            help='physical time per model inference step (s)')
    parser.add_argument('--dt_data', type=float, default=1.0,
            help='physical time between data frames (s)')
    parser.add_argument('--test_steps', type=int, default=10,
            help='evaluation rollout steps')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--num_iterations', type=int, default=500000)
    parser.add_argument('--input_noise_std', type=float, default=0.0,
            help='Gaussian noise std on input during training (0=off)')
    parser.add_argument('--rhs_loss_weight', type=float, default=0.0,
            help='weight for RHS-anchored loss (0=off, mhd_sim style)')
    parser.add_argument('--train_loss_type', type=str, default='mhd_sim',
            choices=['mhd_sim', 'omnifluids'],
            help='training loss type: mhd_sim anchored loss or OmniFluids PSM')
    parser.add_argument('--operator_discretization', type=str, default='mhd_sim',
            choices=['mhd_sim', 'spectral'],
            help='physics operator discretization for RHS/PSM')
    parser.add_argument('--mhd_sim_root', type=str, default='/zhangtao/project2026/mhd_sim',
            help='path to mhd_sim repo root (used when operator_discretization=mhd_sim)')
    parser.add_argument('--train_use_inference_path', action='store_true',
            help='train directly on inference=True path (default: off for rollback)')
    parser.add_argument('--train_unroll_steps', type=int, default=1,
            help='autoregressive unroll steps when train_use_inference_path is on')

    # ---- 旧参数 (兼容性, 不再使用) ----
    parser.add_argument('--model_name', type=str, default='')

    cfg = parser.parse_args()
    main(cfg)
