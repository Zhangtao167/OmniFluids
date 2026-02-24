from tools import Init_generation, HW_Init_generation
import math
import sys
import copy
import json
import random
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
plt.rcParams["animation.html"] = "jshtml"


EPS = 1e-7


def test(config, net, test_data, test_param, test_data_dict):
    """
    test_data: [N, S, S, T+1, 2]
    test_param: [N, 3] → 补全为 [N, 5]
    Student2D 无 inference 参数，输出 [N,S,S,2]
    """
    device = config.device
    T_final = test_data_dict['T']
    data_ratio = test_data_dict['record_ratio']
    rollout_DT = config.rollout_DT
    model_ratio = round(1.0 / rollout_DT)
    sub = round(data_ratio / model_ratio)
    total_iter = round(T_final / rollout_DT)

    N = test_param.shape[0]
    param_full = torch.zeros(N, 5, device=device)
    param_full[:, :3] = test_param[:, :3].to(device)
    param_full[:, 3] = 0.75
    param_full[:, 4] = 0.15

    w_current = test_data[:, :, :, 0, :].to(device)   # [N,S,S,2]
    predictions = [w_current]
    net.eval()
    with torch.no_grad():
        for _ in range(total_iter):
            w_current = net(w_current, param_full).detach()  # Student: [N,S,S,2]
            predictions.append(w_current)
    w_pre = torch.stack(predictions, dim=3)   # [N,S,S,T+1,2]

    rela_err = []
    print('_________Test__________')
    for time_step in range(1, total_iter + 1):
        w = w_pre[:, :, :, time_step, :]
        w_t = test_data[:, :, :, sub * time_step, :].to(device)
        rela_err.append(
            (torch.norm((w - w_t).reshape(N, -1), dim=1)
             / torch.norm(w_t.reshape(N, -1), dim=1)).mean().item())
        if time_step % 50 == 0 or time_step == 1:
            print(time_step, 'relative l_2 error', rela_err[time_step - 1])
    print('Mean Relative l_2 Error', np.mean(rela_err))


def train(config, net_t, net_s):
    device = config.device
    data_name = config.data_name
    data_dict_path = config.data_path + f'log/cfg_test_{data_name}.txt'
    with open(data_dict_path, 'r') as file:
        file_content = file.read()
    test_data_dict = json.loads(file_content)
    alpha_range = test_data_dict['alpha_range']
    kappa_range = test_data_dict['kappa_range']
    lognu_range = test_data_dict['lognu_range']
    data_size = test_data_dict['s'] // test_data_dict['sub']
    batch_size = config.batch_size
    size = config.size
    sub = max(1, data_size // config.student_size)
    test_data = torch.load(config.data_path + f'dataset/test_{data_name}')[:, ::sub, ::sub, ...].float()
    test_param = torch.load(config.data_path + f'dataset/param_test_{data_name}').float()
    print(test_data.shape)
    student_sub = max(1, config.size // config.student_size)

    HW_GRF = HW_Init_generation(size, device=device, dtype=torch.float32)
    teacher_step = round(config.rollout_DT / config.rollout_DT_teacher)

    num_iterations = config.num_iterations
    net_s = net_s.to(device)
    optimizer = optim.Adam(net_s.parameters(), config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=config.num_iterations + 1, max_lr=config.lr)

    for step in range(num_iterations + 1):
        z0, n0 = HW_GRF(batch_size)
        w0_train = torch.stack([z0, n0], dim=-1)   # [B,S,S,2]
        param = torch.zeros(batch_size, 5, device=device)
        param[:, 0] = (alpha_range[1] - alpha_range[0]) * torch.rand(batch_size, device=device) + alpha_range[0]
        param[:, 1] = (kappa_range[1] - kappa_range[0]) * torch.rand(batch_size, device=device) + kappa_range[0]
        param[:, 2] = (lognu_range[1] - lognu_range[0]) * torch.rand(batch_size, device=device) + lognu_range[0]
        param[:, 3] = 0.75
        param[:, 4] = 0.15

        w_gth = w0_train.clone()
        net_t.eval()
        with torch.no_grad():
            for i in range(teacher_step):
                w_gth = net_t(w_gth, param, inference=True).detach()  # [B,S,S,2]
        w_gth = w_gth[:, ::student_sub, ::student_sub, :]
        w0_train = w0_train[:, ::student_sub, ::student_sub, :]

        for _ in range(10):
            net_s.train()
            w_s = net_s(w0_train, param)   # [B,S,S,2]
            loss = torch.mean((w_s - w_gth) ** 2 + EPS).sqrt()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        if step % 10 == 0:
            print('training loss', step, loss.detach().item())
            torch.save(net_s.state_dict(), f'model/{config.model_name}.pt')
            test(config, net_s, test_data, test_param, test_data_dict)
        sys.stdout.flush()
    print('----------------------------FINAL_RESULT-----------------------------')
    net_s.load_state_dict(torch.load(f'model/{config.model_name}.pt'))
    test(config, net_s, test_data, test_param, test_data_dict)
    sys.stdout.flush()
