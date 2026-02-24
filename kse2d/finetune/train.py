from tools import Init_generation, HW_Init_generation, Force_generation

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


def _make_full_param(param_3, device):
    """[N,3] → [N,5] 补全固定参数 N/4=0.75, k0=0.15"""
    N = param_3.shape[0]
    p = torch.zeros(N, 5, device=device)
    p[:, :3] = param_3[:, :3].to(device)
    p[:, 3] = 0.75
    p[:, 4] = 0.15
    return p


def test(config, net, test_data, test_param, test_data_dict):
    """
    test_data: [N, S, S, T+1, 2]
    test_param: [N, 5] (已补全)
    """
    device = config.device
    T_final = test_data_dict['T']
    data_ratio = test_data_dict['record_ratio']
    rollout_DT = config.rollout_DT
    model_ratio = round(1.0 / rollout_DT)
    sub = round(data_ratio / model_ratio)
    total_iter = round(T_final / rollout_DT)

    N = test_param.shape[0]
    w_current = test_data[:, :, :, 0, :].to(device)   # [N,S,S,2]
    predictions = [w_current]
    net.eval()
    with torch.no_grad():
        for _ in range(total_iter):
            w_current = net(w_current, test_param.to(device)).detach()  # Student: [N,S,S,2]
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
    return np.mean(rela_err)


def train(config, net):
    device = config.device
    data_name = config.data_name

    data_dict_path = config.data_path + f'log/cfg_test{data_name}.txt'
    with open(data_dict_path, 'r') as file:
        file_content = file.read()
    test_data_dict = json.loads(file_content)

    data_dict_path = config.data_path + f'log/cfg_val{data_name}.txt'
    with open(data_dict_path, 'r') as file:
        file_content = file.read()
    val_data_dict = json.loads(file_content)

    data_dict_path = config.data_path + f'log/cfg_train{data_name}.txt'
    with open(data_dict_path, 'r') as file:
        file_content = file.read()
    train_data_dict = json.loads(file_content)

    data_size = test_data_dict['s'] // test_data_dict['sub']
    size = config.size
    sub = max(1, data_size // size)
    num_train = config.num_train

    # 数据 shape: [N, S, S, T+1, 2]
    train_data = torch.load(config.data_path + f'dataset/train{data_name}')[:num_train, ::sub, ::sub, ...].float().to(device)
    # param shape: [N, 3] → 补全为 [N, 5]
    train_param_raw = torch.load(config.data_path + f'dataset/param_train{data_name}')[:num_train].float()
    train_param = _make_full_param(train_param_raw, device)
    print(train_data.shape)

    test_data = torch.load(config.data_path + f'dataset/test{data_name}')[:, ::sub, ::sub, ...].float().to(device)
    test_param_raw = torch.load(config.data_path + f'dataset/param_test{data_name}').float()
    test_param = _make_full_param(test_param_raw, device)
    print(test_data.shape)

    val_data = torch.load(config.data_path + f'dataset/val{data_name}')[:, ::sub, ::sub, ...].float().to(device)
    val_param_raw = torch.load(config.data_path + f'dataset/param_val{data_name}').float()
    val_param = _make_full_param(val_param_raw, device)
    print(val_data.shape)

    rollout_DT = config.rollout_DT
    train_step = round(train_data_dict['record_ratio'] * rollout_DT)  # =1 for HW
    optimizer = optim.Adam(net.parameters(), config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=config.num_iterations + 1, max_lr=config.lr)

    print('-----------START_VAL_ERROR-----------')
    val_error = test(config, net, val_data, val_param, val_data_dict)
    print('-----------START_TEST_ERROR-----------')
    test_error = test(config, net, test_data, test_param, test_data_dict)
    torch.save(net.state_dict(), f'model/{config.model_name}.pt')

    # train_data: [N, S, S, T+1, 2] — 时间维在倒数第2维
    T_frames = train_data.shape[3]
    for step in range(config.num_iterations):
        net.train()
        w0 = torch.zeros(num_train, size, size, 2, device=device)
        w_gth1 = torch.zeros(num_train, size, size, 2, device=device)
        w_gth2 = torch.zeros(num_train, size, size, 2, device=device)
        for i in range(num_train):
            t = random.randint(0, T_frames - 2 * train_step - 1)
            w0[i, :, :, :] = train_data[i, :, :, t, :]
            w_gth1[i, :, :, :] = train_data[i, :, :, t + train_step, :]
            w_gth2[i, :, :, :] = train_data[i, :, :, t + 2 * train_step, :]

        w_pre = net(w0, train_param)   # Student: [N,S,S,2]
        loss = torch.square(w_pre - w_gth1).mean()
        loss = (loss + EPS).sqrt()
        w_pre = net(w_pre, train_param)  # Student: [N,S,S,2]
        loss = loss + (torch.square(w_pre - w_gth2).mean() + EPS).sqrt()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step % 10 == 0:
            print(step, '########################')
            print('training loss', step, loss.detach().item())
            print('-----------VAL_ERROR-----------')
            val_error_now = test(config, net, val_data, val_param, val_data_dict)
            if val_error_now < val_error:
                val_error = val_error_now
                print('-----------TEST_ERROR-----------')
                test(config, net, test_data, test_param, test_data_dict)
                print('-----------SAVING NEW MODEL-----------')
                torch.save(net.state_dict(), f'model/{config.model_name}.pt')
            sys.stdout.flush()
    print('----------------------------FINAL_RESULT-----------------------------')
    net.load_state_dict(torch.load(f'model/{config.model_name}.pt'))
    test(config, net, test_data, test_param, test_data_dict)
    sys.stdout.flush()