#!/usr/bin/env python3
"""Generate comprehensive evaluation table from all inference results."""
import os

RESULTS_DIR = '/zhangtao/project2026/OmniFluids/nse2d/pretrain/results'

CATEGORIES = {
    '1. Simulation Data (Offline Supervised)': [
        ('mhd5_omnifluids_v1', 'mhd5_omnifluids_v1 (baseline)'),
        ('mhd5_offline_mae01', 'mhd5_offline_mae01 (sup+MAE)'),
        ('exp1a_pure_supervised_dt01s', 'exp1a (sup dt=0.1s)'),
        ('exp1b_pure_supervised_dt1s_signle_gpu', 'exp1b (sup dt=1s, 1GPU)'),
        ('exp1b_pure_supervised_dt1s_multigpu', 'exp1b (sup dt=1s, 2GPU)'),
        ('exp1c_pure_supervised_dt0p1_interp_multigpu', 'exp1c (sup dt=0.1 interp, 2GPU)'),
        ('exp_offline_pde_only', 'exp_offline_pde (PDE loss, 2GPU)'),
        ('exp_offline_pde_only_single_gpu', 'exp_offline_pde (PDE loss, 1GPU)'),
        ('exp2_physics_overfitting', 'exp2 (physics overfit)'),
        ('exp3_combined_physics_supervised', 'exp3 (physics+supervised)'),
    ],
    '2. GRF+Simulation Data (Staged/Mixed)': [
        ('mhd5_staged_v1', 'mhd5_staged_v1 (staged)'),
        ('exp6_grf_staged', 'exp6 (GRF staged)'),
        ('exp11_grf_staged_multigpu', 'exp11 (GRF staged, 2GPU)'),
        ('exp12_mixed_integrator', 'exp12 (mixed integrator)'),
    ],
    '3. Self-Training': [
        ('exp10_self_training_4gpu', 'exp10 (self-train, 4GPU)'),
        ('exp13_grf_self_training', 'exp13 (GRF self-train)'),
        ('exp18_self_training_v2', 'exp18 (self-train v2)'),
        ('exp22_self_training_linf_multistep', 'exp22 (self-train+Linf+multistep)'),
    ],
    '4. Learnable GRF': [
        ('exp14_learnable_grf', 'exp14 (learnable GRF)'),
        ('exp20_learnable_grf_10k_soft_linf', 'exp20 (learn-GRF+Linf)'),
        ('exp21_learnable_grf_10k_soft_linf_multistep', 'exp21 (learn-GRF+Linf+multistep)'),
    ],
}


def parse_termwise_metrics(filepath):
    results = {}
    if not filepath or not os.path.exists(filepath):
        return results
    with open(filepath, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        parts = line.split('\t')
        if len(parts) >= 8:
            key = parts[0].strip()
            if key in ('1', '3', '5', '10') or key.startswith('mean('):
                label = int(key) if key.isdigit() else 'mean'
                results[label] = {
                    'total': float(parts[1]),
                    'ch_mean': float(parts[2]),
                    'n': float(parts[3]),
                    'U': float(parts[4]),
                    'vpar': float(parts[5]),
                    'psi': float(parts[6]),
                    'Ti': float(parts[7]),
                }
    return results


def find_latest_eval(exp_path, eval_type):
    matches = []
    for root, dirs, files in os.walk(exp_path):
        if 'inference' in root and 'termwise_metrics.txt' in files:
            parent = os.path.basename(root)
            if (eval_type == 'mhd' and 'mhd' in parent) or (eval_type == 'grf' and 'grf' in parent):
                matches.append(os.path.join(root, 'termwise_metrics.txt'))
    if matches:
        matches.sort()
        return matches[-1]
    return None


def fmt(val):
    if val is None:
        return '--'
    if val >= 100:
        return f'{val:.1f}'
    elif val >= 10:
        return f'{val:.2f}'
    elif val >= 1:
        return f'{val:.3f}'
    else:
        return f'{val:.4f}'


def get_val(data, step, key='ch_mean'):
    if step in data and key in data[step]:
        return data[step][key]
    return None


def generate_table():
    out = []
    out.append('# Evaluation Summary Table')
    out.append('')
    out.append('Metric: **Per-channel Mean Relative L2 Error** (avg of 5 fields), averaged over B=10 test trajectories')
    out.append('')
    out.append('Eval steps: rollout step 1, 3, 5, 10; Mean = average over these 4 steps')
    out.append('')

    all_exps = []

    for cat_name, experiments in CATEGORIES.items():
        out.append(f'## {cat_name}')
        out.append('')
        out.append('| Setting | Test | step-1 | step-3 | step-5 | step-10 | **Mean** |')
        out.append('|---------|------|--------|--------|--------|---------|----------|')

        for exp_dir_name, display_name in experiments:
            exp_path = os.path.join(RESULTS_DIR, exp_dir_name)
            if not os.path.isdir(exp_path):
                continue

            mhd_file = find_latest_eval(exp_path, 'mhd')
            grf_file = find_latest_eval(exp_path, 'grf')
            mhd_data = parse_termwise_metrics(mhd_file)
            grf_data = parse_termwise_metrics(grf_file)

            s1 = fmt(get_val(mhd_data, 1))
            s3 = fmt(get_val(mhd_data, 3))
            s5 = fmt(get_val(mhd_data, 5))
            s10 = fmt(get_val(mhd_data, 10))
            sm = fmt(get_val(mhd_data, 'mean'))
            out.append(f'| {display_name} | MHD | {s1} | {s3} | {s5} | {s10} | **{sm}** |')

            s1 = fmt(get_val(grf_data, 1))
            s3 = fmt(get_val(grf_data, 3))
            s5 = fmt(get_val(grf_data, 5))
            s10 = fmt(get_val(grf_data, 10))
            sm = fmt(get_val(grf_data, 'mean'))
            out.append(f'| | GRF | {s1} | {s3} | {s5} | {s10} | **{sm}** |')

            all_exps.append((display_name, mhd_data, grf_data,
                             get_val(mhd_data, 'mean'), get_val(grf_data, 'mean')))
        out.append('')

    # Ranking
    out.append('## Ranking (by MHD Per-ch Mean)')
    out.append('')
    out.append('| # | Experiment | MHD Mean | GRF Mean | MHD s1 | MHD s10 | GRF s1 | GRF s10 |')
    out.append('|---|-----------|----------|----------|--------|---------|--------|---------|')
    all_exps.sort(key=lambda x: x[3] if x[3] is not None else 9999)
    for rank, (name, mhd_d, grf_d, mm, gm) in enumerate(all_exps, 1):
        out.append(f'| {rank} | {name} | **{fmt(mm)}** | **{fmt(gm)}** | '
                   f'{fmt(get_val(mhd_d, 1))} | {fmt(get_val(mhd_d, 10))} | '
                   f'{fmt(get_val(grf_d, 1))} | {fmt(get_val(grf_d, 10))} |')
    out.append('')

    # Per-field for top 5
    out.append('## Top-5 Per-field Breakdown (Mean over 4 steps)')
    out.append('')
    out.append('| Experiment | Test | n | U | vpar | psi | Ti | **ch_mean** |')
    out.append('|-----------|------|---|---|------|-----|-----|------------|')
    for name, mhd_d, grf_d, mm, gm in all_exps[:5]:
        if 'mean' in mhd_d:
            d = mhd_d['mean']
            out.append(f'| {name} | MHD | {fmt(d["n"])} | {fmt(d["U"])} | {fmt(d["vpar"])} | {fmt(d["psi"])} | {fmt(d["Ti"])} | **{fmt(d["ch_mean"])}** |')
        if 'mean' in grf_d:
            d = grf_d['mean']
            out.append(f'| | GRF | {fmt(d["n"])} | {fmt(d["U"])} | {fmt(d["vpar"])} | {fmt(d["psi"])} | {fmt(d["Ti"])} | **{fmt(d["ch_mean"])}** |')
    out.append('')

    return '\n'.join(out)


if __name__ == '__main__':
    table = generate_table()
    out_path = os.path.join(RESULTS_DIR, 'EVAL_SUMMARY_TABLE.md')
    with open(out_path, 'w') as f:
        f.write(table)
    print(table)
    print(f'\n\nSaved to: {out_path}')
