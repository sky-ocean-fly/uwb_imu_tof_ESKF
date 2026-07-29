#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UWB 距离值分布分析与高斯特性检验
读取 record_uwb_distance.py 录制的 CSV 数据，绘制:
  1. 距离直方图 + 高斯拟合曲线
  2. Q-Q 图（正态性检验）
  3. 距离随时间变化图
并输出统计学指标（均值、方差、偏度、峰度、Shapiro-Wilk 检验等）

用法:
    python3 plot_uwb_distribution.py uwb_distance_20250723_120000.csv
    python3 plot_uwb_distribution.py uwb_data.csv --node-id 1 --bins 50
    python3 plot_uwb_distribution.py uwb_data.csv --save
"""

import argparse
import csv
import sys
import os
from collections import defaultdict

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')  # 无头模式，可替换为 'TkAgg' 交互查看
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec

# ── 全局样式（dataviz 原则: thin marks, recessive grid） ──
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 10,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'grid.linewidth': 0.4,
    'grid.alpha': 0.3,
    'lines.linewidth': 1.8,
    'legend.fontsize': 9,
    'legend.frameon': True,
    'legend.framealpha': 0.85,
    'legend.edgecolor': '#cccccc',
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# ── dataviz 默认色板 ──
C_HIST      = '#5B9BD5'   # 直方图填充 (sequential blue mid)
C_HIST_EDGE = '#2B6A9C'   # 直方图边缘 (sequential blue dark)
C_GAUSS     = '#E15759'   # 高斯拟合曲线 (distinct from blue)
C_QQ_LINE   = '#E15759'   # Q-Q 参考线
C_QQ_POINT  = '#5B9BD5'   # Q-Q 散点
C_TIMESERIES = '#5B9BD5'  # 时间序列线
C_MEAN      = '#F28E2B'   # 均值线


def load_data(csv_path):
    """从 CSV 加载数据，按 node_id 分组"""
    data_by_node = defaultdict(list)
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            node_id = int(float(row['node_id']))
            dis = float(row['dis'])
            ts = float(row['timestamp'])
            data_by_node[node_id].append({'ts': ts, 'dis': dis})
    return data_by_node


def compute_statistics(distances):
    """计算距离数据的统计指标"""
    arr = np.array(distances)
    n = len(arr)
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)  # 样本标准差
    var = np.var(arr, ddof=1)
    skew = stats.skew(arr)
    kurt = stats.kurtosis(arr, fisher=True)  # excess kurtosis

    # 正态性检验
    if n >= 3 and n <= 5000:
        shapiro_stat, shapiro_p = stats.shapiro(arr)
    else:
        shapiro_stat, shapiro_p = np.nan, np.nan

    # Jarque-Bera 检验（适用于大样本）
    jb_stat, jb_p = stats.jarque_bera(arr)

    # 分位数
    q25, q50, q75 = np.percentile(arr, [25, 50, 75])
    iqr = q75 - q25

    return {
        'n': n,
        'mean': mean,
        'std': std,
        'var': var,
        'skewness': skew,
        'kurtosis': kurt,
        'shapiro_stat': shapiro_stat,
        'shapiro_p': shapiro_p,
        'jb_stat': jb_stat,
        'jb_p': jb_p,
        'min': np.min(arr),
        'max': np.max(arr),
        'q25': q25,
        'q50': q50,
        'q75': q75,
        'iqr': iqr,
        'range': np.max(arr) - np.min(arr),
        'cv': std / mean if mean != 0 else np.nan,  # 变异系数
    }


def plot_distribution(records, node_id, output_dir, bins='auto'):
    """绘制距离直方图 + 高斯拟合 + Q-Q 图 + 时间序列
    records: list of {'ts': ..., 'dis': ...} dicts"""
    arr = np.array([r['dis'] for r in records])
    stats_dict = compute_statistics(arr)
    n = stats_dict['n']

    # ── 图1: 直方图 + 高斯拟合 ──
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           width_ratios=[3, 1],
                           height_ratios=[3, 1],
                           hspace=0.30, wspace=0.28)

    ax_hist = fig.add_subplot(gs[0, 0])
    ax_stats = fig.add_subplot(gs[0, 1])
    ax_qq = fig.add_subplot(gs[1, 0])
    ax_ts = fig.add_subplot(gs[1, 1])

    # ─── (a) 直方图 + 高斯拟合 ───
    n_bins = bins if isinstance(bins, int) else max(20, int(np.sqrt(n)))
    hist_counts, hist_edges, _ = ax_hist.hist(
        arr, bins=n_bins, density=True, alpha=0.65,
        color=C_HIST, edgecolor=C_HIST_EDGE, linewidth=0.6, zorder=2
    )

    # 高斯拟合曲线
    x_fit = np.linspace(arr.min(), arr.max(), 300)
    gauss_y = stats.norm.pdf(x_fit, stats_dict['mean'], stats_dict['std'])
    ax_hist.plot(x_fit, gauss_y, color=C_GAUSS, linewidth=2.0,
                 label='Gaussian fit\n($\mu$=%.3f, $\sigma$=%.3f)' % (
                     stats_dict['mean'], stats_dict['std']),
                 zorder=3)

    # 均值线
    ax_hist.axvline(stats_dict['mean'], color=C_MEAN, linewidth=1.2,
                    linestyle='--', alpha=0.85, zorder=3)
    ax_hist.text(stats_dict['mean'], ax_hist.get_ylim()[1] * 0.95,
                 ' $\mu$', color=C_MEAN, fontsize=8, va='top', ha='left')

    ax_hist.set_xlabel('Distance (m)')
    ax_hist.set_ylabel('Density')
    ax_hist.set_title('UWB Distance Distribution (node_id=%d, n=%d)' % (node_id, n))
    ax_hist.legend(loc='upper right', fontsize=8)
    ax_hist.grid(axis='y', alpha=0.3, linewidth=0.4)

    # ─── (b) 统计信息面板 ───
    ax_stats.axis('off')
    stat_lines = [
        ("Statistics", 'title'),
        ("", 'blank'),
        ("n", stats_dict['n']),
        ("Mean", "%.4f m" % stats_dict['mean']),
        ("Std", "%.4f m" % stats_dict['std']),
        ("Var", "%.6f m²" % stats_dict['var']),
        ("CV", "%.4f" % stats_dict['cv']),
        ("Min", "%.4f m" % stats_dict['min']),
        ("Max", "%.4f m" % stats_dict['max']),
        ("Range", "%.4f m" % stats_dict['range']),
        ("Q₁ (25%)", "%.4f m" % stats_dict['q25']),
        ("Median", "%.4f m" % stats_dict['q50']),
        ("Q₃ (75%)", "%.4f m" % stats_dict['q75']),
        ("IQR", "%.4f m" % stats_dict['iqr']),
        ("", 'blank'),
        ("Skewness", "%.4f" % stats_dict['skewness']),
        ("Kurtosis", "%.4f" % stats_dict['kurtosis']),
        ("", 'blank'),
        ("Normality", 'title'),
        ("Shapiro-Wilk p", "%.4f" % stats_dict['shapiro_p'] if not np.isnan(stats_dict['shapiro_p']) else "N/A"),
        ("Jarque-Bera p", "%.4f" % stats_dict['jb_p']),
    ]

    for i, (label, value) in enumerate(stat_lines):
        y = 1.0 - i * 0.047
        if value == 'title':
            ax_stats.text(0.05, y, label, transform=ax_stats.transAxes,
                          fontsize=11, fontweight='bold', va='top',
                          color='#333333')
        elif value == 'blank':
            pass
        else:
            ax_stats.text(0.08, y, label + ':', transform=ax_stats.transAxes,
                          fontsize=8.5, va='top', color='#666666')
            if isinstance(value, float):
                val_str = "%.4f" % value
            else:
                val_str = str(value)
            ax_stats.text(0.95, y, val_str, transform=ax_stats.transAxes,
                          fontsize=8.5, va='top', ha='right',
                          fontweight='bold', color='#222222')

    ax_stats.set_xlim(0, 1)
    ax_stats.set_ylim(0, 1)

    # ─── (c) Q-Q Plot ───
    (osm, osr), (slope, intercept, r) = stats.probplot(arr, dist="norm")
    ax_qq.scatter(osm, osr, s=10, color=C_QQ_POINT, alpha=0.5,
                  edgecolors='none', zorder=2)
    ax_qq.plot(osm, slope * osm + intercept, color=C_QQ_LINE, linewidth=1.8,
               zorder=3, label='$R^2$=%.4f' % (r * r))
    ax_qq.set_xlabel('Theoretical quantiles')
    ax_qq.set_ylabel('Sample quantiles')
    ax_qq.set_title('Q-Q Plot (Normal)')
    ax_qq.legend(loc='upper left', fontsize=9)
    ax_qq.grid(alpha=0.3, linewidth=0.4)

    # ─── (d) 时间序列 ───
    ts_list = [r['ts'] for r in records]
    dis_list = [r['dis'] for r in records]
    # 以第一条记录为时间零点
    if ts_list:
        t0 = ts_list[0]
        rel_ts = [t - t0 for t in ts_list]
        ax_ts.plot(rel_ts, dis_list, color=C_TIMESERIES, linewidth=0.8,
                   alpha=0.7, zorder=2)
        ax_ts.axhline(stats_dict['mean'], color=C_MEAN, linewidth=1.0,
                      linestyle='--', alpha=0.8, label='mean')
        ax_ts.set_xlabel('Time (s)')
        ax_ts.set_ylabel('Distance (m)')
        ax_ts.set_title('Distance vs Time')
        ax_ts.legend(loc='best', fontsize=8)
        ax_ts.grid(alpha=0.3, linewidth=0.4)

    fig.suptitle('UWB Distance Analysis — Node %d' % node_id,
                 fontsize=15, fontweight='bold', y=1.01)

    out_path = os.path.join(output_dir,
                            'uwb_distribution_node%d.png' % node_id)
    fig.savefig(out_path)
    plt.close(fig)
    print("Chart saved: %s" % out_path)

    return stats_dict


def print_report(all_stats):
    """打印汇总报告"""
    print("\n" + "=" * 72)
    print("  UWB 距离分布分析报告 — 高斯特性检验")
    print("=" * 72)
    for node_id, s in all_stats.items():
        print("\n── Node %d ──" % node_id)
        print("  样本量 n           = %d" % s['n'])
        print("  均值 μ             = %.4f m" % s['mean'])
        print("  标准差 σ           = %.4f m" % s['std'])
        print("  方差 σ²            = %.6f m²" % s['var'])
        print("  变异系数 CV        = %.4f" % s['cv'])
        print("  最小值             = %.4f m" % s['min'])
        print("  最大值             = %.4f m" % s['max'])
        print("  Q1 / Median / Q3   = %.4f / %.4f / %.4f m" % (s['q25'], s['q50'], s['q75']))
        print("  IQR                = %.4f m" % s['iqr'])
        print("  ─────────────────────────────")
        print("  偏度 Skewness      = %.4f" % s['skewness'])
        print("  峰度 Kurtosis      = %.4f (excess)" % s['kurtosis'])
        print("  Shapiro-Wilk p     = %s" % (
            "%.6f" % s['shapiro_p'] if not np.isnan(s['shapiro_p']) else 'N/A (n too large)'))
        print("  Jarque-Bera p      = %.6f" % s['jb_p'])

        # 高斯特性判定
        print("  ─────────────────────────────")
        print("  高斯特性评估:")
        skew_ok = abs(s['skewness']) < 0.5
        kurt_ok = abs(s['kurtosis']) < 1.0
        sw_ok = not np.isnan(s['shapiro_p']) and s['shapiro_p'] > 0.05
        jb_ok = s['jb_p'] > 0.05

        if skew_ok and kurt_ok:
            print("    ✓ 偏度和峰度接近正态分布 (|skew|<0.5, |kurt|<1.0)")
        else:
            print("    ✗ 偏度或峰度偏离正态分布")
            if not skew_ok:
                print("      - 偏度=%.4f 超出 ±0.5 范围" % s['skewness'])
            if not kurt_ok:
                print("      - 峰度=%.4f 超出 ±1.0 范围" % s['kurtosis'])

        if sw_ok:
            print("    ✓ Shapiro-Wilk 检验: 不能拒绝正态分布假设 (p=%.4f > 0.05)" % s['shapiro_p'])
        elif not np.isnan(s['shapiro_p']):
            print("    ✗ Shapiro-Wilk 检验: 拒绝正态分布假设 (p=%.4f ≤ 0.05)" % s['shapiro_p'])

        if jb_ok:
            print("    ✓ Jarque-Bera 检验: 不能拒绝正态分布假设 (p=%.4f > 0.05)" % s['jb_p'])
        else:
            print("    ✗ Jarque-Bera 检验: 拒绝正态分布假设 (p=%.6f ≤ 0.05)" % s['jb_p'])

    print("\n" + "=" * 72)


def main():
    parser = argparse.ArgumentParser(
        description='Plot UWB distance distribution with Gaussian analysis')
    parser.add_argument('csv_file', help='Path to CSV file from record_uwb_distance')
    parser.add_argument('--node-id', type=int, default=None,
                        help='Analyze specific node_id only (default: all nodes)')
    parser.add_argument('--bins', type=int, default=50,
                        help='Number of histogram bins (default: 50)')
    parser.add_argument('--output-dir', '-o', default='.',
                        help='Output directory for charts (default: current dir)')
    parser.add_argument('--save', action='store_true', default=True,
                        help='Save charts to PNG (default: True)')
    args = parser.parse_args()

    if not os.path.exists(args.csv_file):
        print("Error: file not found: %s" % args.csv_file, file=sys.stderr)
        sys.exit(1)

    print("Loading data from: %s" % args.csv_file)
    data_by_node = load_data(args.csv_file)

    if not data_by_node:
        print("Error: No data found in CSV file", file=sys.stderr)
        sys.exit(1)

    print("Found %d node(s): %s" % (len(data_by_node),
                                     list(data_by_node.keys())))

    os.makedirs(args.output_dir, exist_ok=True)

    # 过滤
    nodes_to_plot = ([args.node_id] if args.node_id is not None
                     else sorted(data_by_node.keys()))

    all_stats = {}
    for node_id in nodes_to_plot:
        if node_id not in data_by_node:
            print("Warning: node_id=%d not found in data, skipping" % node_id)
            continue
        records = data_by_node[node_id]
        print("\nProcessing node_id=%d: %d records" % (node_id, len(records)))
        stats_dict = plot_distribution(records, node_id, args.output_dir,
                                       bins=args.bins)
        all_stats[node_id] = stats_dict

    print_report(all_stats)


if __name__ == '__main__':
    main()
