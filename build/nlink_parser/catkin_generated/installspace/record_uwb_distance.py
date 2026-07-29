#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UWB 距离录制 + 分布分析（录制结束即自动分析）

功能:
  1. 订阅 nlink_parser 的 nodeframe 话题，实时记录距离数据到 CSV
  2. Ctrl+C 停止录制后，自动分析数据并生成:
     - 距离直方图 + 高斯拟合曲线
     - Q-Q 图（正态性检验）
     - 距离随时间变化图
     - 统计指标报告（均值、方差、偏度、峰度、Shapiro-Wilk 检验等）

用法:
    rosrun nlink_parser record_uwb_distance.py
    rosrun nlink_parser record_uwb_distance.py _topic:=/nlink_linktrack_nodeframe2
    rosrun nlink_parser record_uwb_distance.py _topic:=/nlink_linktrack_nodeframe3 _bins:=80
"""

import rospy
import csv
import os
import signal
import sys
from datetime import datetime
from collections import defaultdict

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# 尝试导入消息类型
try:
    from nlink_parser.msg import LinktrackNodeframe2, LinktrackNodeframe3, LinktrackNodeframe4
except ImportError:
    rospy.logwarn("Cannot import nlink_parser messages, trying dynamic import...")

# ── 全局变量 ──
g_data_records = []
g_output_file = ""
g_csv_writer = None
g_csv_file = None
g_output_dir = "."
g_recording_stopped = False  # 防止关闭 CSV 后回调继续写入

# ── matplotlib 全局样式 ──
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

# ── 色板 ──
C_HIST       = '#5B9BD5'
C_HIST_EDGE  = '#2B6A9C'
C_GAUSS      = '#E15759'
C_QQ_POINT   = '#5B9BD5'
C_TIMESERIES = '#5B9BD5'
C_MEAN       = '#F28E2B'


# ══════════════════════════════════════════════════════════════════════
# 录制部分
# ══════════════════════════════════════════════════════════════════════

def callback_nodeframe2(msg):
    """处理 LinktrackNodeframe2/3 消息"""
    if g_recording_stopped:
        return
    for node in msg.nodes:
        record = {
            'timestamp': rospy.get_time(),
            'frame_id': msg.id,
            'node_role': node.role,
            'node_id': node.id,
            'dis': node.dis,
            'fp_rssi': node.fp_rssi,
            'rx_rssi': node.rx_rssi
        }
        g_data_records.append(record)
        if g_csv_writer:
            g_csv_writer.writerow([
                record['timestamp'], record['frame_id'],
                record['node_role'], record['node_id'],
                record['dis'], record['fp_rssi'], record['rx_rssi']
            ])


def callback_nodeframe3(msg):
    callback_nodeframe2(msg)


def callback_nodeframe4(msg):
    """处理 LinktrackNodeframe4 消息（tag→anchor 距离）"""
    if g_recording_stopped:
        return
    for tag in msg.tags:
        for anchor in tag.anchors:
            record = {
                'timestamp': rospy.get_time(),
                'frame_id': msg.id,
                'node_role': 'tag',
                'node_id': tag.id,
                'dis': anchor.dis,
                'fp_rssi': 0.0,
                'rx_rssi': 0.0
            }
            g_data_records.append(record)
            if g_csv_writer:
                g_csv_writer.writerow([
                    record['timestamp'], record['frame_id'],
                    record['node_role'], record['node_id'],
                    record['dis'], record['fp_rssi'], record['rx_rssi']
                ])


# ══════════════════════════════════════════════════════════════════════
# 分析部分
# ══════════════════════════════════════════════════════════════════════

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
    std = np.std(arr, ddof=1)
    var = np.var(arr, ddof=1)
    skew = stats.skew(arr)
    kurt = stats.kurtosis(arr, fisher=True)

    if 3 <= n <= 5000:
        shapiro_stat, shapiro_p = stats.shapiro(arr)
    else:
        shapiro_stat, shapiro_p = np.nan, np.nan

    jb_stat, jb_p = stats.jarque_bera(arr)

    q25, q50, q75 = np.percentile(arr, [25, 50, 75])
    iqr = q75 - q25

    return {
        'n': n, 'mean': mean, 'std': std, 'var': var,
        'skewness': skew, 'kurtosis': kurt,
        'shapiro_stat': shapiro_stat, 'shapiro_p': shapiro_p,
        'jb_stat': jb_stat, 'jb_p': jb_p,
        'min': np.min(arr), 'max': np.max(arr),
        'q25': q25, 'q50': q50, 'q75': q75, 'iqr': iqr,
        'range': np.max(arr) - np.min(arr),
        'cv': std / mean if mean != 0 else np.nan,
    }


def plot_distribution(records, node_id, output_dir, bins=50):
    """绘制距离直方图 + 高斯拟合 + Q-Q 图 + 时间序列
    records: list of {'ts': ..., 'dis': ...} dicts"""
    arr = np.array([r['dis'] for r in records])
    s = compute_statistics(arr)
    n = s['n']

    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           width_ratios=[3, 1], height_ratios=[3, 1],
                           hspace=0.30, wspace=0.28)

    ax_hist  = fig.add_subplot(gs[0, 0])
    ax_stats = fig.add_subplot(gs[0, 1])
    ax_qq    = fig.add_subplot(gs[1, 0])
    ax_ts    = fig.add_subplot(gs[1, 1])

    # ── (a) 直方图 + 高斯拟合 ──
    n_bins = bins if isinstance(bins, int) else max(20, int(np.sqrt(n)))
    ax_hist.hist(arr, bins=n_bins, density=True, alpha=0.65,
                 color=C_HIST, edgecolor=C_HIST_EDGE, linewidth=0.6, zorder=2)

    x_fit = np.linspace(arr.min(), arr.max(), 300)
    gauss_y = stats.norm.pdf(x_fit, s['mean'], s['std'])
    ax_hist.plot(x_fit, gauss_y, color=C_GAUSS, linewidth=2.0,
                 label='Gaussian fit\n($\mu$=%.3f, $\sigma$=%.3f)' % (s['mean'], s['std']),
                 zorder=3)

    ax_hist.axvline(s['mean'], color=C_MEAN, linewidth=1.2,
                    linestyle='--', alpha=0.85, zorder=3)
    ax_hist.text(s['mean'], ax_hist.get_ylim()[1] * 0.95,
                 ' $\mu$', color=C_MEAN, fontsize=8, va='top', ha='left')

    ax_hist.set_xlabel('Distance (m)')
    ax_hist.set_ylabel('Density')
    ax_hist.set_title('UWB Distance Distribution (node_id=%d, n=%d)' % (node_id, n))
    ax_hist.legend(loc='upper right', fontsize=8)
    ax_hist.grid(axis='y', alpha=0.3, linewidth=0.4)

    # ── (b) 统计信息面板 ──
    ax_stats.axis('off')
    stat_lines = [
        ("Statistics", 'title'),
        ("", 'blank'),
        ("n",              s['n']),
        ("Mean",           "%.4f m" % s['mean']),
        ("Std",            "%.4f m" % s['std']),
        ("Var",            "%.6f m²" % s['var']),
        ("CV",             "%.4f" % s['cv']),
        ("Min",            "%.4f m" % s['min']),
        ("Max",            "%.4f m" % s['max']),
        ("Range",          "%.4f m" % s['range']),
        ("Q₁ (25%)",       "%.4f m" % s['q25']),
        ("Median",         "%.4f m" % s['q50']),
        ("Q₃ (75%)",       "%.4f m" % s['q75']),
        ("IQR",            "%.4f m" % s['iqr']),
        ("", 'blank'),
        ("Skewness",       "%.4f" % s['skewness']),
        ("Kurtosis",       "%.4f" % s['kurtosis']),
        ("", 'blank'),
        ("Normality", 'title'),
        ("Shapiro-Wilk p", "%.4f" % s['shapiro_p'] if not np.isnan(s['shapiro_p']) else "N/A"),
        ("Jarque-Bera p",  "%.4f" % s['jb_p']),
    ]

    for i, (label, value) in enumerate(stat_lines):
        y = 1.0 - i * 0.047
        if value == 'title':
            ax_stats.text(0.05, y, label, transform=ax_stats.transAxes,
                          fontsize=11, fontweight='bold', va='top', color='#333333')
        elif value == 'blank':
            pass
        else:
            ax_stats.text(0.08, y, label + ':', transform=ax_stats.transAxes,
                          fontsize=8.5, va='top', color='#666666')
            val_str = "%.4f" % value if isinstance(value, float) else str(value)
            ax_stats.text(0.95, y, val_str, transform=ax_stats.transAxes,
                          fontsize=8.5, va='top', ha='right',
                          fontweight='bold', color='#222222')

    ax_stats.set_xlim(0, 1)
    ax_stats.set_ylim(0, 1)

    # ── (c) Q-Q Plot ──
    (osm, osr), (slope, intercept, r) = stats.probplot(arr, dist="norm")
    ax_qq.scatter(osm, osr, s=10, color=C_QQ_POINT, alpha=0.5,
                  edgecolors='none', zorder=2)
    ax_qq.plot(osm, slope * osm + intercept, color=C_GAUSS, linewidth=1.8,
               zorder=3, label='$R^2$=%.4f' % (r * r))
    ax_qq.set_xlabel('Theoretical quantiles')
    ax_qq.set_ylabel('Sample quantiles')
    ax_qq.set_title('Q-Q Plot (Normal)')
    ax_qq.legend(loc='upper left', fontsize=9)
    ax_qq.grid(alpha=0.3, linewidth=0.4)

    # ── (d) 时间序列 ──
    ts_list  = [r['ts'] for r in records]
    dis_list = [r['dis'] for r in records]
    if ts_list:
        t0 = ts_list[0]
        rel_ts = [t - t0 for t in ts_list]
        ax_ts.plot(rel_ts, dis_list, color=C_TIMESERIES, linewidth=0.8,
                   alpha=0.7, zorder=2)
        ax_ts.axhline(s['mean'], color=C_MEAN, linewidth=1.0,
                      linestyle='--', alpha=0.8, label='mean')
        ax_ts.set_xlabel('Time (s)')
        ax_ts.set_ylabel('Distance (m)')
        ax_ts.set_title('Distance vs Time')
        ax_ts.legend(loc='best', fontsize=8)
        ax_ts.grid(alpha=0.3, linewidth=0.4)

    fig.suptitle('UWB Distance Analysis — Node %d' % node_id,
                 fontsize=15, fontweight='bold', y=1.01)

    out_path = os.path.join(output_dir, 'uwb_distribution_node%d.png' % node_id)
    fig.savefig(out_path)
    plt.close(fig)
    print("  Chart saved: %s" % out_path)

    return s


def print_report(all_stats):
    """打印统计报告"""
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
        print("  最小值 / 最大值    = %.4f / %.4f m" % (s['min'], s['max']))
        print("  Q1 / Median / Q3   = %.4f / %.4f / %.4f m" % (s['q25'], s['q50'], s['q75']))
        print("  IQR                = %.4f m" % s['iqr'])
        print("  ─────────────────────────────")
        print("  偏度 Skewness      = %.4f" % s['skewness'])
        print("  峰度 Kurtosis      = %.4f (excess)" % s['kurtosis'])
        print("  Shapiro-Wilk p     = %s" % (
            "%.6f" % s['shapiro_p'] if not np.isnan(s['shapiro_p']) else 'N/A (n too large)'))
        print("  Jarque-Bera p      = %.6f" % s['jb_p'])

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


def run_analysis(csv_path, output_dir, bins=50, node_id_filter=None):
    """运行完整分析: 加载数据 → 绘图 → 打印报告"""
    print("\n" + "-" * 60)
    print("  开始分析 UWB 距离数据...")
    print("-" * 60)

    if not os.path.exists(csv_path):
        print("Error: file not found: %s" % csv_path, file=sys.stderr)
        return

    data_by_node = load_data(csv_path)
    if not data_by_node:
        print("Error: No data found in CSV file", file=sys.stderr)
        return

    print("Found %d node(s): %s" % (len(data_by_node), sorted(data_by_node.keys())))

    os.makedirs(output_dir, exist_ok=True)

    nodes_to_plot = ([node_id_filter] if node_id_filter is not None
                     else sorted(data_by_node.keys()))

    all_stats = {}
    for node_id in nodes_to_plot:
        if node_id not in data_by_node:
            print("Warning: node_id=%d not found, skipping" % node_id)
            continue

        records = data_by_node[node_id]
        print("\nProcessing node_id=%d: %d records" % (node_id, len(records)))
        s = plot_distribution(records, node_id, output_dir, bins=bins)
        all_stats[node_id] = s

    print_report(all_stats)


# ══════════════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════════════

def save_and_exit(signum=None, frame=None):
    """Ctrl+C 处理: 保存 CSV → 自动分析 → 退出"""
    global g_csv_file, g_recording_stopped

    # 先标记停止，防止后续回调写入已关闭的文件
    g_recording_stopped = True

    if g_csv_file:
        g_csv_file.close()
        g_csv_file = None

    n_total = len(g_data_records)
    print("\n\n录制已停止，共 %d 条记录" % n_total)
    print("数据已保存: %s" % g_output_file)

    if n_total == 0:
        print("没有数据，跳过分析")
        sys.exit(0)

    # 获取参数
    bins = rospy.get_param('~bins', 50)
    node_id_filter = rospy.get_param('~node_id', None)

    # 运行分析
    run_analysis(g_output_file, g_output_dir, bins=bins, node_id_filter=node_id_filter)

    print("\n完成。")
    sys.exit(0)


def main():
    global g_output_file, g_csv_writer, g_csv_file, g_output_dir

    rospy.init_node('uwb_distance_recorder', anonymous=True)

    # 参数
    topic = rospy.get_param('~topic', '/nlink_linktrack_nodeframe2')
    g_output_file = rospy.get_param('~output', '')
    g_output_dir  = rospy.get_param('~output_dir', '')

    if not g_output_file:
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        g_output_file = os.path.join(
            os.path.expanduser('~'), 'uwb_distance_%s.csv' % timestamp_str
        )

    if not g_output_dir:
        g_output_dir = os.path.dirname(g_output_file)

    # 打开 CSV 文件
    g_csv_file = open(g_output_file, 'w', newline='')
    g_csv_writer = csv.writer(g_csv_file)
    g_csv_writer.writerow(['timestamp', 'frame_id', 'node_role', 'node_id',
                           'dis', 'fp_rssi', 'rx_rssi'])

    # 根据 topic 选择消息类型和回调
    topic_lower = topic.lower()
    if 'nodeframe4' in topic_lower:
        msg_type = LinktrackNodeframe4
        callback = callback_nodeframe4
    elif 'nodeframe2' in topic_lower:
        msg_type = LinktrackNodeframe2
        callback = callback_nodeframe2
    else:
        msg_type = LinktrackNodeframe3
        callback = callback_nodeframe3

    # 信号处理: Ctrl+C → 保存并分析
    signal.signal(signal.SIGINT, save_and_exit)
    signal.signal(signal.SIGTERM, save_and_exit)

    # 订阅话题
    rospy.Subscriber(topic, msg_type, callback)

    rospy.loginfo("=" * 60)
    rospy.loginfo("UWB Distance Recorder + Analyzer")
    rospy.loginfo("  订阅话题: %s", topic)
    rospy.loginfo("  数据保存: %s", g_output_file)
    rospy.loginfo("  图表输出: %s", g_output_dir)
    rospy.loginfo("  Ctrl+C 停止录制并自动分析...")
    rospy.loginfo("=" * 60)

    rospy.spin()


if __name__ == '__main__':
    main()
