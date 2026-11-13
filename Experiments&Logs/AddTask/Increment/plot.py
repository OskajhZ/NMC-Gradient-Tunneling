import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

# =============================================================================
# 1. 配置 NeurIPS 出版级绘图样式
# =============================================================================
def set_neurips_style():
    """设置符合 NeurIPS 会议要求的绘图样式"""
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'legend.fontsize': 9,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'lines.linewidth': 2.0,
        'figure.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.format': 'pdf',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'grid.linestyle': '--',
        'grid.alpha': 0.3,
        'axes.axisbelow': True,
        'grid.color': '#E0E0E0',
    })

# =============================================================================
# 2. 数据提取功能
# =============================================================================
def extract_tensorboard_data(logdir, tag):
    """从 TensorBoard 日志中提取指定 tag 的数据"""
    if not os.path.exists(logdir):
        raise FileNotFoundError(f"日志目录不存在：{logdir}")
    
    ea = event_accumulator.EventAccumulator(
        logdir, 
        size_guidance={event_accumulator.SCALARS: 0}
    )
    ea.Reload()
    
    if tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        df = pd.DataFrame([{'step': e.step, 'value': e.value} for e in events])
        print(f"✓ {logdir}: 提取 {len(df)} 个数据点 | 标签：{tag}")
        return df
    else:
        raise ValueError(f"未找到标签：{tag} in {logdir}")

# =============================================================================
# 3. 数据平滑与降采样（核心功能① - 修复边界暴跌问题）
# =============================================================================
def smooth_with_edge_fill(values, window_size=20):
    """
    对数据进行窗口平滑，边界处用原始值填充（避免暴跌）
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    
    if n == 0:
        return values
    
    series = pd.Series(values)
    smoothed = series.rolling(
        window=window_size,
        center=True,
        min_periods=1
    ).mean().values
    
    return smoothed


def smooth_and_downsample(df, window_size=20, target_points=1000):
    """
    对数据进行窗口平滑和降采样
    """
    values = df['value'].values
    steps = df['step'].values
    n_original = len(values)
    
    smoothed_values = smooth_with_edge_fill(values, window_size)
    
    downsample_indices = np.linspace(0, n_original-1, target_points, dtype=int)
    
    downsampled_steps = steps[downsample_indices]
    downsampled_values = smoothed_values[downsample_indices]
    
    downsampled_df = pd.DataFrame({
        'step': downsampled_steps,
        'value': downsampled_values
    })
    
    print(f"✓ 降采样完成：{n_original} → {target_points} 个点 | 窗口：{window_size} | 边界填充：edge-value")
    
    return downsampled_df, downsampled_steps

# =============================================================================
# 4. 背景色设置功能
# =============================================================================
def set_alternating_background(ax, boundaries, colors):
    """设置交替背景色"""
    for i in range(len(boundaries)-1):
        ax.axvspan(boundaries[i], boundaries[i+1], 
                   color=colors[i % len(colors)], 
                   alpha=0.3, 
                   zorder=0)

# =============================================================================
# 5. Y 轴自适应范围计算
# =============================================================================
def set_adaptive_ylim(ax, dataframes, margin_ratio=0.05, force_non_negative=True):
    """根据数据自适应设置 Y 轴范围，使曲线充斥画面"""
    all_values = []
    
    for df in dataframes:
        if not df.empty and 'value' in df.columns:
            valid = df['value'].dropna().values
            if len(valid) > 0:
                all_values.extend(valid)
    
    if len(all_values) == 0:
        return
    
    min_val, max_val = np.min(all_values), np.max(all_values)
    range_val = max_val - min_val
    
    if range_val < 1e-8:
        range_val = abs(max_val) * 0.2 + 1e-8
    
    margin = range_val * margin_ratio
    new_min = min_val - margin
    new_max = max_val + margin
    
    if force_non_negative and new_min < 0:
        new_min = 0
    
    ax.set_ylim(new_min, new_max)

# =============================================================================
# 6. 计算指定点的均值和标准误（核心功能②）
# =============================================================================
def compute_mean_se_at_points(df, display_steps, window_before=10, 
                              total_original_steps=20000, max_display_step=75):
    """
    在指定显示横坐标处，计算对应原始数据位置的均值和标准误
    """
    means, ses = [], []
    valid_display_steps = []
    
    for display_step in display_steps:
        original_idx = int(np.round(display_step / max_display_step * total_original_steps))
        original_idx = min(original_idx, total_original_steps - 1)
        
        start_idx = max(0, original_idx - window_before)
        end_idx = original_idx
        
        if end_idx > start_idx:
            values_window = df['value'].values[start_idx:end_idx]
            mean_val = np.mean(values_window)
            std_val = np.std(values_window, ddof=1)
            se_val = std_val / np.sqrt(len(values_window))
            
            means.append(mean_val)
            ses.append(se_val)
            valid_display_steps.append(display_step)
        else:
            print(f"⚠ 无法提取显示横坐标 {display_step} (原始索引 {original_idx}) 处的数据窗口")
    
    result_df = pd.DataFrame({
        'step': valid_display_steps,
        'mean': means,
        'se': ses
    })
    
    print(f"✓ 计算均值和标准误：{len(valid_display_steps)} 个点 | 窗口大小：{window_before}")
    
    return result_df

# =============================================================================
# 7. 绘图功能①：原始数据绘图（画布 1）
# =============================================================================
def plot_full_training_error(df1, df2, label1, label2, color1, color2, 
                            output_path='figure1_full_training.pdf'):
    """绘制完整训练误差曲线（带交替背景色）"""
    set_neurips_style()
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 3))
    
    boundaries = [0, 4000, 8000, 12000, 16000, 20000]
    colors = ['#ADD8E6', '#E6F3FF']
    set_alternating_background(ax, boundaries, colors)
    
    ax.plot(df1['step'], df1['value'], 
            color=color1, linewidth=2.0, label=label1, zorder=3)
    ax.plot(df2['step'], df2['value'], 
            color=color2, linewidth=2.0, label=label2, zorder=3)
    
    ax.set_title('Error during Add Task Training', 
                 fontsize=14, fontweight='bold', pad=10)
    ax.set_ylabel('Root Mean Square Error', fontsize=11, fontweight='medium')
    ax.set_xlim(0, 20000)
    
    ax.grid(axis='y', linestyle='--', alpha=0.3, color='#E0E0E0')
    ax.legend(loc='best', frameon=False, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"✓ 图 1 已保存：{output_path}")
    plt.show()

# =============================================================================
# 8. 绘图功能②：部分数据绘图（画布 2）- 修复版本
# =============================================================================
def plot_partial_training_error(df1, df2, label1, label2, color1, color2,
                               x_min=3000, x_max=20000,
                               output_path='figure2_partial_training.pdf'):
    """
    绘制部分训练误差曲线（截取 3000-20000，纵坐标自适应，带交替背景色）
    """
    set_neurips_style()
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 3))
    
    # 定义全局边界（与画布 1 完全一致）
    global_boundaries = [0, 4000, 8000, 12000, 16000, 20000]
    colors = ['#ADD8E6', '#E6F3FF']
    
    # 只绘制在当前视图范围内的背景色块
    for i in range(len(global_boundaries)-1):
        start = global_boundaries[i]
        end = global_boundaries[i+1]
        
        visible_start = max(start, x_min)
        visible_end = min(end, x_max)
        
        if visible_end > visible_start:
            ax.axvspan(visible_start, visible_end, 
                       color=colors[i % len(colors)], 
                       alpha=0.3, 
                       zorder=0)
    
    # 截取数据
    df1_partial = df1[(df1['step'] >= x_min) & (df1['step'] <= x_max)]
    df2_partial = df2[(df2['step'] >= x_min) & (df2['step'] <= x_max)]
    
    # 绘制两条曲线
    ax.plot(df1_partial['step'], df1_partial['value'], 
            color=color1, linewidth=2.0, label=label1, zorder=3)
    ax.plot(df2_partial['step'], df2_partial['value'], 
            color=color2, linewidth=2.0, label=label2, zorder=3)
    
    ax.set_title('Error during Add Task Training (Partial View)', 
                 fontsize=14, fontweight='bold', pad=10)
    ax.set_ylabel('Root Mean Square Error', fontsize=11, fontweight='medium')
    ax.set_xlabel('Iteration', fontsize=11, fontweight='medium')
    ax.set_xlim(x_min, x_max)
    
    set_adaptive_ylim(ax, [df1_partial, df2_partial], margin_ratio=0.03)
    
    ax.grid(axis='y', linestyle='--', alpha=0.3, color='#E0E0E0')
    ax.legend(loc='best', frameon=False, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"✓ 图 2 已保存：{output_path}")
    plt.show()

# =============================================================================
# 9. 绘图功能③：带标准误的折线图（画布 3）- 修复版本
# =============================================================================
def plot_mean_se_with_errorbars(df1, df2, label1, label2, color1, color2,
                                display_steps=[15, 30, 45, 60, 75],
                                output_path='figure3_mean_se.pdf'):
    """
    绘制带标准误的折线图（5 个点，横坐标为有效序列长度）
    """
    set_neurips_style()
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 3))
    
    result1 = compute_mean_se_at_points(df1, display_steps, window_before=10)
    result2 = compute_mean_se_at_points(df2, display_steps, window_before=10)
    
    ax.errorbar(result1['step'], result1['mean'], 
                yerr=result1['se'],
                color=color1, linewidth=2.0, marker='o', markersize=8,
                capsize=5, elinewidth=1.5, label=label1, zorder=3)
    ax.errorbar(result2['step'], result2['mean'], 
                yerr=result2['se'],
                color=color2, linewidth=2.0, marker='o', markersize=8,
                capsize=5, elinewidth=1.5, label=label2, zorder=3)
    
    ax.set_title('Final Error for Different Sequence Length', 
                 fontsize=14, fontweight='bold', pad=10)
    ax.set_ylabel('Root Mean Square Error', fontsize=11, fontweight='medium')
    ax.set_xlabel('Effective Sequence Length', fontsize=11, fontweight='medium')
    
    ax.set_xticks(display_steps)
    ax.set_xlim(min(display_steps) - 2, max(display_steps) + 2)
    
    all_values = []
    for df in [result1, result2]:
        all_values.extend((df['mean'] - df['se']).values)
        all_values.extend((df['mean'] + df['se']).values)
    
    min_val, max_val = np.min(all_values), np.max(all_values)
    range_val = max_val - min_val
    margin = range_val * 0.05
    ax.set_ylim(max(0, min_val - margin), max_val + margin)
    
    # ========== 关键修改：优化网格样式（和其他图保持一致） ==========
    # ========== SVG 兼容版网格（永不消失，首选） ==========
    ax.grid(axis='both', linestyle='--', alpha=1, color='#C0C0C0', linewidth=0.8)
    ax.set_axisbelow(True)
# ======================================================
    # ==============================================================
    
    ax.legend(loc='best', frameon=False, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"✓ 图 3 已保存：{output_path}")
    plt.show()

# =============================================================================
# 10. 主程序 - 修复版本
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Plot RMSE training curves with downsampling and SE analysis')
    parser.add_argument('--logdir1', type=str, required=True,
                       help='First TensorBoard log directory (GT-trained)')
    parser.add_argument('--logdir2', type=str, required=True,
                       help='Second TensorBoard log directory (Lyapunov-tuned)')
    parser.add_argument('--tag', type=str, default='metrics/RMSE',
                       help='TensorBoard tag to extract')
    parser.add_argument('--output-prefix', type=str, default='rmse_analysis',
                       help='Output file prefix')
    parser.add_argument('--window', type=int, default=20,
                       help='Smoothing window size (default: 20)')
    args = parser.parse_args()
    
    TAG = args.tag
    WINDOW_SIZE = args.window
    TARGET_POINTS = 1000
    COLOR1 = '#1f77b4'
    COLOR2 = '#ff7f0e'
    LABEL1 = 'GT-trained Feedback NMC'
    LABEL2 = 'Lyapnov-tuned LSM'
    
    print(f"\n{'='*60}")
    print(f"📊 RMSE Training Curve Analysis")
    print(f"{'='*60}")
    print(f"📁 Log Directory 1: {args.logdir1}")
    print(f"📁 Log Directory 2: {args.logdir2}")
    print(f"🏷️  Tag: {TAG}")
    print(f"🔧 Smooth Window: {WINDOW_SIZE} (edge-value fill)")
    print(f"📉 Target Points: {TARGET_POINTS}")
    print(f"{'='*60}\n")
    
    try:
        print("【步骤 1】提取 TensorBoard 原始数据...")
        df1_raw = extract_tensorboard_data(args.logdir1, TAG)
        df2_raw = extract_tensorboard_data(args.logdir2, TAG)
        
        if len(df1_raw) != 20000 or len(df2_raw) != 20000:
            print(f"⚠ 警告：数据点数量不是 20000 (df1: {len(df1_raw)}, df2: {len(df2_raw)})")
        
        print("\n【步骤 2】数据平滑与降采样...")
        print(f"   平滑策略：pandas.rolling(center=True, min_periods=1)")
        print(f"   边界处理：用原始值填充（避免补 0 导致暴跌）")
        
        df1_down, steps1 = smooth_and_downsample(df1_raw, WINDOW_SIZE, TARGET_POINTS)
        df2_down, steps2 = smooth_and_downsample(df2_raw, WINDOW_SIZE, TARGET_POINTS)
        
        print("\n【步骤 3】生成图表...")
        
        plot_full_training_error(
            df1_down, df2_down, LABEL1, LABEL2, COLOR1, COLOR2,
            output_path=f'{args.output_prefix}_figure1_full.svg'
        )
        
        plot_partial_training_error(
            df1_down, df2_down, LABEL1, LABEL2, COLOR1, COLOR2,
            x_min=3000, x_max=20000,
            output_path=f'{args.output_prefix}_figure2_partial.svg'
        )
        
        plot_mean_se_with_errorbars(
            df1_raw, df2_raw, LABEL1, LABEL2, COLOR1, COLOR2,
            display_steps=[15, 30, 45, 60, 75],
            output_path=f'{args.output_prefix}_figure3_mean_se.svg'
        )
        
        print(f"\n{'='*60}")
        print(f"✅ 所有图表生成完成！")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"\n❌ 错误：{e}")
        import traceback
        traceback.print_exc()
        exit(1)
