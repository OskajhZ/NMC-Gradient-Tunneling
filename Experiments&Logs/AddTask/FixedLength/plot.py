import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

# =============================================================================
# 1. 配置 NeurIPS/NMI 出版级绘图样式
# =============================================================================
def set_neurips_style():
    """设置符合 NeurIPS 会议要求的绘图样式"""
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'legend.fontsize': 10,
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
        print(f"✓ {os.path.basename(logdir)}: 提取 {len(df)} 个点 | 标签: {tag}")
        return df
    else:
        raise ValueError(f"未找到标签：{tag} in {logdir}")

# =============================================================================
# 3. 数据平滑与降采样（复用 plot.py 中的方法）
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
    
    # 步骤 1: 窗口平滑
    smoothed_values = smooth_with_edge_fill(values, window_size)
    
    # 步骤 2: 等间隔降采样
    downsample_indices = np.linspace(0, n_original - 1, target_points, dtype=int)
    
    downsampled_steps = steps[downsample_indices]
    downsampled_values = smoothed_values[downsample_indices]
    
    downsampled_df = pd.DataFrame({
        'step': downsampled_steps,
        'value': downsampled_values
    })
    
    print(f"✓ 降采样完成: {n_original} → {target_points} 个点 | 窗口: {window_size}")
    
    return downsampled_df

# =============================================================================
# 4. 绘图功能：3 条 RMSE 曲线对比（无背景色，固定 ylim）
# =============================================================================
def plot_rmse_comparison(df_list, labels, colors, output_path='rmse_comparison.pdf'):
    """
    绘制 3 条 RMSE 曲线的对比图（出版级样式）
    
    参数:
        df_list: list of DataFrame, 3 个已平滑降采样的数据
        labels: list of str, 3 个标签 ['Warmed-up NMC', 'Original NMC', 'Lyapunov-tuned LSM']
        colors: list of str, 3 种颜色
        output_path: str, 输出文件路径
    """
    set_neurips_style()
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 3))
    
    # 绘制 3 条曲线
    for df, label, color in zip(df_list, labels, colors):
        ax.plot(df['step'], df['value'], 
                color=color, linewidth=2.0, label=label)
    
    # 标题和标签
    ax.set_title('Error with Fixed Effective Length', 
                 fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Root Mean Square Error', fontsize=11)
    
    # 【关键】固定 Y 轴范围为 [0, 0.3]，超出部分自动裁剪
    ax.set_ylim(0.1, 0.2)
    
    # 网格和图例
    ax.grid(axis='y', linestyle='--', alpha=0.3, color='#E0E0E0')
    ax.legend(loc='best', frameon=False, fontsize=9)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"✓ 图表已保存：{output_path}")
    plt.show()

# =============================================================================
# 5. 主程序
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Plot RMSE comparison from 3 TensorBoard logs')
    parser.add_argument('--logdirs', type=str, required=True,
                       help='Comma-separated list of 3 TensorBoard log directories')
    parser.add_argument('--output', type=str, default='rmse_comparison.pdf',
                       help='Output PDF filename')
    parser.add_argument('--smooth', type=int, default=20,
                       help='Smoothing window size (default: 20)')
    parser.add_argument('--downsample', type=int, default=1000,
                       help='Target points after downsampling (default: 1000)')
    args = parser.parse_args()
    
    # 配置参数
    TAG = 'metrics/RMSE'
    SMOOTH_WINDOW = args.smooth
    DOWNSAMPLE_TARGET = args.downsample
    
    # 标签和配色（按顺序对应 3 个 logdir）
    LABELS = [
        'Warmed-up NMC',
        'Original NMC', 
        'Lyapunov-tuned LSM'
    ]
    COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 蓝/橙/绿
    
    # 解析 3 个 logdir
    if ',' in args.logdirs:
        logdirs = [d.strip() for d in args.logdirs.split(',')]
    else:
        from glob import glob
        logdirs = glob(args.logdirs)
    
    if len(logdirs) != 3:
        print(f"❌ 错误：需要传入 3 个日志目录，当前：{len(logdirs)}")
        exit(1)
    
    # 验证路径存在
    valid_logdirs = [d for d in logdirs if os.path.exists(d)]
    if len(valid_logdirs) != 3:
        print(f"❌ 错误：需要 3 个有效目录，当前有效：{len(valid_logdirs)}")
        exit(1)
    
    print(f"\n{'='*60}")
    print(f"📊 RMSE 对比可视化")
    print(f"{'='*60}")
    print(f"📁 日志目录：")
    for i, d in enumerate(valid_logdirs):
        print(f"   [{i+1}] {LABELS[i]}: {os.path.basename(d)}")
    print(f"🔧 平滑窗口：{SMOOTH_WINDOW} | 降采样点数：{DOWNSAMPLE_TARGET}")
    print(f"📄 输出文件：{args.output}")
    print(f"{'='*60}\n")
    
    try:
        # 步骤 1: 提取 + 平滑 + 降采样
        print("【步骤 1】处理数据...")
        df_list = []
        for logdir in valid_logdirs:
            df_raw = extract_tensorboard_data(logdir, TAG)
            df_processed = smooth_and_downsample(
                df_raw, 
                window_size=SMOOTH_WINDOW, 
                target_points=DOWNSAMPLE_TARGET
            )
            df_list.append(df_processed)
        
        # 步骤 2: 绘图
        print("\n【步骤 2】绘制对比图...")
        plot_rmse_comparison(
            df_list=df_list,
            labels=LABELS,
            colors=COLORS,
            output_path=args.output
        )
        
        print(f"\n{'='*60}")
        print(f"✅ 图表生成完成！")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"\n❌ 错误：{e}")
        import traceback
        traceback.print_exc()
        exit(1)
