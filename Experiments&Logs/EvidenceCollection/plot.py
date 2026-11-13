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
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'legend.fontsize': 8,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'lines.linewidth': 2.0,
        'lines.markersize': 4,
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
    """
    从单个 TensorBoard 日志中提取指定 tag 的数据
    
    参数:
        logdir: str, TensorBoard 日志目录
        tag: str, 需要提取的标签
    
    返回:
        df: DataFrame, 包含'step'和'value'列
    """
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
# 3. 平滑与降采样功能
# =============================================================================
def smooth_with_rolling(values, window_size=20):
    """
    使用 pandas.rolling 进行滑动平均平滑（边界用原始值填充）
    
    参数:
        values: array-like, 原始数据
        window_size: int, 滑动窗口大小
    
    返回:
        smoothed: 平滑后的数据
    """
    series = pd.Series(values)
    smoothed = series.rolling(
        window=window_size,
        center=True,      # 窗口中心对齐当前点
        min_periods=1     # 边界处至少1个点就计算均值
    ).mean().values
    return smoothed


def downsample_with_original_steps(df, interval=20):
    """
    等间隔降采样，保留原始横坐标
    
    参数:
        df: DataFrame, 包含'step'和'value'列（已平滑）
        interval: int, 降采样间隔（每隔interval个点取1个）
    
    返回:
        downsampled_df: DataFrame, 降采样后的数据
    """
    n_original = len(df)
    
    # 生成降采样索引（等间隔，包含首尾）
    downsample_indices = np.linspace(0, n_original - 1, 
                                     num=n_original // interval + 1, 
                                     dtype=int)
    
    # 提取降采样后的数据和对应原始横坐标
    downsampled_steps = df['step'].values[downsample_indices]
    downsampled_values = df['value'].values[downsample_indices]
    
    return pd.DataFrame({
        'step': downsampled_steps,
        'value': downsampled_values
    })


def process_curve(logdir, tag, smooth_window=20, downsample_interval=20):
    """
    完整处理流程：提取 → 平滑 → 降采样
    
    参数:
        logdir: str, TensorBoard 日志目录
        tag: str, 需要提取的标签
        smooth_window: int, 平滑窗口大小
        downsample_interval: int, 降采样间隔
    
    返回:
        df: DataFrame, 处理后的数据（包含原始横坐标'step'和降采样后的'value'）
    """
    # 步骤1: 提取原始数据
    df_raw = extract_tensorboard_data(logdir, tag)
    
    # 步骤2: 平滑
    df_raw['smoothed'] = smooth_with_rolling(df_raw['value'].values, smooth_window)
    
    # 步骤3: 降采样（基于平滑后的值）
    df_down = downsample_with_original_steps(
        df_raw[['step', 'smoothed']].rename(columns={'smoothed': 'value'}), 
        interval=downsample_interval
    )
    
    return df_down

# =============================================================================
# 4. Y 轴自适应范围计算
# =============================================================================
def set_adaptive_ylim(ax, dataframes, margin_ratio=0.1, force_non_negative=False):
    """根据多个 DataFrame 的值自适应设置 Y 轴范围"""
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
# 5. 绘图功能（1行2列：Loss + Accuracy）
# =============================================================================
def plot_training_curves(loss_data_list, acc_data_list, labels, colors,
                         output_path='training_curves.pdf'):
    """
    绘制 Loss 和 Accuracy 训练曲线
    
    参数:
        loss_data_list: list of DataFrame, 4 条 Loss 曲线数据
        acc_data_list: list of DataFrame, 4 条 Accuracy 曲线数据
        labels: list of str, 4 个标签
        colors: list of str, 4 种颜色
        output_path: str, 输出文件路径
    """
    set_neurips_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 3))
    ax_loss, ax_acc = axes
    
    # ========== 子图1：Loss ==========
    for df, label, color in zip(loss_data_list, labels, colors):
        ax_loss.plot(df['step'], df['value'], 
                    color=color, linewidth=2.0, label=label)
    
    ax_loss.set_xlabel('Iteration', fontsize=10)
    ax_loss.set_ylabel('Loss', fontsize=10)
    ax_loss.set_xlim(left=0)
    set_adaptive_ylim(ax_loss, loss_data_list, margin_ratio=0.1, force_non_negative=True)
    ax_loss.grid(axis='y', linestyle='--', alpha=0.3, color='#E0E0E0')
    ax_loss.legend(loc='best', frameon=False, fontsize=8)
    ax_loss.set_title('Training Loss', fontsize=11, fontweight='bold', pad=10)
    
    # ========== 子图2：Accuracy ==========
    for df, label, color in zip(acc_data_list, labels, colors):
        ax_acc.plot(df['step'], df['value'], 
                   color=color, linewidth=2.0, label=label)
    
    ax_acc.set_xlabel('Iteration', fontsize=10)
    ax_acc.set_ylabel('Accuracy', fontsize=10)
    ax_acc.set_xlim(left=0)
    set_adaptive_ylim(ax_acc, acc_data_list, margin_ratio=0.08, force_non_negative=True)
    
    # 确保 Accuracy 不超过 100%
    ymax = ax_acc.get_ylim()[1]
    if ymax > 100:
        ax_acc.set_ylim(0, min(ymax, 105))
    
    ax_acc.grid(axis='y', linestyle='--', alpha=0.3, color='#E0E0E0')
    ax_acc.legend(loc='best', frameon=False, fontsize=8)
    ax_acc.set_title('Training Accuracy', fontsize=11, fontweight='bold', pad=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"✓ 图表已保存：{output_path}")
    plt.show()

# =============================================================================
# 6. 主程序
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Plot training curves from 4 TensorBoard logs with smoothing and downsampling')
    parser.add_argument('--logdirs', type=str, required=True,
                       help='Comma-separated list of 4 TensorBoard log directories')
    parser.add_argument('--output', type=str, default='training_curves.pdf',
                       help='Output PDF filename')
    parser.add_argument('--smooth', type=int, default=20,
                       help='Smoothing window size (default: 20)')
    parser.add_argument('--downsample', type=int, default=20,
                       help='Downsampling interval (default: 20)')
    args = parser.parse_args()
    
    # 配置参数
    LOSS_TAG = 'metrics/loss'
    ACC_TAG = 'metrics/accuracy'
    SMOOTH_WINDOW = args.smooth
    DOWNSAMPLE_INTERVAL = args.downsample
    
    # 标签和配色（按顺序对应 4 个 logdir）
    LABELS = [
        'Feedback Ratio 5%',
        'Feedback Ratio 15%', 
        'Feedback Ratio 25%',
        'Lyapunov-tuned LSM'
    ]
    COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # 解析 4 个 logdir
    if ',' in args.logdirs:
        logdirs = [d.strip() for d in args.logdirs.split(',')]
    else:
        from glob import glob
        logdirs = glob(args.logdirs)
    
    if len(logdirs) != 4:
        print(f"❌ 错误：需要传入 4 个日志目录，当前：{len(logdirs)}")
        exit(1)
    
    # 验证路径存在
    valid_logdirs = [d for d in logdirs if os.path.exists(d)]
    if len(valid_logdirs) != 4:
        print(f"❌ 错误：需要 4 个有效目录，当前有效：{len(valid_logdirs)}")
        exit(1)
    
    print(f"\n{'='*60}")
    print(f"📊 训练曲线可视化 (平滑 + 降采样)")
    print(f"{'='*60}")
    print(f"📁 日志目录：")
    for i, d in enumerate(valid_logdirs):
        print(f"   [{i+1}] {LABELS[i]}: {os.path.basename(d)}")
    print(f"🔧 平滑窗口：{SMOOTH_WINDOW} | 降采样间隔：{DOWNSAMPLE_INTERVAL}")
    print(f"📄 输出文件：{args.output}")
    print(f"{'='*60}\n")
    
    try:
        # ========== 处理 Loss 曲线 ==========
        print("【步骤 1】处理 Loss 曲线...")
        loss_data_list = []
        for logdir in valid_logdirs:
            df = process_curve(logdir, LOSS_TAG, 
                              smooth_window=SMOOTH_WINDOW,
                              downsample_interval=DOWNSAMPLE_INTERVAL)
            loss_data_list.append(df)
        print(f"✓ Loss 降采样后点数：{[len(df) for df in loss_data_list]}\n")
        
        # ========== 处理 Accuracy 曲线 ==========
        print("【步骤 2】处理 Accuracy 曲线...")
        acc_data_list = []
        for logdir in valid_logdirs:
            df = process_curve(logdir, ACC_TAG,
                              smooth_window=SMOOTH_WINDOW,
                              downsample_interval=DOWNSAMPLE_INTERVAL)
            acc_data_list.append(df)
        print(f"✓ Accuracy 降采样后点数：{[len(df) for df in acc_data_list]}\n")
        
        # ========== 绘图 ==========
        print("【步骤 3】绘制图表...")
        plot_training_curves(
            loss_data_list, 
            acc_data_list, 
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
