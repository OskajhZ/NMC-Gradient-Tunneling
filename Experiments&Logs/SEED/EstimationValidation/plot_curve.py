import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
from glob import glob

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
        'legend.fontsize': 9,
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
def extract_tensorboard_data(logdir, tags):
    """从单个 TensorBoard 日志中提取指定 tag 的数据"""
    if not os.path.exists(logdir):
        raise FileNotFoundError(f"日志目录不存在：{logdir}")
    
    ea = event_accumulator.EventAccumulator(
        logdir, 
        size_guidance={event_accumulator.SCALARS: 0}
    )
    ea.Reload()
    
    data_dict = {}
    
    for tag in tags:
        if tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            df = pd.DataFrame([{'step': e.step, 'value': e.value} for e in events])
            
            if len(df) > 0:
                data_dict[tag] = df
            else:
                data_dict[tag] = pd.DataFrame(columns=['step', 'value'])
        else:
            print(f"⚠ 未找到标签：{tag} in {logdir}")
            data_dict[tag] = pd.DataFrame(columns=['step', 'value'])
    
    return data_dict

# =============================================================================
# 3. 多 log 聚合：按组计算均值和标准误（核心修改）
# =============================================================================
def aggregate_logs_by_group(logdir_groups, tags):
    """
    从多组 logdir 提取数据，每组计算均值和标准误（SE）
    
    参数:
        logdir_groups: list of list of str, 每组包含4个日志目录
        tags: list of str, 需要提取的标签
    
    返回:
        aggregated_data: dict, 每个 tag 对应一个 dict，包含
                        每个组的 'mean', 'se', 'ci_lower', 'ci_upper'
    """
    print(f"\n📊 聚合 {len(logdir_groups)} 组实验日志（每组 {len(logdir_groups[0])} 个种子）...")
    
    aggregated_data = {tag: {} for tag in tags}
    
    group_names = ['Feedback Ratio 5%', 'Feedback Ratio 15%', 'Feedback Ratio 25%']
    
    for group_idx, logdirs in enumerate(logdir_groups):
        group_name = group_names[group_idx]
        print(f"\n  【组 {group_idx+1}】{group_name}:")
        
        all_logs_data = {tag: [] for tag in tags}
        
        # 从该组的每个 logdir 提取数据
        for i, logdir in enumerate(logdirs):
            print(f"    [{i+1}/{len(logdirs)}] 读取：{os.path.basename(logdir)}")
            data = extract_tensorboard_data(logdir, tags)
            
            for tag in tags:
                if not data[tag].empty:
                    all_logs_data[tag].append(data[tag])
        
        # 对每个 tag 计算均值和标准误
        for tag in tags:
            log_dfs = all_logs_data[tag]
            
            if len(log_dfs) == 0:
                print(f"    ⚠ {tag}: 无有效数据")
                continue
            
            # 确保所有 log 的 step 对齐（取交集）
            all_steps = log_dfs[0]['step'].values
            for df in log_dfs[1:]:
                all_steps = np.intersect1d(all_steps, df['step'].values)
            
            if len(all_steps) == 0:
                print(f"    ⚠ {tag}: step 无交集")
                continue
            
            # 对每个 step 计算均值和标准误
            means, ses = [], []
            n_logs = len(log_dfs)
            
            for step in all_steps:
                values_at_step = []
                for df in log_dfs:
                    row = df[df['step'] == step]
                    if len(row) > 0:
                        values_at_step.append(row['value'].values[0])
                
                if len(values_at_step) >= 2:
                    mean_val = np.mean(values_at_step)
                    std_val = np.std(values_at_step, ddof=1)
                    se_val = std_val / np.sqrt(n_logs)
                elif len(values_at_step) == 1:
                    mean_val = values_at_step[0]
                    se_val = 0
                else:
                    continue
                
                means.append(mean_val)
                ses.append(se_val)
            
            aggregated_data[tag][group_name] = pd.DataFrame({
                'step': all_steps,
                'mean': means,
                'se': ses,
                'ci_lower': np.array(means) - np.array(ses),
                'ci_upper': np.array(means) + np.array(ses)
            })
            
            print(f"    ✓ {tag}: {len(all_steps)} 个点 | 种子数：{n_logs}")
    
    return aggregated_data

# =============================================================================
# 4. Y 轴自适应范围计算
# =============================================================================
def set_adaptive_ylim(ax, dataframes, margin_ratio=0.1, force_non_negative=False):
    """根据多个 DataFrame 的 mean±se 自适应设置 Y 轴范围"""
    all_values = []
    
    for df in dataframes:
        if not df.empty and 'mean' in df.columns and 'se' in df.columns:
            valid = df.dropna(subset=['mean', 'se'])
            if len(valid) > 0:
                all_values.extend((valid['mean'] - valid['se']).values)
                all_values.extend((valid['mean'] + valid['se']).values)
    
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
# 5. 绘图功能（3个子图：Loss、Accuracy、Firing Rate）
# =============================================================================
def plot_publication_figure(data_dict, output_path='results.pdf',
                           shadow_alpha=0.15, n_seeds_per_group=4):
    """
    绘制 1 行 3 列的子图：Loss、Accuracy、Firing Rate
    - 实线：每组4个种子的均值
    - 阴影：标准误（SE = Std / sqrt(n)），无外轮廓线
    """
    set_neurips_style()
    
    # 配置3条曲线的颜色
    config = {
        'Feedback Ratio 5%':  {'color': '#1f77b4', 'label': 'Feedback Ratio 5%'},
        'Feedback Ratio 15%': {'color': '#ff7f0e', 'label': 'Feedback Ratio 15%'},
        'Feedback Ratio 25%': {'color': '#2ca02c', 'label': 'Feedback Ratio 25%'}
    }
    
    group_names = ['Feedback Ratio 5%', 'Feedback Ratio 15%', 'Feedback Ratio 25%']
    
    fig, axes = plt.subplots(1, 3, figsize=(18*0.8, 4*0.8))
    ax_loss, ax_acc, ax_rate = axes
    
    # ========== 子图1：Loss ==========
    loss_data = data_dict.get('train/loss', {})
    
    for group_name in group_names:
        if group_name in loss_data and not loss_data[group_name].empty:
            df = loss_data[group_name]
            ax_loss.fill_between(df['step'],
                                df['ci_lower'],
                                df['ci_upper'],
                                color=config[group_name]['color'],
                                alpha=shadow_alpha,
                                edgecolor='none',
                                label='_nolegend_')
            ax_loss.plot(df['step'], df['mean'],
                        color=config[group_name]['color'],
                        linewidth=2.0,
                        label=config[group_name]['label'])
    
    ax_loss.set_xlabel('Epoch', fontsize=10)
    ax_loss.set_ylabel('Loss', fontsize=10)
    ax_loss.set_xlim(left=0)
    set_adaptive_ylim(ax_loss, list(loss_data.values()), margin_ratio=0.1, force_non_negative=True)
    ax_loss.grid(axis='y', linestyle='--', alpha=0.3, color='#E0E0E0')
    ax_loss.legend(loc='best', frameon=False, fontsize=8)
    ax_loss.set_title('Loss Curve with Frozen Readout', 
                      fontsize=11, fontweight='bold', pad=10)
    
    # ========== 子图2：Accuracy ==========
    acc_data = data_dict.get('train/accuracy', {})
    
    for group_name in group_names:
        if group_name in acc_data and not acc_data[group_name].empty:
            df = acc_data[group_name]
            ax_acc.fill_between(df['step'],
                               df['ci_lower'],
                               df['ci_upper'],
                               color=config[group_name]['color'],
                               alpha=shadow_alpha,
                               edgecolor='none',
                               label='_nolegend_')
            ax_acc.plot(df['step'], df['mean'],
                       color=config[group_name]['color'],
                       linewidth=2.0,
                       label=config[group_name]['label'])
    
    ax_acc.set_xlabel('Epoch', fontsize=10)
    ax_acc.set_ylabel('Accuracy', fontsize=10)
    ax_acc.set_xlim(left=0)
    set_adaptive_ylim(ax_acc, list(acc_data.values()), margin_ratio=0.08, force_non_negative=True)
    
    # 确保 Accuracy 不超过 100%
    ymax = ax_acc.get_ylim()[1]
    if ymax > 100:
        ax_acc.set_ylim(0, min(ymax, 105))
    
    ax_acc.grid(axis='y', linestyle='--', alpha=0.3, color='#E0E0E0')
    ax_acc.legend(loc='best', frameon=False, fontsize=8)
    ax_acc.set_title('Accuracy Curve with Frozen Readout', 
                     fontsize=11, fontweight='bold', pad=10)
    
    # ========== 子图3：Firing Rate ==========
    rate_data = data_dict.get('output_firing_rates/tunneler0', {})
    
    for group_name in group_names:
        if group_name in rate_data and not rate_data[group_name].empty:
            df = rate_data[group_name]
            ax_rate.fill_between(df['step'],
                                df['ci_lower'],
                                df['ci_upper'],
                                color=config[group_name]['color'],
                                alpha=shadow_alpha,
                                edgecolor='none',
                                label='_nolegend_')
            ax_rate.plot(df['step'], df['mean'],
                        color=config[group_name]['color'],
                        linewidth=2.0,
                        label=config[group_name]['label'])
    
    ax_rate.set_xlabel('Epoch', fontsize=10)
    ax_rate.set_ylabel('Firing Rate', fontsize=10)
    ax_rate.set_xlim(left=0)
    set_adaptive_ylim(ax_rate, list(rate_data.values()), margin_ratio=0.1, force_non_negative=True)
    ax_rate.grid(axis='y', linestyle='--', alpha=0.3, color='#E0E0E0')
    ax_rate.legend(loc='best', frameon=False, fontsize=8)
    ax_rate.set_title('Firing Rate Evolution', 
                      fontsize=11, fontweight='bold', pad=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"\n✓ 图表已保存：{output_path}")
    plt.show()

# =============================================================================
# 6. 主程序
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Plot TensorBoard metrics from 12 runs (3 Feedback Ratios × 4 seeds)')
    parser.add_argument('--logdirs', type=str, required=True,
                       help='Comma-separated list of 12 TensorBoard log directories')
    parser.add_argument('--output', type=str, default='training_curves_multi.pdf',
                       help='Output PDF filename')
    parser.add_argument('--shadow', type=float, default=0.15,
                       help='Alpha value for SE shadow (0.0~1.0)')
    args = parser.parse_args()
    
    # 定义需要提取的标签（不再包含验证集）
    TARGET_TAGS = [
        'train/loss',
        'train/accuracy',
        'output_firing_rates/tunneler0'
    ]
    
    # 解析 12 个 logdir
    if ',' in args.logdirs:
        logdirs = [d.strip() for d in args.logdirs.split(',')]
    else:
        logdirs = glob(args.logdirs)
    
    # 验证数量
    if len(logdirs) != 12:
        print(f"❌ 错误：需要传入 12 个日志目录（3 种 Feedback Ratio × 4 个种子），当前：{len(logdirs)}")
        exit(1)
    
    # 验证路径存在
    valid_logdirs = [d for d in logdirs if os.path.exists(d)]
    
    if len(valid_logdirs) != 12:
        print(f"❌ 错误：需要 12 个有效目录，当前有效：{len(valid_logdirs)}")
        print(f"   输入：{logdirs}")
        exit(1)
    
    # 按顺序分组：每4个为一组（对应 5%、15%、25%）
    logdir_groups = [
        valid_logdirs[0:4],   # Feedback Ratio 5%
        valid_logdirs[4:8],   # Feedback Ratio 15%
        valid_logdirs[8:12]   # Feedback Ratio 25%
    ]
    
    print(f"\n{'='*60}")
    print(f"📊 多实验训练曲线可视化 (Mean ± SE)")
    print(f"{'='*60}")
    print(f"📁 总实验数：{len(valid_logdirs)} (3 组 × 4 种子)")
    print(f"📂 分组情况：")
    group_names = ['Feedback Ratio 5%', 'Feedback Ratio 15%', 'Feedback Ratio 25%']
    for i, group in enumerate(logdir_groups):
        print(f"   【组 {i+1}】{group_names[i]}:")
        for d in group:
            print(f"      • {os.path.basename(d)}")
    print(f"🎨 阴影透明度：alpha={args.shadow}")
    print(f"📄 输出文件：{args.output}")
    print(f"{'='*60}\n")
    
    try:
        # 聚合多组 log 的数据
        aggregated_data = aggregate_logs_by_group(logdir_groups, TARGET_TAGS)
        
        # 检查是否有有效数据
        has_data = any(len(group_data) > 0 for group_data in aggregated_data.values())
        if not has_data:
            print("❌ 未找到有效数据，请检查标签或路径")
            exit(1)
        
        # 绘图
        plot_publication_figure(
            aggregated_data,
            output_path=args.output,
            shadow_alpha=args.shadow,
            n_seeds_per_group=4
        )
        
        print(f"\n{'='*60}")
        print(f"✅ 所有图表生成完成！")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"\n❌ 错误：{e}")
        import traceback
        traceback.print_exc()
        exit(1)
