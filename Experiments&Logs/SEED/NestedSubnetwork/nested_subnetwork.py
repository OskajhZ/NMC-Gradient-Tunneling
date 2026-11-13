import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# ================= 数据准备 =================
# =============================================================================

# --- Per Session 数据 (plot_per_session.py) ---
sessions = ['Session 1', 'Session 2', 'Session 3']
x_session = np.arange(len(sessions))
width_session = 0.35

# F1 Score - Per Session
f1_sub_mean = [55.15, 55.17, 55.52]
f1_sub_err = [4.16, 3.31, 2.90]
f1_triv_mean = [45.04, 50.65, 48.38]
f1_triv_err = [3.99, 3.62, 3.60]

# Accuracy - Per Session
acc_sub_mean = [56.37, 57.29, 57.23]
acc_sub_err = [4.08, 3.04, 2.71]
acc_triv_mean = [45.95, 52.04, 49.11]
acc_triv_err = [3.77, 3.60, 3.66]

# --- Average 数据 (plot_avg.py) ---
categories = ['Full Output', 'Sub-network\nOutput', 'Peripheral\nOutput']
x_avg = np.arange(len(categories))
width_avg = 0.5

# Accuracy - Average
acc_avg_mean = [60.22, 56.96, 49.03]
acc_avg_err = [2.52, 1.92, 2.12]

# F1 Score - Average
f1_avg_mean = [58.04, 55.28, 48.38]
f1_avg_err = [2.74, 2.02, 3.60]

# =============================================================================
# ================= 计算自适应 Y 轴范围 =================
# =============================================================================

def calculate_ylim(means_list, errors_list, margin_ratio=0.15):
    """Per Session: 计算Y轴范围（从0开始）"""
    all_means = []
    all_errors = []
    for means, errors in zip(means_list, errors_list):
        all_means.extend(means)
        all_errors.extend(errors)
    
    all_means = np.array(all_means)
    all_errors = np.array(all_errors)
    
    min_val = np.min(all_means - all_errors)
    max_val = np.max(all_means + all_errors)
    
    data_range = max_val - min_val
    margin = data_range * margin_ratio
    
    y_min = max(0, min_val - margin)
    y_max = max_val + margin
    y_max = np.ceil(y_max / 5) * 5
    
    return y_min, y_max

def calculate_ylim_zoomed(means, errors, margin_ratio=0.08):
    """Average: 计算紧凑Y轴范围（放大差异）"""
    means = np.array(means)
    errors = np.array(errors)
    
    min_val = np.min(means - errors)
    max_val = np.max(means + errors)
    
    data_range = max_val - min_val
    margin = data_range * margin_ratio
    
    y_min = min_val - margin
    y_max = max_val + margin
    
    y_min = np.floor(y_min / 2) * 2
    y_max = np.ceil(y_max / 2) * 2
    
    return y_min, y_max

# 计算各子图Y轴范围
f1_session_ylim_min, f1_session_ylim_max = calculate_ylim(
    [f1_sub_mean, f1_triv_mean], [f1_sub_err, f1_triv_err]
)
acc_session_ylim_min, acc_session_ylim_max = calculate_ylim(
    [acc_sub_mean, acc_triv_mean], [acc_sub_err, acc_triv_err]
)
acc_avg_ylim_min, acc_avg_ylim_max = calculate_ylim_zoomed(acc_avg_mean, acc_avg_err)
f1_avg_ylim_min, f1_avg_ylim_max = calculate_ylim_zoomed(f1_avg_mean, f1_avg_err)

# =============================================================================
# ================= NeurIPS 风格设置 =================
# =============================================================================

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif', 'serif'],
    'font.size': 11,
    'axes.labelsize': 16,
    'axes.titlesize': 12,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 13,
    'figure.dpi': 300,
    'figure.figsize': (7.5, 7.5),
    'lines.linewidth': 2.5,
    'lines.markersize': 8,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'grid.color': '#E0E0E0',
    'grid.linestyle': '-',
    'axes.axisbelow': True,
})

# 定义配色
color_sub = '#1f77b4'   # 蓝色 - Sub-network / Trend
color_triv = '#ff7f0e'  # 橙色 - Peripheral / Bar

# =============================================================================
# ================= 创建两个独立画布（宽高比 1.5:1） =================
# =============================================================================

fig1, ax_left = plt.subplots(figsize=(7.5, 5))  # 宽高比 7.5:5 = 1.5:1
fig2, ax_right = plt.subplots(figsize=(7.5, 5))  # 宽高比 7.5:5 = 1.5:1

# =============================================================================
# ================= 子图 (左): Per Session - Accuracy =================
# =============================================================================

bars_left_1 = ax_left.bar(x_session - width_session/2, acc_sub_mean, width_session,
                          yerr=acc_sub_err, color=color_sub, ecolor=color_sub,
                          error_kw={'capsize': 4, 'elinewidth': 1.2},
                          edgecolor='white', linewidth=0.5, zorder=3)
bars_left_2 = ax_left.bar(x_session + width_session/2, acc_triv_mean, width_session,
                          yerr=acc_triv_err, color=color_triv, ecolor=color_triv,
                          error_kw={'capsize': 4, 'elinewidth': 1.2},
                          edgecolor='white', linewidth=0.5, zorder=3)

ax_left.set_ylabel('Accuracy (%)', fontsize=16, fontweight='medium')
ax_left.set_xticks(x_session)
ax_left.set_xticklabels(sessions, fontsize=14)
ax_left.set_ylim(acc_session_ylim_min, acc_session_ylim_max)
ax_left.grid(axis='y', linestyle='-', alpha=0.6, color='#E0E0E0')
ax_left.set_axisbelow(True)
ax_left.legend(['Sub-network Output', 'Peripheral Output'], loc='upper right',
               frameon=False, fontsize=13)

# =============================================================================
# ================= 子图 (右): Average - Accuracy =================
# =============================================================================

bars_right = ax_right.bar(x_avg, acc_avg_mean, width_avg,
                          yerr=acc_avg_err, color=color_triv, ecolor=color_triv,
                          error_kw={'capsize': 4, 'elinewidth': 1.2, 'capthick': 1},
                          edgecolor='white', linewidth=0.5, zorder=2)

ax_right.plot(x_avg, acc_avg_mean, color=color_sub, marker='o', markersize=8,
              markerfacecolor='white', markeredgecolor=color_sub,
              markeredgewidth=1.5, linewidth=2.5, zorder=4)

ax_right.set_ylabel('Accuracy (%)', fontsize=16, fontweight='medium')
ax_right.set_xticks(x_avg)
ax_right.set_xticklabels(categories, fontsize=13, ha='center')
ax_right.set_ylim(acc_avg_ylim_min, acc_avg_ylim_max)
ax_right.grid(axis='y', linestyle='-', alpha=0.6, color='#E0E0E0')
ax_right.set_axisbelow(True)

# =============================================================================
# ================= 整体美化 =================
# =============================================================================

fig1.patch.set_facecolor('white')
fig2.patch.set_facecolor('white')

# =============================================================================
# ================= 分别保存两个子图为 PDF（页面宽高比精确 1.5:1） =================
# =============================================================================

def save_figure_ratio(fig, filename, target_ratio=1.5, pad=0.1):
    """迭代调整画布尺寸，使最终PDF页面宽高比精确为 target_ratio。"""
    for _ in range(60):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bbox = fig.get_tightbbox(renderer)
        w_in = bbox.width + 2 * pad
        h_in = bbox.height + 2 * pad
        ratio = w_in / h_in
        if abs(ratio - target_ratio) < 0.001:
            break
        W, H = fig.get_size_inches()
        fig.set_size_inches(W, H * ratio / target_ratio, forward=True)
    fig.savefig(f"{filename}.svg", format='svg', dpi=300, bbox_inches='tight',
                pad_inches=pad, facecolor='white', edgecolor='none')
    fig.savefig(f"{filename}.pdf", format='pdf', dpi=300, bbox_inches='tight',
                pad_inches=pad, facecolor='white', edgecolor='none')

# 子图1: Accuracy Per Session
save_figure_ratio(fig1, 'nested_subnetwork_per_session')

# 子图2: Average Accuracy
save_figure_ratio(fig2, 'nested_subnetwork_average')

print("✓ Figure 1 saved as 'nested_subnetwork_per_session'")
print("✓ Figure 2 saved as 'nested_subnetwork_average'")
print(f"\nPer Session F1 Y-axis: [{f1_session_ylim_min:.1f}, {f1_session_ylim_max:.1f}]")
print(f"Per Session Acc Y-axis: [{acc_session_ylim_min:.1f}, {acc_session_ylim_max:.1f}]")
print(f"Average Acc Y-axis: [{acc_avg_ylim_min:.1f}, {acc_avg_ylim_max:.1f}] (zoomed)")
print(f"Average F1 Y-axis: [{f1_avg_ylim_min:.1f}, {f1_avg_ylim_max:.1f}] (zoomed)")

plt.show()
