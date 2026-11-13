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
categories = ['Full Output', 'Sub-network\nOutput', 'Trivial\nOutput']
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
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'figure.figsize': (15, 9),  # 宽15高9
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
color_triv = '#ff7f0e'  # 橙色 - Trivial / Bar

# =============================================================================
# ================= 创建画布 (2行2列) =================
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(18, 5))
axes = axes.reshape(-1)
ax1, ax2 = axes[0], axes[2]  # 上行：Per Session
ax3, ax4 = axes[1], axes[3]  # 下行：Average

# =============================================================================
# ================= 子图 (a): Per Session - F1 Score =================
# =============================================================================

bars1_1 = ax1.bar(x_session - width_session/2, f1_sub_mean, width_session,
                  yerr=f1_sub_err, color=color_sub, ecolor=color_sub,
                  error_kw={'capsize': 4, 'elinewidth': 1.2},
                  edgecolor='white', linewidth=0.5, zorder=3)
bars1_2 = ax1.bar(x_session + width_session/2, f1_triv_mean, width_session,
                  yerr=f1_triv_err, color=color_triv, ecolor=color_triv,
                  error_kw={'capsize': 4, 'elinewidth': 1.2},
                  edgecolor='white', linewidth=0.5, zorder=3)

ax1.set_ylabel('F1 Score (%)', fontsize=12, fontweight='medium')
ax1.set_title('F1 Score Per Session', fontsize=12, fontweight='bold', pad=10)
ax1.set_xticks(x_session)
ax1.set_xticklabels(sessions, fontsize=10)
ax1.set_ylim(f1_session_ylim_min, f1_session_ylim_max)
ax1.grid(axis='y', linestyle='-', alpha=0.6, color='#E0E0E0')
ax1.set_axisbelow(True)
ax1.legend(['Sub-network Output', 'Trivial Output'], loc='upper right', 
           frameon=False, fontsize=9)

# =============================================================================
# ================= 子图 (b): Per Session - Accuracy =================
# =============================================================================

bars2_1 = ax2.bar(x_session - width_session/2, acc_sub_mean, width_session,
                  yerr=acc_sub_err, color=color_sub, ecolor=color_sub,
                  error_kw={'capsize': 4, 'elinewidth': 1.2},
                  edgecolor='white', linewidth=0.5, zorder=3)
bars2_2 = ax2.bar(x_session + width_session/2, acc_triv_mean, width_session,
                  yerr=acc_triv_err, color=color_triv, ecolor=color_triv,
                  error_kw={'capsize': 4, 'elinewidth': 1.2},
                  edgecolor='white', linewidth=0.5, zorder=3)

ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='medium')
ax2.set_title('Accuracy Per Session', fontsize=12, fontweight='bold', pad=10)
ax2.set_xticks(x_session)
ax2.set_xticklabels(sessions, fontsize=10)
ax2.set_ylim(acc_session_ylim_min, acc_session_ylim_max)
ax2.grid(axis='y', linestyle='-', alpha=0.6, color='#E0E0E0')
ax2.set_axisbelow(True)
ax2.legend(['Sub-network Output', 'Trivial Output'], loc='upper right', 
           frameon=False, fontsize=9)

# =============================================================================
# ================= 子图 (c): Average - Accuracy =================
# =============================================================================

bars3 = ax3.bar(x_avg, acc_avg_mean, width_avg,
                yerr=acc_avg_err, color=color_triv, ecolor=color_triv,
                error_kw={'capsize': 4, 'elinewidth': 1.2, 'capthick': 1},
                edgecolor='white', linewidth=0.5, zorder=2)

ax3.plot(x_avg, acc_avg_mean, color=color_sub, marker='o', markersize=8,
         markerfacecolor='white', markeredgecolor=color_sub,
         markeredgewidth=1.5, linewidth=2.5, zorder=4)

ax3.set_ylabel('Accuracy (%)', fontsize=12, fontweight='medium')
ax3.set_title('Average Accuracy', 
              fontsize=12, fontweight='bold', pad=10)
ax3.set_xticks(x_avg)
ax3.set_xticklabels(categories, fontsize=9, ha='center')
ax3.set_ylim(acc_avg_ylim_min, acc_avg_ylim_max)
ax3.grid(axis='y', linestyle='-', alpha=0.6, color='#E0E0E0')
ax3.set_axisbelow(True)

# =============================================================================
# ================= 子图 (d): Average - F1 Score =================
# =============================================================================

bars4 = ax4.bar(x_avg, f1_avg_mean, width_avg,
                yerr=f1_avg_err, color=color_triv, ecolor=color_triv,
                error_kw={'capsize': 4, 'elinewidth': 1.2, 'capthick': 1},
                edgecolor='white', linewidth=0.5, zorder=2)

ax4.plot(x_avg, f1_avg_mean, color=color_sub, marker='o', markersize=8,
         markerfacecolor='white', markeredgecolor=color_sub,
         markeredgewidth=1.5, linewidth=2.5, zorder=4)

ax4.set_ylabel('F1 Score (%)', fontsize=12, fontweight='medium')
ax4.set_title('Average F1 Score', 
              fontsize=12, fontweight='bold', pad=10)
ax4.set_xticks(x_avg)
ax4.set_xticklabels(categories, fontsize=9, ha='center')
ax4.set_ylim(f1_avg_ylim_min, f1_avg_ylim_max)
ax4.grid(axis='y', linestyle='-', alpha=0.6, color='#E0E0E0')
ax4.set_axisbelow(True)

# =============================================================================
# ================= 整体美化 =================
# =============================================================================

plt.subplots_adjust(hspace=0.35, wspace=0.2)
fig.patch.set_facecolor('white')

# =============================================================================
# ================= 保存为 PDF =================
# =============================================================================

output_filename = 'nested_subnetwork'
plt.savefig(f"{output_filename}.svg", format='svg', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig(f"{output_filename}.pdf", format='pdf', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')

print(f"✓ Figure saved as '{output_filename}'")
print(f"\nPer Session F1 Y-axis: [{f1_session_ylim_min:.1f}, {f1_session_ylim_max:.1f}]")
print(f"Per Session Acc Y-axis: [{acc_session_ylim_min:.1f}, {acc_session_ylim_max:.1f}]")
print(f"Average Acc Y-axis: [{acc_avg_ylim_min:.1f}, {acc_avg_ylim_max:.1f}] (zoomed)")
print(f"Average F1 Y-axis: [{f1_avg_ylim_min:.1f}, {f1_avg_ylim_max:.1f}] (zoomed)")

plt.show()
