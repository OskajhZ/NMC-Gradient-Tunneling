import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 数据准备 (从 LaTeX 表格 Avg 行提取)
# ==========================================
edges = np.array([6, 7, 8, 9])

# GT Method Data
gt_f1_mean = np.array([52.27, 56.87, 58.04, 56.71])
gt_f1_se   = np.array([2.86, 2.96, 2.74, 2.83])
gt_acc_mean = np.array([54.39, 58.45, 60.22, 58.57])
gt_acc_se   = np.array([2.66, 2.85, 2.52, 2.72])

# Lyapunov Method Data
lyap_f1_mean = np.array([52.70, 54.88, 55.22, 49.13])
lyap_f1_se   = np.array([3.23, 3.05, 2.98, 3.50])
lyap_acc_mean = np.array([56.68, 57.55, 58.42, 54.79])
lyap_acc_se   = np.array([2.87, 2.79, 2.78, 2.95])

# ==========================================
# 2. 绘图风格设置 (NeurIPS Style)
# ==========================================
plt.style.use('default') # 重置默认风格，避免冲突
font_coef = 1.5
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 11*font_coef,
    'axes.labelsize': 12*font_coef,
    'axes.titlesize': 14*font_coef,
    'xtick.labelsize': 11*font_coef,
    'ytick.labelsize': 11*font_coef,
    'legend.fontsize': 11*font_coef,
    'figure.dpi': 300,          # 高分辨率
    'figure.figsize': (18, 9),
    'lines.linewidth': 2.5,     # 线条加粗
    'lines.markersize': 8,      # 标记点大小
    'axes.spines.top': False,   # 去除上边框
    'axes.spines.right': False, # 去除右边框
    'grid.linestyle': '--',     # 网格虚线
    'grid.alpha': 0.4,          # 网格透明度
    'grid.color': '#CCCCCC',    # 网格浅灰色
})

# 定义配色 (色盲友好且对比度高)
color_gt = '#1f77b4'      # 专业蓝 (GT)
color_lyap = '#ff7f0e'    # 活力橙 (Lyapunov)

# ==========================================
# 3. 创建画布与子图
# ==========================================
fig, axes = plt.subplots(1, 2, sharey=False) # sharey=False 因为 F1 和 Acc 刻度可能不同

# 子图配置列表 (便于循环绘制)
plots_config = [
    {
        'ax': axes[0],
        'title': 'F1 Score on SEED Dataset',
        'ylabel': 'F1 Score (%)',
        'data_gt': (gt_f1_mean, gt_f1_se),
        'data_lyap': (lyap_f1_mean, lyap_f1_se),
        'ylim': [45, 62]  # 根据数据范围微调，使趋势更明显
    },
    {
        'ax': axes[1],
        'title': 'Accuracy on SEED Dataset',
        'ylabel': 'Accuracy (%)',
        'data_gt': (gt_acc_mean, gt_acc_se),
        'data_lyap': (lyap_acc_mean, lyap_acc_se),
        'ylim': [51, 64]
    }
]

# ==========================================
# 4. 绘制循环
# ==========================================
for cfg in plots_config:
    ax = cfg['ax']
    
    # --- 绘制 GT (置于下层阴影，上层线) ---
    # 阴影区域 (Mean ± SE)
    ax.fill_between(edges, 
                    cfg['data_gt'][0] - cfg['data_gt'][1], 
                    cfg['data_gt'][0] + cfg['data_gt'][1], 
                    color=color_gt, alpha=0.15, label='_nolegend_') # _nolegend_ 避免图例重复
    # 均值线
    ax.plot(edges, cfg['data_gt'][0], 
            color=color_gt, marker='o', label='GT', zorder=3)
    
    # --- 绘制 Lyapunov ---
    # 阴影区域
    ax.fill_between(edges, 
                    cfg['data_lyap'][0] - cfg['data_lyap'][1], 
                    cfg['data_lyap'][0] + cfg['data_lyap'][1], 
                    color=color_lyap, alpha=0.15, label='_nolegend_')
    # 均值线
    ax.plot(edges, cfg['data_lyap'][0], 
            color=color_lyap, marker='s', label='Lyapunov', zorder=3)
    
    # --- 美化子图 ---
    ax.set_title(cfg['title'], pad=10, fontweight='bold')
    ax.set_ylabel(cfg['ylabel'])
    ax.set_xlabel('Edge Length of Reservoir')
    ax.set_xticks(edges)
    ax.set_ylim(cfg['ylim'])
    ax.grid(axis='y', which='major') # 仅 Y 轴网格
    
    # 添加图例 (仅在第一个子图添加，或分别添加，此处为了清晰分别在右下角添加)
    # 如果希望统一图例，可只在 axes[0] 添加，此处为了子图独立性分别添加
    leg = ax.legend(frameon=False, loc='lower right')
    leg.set_zorder(10)

# ==========================================
# 5. 整体调整与保存
# ==========================================
plt.tight_layout(pad=2.0) # 自动调整间距

# 保存为矢量图 (推荐投稿用)
#plt.savefig('reservoir_performance.pdf', format='pdf', bbox_inches='tight')
# 保存为高分辨率位图
#plt.savefig('reservoir_performance.png', dpi=300, bbox_inches='tight')
plt.savefig('reservoir_performance.pdf')

plt.show()
