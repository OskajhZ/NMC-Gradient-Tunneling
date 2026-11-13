import math

# 定义三组数据：每组是一个字典，包含4个指标的 (Mean, SE) 元组
# 注意：数据已去除 "%" 符号，仅保留数值
groups = [
    {  # Group 1
        "Accuracy": (50.63, 4.96),
        "Precision": (50.77, 5.55),
        "Recall": (50.63, 4.96),
        "F1": (49.29, 5.22)
    },
    {  # Group 2
        "Accuracy": (53.42, 5.63),
        "Precision": (56.11, 5.82),
        "Recall": (53.42, 5.63),
        "F1": (51.45, 5.78)
    },
    {  # Group 3
        "Accuracy": (58.07, 4.55),
        "Precision": (57.23, 5.04),
        "Recall": (58.07, 4.55),
        "F1": (56.17, 4.82)
    }
]

def calculate_grand_avg(groups, metric, n=3):
    """
    计算某个指标的总体平均：
    Mean = arithmetic mean of the 3 group means
    SE = sqrt(sum(SE_i^2)) / N  (基于独立统计量方差可加性)
    """
    means = [g[metric][0] for g in groups]
    ses = [g[metric][1] for g in groups]
    grand_mean = sum(means) / n
    grand_se = math.sqrt(sum(se**2 for se in ses)) / n
    return grand_mean, grand_se

def fmt(mean, se):
    """格式化输出：保留2位小数，LaTeX 兼容"""
    return f"{mean:.2f}$\\pm${se:.2f}"

# 主计算循环
print("% 三组数据的总体平均计算结果")
print("% 公式：Mean = Σμᵢ/3,  SE = √(ΣSEᵢ²)/3")
print("=" * 70)

metrics = ["Accuracy", "Precision", "Recall", "F1"]
results = {}

for metric in metrics:
    m, s = calculate_grand_avg(groups, metric)
    results[metric] = (m, s)
    print(f"{metric:12s}: {fmt(m, s)} %")

print("\n% 表格兼容输出行：")
print("-" * 70)
# 输出为表格行格式（可根据需要调整列顺序）
table_row = (f"Overall Avg & "
             f"{fmt(*results['Accuracy'])} \\% & "
             f"{fmt(*results['Precision'])} \\% & "
             f"{fmt(*results['Recall'])} \\% & "
             f"{fmt(*results['F1'])} \\% \\\\")
print(table_row)

# 验证模块：手动计算 Accuracy 作为示例
print("\n% === 手动验证：Accuracy ===")
means = [50.63, 53.42, 58.07]
ses = [4.96, 5.63, 4.55]
calc_m = sum(means) / 3
calc_s = math.sqrt(sum(se**2 for se in ses)) / 3
print(f"Mean: ({' + '.join(map(str, means))}) / 3 = {calc_m:.4f} → {calc_m:.2f}")
print(f"SE: √({ses[0]}² + {ses[1]}² + {ses[2]}²) / 3 = √{sum(se**2 for se in ses):.4f} / 3 = {calc_s:.4f} → {calc_s:.2f}")
print(f"Final: {fmt(calc_m, calc_s)} %")
