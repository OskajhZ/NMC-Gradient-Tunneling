import math

# 定义三组数据：每组是一个字典，包含4个指标的 (Mean, SE) 元组
groups = [
    {  # Group 1
        "Accuracy": (51.92, 6.53),
        "Precision": (53.36, 6.64),
        "Recall": (51.92, 6.53),
        "F1": (50.49, 6.58)
    },
    {  # Group 2
        "Accuracy": (59.66, 4.84),
        "Precision": (60.72, 5.11),
        "Recall": (59.66, 4.84),
        "F1": (58.15, 5.07)
    },
    {  # Group 3
        "Accuracy": (58.29, 4.88),
        "Precision": (59.38, 5.44),
        "Recall": (58.29, 4.88),
        "F1": (56.62, 5.22)
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
# 输出为表格行格式
table_row = (f"Overall Avg & "
             f"{fmt(*results['Accuracy'])} \\% & "
             f"{fmt(*results['Precision'])} \\% & "
             f"{fmt(*results['Recall'])} \\% & "
             f"{fmt(*results['F1'])} \\% \\\\")
print(table_row)

# 验证模块：手动计算所有指标
print("\n% === 手动验证（所有指标）===")
for metric in metrics:
    means = [g[metric][0] for g in groups]
    ses = [g[metric][1] for g in groups]
    calc_m = sum(means) / 3
    calc_s = math.sqrt(sum(se**2 for se in ses)) / 3
    print(f"{metric:12s}: Mean={calc_m:.4f}→{calc_m:.2f}, "
          f"SE=√{sum(se**2 for se in ses):.4f}/3={calc_s:.4f}→{calc_s:.2f} → {fmt(calc_m, calc_s)} %")
