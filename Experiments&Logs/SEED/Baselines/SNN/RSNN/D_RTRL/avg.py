import math

# 提取本组数据（3个Session，4个指标）
data = {
    "Accuracy": [(47.50, 4.61), (50.95, 3.05), (48.16, 3.28)],
    "Precision": [(49.46, 5.04), (50.94, 3.25), (50.05, 3.30)],
    "Recall": [(47.50, 4.61), (50.95, 3.05), (48.16, 3.28)],  # 与Accuracy相同
    "F1 Score": [(46.33, 4.58), (49.20, 3.33), (45.27, 4.30)]
}

def calc_avg(sessions, n=3):
    """
    计算 Avg 行：
    Mean = arithmetic mean of μ_i
    SE = sqrt(sum(SE_i^2)) / N  (基于独立统计量方差可加性)
    """
    means = [s[0] for s in sessions]
    ses = [s[1] for s in sessions]
    avg_m = sum(means) / n
    avg_se = math.sqrt(sum(se**2 for se in ses)) / n
    return avg_m, avg_se

def fmt(m, s):
    """格式化输出为 LaTeX 兼容的 'XX.XX$\\pm$XX.XX' 格式"""
    return f"{m:.2f}$\\pm${s:.2f}"

print("% Avg 行计算结果")
print("-" * 60)

results = {}
for metric, sessions in data.items():
    m, s = calc_avg(sessions)
    results[metric] = (m, s)
    print(f"{metric:12s}: Mean={m:.4f}, SE={s:.4f} → {fmt(m, s)}")

print("-" * 60)
# 输出LaTeX行 (注意表格结构：跳过第一列，从Session列开始)
latex_row = (f"& Avg & "
             f"{fmt(*results['Accuracy'])} & "
             f"{fmt(*results['Precision'])} & "
             f"{fmt(*results['Recall'])} & "
             f"{fmt(*results['F1 Score'])} \\\\")
print("LaTeX 可直接复制行：")
print(latex_row)
