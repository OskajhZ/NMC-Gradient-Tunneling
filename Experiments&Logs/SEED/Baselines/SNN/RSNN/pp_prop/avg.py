import math

# 提取本组数据（3个Session，4个指标）
data = {
    "Accuracy": [(38.82, 5.32), (46.22, 3.60), (48.34, 4.33)],
    "Precision": [(38.82, 6.45), (45.37, 4.79), (47.04, 5.61)],
    "Recall": [(38.82, 5.32), (46.22, 3.60), (48.34, 4.33)],  # 与Accuracy相同
    "F1 Score": [(34.70, 5.18), (41.23, 4.75), (44.75, 5.11)]
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
