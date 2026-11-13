import math

# 定义数据结构：Model -> Metric -> [(S1_Mean, S1_SE), (S2_Mean, S2_SE), (S3_Mean, S3_SE)]
data = {
    "Trivial Output": {
        "Accuracy": [(45.95, 3.77), (52.04, 3.60), (49.11, 3.66)],
        "Precision": [(45.98, 4.25), (52.90, 3.82), (49.38, 3.52)],
        "Recall": [(45.95, 3.77), (52.04, 3.60), (49.11, 3.66)],  # 与Accuracy相同
        "F1 Score": [(45.04, 3.99), (50.65, 3.62), (48.38, 3.60)]
    },
    "Subnetwork Output": {
        "Accuracy": [(56.37, 4.08), (57.29, 3.04), (57.23, 2.71)],
        "Precision": [(57.02, 3.96), (57.91, 3.22), (57.88, 2.27)],
        "Recall": [(56.37, 4.08), (57.29, 3.04), (57.23, 2.71)],  # 与Accuracy相同
        "F1 Score": [(55.15, 4.16), (55.17, 3.31), (55.52, 2.90)]
    }
}

def calculate_avg(sessions_data, n=3):
    """
    计算 Avg 行：
    Mean = arithmetic mean of μ_i
    SE = sqrt(sum(SE_i^2)) / N  (基于独立统计量方差可加性)
    """
    means = [s[0] for s in sessions_data]
    ses = [s[1] for s in sessions_data]
    avg_mean = sum(means) / n
    avg_se = math.sqrt(sum(se**2 for se in ses)) / n
    return avg_mean, avg_se

def fmt(mean, se):
    """格式化输出为 'XX.XX±XX.XX' 格式（适配表格文本）"""
    return f"{mean:.2f}$\\pm${se:.2f}"

# 主计算循环
print("% 计算结果：Avg 行（可直接复制到表格）")
print("% 公式：Mean = Σμᵢ/N,  SE = √(ΣSEᵢ²)/N")
print("=" * 100)

for model, metrics in data.items():
    print(f"\n% Model: {model}")
    
    # 计算四个指标
    results = {}
    for metric_name, sessions in metrics.items():
        m, s = calculate_avg(sessions)
        results[metric_name] = (m, s)
        print(f"  {metric_name:12s}: {fmt(m, s)}")
    
    # 输出表格兼容行（文本/Markdown格式）
    # 格式：Avg & Mean±SE & Mean±SE & Mean±SE & Mean±SE \\
    table_row = (f"\tAvg & "
                 f"{fmt(*results['Accuracy'])} & "
                 f"{fmt(*results['Precision'])} & "
                 f"{fmt(*results['Recall'])} & "
                 f"{fmt(*results['F1 Score'])} \\\\")
    print(f"\n  表格行格式：\n  {table_row}")
    print("-" * 100)

# 验证模块：手动计算示例
print("\n% === 手动验证示例：Trivial Output, Accuracy ===")
means = [45.95, 52.04, 49.11]
ses = [3.77, 3.60, 3.66]
calc_mean = sum(means) / 3
calc_se = math.sqrt(sum(se**2 for se in ses)) / 3
print(f"Mean: ({' + '.join(map(str, means))}) / 3 = {calc_mean:.4f} → {calc_mean:.2f}")
print(f"SE: √({ses[0]}² + {ses[1]}² + {ses[2]}²) / 3 = √{sum(se**2 for se in ses):.4f} / 3 = {calc_se:.4f} → {calc_se:.2f}")
print(f"Result: {fmt(calc_mean, calc_se)}")
