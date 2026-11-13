import math

# 定义数据结构：(Edge, TuningMethod) -> Metric -> [(S1_Mean, S1_SE), (S2_Mean, S2_SE), (S3_Mean, S3_SE)]
data = {
    (6, "Lyapunov"): {
        "Accuracy": [(52.82, 4.48), (57.48, 5.63), (59.74, 4.74)],
        "Precision": [(54.43, 4.71), (59.93, 5.90), (60.01, 4.92)],
        "Recall": [(52.82, 4.48), (57.48, 5.63), (59.74, 4.74)],
        "F1 Score": [(49.25, 5.13), (53.99, 6.14), (54.86, 5.48)]
    },
    (6, "GT"): {
        "Accuracy": [(50.59, 5.20), (54.98, 4.78), (57.61, 3.72)],
        "Precision": [(50.55, 5.57), (56.01, 5.33), (56.74, 4.06)],
        "Recall": [(50.59, 5.20), (54.98, 4.78), (57.61, 3.72)],
        "F1 Score": [(48.64, 5.52), (52.75, 5.14), (55.41, 4.12)]
    },
    (7, "Lyapunov"): {
        "Accuracy": [(55.08, 5.85), (58.93, 3.87), (58.64, 4.56)],
        "Precision": [(55.06, 6.94), (62.33, 4.35), (58.82, 5.13)],
        "Recall": [(55.08, 5.85), (58.93, 3.87), (58.64, 4.56)],
        "F1 Score": [(51.76, 6.46), (56.86, 4.17), (56.02, 4.95)]
    },
    (7, "GT"): {
        "Accuracy": [(55.64, 5.31), (58.71, 4.61), (61.00, 4.86)],
        "Precision": [(55.44, 5.32), (60.78, 4.69), (62.26, 5.09)],
        "Recall": [(55.64, 5.31), (58.71, 4.61), (61.00, 4.86)],
        "F1 Score": [(54.43, 5.42), (56.66, 4.91), (59.52, 5.04)]
    },
    (8, "Lyapunov"): {
        "Accuracy": [(56.18, 6.07), (57.19, 3.67), (61.89, 4.36)],
        "Precision": [(54.53, 7.38), (60.14, 4.34), (60.84, 4.83)],
        "Recall": [(56.18, 6.07), (57.19, 3.67), (61.89, 4.36)],
        "F1 Score": [(52.60, 6.49), (53.78, 3.97), (59.27, 4.70)]
    },
    (8, "GT"): {
        "Accuracy": [(58.20, 4.80), (61.41, 4.46), (61.05, 3.76)],
        "Precision": [(59.24, 5.05), (63.47, 4.92), (62.10, 4.01)],
        "Recall": [(58.20, 4.80), (61.41, 4.46), (61.05, 3.76)],
        "F1 Score": [(56.39, 4.99), (58.80, 5.11), (58.92, 4.04)]
    },
    # Edge=9 的数据用于验证（表格中已给出答案）
    (9, "Lyapunov"): {
        "Accuracy": [(55.58, 5.66), (51.34, 4.51), (57.45, 5.10)],
        "Precision": [(54.87, 7.33), (50.09, 6.45), (55.20, 6.25)],
        "Recall": [(55.58, 5.66), (51.34, 4.51), (57.45, 5.10)],
        "F1 Score": [(50.16, 6.49), (46.26, 5.42), (50.97, 6.25)]
    },
    (9, "GT"): {
        "Accuracy": [(55.81, 5.98), (58.70, 3.85), (61.21, 3.97)],
        "Precision": [(56.06, 6.41), (61.74, 4.23), (61.45, 4.34)],
        "Recall": [(55.81, 5.98), (58.70, 3.85), (61.21, 3.97)],
        "F1 Score": [(54.15, 6.14), (56.31, 4.14), (59.67, 4.13)]
    }
}

def calculate_avg(sessions_data, n=3):
    """计算 Avg 行：Mean = avg(M_i), SE = sqrt(sum(SE_i^2)) / N"""
    means = [s[0] for s in sessions_data]
    ses = [s[1] for s in sessions_data]
    avg_mean = sum(means) / n
    avg_se = math.sqrt(sum(se**2 for se in ses)) / n
    return avg_mean, avg_se

def fmt(m, s):
    return f"{m:.2f}$\\pm${s:.2f}"

# 主计算循环
print("% 计算结果：可直接复制到 LaTeX 表格的 Avg 行")
print("% 公式：Mean = Σμᵢ/N,  SE = √(ΣSEᵢ²)/N")
print("=" * 90)

for (edge, method), metrics in data.items():
    # 计算四个指标
    acc = calculate_avg(metrics["Accuracy"])
    pre = calculate_avg(metrics["Precision"])
    rec = calculate_avg(metrics["Recall"])  # 注意：Recall 数据与 Accuracy 完全相同（根据表格）
    f1 = calculate_avg(metrics["F1 Score"])
    
    # 生成 LaTeX 行 (注意表格结构：Edge列multirow, Tuning列multirow, 所以Avg行用 & & 跳过前两列)
    latex_line = (f"& & Avg & "
                  f"{fmt(*acc)} & "
                  f"{fmt(*pre)} & "
                  f"{fmt(*rec)} & "
                  f"{fmt(*f1)} \\\\")
    
    print(f"% Edge={edge}, Method={method}")
    print(latex_line)
    print("-" * 90)

# 验证模块：检查 Edge=9 的计算结果是否与表格中已给答案一致
print("\n% === 验证模块：Edge=9 (表格中已有答案) ===")
for method in ["Lyapunov", "GT"]:
    key = (9, method)
    print(f"% {method}:")
    for metric in ["Accuracy", "Precision", "F1 Score"]:  # 抽查关键指标
        calc_m, calc_s = calculate_avg(data[key][metric])
        print(f"  {metric}: 计算值={fmt(calc_m, calc_s)}")
