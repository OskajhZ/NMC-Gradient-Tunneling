import math

# 定义数据结构：算法 -> 指标 -> [ (Session1_Mean, Session1_SE), (Session2_Mean, Session2_SE), (Session3_Mean, Session3_SE) ]
data = {
    "LSTM": {
        "Accuracy": [(53.26, 6.76), (57.41, 4.41), (59.36, 4.81)],
        "Precision": [(52.87, 7.10), (60.24, 4.11), (59.61, 4.73)],
        "Recall": [(53.26, 6.76), (57.41, 4.41), (59.36, 4.81)],
        "F1 Score": [(51.78, 7.07), (55.87, 4.52), (58.05, 4.86)]
    },
    "GRU": {
        "Accuracy": [(51.23, 6.48), (57.67, 5.12), (60.14, 4.62)],
        "Precision": [(50.48, 6.98), (60.33, 4.69), (59.95, 4.72)],
        "Recall": [(51.23, 6.48), (57.67, 5.12), (60.14, 4.62)],
        "F1 Score": [(49.79, 6.69), (56.42, 5.12), (58.67, 4.84)]
    },
    "EEGNet": {
        "Accuracy": [(52.62, 7.12), (57.30, 4.78), (62.08, 4.36)],
        "Precision": [(50.39, 7.82), (53.10, 6.41), (61.83, 4.72)],
        "Recall": [(52.62, 7.12), (57.30, 4.78), (62.08, 4.36)],
        "F1 Score": [(49.39, 7.59), (52.98, 5.42), (59.76, 4.68)]
    },
    "Lyapunov-tuned LSM": {
        "Accuracy": [(55.58, 5.66), (51.34, 4.51), (57.45, 5.10)],
        "Precision": [(54.87, 7.33), (50.09, 6.45), (55.20, 6.25)],
        "Recall": [(55.58, 5.66), (51.34, 4.51), (57.45, 5.10)],
        "F1 Score": [(50.16, 6.49), (46.26, 5.42), (50.97, 6.25)]
    },
    "Ours (Feedback LSM + GT)": {
        "Accuracy": [(55.81, 5.98), (58.70, 3.85), (61.21, 3.97)],
        "Precision": [(56.06, 6.41), (61.74, 4.23), (61.45, 4.34)],
        "Recall": [(55.81, 5.98), (58.70, 3.85), (61.21, 3.97)],
        "F1 Score": [(54.15, 6.14), (56.31, 4.14), (59.67, 4.13)]
    }
}

def calculate_avg_row(sessions_data, n_sessions=3):
    """
    根据上一轮推导的结论计算 Avg 行：
    Mean_Avg = sum(Means) / N
    SE_Avg = sqrt(sum(SEs^2)) / N
    """
    means = [s[0] for s in sessions_data]
    ses = [s[1] for s in sessions_data]
    
    # 1. 计算新均值
    avg_mean = sum(means) / n_sessions
    
    # 2. 计算新标准误 (根据独立统计量平均的方差性质)
    # Var_new = (1/N^2) * sum(Var_i) => SE_new = (1/N) * sqrt(sum(SE_i^2))
    sum_sq_se = sum([se**2 for se in ses])
    avg_se = math.sqrt(sum_sq_se) / n_sessions
    
    return avg_mean, avg_se

def format_latex(mean, se):
    return f"{mean:.2f}$\\pm${se:.2f}"

# 主程序：遍历并计算
print("% 以下是计算出的 Avg 行数据，可直接替换 LaTeX 表格中的空白 Avg 行")
print("% 计算公式：Mean = avg(M_i), SE = sqrt(sum(SE_i^2)) / N")
print("-" * 80)

for algo, metrics in data.items():
    print(f"% Algorithm: {algo}")
    row_parts = ["& & Avg"] # 对应 LaTeX 表格的前两列 (Type 和 Algorithm 列在 multirow 中已处理，这里只需补全后部分)
    
    # 注意：LaTeX 表格中，Type 和 Algorithm 列是 multirow 合并的，
    # 所以 Avg 行通常只需要填充 Session 列及之后的指标列。
    # 根据原表结构：Type & Algorithm & Session & Acc & Pre & Rec & F1
    # Avg 行对应位置：空 & 空 & Avg & val & val & val & val
    
    for metric_name, sessions in metrics.items():
        m, s = calculate_avg_row(sessions)
        cell_str = format_latex(m, s)
        row_parts.append(cell_str)
    
    # 拼接成 LaTeX 行格式 (注意开头的 & & 对应跳过 Type 和 Algorithm 列)
    latex_row = " & & ".join(row_parts) + " \\\\"
    # 修正格式以匹配原表：原表 Avg 行在 Session 列写的是 Avg
    # 上面 row_parts 初始化为 ["& & Avg"] 是为了占位，现在需要重新组装以匹配列对齐
    
    # 重新组装以完全匹配 LaTeX 列数 (共 7 列)
    # Col1(Type-merged), Col2(Algo-merged), Col3(Session), Col4(Acc), Col5(Pre), Col6(Rec), Col7(F1)
    final_row = f"& & Avg & {format_latex(*calculate_avg_row(metrics['Accuracy']))} & {format_latex(*calculate_avg_row(metrics['Precision']))} & {format_latex(*calculate_avg_row(metrics['Recall']))} & {format_latex(*calculate_avg_row(metrics['F1 Score']))} \\\\"
    
    print(final_row)
    print("-" * 80)
