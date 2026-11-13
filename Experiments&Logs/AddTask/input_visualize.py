import numpy as np
import matplotlib.pyplot as plt
import LSM_lib

import numpy as np
import matplotlib.pyplot as plt

def plot_input_spikes(spikes: np.ndarray, effective_range: tuple, n_label: int) -> plt.Figure:
    """
    绘制2个神经元的输入脉冲图（Raster Plot），全文字体为 Times New Roman
    
    参数:
        spikes: np.ndarray, 形状必须为 [2, 90]，1表示发放脉冲，0表示无脉冲
        effective_range: tuple, (start, end)，0~89之间的整数，start<end，高亮区域
        n_label: int, 标题中的标签数值
    
    返回:
        plt.Figure: 绘制好的图表对象
    """
    # ------------- 核心：全局设置 Times New Roman 字体 -------------
    plt.rcParams['font.family'] = 'Times New Roman'  # 所有文本字体
    plt.rcParams['mathtext.fontset'] = 'stix'        # LaTeX数学符号匹配 Times New Roman
    plt.rcParams['axes.unicode_minus'] = False       # 兼容负号

    # ------------- 输入参数校验 -------------
    assert spikes.shape == (2, 90), "脉冲数组必须是形状为 [2, 90] 的numpy数组！"
    start, end = effective_range
    assert 0 <= start < end <= 89, "effective_range必须是0-89之间的整数，且左值小于右值！"

    # ------------- 转换数据格式 -------------
    spike_times = []
    for neuron in spikes:
        times = np.where(neuron == 1)[0]
        spike_times.append(times)

    # ------------- 创建画布 -------------
    fig, ax = plt.subplots(figsize=(15*0.8, 3*0.8), dpi=120)

    # ------------- 绘制高亮背景色 -------------
    ax.axvspan(start, end, color='#E6F3FF', alpha=0.8, zorder=0)

    # ------------- 绘制黑色纵向脉冲长方体 -------------
    ax.eventplot(
        spike_times,
        colors='black',
        lineoffsets=[0, 1],
        linelengths=0.6,
        linewidths=5,
        zorder=2
    )

    # ------------- 坐标轴设置（字体统一为 Times New Roman） -------------
    ax.set_xlabel('Time', fontsize=12, fontfamily='Times New Roman')
    ax.set_xlim(0, 89)
    ax.set_yticks([0, 1])
    # LaTeX角标 + Times New Roman字体
    ax.set_yticklabels([r'$x_1$', r'$x_2$'], fontsize=14, fontfamily='Times New Roman')
    ax.set_ylim(-0.5, 1.5)

    # ------------- 标题设置（字体统一为 Times New Roman） -------------
    ax.set_title(
        f"Input Spike Trains of Add Task (label = {n_label})",
        fontsize=14,
        pad=15,
        fontweight='bold',
        fontfamily='Times New Roman'
    )

    plt.tight_layout()
    return fig

if __name__ == "__main__":
    data, labels = LSM_lib.generate_add_task_pair_v2(90, 1, 4, 75)
    data = data[0]
    labels = labels[0]
    fig = plot_input_spikes(data, (89-75, 89), labels)
    fig.savefig("input_spike_trains.svg")
    fig.show()
