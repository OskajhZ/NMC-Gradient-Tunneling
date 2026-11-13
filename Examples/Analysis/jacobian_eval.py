import NMC_lib
import torch
import numpy as np
import matplotlib.pyplot as plt

CHANNEL = 1

def generate_spike_pair(base_rate = 0.5, gain = 0.01):
    spikes = NMC_lib.generate_spike_train(CHANNEL, 200, 0.1)
    spikes_0 = NMC_lib.generate_spike_train(1, 200, base_rate)
    spikes_0_var = NMC_lib.generate_spike_train(1, 200, base_rate+gain)
    spikes_var = spikes.clone()
    spikes_var[0] = spikes_0_var
    spikes[0] = spikes_0
    return spikes.unsqueeze(0), spikes_var.unsqueeze(0)

def get_jacobian(reservoir, base_rate = 0.5, gain = 0.01):
    spikes, spikes_var = generate_spike_pair(base_rate, gain)
    reservoir_wrapped = NMC_lib.reservoir.ReservoirOfflineWrapper(reservoir)
    output_rate = reservoir_wrapped(spikes)[..., -1].detach()
    _, jacobian_estimated = reservoir.get_eligibility_jacobian_traces()
    jacobian_estimated = jacobian_estimated[..., 0] # with first input only
    output_rate_var = reservoir_wrapped(spikes_var)[..., -1].detach()
    jacobian_num = (output_rate_var - output_rate) / gain
    connection_mask = (reservoir.input_weights[:,0]!=0)
    return jacobian_estimated[0].numpy(), jacobian_num[0].numpy(), connection_mask.numpy()

def reservoir_prepare(rec_weight_coef):
    reservoir = NMC_lib.reservoir.Reservoir(CHANNEL, 8)
    reservoir.generate_input_weights(density=0.1)
    reservoir.set_monitor_filter_coef(1 - (0.05)**(1/200))
    reservoir.recursive_weights.data *= rec_weight_coef
    return reservoir

def estimate(base_rate, gain=0.03):
    jacobian = {"Reservoir": (None, None), "LIF Layer": (None, None)}
    for i, coef in enumerate([1, 0]):
        reservoir = reservoir_prepare(coef)
        jacobian_estimated, jacobian_num, connection_mask = get_jacobian(reservoir, base_rate, gain)
        jacobian_estimated = jacobian_estimated[connection_mask]
        jacobian_num = jacobian_num[connection_mask]
        key = "Reservoir" if i==0 else "LIF Layer"
        jacobian[key] = (jacobian_estimated, jacobian_num)
    return jacobian

def draw(jacobian, ax_dict):
    color_dict = {"Reservoir": "#1f77b4", "LIF Layer": "#ff7f0e"}
    
    for key, (jacobian_estimated, jacobian_num) in jacobian.items():
        # 线性回归
        slope, intercept = np.polyfit(jacobian_num.reshape(-1), jacobian_estimated.reshape(-1), 1)
        estimated_pred = jacobian_num * slope + intercept
        
        # 计算相关系数 (使用 np.corrcoef)
        corr_matrix = np.corrcoef(jacobian_num.reshape(-1), jacobian_estimated.reshape(-1))
        corr_coef = corr_matrix[0, 1]
        
        # 绘制散点图和回归线
        ax_dict[key].scatter(jacobian_num, jacobian_estimated, 
                            color=color_dict[key], alpha=0.6, s=20, label='Data points')
        ax_dict[key].plot(jacobian_num, estimated_pred, 
                         color=color_dict[key], linewidth=2, 
                         label=f'Fit line')
        
        # 标注相关系数在回归线附近
        # 方法1：动态计算标注位置（在回归线左侧上方）
        x_min, x_max = jacobian_num.min(), jacobian_num.max()
        x_range = x_max - x_min
        x_pos = x_min + 0.1 * x_range  # x轴10%位置
        y_pos = slope * x_pos + intercept  # 对应回归线上的y值
        
        # 添加带背景的相关系数文本
        ax_dict[key].text(x_pos, y_pos, f'r = {corr_coef:.3f}', 
                         color=color_dict[key], fontsize=9, fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                  edgecolor=color_dict[key], alpha=0.8))
        
        # 可选：添加对角线参考线 (y=x)
        min_val = min(jacobian_num.min(), jacobian_estimated.min())
        max_val = max(jacobian_num.max(), jacobian_estimated.max())
        ax_dict[key].plot([min_val, max_val], [min_val, max_val], 
                         'k--', alpha=0.2, linewidth=1, label='y=x')
        
        # 设置标签、标题和图例
        ax_dict[key].set_xlabel("Numerical Jacobian", fontsize=9)
        ax_dict[key].set_ylabel("GT Estimated Jacobian", fontsize=9)
        ax_dict[key].tick_params(labelsize=8)
        ax_dict[key].legend(loc='lower right', fontsize=7, framealpha=0.9)
        ax_dict[key].grid(True, alpha=0.3, linestyle='--')

if __name__=="__main__":
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
    rate_list = [0.1, 0.2, 0.3]
    for i, rate in enumerate(rate_list):
        ax_dict = {"Reservoir": axes[0, i], "LIF Layer": axes[1, i]}
        ax_dict["Reservoir"].set_title(f"Reservoir with Input Firing Rate {rate}")
        ax_dict["LIF Layer"].set_title(f"LIF Layer with Input Firing Rate {rate}")
        jacobian = estimate(rate, 0.05)
        draw(jacobian, ax_dict)
    plt.tight_layout()
    fig.savefig("jacobian_estimation.pdf")
    plt.show()
