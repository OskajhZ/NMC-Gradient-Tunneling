import numpy as np
import matplotlib.pyplot as plt
import h5py

# =============================================================================
# 1. 功率谱检验（归一化：每个频率点功率占总能量比重）
# =============================================================================
def compute_power_spectrum(signals, window_type='hann'):
    """
    计算多通道信号的功率谱（加窗减少频谱泄漏）
    
    参数:
        signals: np.ndarray, 形状 [bands, timesteps]
        window_type: str, 窗口类型 ('hann', 'hamming', 'blackman')
    
    返回:
        power_spectrum: np.ndarray, 形状 [bands, timesteps]
                       每个频率点的功率占总能量的比重（归一化到 [0,1]）
    """
    bands, timesteps = signals.shape
    
    # 选择窗函数
    if window_type == 'hann':
        window = np.hanning(timesteps)
    elif window_type == 'hamming':
        window = np.hamming(timesteps)
    elif window_type == 'blackman':
        window = np.blackman(timesteps)
    else:
        window = np.ones(timesteps)
    
    # 加窗
    windowed_signals = signals * window
    
    # FFT 计算
    fft_result = np.fft.fft(windowed_signals, axis=1)
    
    # 计算功率谱（取模的平方）
    power_spectrum = np.abs(fft_result) ** 2
    
    # 归一化：每个频率点的功率占总能量的比重
    total_energy = power_spectrum.sum(axis=-1, keepdims=True)
    power_spectrum_normalized = power_spectrum / (total_energy + 1e-10)
    
    return power_spectrum_normalized


# =============================================================================
# 2. 相关性系数检验
# =============================================================================
def compute_correlation_with_delays(signals, delays=[0, 2, 4, 6, 8, 10]):
    """
    计算信号与不同时间滞后信号的相关系数绝对值
    
    参数:
        signals: np.ndarray, 形状 [bands, channels, timesteps]
        delays: list, 时间滞后列表
    
    返回:
        correlation_matrix: np.ndarray, 形状 [bands, len(delays)]
    """
    bands, channels, timesteps = signals.shape
    n_delays = len(delays)
    
    correlation_matrix = np.zeros((bands, n_delays))
    
    for i, delay in enumerate(delays):
        # 创建滞后信号（在时间维度上平移）
        if delay == 0:
            lagged_signals = signals.copy()
        else:
            lagged_signals = np.zeros_like(signals)
            lagged_signals[:, :, delay:] = signals[:, :, :-delay]
        
        # 计算每个 band、每个 channel 的相关系数
        band_corrs = []
        
        for band in range(bands):
            channel_corrs = []
            for ch in range(channels):
                x = signals[band, ch, :]
                y = lagged_signals[band, ch, :]
                
                # 计算相关系数（沿时间维度）
                if np.std(x) > 1e-10 and np.std(y) > 1e-10:
                    corr = np.corrcoef(x, y)[0, 1]
                    channel_corrs.append(np.abs(corr))
                else:
                    channel_corrs.append(0.0)
            
            # 沿 channels 维度求平均
            band_corrs.append(np.mean(channel_corrs))
        
        correlation_matrix[:, i] = band_corrs
    
    return correlation_matrix


# =============================================================================
# 3. 结果绘制（修改：移除固定 ylim，添加白噪声参考线）
# =============================================================================
def plot_white_noise_tests(power_spectrum, correlation_matrix, 
                           output_path='white_noise_test.pdf'):
    """
    绘制功率谱和相关性系数检验结果
    
    参数:
        power_spectrum: np.ndarray, 形状 [bands, timesteps]
                       （已归一化，值域 [0,1]）
        correlation_matrix: np.ndarray, 形状 [bands, n_delays]
        output_path: str, 输出文件路径
    """
    # NeurIPS 风格设置
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'legend.fontsize': 8,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'lines.linewidth': 2.0,
        'figure.dpi': 300,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'grid.linestyle': '--',
        'grid.alpha': 0.3,
    })
    
    bands = power_spectrum.shape[0]
    delays = [0, 2, 4, 6, 8, 10]
    timesteps = power_spectrum.shape[1]
    freq_axis = np.arange(timesteps)
    
    # 白噪声理论参考值：均匀功率谱 → 每个频率点功率 = 1/频率点数
    white_noise_ref = 1.0 / timesteps
    
    # 配色方案
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:bands]
    
    fig, axes = plt.subplots(1, 2, figsize=(18*0.7, 4*0.7))
    ax1, ax2 = axes
    
    # ========== 子图 1：功率谱 ==========
    for i in range(bands):
        ax1.plot(freq_axis, power_spectrum[i, :], 
                color=colors[i], linewidth=2.0, 
                label=f'Band {i+1}')
    
    # 【修改①】添加白噪声参考线（y = 1/200）
    ax1.plot([freq_axis.min(), freq_axis.max()],
             [white_noise_ref, white_noise_ref],
             color='grey', linestyle='--', 
             linewidth=1.5, alpha=0.8, 
             label=f'White Noise Ref.')
    
    ax1.set_title('Power Spectrum of NMC Recurrent Part', 
                  fontsize=11, fontweight='bold', pad=10)
    ax1.set_xlabel('Frequency Bin', fontsize=10)
    ax1.set_ylabel('Normalized Power Ratio', fontsize=10)
    
    # 【修改②】移除固定 ylim，改为自适应（但保持非负）
    # 使用已有的 set_adaptive_ylim 函数，或手动计算
    all_ps_values = power_spectrum.flatten()
    ps_min, ps_max = all_ps_values.min(), all_ps_values.max()
    ps_range = ps_max - ps_min
    margin = ps_range * 0.1
    ax1.set_ylim(max(0, ps_min - margin), ps_max + margin)
    
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    ax1.legend(loc='upper right', frameon=False, fontsize=8)
    
    # ========== 子图 2：相关性系数 ==========
    for i in range(bands):
        ax2.plot(delays, correlation_matrix[i, :], 
                color=colors[i], linewidth=2.0, 
                marker='o', markersize=6,
                label=f'Band {i+1}')
    
    ax2.set_title('Strength of Correlation with Different Time Delay', 
                  fontsize=11, fontweight='bold', pad=10)
    ax2.set_xlabel('Time Delay', fontsize=10)
    ax2.set_ylabel('Absolute Value of Correlation Coefficient', fontsize=10)
    ax2.set_xticks(delays)
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    ax2.legend(loc='upper right', frameon=False, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"✓ 图表已保存：{output_path}")
    plt.show()


# =============================================================================
# 4. 主程序
# =============================================================================
if __name__ == "__main__":
    recurrent_dir = "recurrent_traj.h5"
    with h5py.File(recurrent_dir, "r") as root:
        signals = root["recurrent_traj"][0]
        signals = signals - signals.mean(-1, keepdims=True)
    
    print(f"\n{'='*60}")
    print(f"🔬 白噪声检验")
    print(f"{'='*60}")
    print(f"📊 数据形状：{signals.shape}")
    
    try:
        # 步骤 1：功率谱检验
        print("【步骤 1】计算功率谱...")
        signals_for_spectrum = signals[:, 0, ...]
        print(f"   输入功率谱形状：{signals_for_spectrum.shape}")
        
        power_spectrum = compute_power_spectrum(
            signals_for_spectrum,
            window_type='hann'
        )
        print(f"✓ 功率谱形状：{power_spectrum.shape}")
        print(f"✓ 功率谱值域：[{power_spectrum.min():.4f}, {power_spectrum.max():.4f}]")
        print(f"✓ 每 band 总能量和：{power_spectrum.sum(axis=1)}\n")
        
        # 步骤 2：相关性系数检验
        print("【步骤 2】计算相关性系数...")
        delays = [0, 2, 4, 6, 8, 10]
        correlation_matrix = compute_correlation_with_delays(
            signals, 
            delays=delays
        )
        print(f"✓ 相关性矩阵形状：{correlation_matrix.shape}")
        print(f"✓ 时间滞后：{delays}\n")
        
        # 步骤 3：绘制结果
        print("【步骤 3】绘制检验结果...")
        plot_white_noise_tests(
            power_spectrum, 
            correlation_matrix,
            output_path='white_noise_test.svg'
        )
        
        print(f"\n{'='*60}")
        print(f"✅ 白噪声检验完成！")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"\n❌ 错误：{e}")
        import traceback
        traceback.print_exc()
        exit(1)
