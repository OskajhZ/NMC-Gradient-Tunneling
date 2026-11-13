import LSM_lib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import torch
from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import metrics
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.gridspec as gridspec

def prepare_model(state_dict_dir: str):
    block_length = 20
    CHANNEL_PER_BAND = 25
    INPUT_CHANNEL = 4*CHANNEL_PER_BAND
    NEURON_NUM = 10**3
    feedback_channel = int(NEURON_NUM * 0.25)
    fb_NMC = LSM_lib.reservoir.FeedbackReservoirDefault(
            INPUT_CHANNEL, feedback_channel, NEURON_NUM,
            output_dim=None,  evidence_collection_length=block_length, regenerate_weights=True)
    fb_NMC.config_tunnelers(
            regularization_coef = 1,
            target_firing_rate = 0.2,
            )
    model = LSM_lib.models.AddTaskNMC(fb_NMC, 2)

    state_dict = torch.load(state_dict_dir, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

def select_correct_data(loaded_model, data: torch.Tensor, labels: torch.Tensor):
    loaded_model.eval()
    with torch.no_grad():
        pred_logits = loaded_model(data)
        pred = torch.argmax(pred_logits, dim=-1)
        correct_mask = pred==labels
        pos_mask = labels==1
        neg_mask = ~pos_mask
        correct_pos = pos_mask & correct_mask
        correct_neg = neg_mask & correct_mask

        correct_pos_logits = pred_logits[correct_pos]
        correct_pos_prob = torch.softmax(correct_pos_logits, dim=-1)
        correct_pos_data = data[correct_pos]
        _, pos_idx = torch.median(correct_pos_prob, dim=0)
        pos_data_pair = (correct_pos_data[pos_idx[1]], None)

        correct_neg_logits = pred_logits[correct_neg]
        correct_neg_prob = torch.softmax(correct_neg_logits, dim=-1)
        correct_neg_data = data[correct_neg]
        _, neg_idx = torch.median(correct_neg_prob, dim=0)
        neg_data_pair = (correct_neg_data[pos_idx[0]], None)

        return neg_data_pair, pos_data_pair

def do_inference(loaded_model, single_data: torch.Tensor):
    '''
    return: prob_traj [360, 2], state_traj [360, 1000]
    '''
    loaded_model.eval()
    with torch.no_grad():
        state_traj = loaded_model.reservoir(single_data) # [1, 1000, 360]
        state_traj = torch.transpose(state_traj, 1, 2).squeeze() # [360, 1000]
        logits_traj = loaded_model.readout(state_traj) # [360, 2]
        prob_traj = torch.softmax(logits_traj, dim=-1)
        return prob_traj, state_traj

def traj_decrease(original_traj: np.ndarray):
    '''
    original_traj: [360, 1000]
    return: [360, 3]
    '''
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # 零均值、单位方差
        ("pca", PCA(n_components=50)),
        ('decrease', UMAP(n_components=3, random_state=42, n_neighbors=30, min_dist=0.1))
    ])
    decreased_traj = pipeline.fit_transform(original_traj)
    return decreased_traj
def set_nmi_style():
    plt.rcParams.update({
        # 原有设置保留 ...
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'legend.fontsize': 8,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'lines.linewidth': 2.0,
        'lines.markersize': 4,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.format': 'pdf',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.8,
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        'grid.alpha': 0.3,
        'grid.color': '#CCCCCC',
        'axes.axisbelow': True,
        
        # ========== 新增 SVG 安全设置 ==========
        'path.simplify': False,               # 禁用路径简化
        'path.simplify_threshold': 0.0,       # 彻底关闭
        'svg.fonttype': 'none',               # 避免字体嵌入干扰
        'svg.hashsalt': 'fixed',              # 稳定哈希，避免随机闭合
    })

def plot_spike_train_lanes(spikes: np.ndarray, ax: plt.Axes,
                           title: str = None,
                           lane_labels=['Left', 'Right', 'Recall', 'Noise'],
                           lane_colors=['#FDBF6F', '#1F77B4', '#000000', '#000000'],
                           lane_bg_colors=['#FFF8E6', '#E6F3FF', '#FFFFFF', '#FFFFFF'],
                           spike_marker='.', spike_size=10, spike_alpha=0.8):
    """
    绘制多Lane的Spike Train可视化图（NMI出版级样式）
    """
    # ========== 应用NMI样式 ==========
    ax.tick_params(labelsize=9)

    # ========== 参数验证 ==========
    assert spikes.ndim == 2, f"spikes形状应为 [channels, timesteps]，当前：{spikes.shape}"
    n_channels, n_timesteps = spikes.shape
    n_lanes = len(lane_labels)

    assert n_lanes == 4, f"需要4个Lane标签，当前：{n_lanes}"
    assert n_channels % n_lanes == 0, f"通道数{n_channels}应能被Lane数{n_lanes}整除"

    channels_per_lane = n_channels // n_lanes

    # ========== 绘制Lane背景色 ==========
    for i in range(n_lanes):
        y_bottom = i * channels_per_lane
        y_top = (i + 1) * channels_per_lane - 1
        ax.axhspan(y_bottom - 0.5, y_top + 0.5,
                   color=lane_bg_colors[i], alpha=0.5, zorder=0)

    # ========== 绘制Spike点 ==========
    for lane_idx in range(n_lanes):
        ch_start = lane_idx * channels_per_lane
        ch_end = (lane_idx + 1) * channels_per_lane
        lane_spikes = spikes[ch_start:ch_end, :]
        spike_positions = np.where(lane_spikes == 1)
        spike_channels = spike_positions[0] + ch_start
        spike_times = spike_positions[1]

        if len(spike_times) > 0:
            ax.scatter(spike_times, spike_channels,
                      c=lane_colors[lane_idx],
                      s=spike_size,
                      alpha=spike_alpha,
                      marker=spike_marker,
                      zorder=3,
                      linewidths=0)

    # ========== 设置Y轴 ==========
    ax.set_yticks(np.arange(n_lanes) * channels_per_lane + channels_per_lane // 2)
    ax.set_yticklabels(lane_labels, fontsize=10, fontfamily='Times New Roman')
    ax.set_ylim(-0.5, n_channels - 0.5)

    # ========== 设置X轴 ==========
    ax.set_xlim(0, n_timesteps)
    # 【修改1】已按要求移除第一行子图的 xlabel
    # ax.set_xlabel('Time Step', fontsize=10, fontfamily='Times New Roman')
    ax.tick_params(axis='x', labelsize=9)

    # ========== 标题设置 ==========
    if title:
        ax.set_title(title, fontsize=11, fontweight='bold',
                    fontfamily='Times New Roman')

    # ========== 样式美化 ==========
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.grid(False)
    ax.set_axisbelow(True)

def plot_probability(prob: np.ndarray, ax: plt.Axes,
                     title: str = None,
                     xlabel: str = 'Index',
                     ylabel: str = 'Probability',
                     color: str = '#1f77b4',
                     show_markers: bool = False,
                     fill_area: bool = False,
                     y_lim: tuple = (-0.05, 1.05)) -> plt.Axes:
    """
    绘制0-1范围内的概率分布图，使用逐段线段绘制，避免SVG异常闭合。
    """
    n = len(prob)
    x_axis = np.arange(n)

    # 逐段绘制曲线（每相邻两点独立线段）
    for i in range(n - 1):
        ax.plot([x_axis[i], x_axis[i+1]], [prob[i], prob[i+1]],
                color=color, linewidth=2.0, solid_capstyle='butt',
                label='_nolegend_')

    # 绘制 y=0.5 参考线（单一线段，无闭合风险）
    ax.plot([-0.5, n - 0.5], [0.5, 0.5], color='gray', linestyle='--',
            linewidth=1, label='_nolegend_')

    # 可选：数据点标记
    if show_markers:
        ax.scatter(x_axis, prob, s=12, c=color, marker='o',
                   edgecolors='none', zorder=5, label='_nolegend_')

    # 可选：填充曲线下方面积
    if fill_area:
        ax.fill_between(x_axis, 0, prob, color=color, alpha=0.15,
                        label='_nolegend_')

    # 坐标轴设置
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(y_lim)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(['0', '0.5', '1.0'], fontsize=9)

    if title:
        ax.set_title(title, fontsize=11, fontweight='bold')

    return ax

def plot_3d_trajectory(trajectory: np.ndarray, ax,
                       title: str = None,
                       color_by_time: bool = True,
                       cmap: str = 'viridis',
                       show_start_end: bool = True,
                       start_marker: str = 'o',
                       end_marker: str = 's',
                       marker_size: int = 50,
                       line_alpha: float = 0.8,
                       view_angle: tuple = (25, 45),
                       axis_labels: tuple = ('Dim 1', 'Dim 2', 'Dim 3'),
                       projection_plane: str = 'xy',
                       spike_mask: np.ndarray = None) -> plt.Axes:
    set_nmi_style()
    x, y, z = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]
    timesteps = len(x)
    
    colors = plt.get_cmap(cmap)(np.linspace(0, 1, timesteps))
    ax.scatter(x, y, z, c=colors, s=18, alpha=0.8, edgecolors='none', depthshade=True)
    if spike_mask is not None:
        ax.scatter(x[spike_mask], y[spike_mask], z[spike_mask],
                   c='black', s=30, alpha=1.0, edgecolors='none')

    range_x = x.max() - x.min() or 1.0
    range_y = y.max() - y.min() or 1.0
    range_z = z.max() - z.min() or 1.0
    margin = 0.02
    ax.set_xlim(x.min() - range_x * margin, x.max() + range_x * margin)
    ax.set_ylim(y.min() - range_y * margin, y.max() + range_y * margin)
    ax.set_zlim(z.min() - range_z * margin, z.max() + range_z * margin)

    ax.xaxis.pane.set_visible(True)
    ax.yaxis.pane.set_visible(True)
    ax.zaxis.pane.set_visible(True)
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')
    ax.xaxis.pane.set_facecolor((1,1,1,0.8))
    ax.xaxis.line.set_visible(True)
    ax.yaxis.line.set_visible(True)
    ax.zaxis.line.set_visible(True)
    
    ax.set_xticks([x.min(), x.max()])
    ax.set_yticks([y.min(), y.max()])
    ax.set_zticks([z.min(), z.max()])
    ax.tick_params(axis='both', labelsize=7, pad=-3)

    ax.set_xlabel(axis_labels[0], labelpad=-8, fontsize=9)
    ax.set_ylabel(axis_labels[1], labelpad=-8, fontsize=9)
    ax.set_zlabel(axis_labels[2], labelpad=-8, fontsize=9)

    ax.view_init(elev=view_angle[0], azim=view_angle[1])

    if title:
        ax.set_title(title, fontsize=11, fontweight='bold', y=0.96)

    return ax

def get_all_data(state_dir: str):
    model = prepare_model(state_dir)
    block_length = 40
    CHANNEL_PER_BAND = 25
    data, labels = LSM_lib.generate_evidence_collection(20, cue_block_length=block_length, resting_length=block_length, channel_per_band=CHANNEL_PER_BAND)
    (neg_data, _), (pos_data, _) = select_correct_data(model, data, labels)

    neg_prob_traj, neg_state_traj = do_inference(model, neg_data.unsqueeze(0))
    neg_prob_traj = neg_prob_traj.cpu().numpy()
    neg_state_traj = neg_state_traj.cpu().numpy()
    neg_state_traj = traj_decrease(neg_state_traj)

    pos_prob_traj, pos_state_traj = do_inference(model, pos_data.unsqueeze(0))
    pos_prob_traj = pos_prob_traj.cpu().numpy()
    pos_state_traj = pos_state_traj.cpu().numpy()
    pos_state_traj = traj_decrease(pos_state_traj)

    return (neg_data, neg_prob_traj, neg_state_traj), (pos_data, pos_prob_traj, pos_state_traj)

if __name__ == "__main__":
    state_dir = "FNMC_25/final{step+1}_model.pt"
    (neg_data, neg_prob_traj, neg_state_traj), \
    (pos_data, pos_prob_traj, pos_state_traj) = get_all_data(state_dir)

    set_nmi_style()

    fig = plt.figure(figsize=(14, 10), dpi=300)
    gs = fig.add_gridspec(
        nrows=3, ncols=2,
        height_ratios=[0.9, 0.6, 1.2],
        hspace=0.25,
        wspace=0.25
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, 0], projection='3d')
    ax6 = fig.add_subplot(gs[2, 1], projection='3d')

    # 绘制所有内容
    plot_spike_train_lanes(neg_data, ax1, title="Input Spike Trains (Label = Left)")
    plot_spike_train_lanes(pos_data, ax2, title="Input Spike Trains (Label = Right)")
    plot_probability(neg_prob_traj[:, 0], ax3, title="Output Probability (Left)", xlabel="Time Step")
    plot_probability(pos_prob_traj[:, 1], ax4, title="Output Probability (Right)", xlabel="Time Step")
    plot_3d_trajectory(neg_state_traj, ax5, title="Population State Space (Left)")
    plot_3d_trajectory(pos_state_traj, ax6, title="Population State Space (Right)")

    fig.savefig("inference_results.svg")
    plt.show()
