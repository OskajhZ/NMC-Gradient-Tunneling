import LSM_lib
import numpy as np
import matplotlib.pyplot as plt
import torch

from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import metrics

class FinalStateGetter():

    def __init__(self, model, data: torch.Tensor, labels: torch.Tensor):
        '''
        data: [batchsize, channel, timestep]
        '''
        self.model = model
        self.data = data
        self.labels = labels
        if torch.cuda.is_available():
            self.data = self.data.cuda()
            self.model = self.model.to("cuda")

    def get_state(self, state_dict_dir) -> torch.Tensor:
        if torch.cuda.is_available():
            state_dict = torch.load(state_dict_dir)
        else:
            state_dict = torch.load(state_dict_dir, map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict)
        self.model.eval()
        with torch.no_grad():
            state = self.model.reservoir(self.data)[..., -1]
            pred = self.model.readout(state)
            pred = torch.argmax(pred, dim=-1)
            acc = metrics.accuracy_score(pred, self.labels)
            print(acc)
        return state

def prepare_model(idx):
    model = None
    if idx in [0,1,2]:
        block_length = 20
        CHANNEL_PER_BAND = 25

        INPUT_CHANNEL = 4*CHANNEL_PER_BAND
        NEURON_NUM = 10**3
        feedback_channel = int(NEURON_NUM * [0.05, 0.15, 0.25][idx])
        fb_NMC = LSM_lib.reservoir.FeedbackReservoirDefault(
                INPUT_CHANNEL, feedback_channel, NEURON_NUM,
                output_dim=None,  evidence_collection_length=block_length, regenerate_weights=True)
        fb_NMC.config_tunnelers(
                regularization_coef = 1,
                target_firing_rate = 0.2,
                )
        model = LSM_lib.models.AddTaskNMC(fb_NMC, 2)
    if idx == 3:
        reservoir = LSM_lib.reservoir.get_in100_edge10_avgfilter_reservoir()
        reservoir.skip_monitor = True
        model = LSM_lib.models.AddTaskNMC(reservoir, 2)
    return model

def do_decrease(original_pts: np.ndarray):
    '''
    original_pts: [group_num, data_num, dim]
    return: [group_num, data_num, 2]
    '''
    pipeline = Pipeline([
    ('scaler', StandardScaler()),  # 零均值、单位方差
    ("pca", PCA(n_components=50)),
    ('decrease', UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1, spread=0.5))
    ])

    final_pts = np.zeros((original_pts.shape[0], original_pts.shape[1], 2))
    for group_idx in range(original_pts.shape[0]):
        pts = original_pts[group_idx]
        processed = pipeline.fit_transform(pts)
        final_pts[group_idx] = processed

    return final_pts

def set_publication_style():
    """设置符合学术会议出版要求的绘图样式"""
    plt.rcParams.update({
        # 字体设置
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'legend.fontsize': 8,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,

        # 线条和标记
        'lines.linewidth': 1.5,
        'lines.markersize': 4,
        'scatter.marker': 'o',

        # 图像质量
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.format': 'pdf',

        # 边框和网格
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.8,
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        'grid.alpha': 0.3,
        'grid.color': '#CCCCCC',
        'axes.axisbelow': True,

        # 颜色循环（色盲友好）
        'axes.prop_cycle': plt.cycler(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    })

def do_plot(points: np.ndarray, labels: np.ndarray,
            step_list=[500, 1500, 2500, 3500],
            output_path: str = None) -> plt.Figure:
    '''
    绘制多子图散点图（出版级样式）

    参数:
        points: np.ndarray, 形状 [group_num=4, data_num, 2]
                每个子图的数据点 [x, y]
        labels: np.ndarray, 形状 [data_num,]
                每个数据点的类别标签 (0 或 1)
        step_list: list, 每个子图对应的迭代步数标签
        output_path: str, 可选，保存图表的文件路径

    返回:
        fig: matplotlib Figure 对象
    '''
    # 应用出版级样式
    set_publication_style()

    # 配色方案（色盲友好 + 高对比度）
    colors = {
        0: {'color': '#1f77b4', 'label': 'Left', 'alpha': 0.6, 's': 2},   # 蓝色
        1: {'color': '#ff7f0e', 'label': 'Right', 'alpha': 0.6, 's': 2}     # 橙色
    }

    n_subplots = len(step_list)
    fig, axes = plt.subplots(1, n_subplots, figsize=(3 * n_subplots, 3.0))

    # 确保 axes 是列表（单个子图时 matplotlib 返回单个 Axes 对象）
    if n_subplots == 1:
        axes = [axes]

    for i, (ax, step) in enumerate(zip(axes, step_list)):
        # 提取当前子图数据
        data = points[i]  # [data_num, 2]

        # 按标签分类
        neg_mask = (labels == 0)
        pos_mask = (labels == 1)
        neg_data = data[neg_mask]
        pos_data = data[pos_mask]

        # 绘制散点（修复原代码：right_data → pos_data）
        if len(pos_data) > 0:
            ax.scatter(pos_data[:, 0], pos_data[:, 1],
                      c=colors[1]['color'],
                      label=colors[1]['label'],
                      alpha=colors[1]['alpha'],
                      s=colors[1]['s'],
                      zorder=3)

        if len(neg_data) > 0:
            ax.scatter(neg_data[:, 0], neg_data[:, 1],
                      c=colors[0]['color'],
                      label=colors[0]['label'],
                      alpha=colors[0]['alpha'],
                      s=colors[0]['s'],
                      zorder=3)
        
        # 子图标题（添加 (a)(b)(c) 编号）
        subplot_label = chr(ord('a') + i)
        ax.set_title(f'Final States of {["5% Feedback NMC", "15% Feedback NMC", "25% Feedback NMC", "Lyapunov-tuned LSM"][i]}',
                    fontsize=11, fontweight='bold', pad=8)

        # 坐标轴标签
        ax.set_xlabel('Dimension 1', fontsize=10)
        ax.set_ylabel('Dimension 2', fontsize=10)

        # 坐标轴范围自适应（保留 10% 边距）
        all_data = np.vstack([neg_data, pos_data]) if len(neg_data) > 0 and len(pos_data) > 0 else data
        if len(all_data) > 0:
            x_min, x_max = all_data[:, 0].min(), all_data[:, 0].max()
            y_min, y_max = all_data[:, 1].min(), all_data[:, 1].max()
            x_margin = (x_max - x_min) * 0.1
            y_margin = (y_max - y_min) * 0.1
            ax.set_xlim(x_min - x_margin, x_max + x_margin)
            ax.set_ylim(y_min - y_margin, y_max + y_margin)

        # 网格线
        ax.grid(True, linestyle='--', alpha=0.3, zorder=0)

        # 图例（自适应位置 + 无框）
        ax.legend(loc='best', frameon=False, fontsize=8)

    # 整体布局调整
    plt.tight_layout(pad=2.0)

    # 保存图表（如果指定路径）
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"✓ 图表已保存：{output_path}")

    return fig

if __name__ == "__main__":
    state_dict_dir_list = [
            "FNMC_5/final{step+1}_model.pt",
            "FNMC_15/final{step+1}_model.pt",
            "FNMC_25/final{step+1}_model.pt",
            "LSM/step500_model.pt"]
    figure_dir = "state_evolution.svg"

    block_length = 40
    CHANNEL_PER_BAND = 25
    data, labels = LSM_lib.generate_evidence_collection(500, cue_block_length=block_length, resting_length=block_length, channel_per_band=CHANNEL_PER_BAND)

    state_list = []
    for i, state_dict_dir in enumerate(state_dict_dir_list):
        model_ref = prepare_model(i)
        state_getter = FinalStateGetter(model_ref, data, labels)
        state = state_getter.get_state(state_dict_dir).unsqueeze(0)
        state_list.append(state)
    total_state_tensor = torch.cat(state_list, dim=0)
    total_state = total_state_tensor.cpu().numpy()
    print("Inference finished")

    state_2d = do_decrease(total_state)

    do_plot(state_2d, labels, output_path=figure_dir)
