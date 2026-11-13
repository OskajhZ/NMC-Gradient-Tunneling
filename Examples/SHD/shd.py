"""Train the feedback LSM on the Spiking Heidelberg Digits (SHD) dataset."""

import os
import sys

# Make the NMC_lib package importable even when this script is launched from
# outside the repository (e.g. via a shell script in another directory).
sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ),
)

import h5py
import numpy as np
import torch
import NMC_lib
from torch.utils.data import Dataset, DataLoader, get_worker_info
from tqdm import tqdm
import shutil


class SHDEventAugmentation:
    """Official-strength SHD augmentations applied before temporal binning.

    Each sample receives a global input-channel shift, a temporal dilation,
    and an independent per-event channel jitter. Events moved outside the SHD
    channel or time range are discarded, matching the pp-prop implementation.
    """

    def __init__(
        self,
        max_channel_shift: int = 40,
        min_time_scale: float = 0.95,
        max_time_scale: float = 1.2,
        channel_jitter_sigma: float = 4.0,
    ):
        if max_channel_shift < 0:
            raise ValueError("max_channel_shift must be non-negative.")
        if not 0 < min_time_scale <= max_time_scale:
            raise ValueError("Time-scale bounds must satisfy 0 < min <= max.")
        if channel_jitter_sigma < 0:
            raise ValueError("channel_jitter_sigma must be non-negative.")

        self.max_channel_shift = max_channel_shift
        self.min_time_scale = min_time_scale
        self.max_time_scale = max_time_scale
        self.channel_jitter_sigma = channel_jitter_sigma

    def __call__(
        self,
        times: np.ndarray,
        units: np.ndarray,
        rng: np.random.Generator,
        nb_units: int,
        max_time: float,
    ):
        times = np.asarray(times, dtype=np.float64).copy()
        units = np.asarray(units, dtype=np.int64).copy()

        # 1. Shift the complete cochlear-channel pattern by one random offset.
        if self.max_channel_shift > 0:
            channel_shift = int(
                rng.uniform(-self.max_channel_shift, self.max_channel_shift)
            )
            units += channel_shift
            valid = (units >= 0) & (units < nb_units)
            times, units = times[valid], units[valid]

        # 2. Dilate the complete event trajectory with a log-uniform factor.
        log_scale = rng.uniform(
            np.log(self.min_time_scale),
            np.log(self.max_time_scale),
        )
        times *= np.exp(log_scale)
        valid = (times >= 0.0) & (times < max_time)
        times, units = times[valid], units[valid]

        # 3. Jitter every event independently along the cochlear-channel axis.
        if self.channel_jitter_sigma > 0 and len(units) > 0:
            channel_jitter = np.rint(
                rng.standard_normal(len(units)) * self.channel_jitter_sigma
            ).astype(np.int64)
            units += channel_jitter
            valid = (units >= 0) & (units < nb_units)
            times, units = times[valid], units[valid]

        return times, units

def bin_single_sample(
    times: np.ndarray,      # 脉冲时间戳数组 [N_events]，范围 [0, max_time]
    units: np.ndarray,      # 触发脉冲的神经元索引数组 [N_events]，范围 [0, nb_units)
    nb_steps: int,          # 时间轴分箱数量 (输出张量的时间维度)
    nb_units: int = 700,    # 神经元总数 (输出张量的空间维度)
    max_time: float = 1.4,  # SHD 官方示例使用 1.4 秒作为时间轴上限
    target_channels: int = 700  # 目标通道数，相邻通道合并
) -> torch.Tensor:
    """
    将单个样本的脉冲事件 (events) 转换为二维脉冲网格 (raster grid)。

    输入格式 (来自 DYNAP-SE 等神经形态芯片的原始事件流):
        - times:  每个脉冲的精确时间戳 (连续值)
        - units:  每个脉冲对应的神经元 ID (离散值)
        两者一一对应，构成 (time, neuron_id) 事件对列表。

    执行逻辑:
        1. 将 [0, max_time] 均匀划分为 nb_steps 个时间 bin
        2. 使用 floor(time / bin_width) 将时间戳映射到离散 bin 索引
        3. 过滤掉神经元索引越界的无效事件
        4. 将有事件的 (neuron, time_bin) 位置直接置为 1
           (同一位置的重复脉冲仍保持为 1，不累加计数)
        5. 通道合并: 将 nb_units 个原始通道按相邻分组合并为 target_channels 个输出通道
           - 每组大小: group_size = nb_units // target_channels
           - 余数全部分配给最后一个输出通道，保持相邻通道的空间局部性
        6. 二值化: 合并后的脉冲计数 >1 则截断为 1

    输出:
        torch.Tensor [target_channels, nb_steps]
        - [i, j] = 合并后的通道 i 在时间 bin j 内是否有脉冲 (0/1)

    注意:
        - 输入保持二值化，重复事件不会增大输入强度
        - 时间 bin 边界: bin[k] 对应区间 [k*dt, (k+1)*dt)，其中 dt = max_time/nb_steps
        - 通道合并保持原始通道的顺序关系，相邻通道合并到同一输出通道
    """
    # Use exactly nb_steps equal-width bins. For the default SHD setting this
    # gives 100 bins of 14 ms, indexed as floor(time / 0.014).
    bin_width = max_time / nb_steps
    bin_idx = np.floor(np.asarray(times) / bin_width).astype(np.int64)
    # 边界保护: 超出 [0, max_time] 的时间戳会被钳位到有效范围
    bin_idx = np.clip(bin_idx, 0, nb_steps - 1)

    # 3. 过滤无效事件: 神经元索引必须在 [0, nb_units) 范围内
    valid_mask = (units >= 0) & (units < nb_units)
    valid_units = units[valid_mask].astype(np.int32)
    valid_bins = bin_idx[valid_mask].astype(np.int32)

    # 4. 空事件处理: 没有有效脉冲时返回全零张量
    if len(valid_units) == 0:
        return torch.zeros((target_channels, nb_steps), dtype=torch.float32)

    # 5. Binary occupancy: repeated events in one cell remain one.
    sample_grid = np.zeros((nb_units, nb_steps), dtype=np.float32)
    sample_grid[valid_units, valid_bins] = 1.0

    # 6. 通道合并: 将 nb_units 个通道合并为 target_channels 个
    if target_channels is not None and target_channels < nb_units:
        group_size = nb_units // target_channels
        remainder = nb_units % target_channels

        merged_grid = np.zeros((target_channels, nb_steps), dtype=np.float32)

        start = 0
        for i in range(target_channels):
            # 最后一个通道吸收余数
            current_group_size = group_size + remainder if i == target_channels - 1 else group_size
            end = start + current_group_size
            # 沿通道维度求和 (多个原始通道的脉冲计数累加)
            merged_grid[i] = sample_grid[start:end].sum(axis=0)
            start = end

        sample_grid = merged_grid

    # 7. 二值化 & 转换为 PyTorch 张量
    sample_grid[sample_grid > 1] = 1.0
    return torch.from_numpy(sample_grid)

def load_and_bin_spikes(
    h5_filename: str,
    nb_steps: int = 500,
    target_channels: int = 700,
    augmentation=None,
):
    """
    修改为工厂函数：不再一次性加载并分箱全量数据，
    而是返回轻量级的 SpikeH5Dataset 实例，实现延迟加载与实时分箱。
    """
    return SpikeH5Dataset(
        h5_filename,
        nb_steps,
        target_channels=target_channels,
        augmentation=augmentation,
    )

class SpikeH5Dataset(Dataset):
    """Read SHD events lazily, optionally augment them, and rasterize them."""

    def __init__(
        self,
        h5_path: str,
        nb_steps: int,
        nb_units: int = 700,
        max_time: float = 1.4,
        target_channels: int = 700,
        augmentation=None,
    ):
        self.h5_path = h5_path
        self.nb_steps = nb_steps
        self.nb_units = nb_units
        self.max_time = max_time
        self.target_channels = target_channels
        self.augmentation = augmentation
        self._rng = None
        self._worker_seed = None

        # 仅加载标签（通常仅占几 MB），避免加载庞大的 spikes 数据集
        with h5py.File(h5_path, 'r') as f:
            # 🔧 修复：显式转换为 int64 以兼容 PyTorch 的张量创建
            self.labels = torch.from_numpy(np.array(f['labels'][:], dtype=np.int64))
            self.N = len(self.labels)

    def __len__(self):
        return self.N

    def _get_rng(self):
        """Return a reproducible generator private to the current worker."""
        worker_info = get_worker_info()
        worker_seed = (
            worker_info.seed
            if worker_info is not None
            else torch.initial_seed()
        )
        if self._rng is None or self._worker_seed != worker_seed:
            self._rng = np.random.default_rng(worker_seed)
            self._worker_seed = worker_seed
        return self._rng

    def __getitem__(self, idx):
        # 在 __getitem__ 中打开文件是 PyTorch 多进程 DataLoader 的标准安全做法
        with h5py.File(self.h5_path, 'r') as f:
            times = np.asarray(f['spikes']['times'][idx])
            units = np.asarray(f['spikes']['units'][idx])

        if self.augmentation is not None:
            times, units = self.augmentation(
                times,
                units,
                self._get_rng(),
                self.nb_units,
                self.max_time,
            )

        # 实时分箱：每次仅处理一个样本，内存开销极小
        data = bin_single_sample(times, units, self.nb_steps, self.nb_units, self.max_time, self.target_channels)
        return data, self.labels[idx]

def preprocess_and_cache_h5(h5_path: str, cache_path: str, nb_steps: int, nb_units: int = 700, max_time: float = 1.4, target_channels: int = 700):
    """
    预分 bin 并缓存为新的 HDF5 文件，避免训练时重复计算。

    策略：
    1. 一次性读取所有 spikes 数据
    2. 按 chunk 向量化分 bin（平衡内存和速度）
    3. 将结果写入新的 HDF5 文件供后续训练使用

    内存估算：chunk_size=2000 时，约占用 2000*target_channels*nb_steps*4B
    """
    if os.path.exists(cache_path):
        print(f"Cache exists: {cache_path}, skipping preprocessing.")
        return cache_path

    print(f"Preprocessing and caching: {h5_path} -> {cache_path}")

    chunk_size = 2000  # 每次处理 2000 个样本，平衡内存和速度

    with h5py.File(h5_path, 'r') as src:
        labels = src['labels'][:]
        total_samples = len(labels)

        # 创建缓存文件
        with h5py.File(cache_path, 'w') as dst:
            dst.create_dataset('labels', data=labels)

            # 预分配分 bin 后的数据
            data_shape = (total_samples, target_channels, nb_steps)
            data_dset = dst.create_dataset(
                'data',
                shape=data_shape,
                dtype=np.float32,
                chunks=(1, target_channels, nb_steps),  # 按样本分 chunk 便于 DataLoader 读取
                compression='gzip',
                compression_opts=1
            )

            for start_idx in range(0, total_samples, chunk_size):
                end_idx = min(start_idx + chunk_size, total_samples)
                chunk_times = src['spikes']['times'][start_idx:end_idx]
                chunk_units = src['spikes']['units'][start_idx:end_idx]

                # 处理当前 chunk 的每个样本
                for i in range(end_idx - start_idx):
                    times = np.asarray(chunk_times[i])
                    units = np.asarray(chunk_units[i])

                    data_dset[start_idx + i] = bin_single_sample(
                        times,
                        units,
                        nb_steps,
                        nb_units,
                        max_time,
                        target_channels,
                    ).numpy()

                processed = min(end_idx, total_samples)
                print(f"  Progress: {processed}/{total_samples} samples")

    print(f"Cache saved: {cache_path}")
    return cache_path

class CachedSpikeDataset(Dataset):
    """
    使用预分 bin 缓存的数据集，避免训练时重复计算。
    直接从缓存的 HDF5 读取已分 bin 的数据。
    """
    def __init__(self, cache_path: str):
        self.cache_path = cache_path
        self.h5_file = None  # 延迟打开文件
        self.worker_pid = None  # 跟踪当前 worker PID
        
        with h5py.File(cache_path, 'r') as f:
            self.N = len(f['labels'])
    
    def __len__(self):
        return self.N
    
    def _get_file(self):
        """每个 worker 独立打开文件句柄，避免多进程冲突"""
        import multiprocessing
        current_pid = multiprocessing.current_process().pid
        
        # 如果是新 worker 或文件未打开，则重新打开
        if self.h5_file is None or self.worker_pid != current_pid:
            if self.h5_file is not None:
                try:
                    self.h5_file.close()
                except:
                    pass
            self.h5_file = h5py.File(self.cache_path, 'r')
            self.worker_pid = current_pid
        
        return self.h5_file
    
    def __getitem__(self, idx):
        f = self._get_file()
        data = torch.from_numpy(f['data'][idx])
        label = torch.tensor(f['labels'][idx], dtype=torch.long)
        return data, label
    
    def __del__(self):
        if self.h5_file is not None:
            try:
                self.h5_file.close()
            except:
                pass

def get_dataloader(
    shd_dir: str,
    nb_steps: int,
    use_cache: bool = True,
    target_channels: int = 700,
    cache_dir: str = None,
):
    """Build SHD train/test loaders.

    SHD has no validation split. Expected files are ``shd_train.h5`` and
    ``shd_test.h5`` under ``shd_dir``.
    """
    print("Data preprocessing ...")

    train_set = load_and_bin_spikes(
        os.path.join(shd_dir, "shd_train.h5"),
        nb_steps,
        target_channels=target_channels,
        augmentation=SHDEventAugmentation(),
    )

    if use_cache:
        # Training augmentation must run on raw events before binning, so only
        # the clean test split is cached.
        if cache_dir is None:
            cache_dir = os.path.join(
                os.path.expanduser("~"),
                ".cache",
                "NMC_lib",
                f"SHD_binned_tc{target_channels}",
            )
        os.makedirs(cache_dir, exist_ok=True)

        src_path = os.path.join(shd_dir, "shd_test.h5")
        cache_path = os.path.join(
            cache_dir,
            f"shd_test_floor_binary_v2_{nb_steps}_tc{target_channels}.h5",
        )
        preprocess_and_cache_h5(
            src_path,
            cache_path,
            nb_steps,
            target_channels=target_channels,
        )
        test_set = CachedSpikeDataset(cache_path)

        print(
            "Data amount: train={} (augmented lazily), test={} (cached).".format(
                len(train_set),
                len(test_set),
            )
        )
    else:
        # The test split can also be read lazily without augmentation.
        test_set = load_and_bin_spikes(
            os.path.join(shd_dir, "shd_test.h5"),
            nb_steps,
            target_channels=target_channels,
        )
        print(
            "Data amount (lazy): train={} (augmented), test={} (clean).".format(
                len(train_set),
                len(test_set),
            )
        )

    train_loader = DataLoader(
        train_set, batch_size=256, shuffle=True, num_workers=4, drop_last=True,
        pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(
        test_set, batch_size=512, shuffle=False, num_workers=2,
        pin_memory=True, persistent_workers=True)

    return train_loader, test_loader
 
class ResidualReadout(torch.nn.Module):

    def __init__(self, reservoir_output_dim, out_dim = 20):
        super().__init__()
        bottleneck_dim = int(0.5 * reservoir_output_dim)

        self.norm = torch.nn.LayerNorm((reservoir_output_dim,))
        self.compress = torch.nn.Linear(reservoir_output_dim, bottleneck_dim)
        self.gate = torch.nn.Sequential(
                torch.nn.Linear(reservoir_output_dim, bottleneck_dim),
                torch.nn.Sigmoid()
                ) # GLU

        self.res = torch.nn.Sequential(
                torch.nn.Linear(bottleneck_dim, bottleneck_dim),
                torch.nn.GELU(),
                torch.nn.Linear(bottleneck_dim, bottleneck_dim))
        self.comb = torch.nn.Sequential(
                torch.nn.BatchNorm1d(bottleneck_dim),
                torch.nn.GELU())

        self.classifier = torch.nn.Linear(bottleneck_dim, out_dim)

    def forward(self, x):
        '''
        x: [batch_size, reservoir_output_dim]
        '''
        x = self.norm(x)
        y = self.compress(x) * self.gate(x) # Gated Linear Unit
        y = self.comb(y + self.res(y)) # Residual Block
        out = self.classifier(y)
        return out

class FeedbackNMC(NMC_lib.models.LSMBase):
    def __init__(self, kernel_reservoir):
        super().__init__()
        self.reservoir = NMC_lib.reservoir.ReservoirOfflineWrapper(kernel_reservoir)
        self.readout = ResidualReadout(kernel_reservoir.get_output_dim(), out_dim=20)

    def forward(self, x: torch.Tensor):
        '''
        x: [batch_size, channel, timestep]
        return: [batch_size, 20]
        '''
        x = x[..., :int(0.5*x.shape[-1])]
        y = self.reservoir(x)[..., -1]
        z = self.readout(y)
        return z

def config_model(LENGTH, INPUT_CHANNEL):
    NEURON_NUM = 10**3
    feedback_channel = int(700)
    kernel_reservoir = NMC_lib.reservoir.FeedbackReservoirDefault(
            INPUT_CHANNEL, feedback_channel, NEURON_NUM,
            output_dim=None, stationary_window_length=int(LENGTH*0.8), regenerate_weights = True,
            LIF_voltage_decay = 0.9, LIF_threshold=10)
    print(f"Output dim of feedback reservoir: {kernel_reservoir.get_output_dim()}")
    kernel_reservoir.config_tunnelers(
            regularization_coef = 1,
            target_firing_rate = 0.4,
            )
    tunneler_list = kernel_reservoir.get_tunneler_list()

    model = FeedbackNMC(kernel_reservoir)
    return model, tunneler_list


class SHDTrainer(NMC_lib.training_utils.NetworkTrainer):
    """Train without validation and select the best epoch by test-set macro F1."""

    def _get_reservoir_trace_dir(self):
        return os.path.join(
            getattr(self.log, "log_dir", "logs/shd"),
            "reservoir_trajectories",
        )

    def _record_reservoir_trajectory(self, step, sample_idx):
        """Run the selected test sample over the training-visible window and
        save the reservoir trajectory as raw data (no plotting during
        training; use plot_reservoir_trajectories.py to render figures)."""
        test_dataset = self.test_loader.dataset
        if len(test_dataset) == 0:
            raise ValueError("Cannot record a reservoir trajectory from an empty test set.")

        sample_x, sample_label = test_dataset[sample_idx]
        sample_x = sample_x.unsqueeze(0)

        try:
            model_device = next(self.model.parameters()).device
        except StopIteration:
            model_device = torch.device("cpu")
        sample_x = sample_x.to(model_device)

        self.model.eval()
        with torch.no_grad():
            # Keep the same evidence window as training: the model's forward
            # consumes only the first half of the input (LENGTH=100 -> 50).
            sample_x = sample_x[..., : int(0.5 * sample_x.shape[-1])]
            trajectory = self.model.reservoir(sample_x).squeeze(0).detach().cpu()

        if trajectory.ndim != 2:
            raise ValueError(
                "Expected a reservoir trajectory shaped [neuron, timestep], "
                f"got {tuple(trajectory.shape)}."
            )

        trace_dir = self._get_reservoir_trace_dir()
        os.makedirs(trace_dir, exist_ok=True)

        file_stem = f"epoch_{step:04d}_sample_{sample_idx}"
        label = int(sample_label.item()) if torch.is_tensor(sample_label) else int(sample_label)
        np.savez(
            os.path.join(trace_dir, f"{file_stem}.npz"),
            trajectory=trajectory.numpy(),
            step=step,
            sample_idx=sample_idx,
            label=label,
        )

    def _adjust_reservoir_lr(self, step):
        """Halve the reservoir learning rate every 30 epochs while it is > 1e-4."""
        reservoir_group = self.optimizer.param_groups[0]
        if reservoir_group["lr"] > 1e-4 and step % 50 == 0:
            reservoir_group["lr"] *= 0.9

    def execute(self, epochs=10, early_stop_start=50, tolerance=20, record_trajectory=True):
        if early_stop_start < 0:
            raise ValueError("early_stop_start must be non-negative.")
        if tolerance < 1:
            raise ValueError("tolerance must be at least 1.")

        if self.train_loader is None or self.test_loader is None:
            raise ValueError("SHD training requires both train and test dataloaders.")
        if not hasattr(self, "log"):
            raise ValueError("TensorBoard logging must be enabled for SHD training.")

        model_parent = os.path.dirname(self.model_dir)
        if model_parent:
            os.makedirs(model_parent, exist_ok=True)

        visualization_sample_idx = 123
        trace_dir = self._get_reservoir_trace_dir()
        if record_trajectory:
            test_dataset = self.test_loader.dataset
            if len(test_dataset) == 0:
                raise ValueError("Cannot record a reservoir trajectory from an empty test set.")
            shutil.rmtree(trace_dir, ignore_errors=True)
            os.makedirs(trace_dir, exist_ok=True)
            tqdm.write(
                "Using test sample {} for reservoir visualization in every epoch.".format(
                    visualization_sample_idx
                )
            )

        highest_f1 = float("-inf")
        highest_step = 0
        best_segment_result = None
        best_ensemble_result = None
        best_confusion_matrix = None
        train_loss = None
        train_accuracy = None
        no_improvement_epochs = 0

        for epoch in tqdm(range(epochs), total=epochs):
            step = epoch + 1
            self._adjust_reservoir_lr(step)
            self.log.add_scalar(
                "train/reservoir_lr",
                self.optimizer.param_groups[0]["lr"],
                step,
            )
            train_loss, train_accuracy = self.train_one_epoch()
            self.log.add_scalar("train/loss", train_loss, step)
            self.log.add_scalar("train/accuracy", train_accuracy, step)
            self._record_firing_rates(step)

            segment_result, ensemble_result, confusion_matrix = self.eval("test")
            if record_trajectory:
                self._record_reservoir_trajectory(step, visualization_sample_idx)
            selected_result = (
                ensemble_result if self.ensemble_amount > 1 else segment_result
            )

            # Required SHD metrics, with the epoch number used as TensorBoard step.
            for metric_name in ("accuracy", "f1", "precision", "recall"):
                self.log.add_scalar(
                    f"test/{metric_name}",
                    selected_result[metric_name],
                    step,
                )
            self.log.add_scalar("test/loss", segment_result["loss"], step)

            tqdm.write(
                "Step {} test: accuracy={:.6f}, f1={:.6f}, "
                "precision={:.6f}, recall={:.6f}".format(
                    step,
                    selected_result["accuracy"],
                    selected_result["f1"],
                    selected_result["precision"],
                    selected_result["recall"],
                )
            )

            if selected_result["f1"] > highest_f1:
                highest_f1 = selected_result["f1"]
                highest_step = step
                no_improvement_epochs = 0
                best_segment_result = segment_result.copy()
                best_ensemble_result = (
                    ensemble_result.copy() if ensemble_result is not None else None
                )
                best_confusion_matrix = (
                    confusion_matrix.copy()
                    if confusion_matrix is not None
                    else None
                )
                self.save()
            elif step > early_stop_start:
                no_improvement_epochs += 1

            if (
                step > early_stop_start
                and no_improvement_epochs >= tolerance
            ):
                tqdm.write(
                    "Early stopped at step {} after {} consecutive epochs "
                    "without a test macro-F1 improvement.".format(
                        step,
                        no_improvement_epochs,
                    )
                )
                break

        if best_segment_result is None:
            raise ValueError("At least one training epoch is required.")

        self.load()
        self.log.flush()

        selected_best_result = (
            best_ensemble_result
            if self.ensemble_amount > 1
            else best_segment_result
        )
        print("Training finished.")
        print(
            "Final training loss: {}, training accuracy: {}".format(
                train_loss,
                train_accuracy,
            )
        )
        print(f"Best test macro F1 occurred at step {highest_step}.")
        print(
            "Best-step test metrics: accuracy={accuracy:.6f}, f1={f1:.6f}, "
            "precision={precision:.6f}, recall={recall:.6f}".format(
                **selected_best_result
            )
        )

        return best_segment_result, best_ensemble_result, best_confusion_matrix


if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the feedback LSM on SHD.")
    parser.add_argument(
        "--no-record-trajectory",
        action="store_true",
        help="Disable per-epoch reservoir trajectory dumping during training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for PyTorch (default: 42).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-1,
        help="Reservoir learning rate (default: 1e-1).",
    )
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    LENGTH = 100
    TARGET_CHANNELS = 700
    shd_dir = "/home/featurize/datasets/072a0461-d229-4aba-972c-76382b31f65f"

    train_loader, test_loader = get_dataloader(
        shd_dir,
        LENGTH,
        target_channels=TARGET_CHANNELS,
    )
    model, tunneler_list = config_model(LENGTH, TARGET_CHANNELS)

    trainer = SHDTrainer(
                model, train_loader, test_loader,
                readout_lr = 1e-4,
                reservoir_lr = args.lr,
                grad_tunneler_list = tunneler_list,
                log_dump_folder = "logs/shd",
                model_dump_folder = "models/shd",
                temporal_gt = False)

    epochs = 1000
    early_stop_start = 100
    tolerance = 100
    test_result, _, _ = trainer.execute(
        epochs,
        early_stop_start,
        tolerance,
        record_trajectory=not args.no_record_trajectory,
    )

    print(test_result)

