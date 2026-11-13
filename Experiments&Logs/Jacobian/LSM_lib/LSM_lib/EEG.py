import os
import re
from typing import List, Tuple

import numpy as np
import scipy
import h5py
from tqdm import tqdm

band_dict = {
        "Theta": (4,8),
        "Alpha": (8, 14),
        "Beta": (14, 31),
        "Gamma": (31, 40)
        }


EEG_axis_note = ["subject", "trial", "electrode", "timestep"]
fEEG_axis_note = ["subject", "trial", "band", "electrode", "timestep"] # filtered EEG

# DEAP channel arrangement can be seen in:
# http://www.eecs.qmul.ac.uk/mmv/datasets/deap/readme.html
# DEAP_14_electrode_indices: Liu 2025 multi-reservoir, Anubhav 2023 Reservoir Splitting
# AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, and AF4.
DEAP_14_electrode_indices = [1, 3, 2, 4, 7, 11, 13, 31, 29, 25, 21, 19, 20, 17]
SEED_14_electrode_indices = [3, 5, 7, 15, 23, 41, 58, 60, 49, 31, 21, 11, 13, 4] 

DEAP_binary_label_threshold = 9/2 # Kolestra 2012 DEAP

class ButterworthFilter():
    def __init__(self, base_signal: np.ndarray, fs, order):
        self.base_signal = base_signal
        self.frequency = fs
        self.order = order
    def filter(self, lowcut, highcut):
        nyq = 0.5 * self.frequency
        low = lowcut / nyq
        high = highcut / nyq
        b, a = scipy.signal.butter(self.order, [low, high], btype="band")
        y = scipy.signal.lfilter(b, a, self.base_signal)
        return y

def filter_1d_signal(
        signal: np.ndarray,
        sample_frequency=128,
        filter_order=6):
    '''
    signal: [len]
    Return: [4, len]
    '''
    frequency_ranges = list(band_dict.values())
    band_num = len(frequency_ranges)
    filterd_signals = np.zeros((band_num, len(signal)))

    ffilter = ButterworthFilter(signal, sample_frequency, filter_order)
    for i, (lowcut, highcut) in enumerate(frequency_ranges):
        filterd_signals[i] = ffilter.filter(lowcut, highcut)

    return filterd_signals

def filter_dataset(signal, sample_frequency=128, filter_order=6):
    '''
    signal: [..., electrode, length]
    return: [..., band, electrode, length]
    '''
    signal_shape = signal.shape
    band_amount = len(band_dict)
    unpacked_signal = signal.reshape(-1, signal_shape[-1])
    filterd_signal = np.zeros((unpacked_signal.shape[0], band_amount, signal_shape[-1]))

    for i, channel in enumerate(unpacked_signal):
        filterd_signal[..., i, :, :] = filter_1d_signal(channel, sample_frequency, filter_order)

    filterd_signal = filterd_signal.reshape(*signal_shape[:-1], band_amount, signal_shape[-1])
    axes = tuple(range(len(filterd_signal.shape)))
    transposed_shape_tuple = (*axes[:-3], axes[-2], axes[-3], axes[-1])
    filterd_signal = np.transpose(filterd_signal, axes=transposed_shape_tuple)
    return filterd_signal

def avg_smooth(signals: np.ndarray, window_len=11):
    '''
    signals: [..., length]
    window_len should be odd
    '''
    if window_len / 2 == window_len // 2:
        raise ValueError("In avg_smooth: window_len should be an odd number.")

    length = signals.shape[-1]
    channel_shape = signals.shape[:-1]

    flat_signals = signals.reshape(-1, length)
    smoothed_signals = np.zeros_like(flat_signals)

    padding_amount = (window_len-1)//2
    start_padding = flat_signals[..., 0][..., np.newaxis]
    start_padding = np.repeat(start_padding, padding_amount, axis=-1)
    end_padding = flat_signals[..., -1][..., np.newaxis]
    end_padding = np.repeat(end_padding, padding_amount, axis=-1)
    padded_signals = np.concatenate([start_padding, flat_signals, end_padding], axis=-1)

    for timestep in range(length):
        start_idx = timestep
        end_idx = timestep + window_len
        window = padded_signals[..., start_idx: end_idx]
        smoothed_signals[..., timestep] = np.mean(window, axis=-1)

    smoothed_signals = smoothed_signals.reshape(*channel_shape, length)
    return smoothed_signals

def batch_wiener(original: np.ndarray, window_len=11):
    '''
    original: [..., timesteps]
    return: the same shape
    '''
    channel_shape = original.shape[:-1]
    length = original.shape[-1]
    original = original.reshape(-1, length)
    filtered = np.zeros_like(original)
    for i,signal in enumerate(original):
        filtered_signal = scipy.signal.wiener(signal, mysize=window_len)
        filtered[i] = filtered_signal
    filtered = filtered.reshape(*channel_shape, length)
    return filtered

def compute_AED(
        filtered_EEG: np.ndarray,
        sample_frequency=128,
        frequency_ranges: List[Tuple[float]]=None
        ) -> np.ndarray:
    '''
    Average Energy Density
    filtered_EEG: [..., band, electrode, timestep]
    If only FES is needed, accurate sample_frequency is not necessary, since the following dimensionless operation.
    '''
    if frequency_ranges is None:
        frequency_ranges = list(band_dict.values())

    AED = filtered_EEG ** 2
    for band_idx, (freq_low, freq_high) in enumerate(frequency_ranges):
        length = filtered_EEG.shape[-1]
        discrete_freq_low = int(freq_low * length / sample_frequency)
        discrete_freq_high = int(freq_high * length / sample_frequency)
        freq_stride = discrete_freq_high - discrete_freq_low + 1
        AED[..., band_idx, :, :] /= freq_stride
    return AED

def compute_dimensionless_coef(AED: np.ndarray):
    '''
    AED: [num, band, electrode, timestep]
    return: [band, electrode]
    '''
    coef = AED.mean(-1).mean(0)
    return coef

def compute_FES(
        AED: np.ndarray,
        dimensionless_coef: np.ndarray):
    '''
    AED: [..., band, electrode, timestep]
    dimensionless_coef: [band, electrode]
    '''
    length = AED.shape[-1]
    coef = dimensionless_coef[..., np.newaxis]
    coef = np.repeat(coef, length, axis=-1)
    dimensionless_AED = AED / coef
    FES = np.log(dimensionless_AED + 1e-9) + 1
    return FES

# Following: Propcess Functions

def preprocess_as_FES(
        train_task: np.ndarray, test_task: np.ndarray,
        use_avg_smooth = True,
        use_wiener = False) -> Tuple[np.ndarray]:
    '''
    Input data should be filterd first.
    data shape: [num, band, electrode, timestep]
    '''
    train_AED = compute_AED(train_task)
    test_AED = compute_AED(test_task)

    if use_avg_smooth:
        train_AED = avg_smooth(train_AED, 11)
        test_AED = avg_smooth(test_AED, 11)

    diml_coef = compute_dimensionless_coef(train_AED)

    train_FES = compute_FES(train_AED, diml_coef)
    test_FES = compute_FES(test_AED, diml_coef)

    if use_wiener:
        train_FES = batch_wiener(train_FES, 11)
        test_FES = batch_wiener(test_FES, 11)

    return train_FES, test_FES

def no_preprocess(train_task, test_task):
    return train_task, test_task

def get_mat_files(folder_path):
    # 存储所有.mat文件名的列表
    mat_files = []
    # 遍历文件夹下的所有条目（文件/目录）
    for entry in os.listdir(folder_path):
        # 拼接完整路径（用于判断是否为文件）
        full_path = os.path.join(folder_path, entry)
        # 筛选：是文件 且 后缀为.mat
        if os.path.isfile(full_path) and entry.endswith('.mat'):
            mat_files.append(full_path)
    return mat_files

def filter_DEAP(folder_path, save_dir):
    '''
    Dump as a single HDF5 file
    '''
    mat_files = get_mat_files(folder_path)
    subject_amount = len(mat_files)
    EEG = np.zeros((subject_amount, 40, 32, 8064))
    labels = np.zeros((subject_amount, 40, 4))
    for i, file_dir in tqdm(enumerate(mat_files), total=len(mat_files)):
        mat = scipy.io.loadmat(file_dir)
        single_data = mat["data"][:, :32, :] # EEG only
        single_labels = mat["labels"]
        EEG[i] = single_data
        labels[i] = single_labels

    labels = (labels > DEAP_binary_label_threshold).astype(float)
    rEEG = filter_dataset(EEG,
            sample_frequency=128,
            filter_order=4)

    print("Saving as {} ...".format(save_dir))
    with h5py.File(save_dir, "w") as root:
        root.create_dataset("data", data=rEEG)
        root.create_dataset("labels", data=labels)

    return rEEG.shape, labels.shape
    
def dump_DEAP(folder_path, save_dir):
    '''
    Dump as a single HDF5 file
    '''
    mat_files = get_mat_files(folder_path)
    subject_amount = len(mat_files)
    EEG = np.zeros((subject_amount, 40, 32, 8064))
    labels = np.zeros((subject_amount, 40, 4))
    for i, file_dir in tqdm(enumerate(mat_files), total=len(mat_files)):
        mat = scipy.io.loadmat(file_dir)
        single_data = mat["data"][:, :32, :] # EEG only
        single_labels = mat["labels"]
        EEG[i] = single_data
        labels[i] = single_labels

    labels = (labels > DEAP_binary_label_threshold).astype(float)
    
    print("Saving as {} ...".format(save_dir))
    with h5py.File(save_dir, "w") as root:
        root.create_dataset("data", data=EEG)
        root.create_dataset("labels", data=labels)

    return EEG.shape, labels.shape

def dump_SEED(folder_path: str, save_dir: str) -> None:
    """
    将SEED数据集Preprocessed_EEG文件夹的EEG数据转储为HDF5文件

    输出HDF5结构：
    - group: session_1, session_2, session_3（按实验日期排序）
      - dataset: data -> 形状[subject, trial, electrode, timestep]，float32类型
      - dataset: labels -> 形状[subject, trial]，float32类型（0负性/1中性/2正性）

    参数:
        folder_path: Preprocessed_EEG文件夹的绝对路径
        save_dir: 输出HDF5文件的绝对路径（含文件名，如"seed_eeg.h5"）
    """
    # -------------------------- 步骤1：读取并处理标签数据 --------------------------
    label_path = os.path.join(folder_path, "label.mat")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"未在{folder_path}中找到label.mat文件")

    # 读取标签（shape: [1, 15]）
    label_data = scipy.io.loadmat(label_path)
    if "label" not in label_data:
        raise KeyError("label.mat文件中未找到'label'关键字")

    raw_labels = label_data["label"]+1  # shape: (1, 15)
    if raw_labels.shape != (1, 15):
        raise ValueError(f"标签形状异常：{raw_labels.shape}，预期(1,15)")

    # 扩展标签为[subject, trial]形状（15个被试×15个trial）
    subject_num = 15  # SEED数据集固定15个被试
    labels = np.tile(raw_labels, (subject_num, 1))  # shape: (15, 15)

    # -------------------------- 步骤2：整理EEG文件（按被试和日期分组） --------------------------
    # 匹配文件名格式："被试编号_日期.mat"（如"1_20131027.mat"）
    file_pattern = re.compile(r"^(\d+)_(\d+)\.mat$")
    eeg_files = [f for f in os.listdir(folder_path) if f.endswith(".mat") and f != "label.mat"]

    # 按被试分组，存储(日期, 文件路径)对
    subject_file_dict = {}
    for filename in eeg_files:
        match = file_pattern.match(filename)
        if not match:
            raise ValueError(f"文件名格式错误：{filename}，预期格式'被试编号_日期.mat'")

        subj_id = int(match.group(1))
        date = int(match.group(2))  # 日期转为整数便于排序
        file_path = os.path.join(folder_path, filename)

        if subj_id not in subject_file_dict:
            subject_file_dict[subj_id] = []
        subject_file_dict[subj_id].append((date, file_path))

    # 验证数据完整性
    if len(subject_file_dict) != subject_num:
        raise ValueError(f"被试数量异常：{len(subject_file_dict)}，预期15")
    for subj_id, files in subject_file_dict.items():
        if len(files) != 3:
            raise ValueError(f"被试{subj_id}的session数异常：{len(files)}，预期3次")

        # 按日期排序（确保session_1→2→3对应实验时间顺序）
        files.sort(key=lambda x: x[0])
        subject_file_dict[subj_id] = files

    # 分配每个session的文件（按被试ID排序，保证顺序一致性）
    session_files = {1: [], 2: [], 3: []}
    for subj_id in sorted(subject_file_dict.keys()):
        for session_idx, (date, file_path) in enumerate(subject_file_dict[subj_id], 1):
            session_files[session_idx].append((subj_id, file_path))

    # -------------------------- 步骤3：处理每个session的EEG数据 --------------------------
    for session in [1, 2, 3]:
        print(f"正在处理session_{session}...")
        current_files = session_files[session]

        # 第一步：收集所有trial的时间步长，确定最小长度（用于对齐）
        all_timesteps = []
        subject_trials_buffer = []  # 临时存储：[被试][trial][62, ts]

        for subj_id, file_path in sorted(current_files, key=lambda x: x[0]):
            # 读取单个被试的单个session数据
            mat_data = scipy.io.loadmat(file_path)
            subj_trials = []

            # ------------- 修改部分：动态匹配{name}_eeg{trial}格式的keys -------------
            # 正则表达式匹配：任意字符+_eeg+数字（捕获trial编号）
            trial_key_pattern = re.compile(r"^(.+)_eeg(\d+)$")
            trial_keys = []
            
            # 遍历所有keys，筛选符合格式的trial key
            for key in mat_data.keys():
                match = trial_key_pattern.match(key)
                if match:
                    trial_num = int(match.group(2))  # 提取trial编号
                    trial_keys.append((trial_num, key))  # 存储(编号, 完整key)
            
            # 按trial编号排序（确保1-15的顺序）
            trial_keys.sort(key=lambda x: x[0])
            # 提取排序后的key列表
            sorted_trial_keys = [key for (num, key) in trial_keys]
            
            # 验证是否包含15个trial
            if len(sorted_trial_keys) != 15:
                raise ValueError(f"文件{file_path}的trial数量异常：{len(sorted_trial_keys)}，预期15")
            # ------------------------------------------------------------------------

            # 提取15个trial（按排序后的key处理）
            for trial_idx, trial_key in enumerate(sorted_trial_keys):
                trial_data = mat_data[trial_key]  # shape: [62, timestep]
                # 验证电极数（SEED为62通道）
                if trial_data.shape[0] != 62:
                    raise ValueError(f"trial{trial_idx+1}（key: {trial_key}）电极数异常：{trial_data.shape[0]}，预期62")

                subj_trials.append(trial_data)
                all_timesteps.append(trial_data.shape[1])

            subject_trials_buffer.append(subj_trials)

        # 确定最小时间步长（统一对齐所有trial）
        min_ts = min(all_timesteps)
        print(f"session_{session}最小时间步长：{min_ts}")

        # 第二步：构建统一形状的data数组 [15, 15, 62, min_ts]
        data_shape = (subject_num, 15, 62, min_ts)
        eeg_data = np.empty(data_shape, dtype=np.float32)

        for subj_idx in range(subject_num):
            for trial_idx in range(15):
                # 裁剪到最小时间步长（保留前min_ts个数据点）
                trial_data = subject_trials_buffer[subj_idx][trial_idx][:, :min_ts]
                eeg_data[subj_idx, trial_idx, :, :] = trial_data

        # -------------------------- 步骤4：写入HDF5文件 --------------------------
        with h5py.File(save_dir, "a") as hdf_file:
            # 创建/覆盖session组
            group_name = f"session_{session}"
            if group_name in hdf_file:
                del hdf_file[group_name]  # 避免重复写入
            session_group = hdf_file.create_group(group_name)

            # 写入数据（使用gzip压缩减少文件体积）
            session_group.create_dataset(
                name="data",
                data=eeg_data,
                dtype=np.float32,
                compression="gzip",
                compression_opts=1  # 压缩级别1（速度与体积平衡）
            )

            # 写入标签
            session_group.create_dataset(
                name="labels",
                data=labels,
                dtype=np.float32
            )

        print(f"session_{session}处理完成\n")

    print(f"所有session处理完成！HDF5文件已保存至：{save_dir}")

def filter_SEED(hdf5_source_dir, obj_dir, session = None):
    '''
    The original file should be the dumped HDF5 SEED file.
    session: string, "session_1" or "session_2" or "session_3" or None
    '''
    if session not in ["session_1", "session_2", "session_3", None]:
        raise ValueError("Invalid session: {}".format(session))
    with h5py.File(obj_dir, "w") as obj:
        pass # Clear the file if it exists
    with h5py.File(hdf5_source_dir, "r") as source:
        for session_no in tqdm(source.keys(), total=3):
            if session is not None and session_no != session:
                continue
            data = source[session_no]["data"][:]
            labels = source[session_no]["labels"][:]
            fdata = filter_dataset(data,
                    sample_frequency=200,
                    filter_order=4)
            with h5py.File(obj_dir, "a") as obj:
                session_group = obj.create_group(session_no)
                session_group.create_dataset("data", data=fdata)
                session_group.create_dataset("labels", data=labels)
    print("Filter Done")

def load_DEAPh5(
        DEAP_dir: str,
        label_type: str,
        selected_electrodes: List[int] = None
        ):
    '''
    return: EEG [subject, trial, band, electrode, timestep],
        label[subject, trial]
    '''
    with h5py.File(DEAP_dir, "r") as root:
        fEEG = root["data"][:]
        labels = root["labels"][:]
    if selected_electrodes is not None:
        fEEG = fEEG[..., selected_electrodes, :]
    label_type_dict = {
            "valence": 0,
            "arousal": 1,
            "dominance": 2,
            "liking": 3}
    label_type_idx = label_type_dict.get(label_type, None)
    if label_type_idx is None:
        raise ValueError("Invalid Label Type: {}.".format(label_type))
    labels = labels[..., label_type_idx]

    return fEEG, labels

def load_three_lead(spectrum_dir):
    '''
    Load Three-lead EEG dataset by BHE.
    spectrum_dir: use original EEG in spectrum.h5
    '''
    with h5py.File(spectrum_dir, "r") as root:
        original_EEG_group = root["OriginalEEG"]
        EEG = original_EEG_group["data"][:]
        labels = original_EEG_group["labels"][:]
    EEG = EEG[:, np.newaxis, ...] # Each subject has only one trial
    return EEG, labels

def load_SEEDh5(
        seed_dir: str,
        session: str = "session_1",
        selected_electrodes: List[int] = None
        ):
    '''
    session: "session_1" or "session_2" or "session_3".
    seed_dir: filtered or unfiltered SEED HDF5 file.
    '''
    with h5py.File(seed_dir, "r") as root:
        session_group = root[session]
        EEG = session_group["data"]
        if selected_electrodes is None:
            EEG = EEG[:]
        else:
            EEG = EEG[:][..., selected_electrodes, :]
        labels = session_group["labels"][:]
    return EEG, labels
