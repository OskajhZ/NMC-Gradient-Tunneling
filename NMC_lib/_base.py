'''
By Xiangnan Zhang, 2025
    School of Future Technologies, Beijing Institute of Technology

Core primitives for Gradient Tunneling (GT) training of spiking Liquid State Machines.
Implements: ALIF adaptive neurons, low-pass / iterative-moving-average filters,
STDP-based causality monitors, firing-rate monitors, the time-domain traversal loop
(_time_traverse), the eligibility/jacobian trace computation, and the GradientTunneler
class that orchestrates the weight-gradient bypass through the causality trace.
Also contains helper functions for synthetic spike-train generation (AddTask, Evidence
Collection) and BSA encoding for analog-to-spike conversion.
'''
import random
import warnings
import numpy as np
import scipy
import torch
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from typing import Callable, Tuple

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

def count_nonzero_trainable_params(module: torch.nn.Module) -> int:
    """
    Count the number of non-zero trainable parameter elements in a module.

    Args:
        module (torch.nn.Module): the neural network module

    Returns:
        int: total number of non-zero scalar elements among trainable params
    """
    total_nonzero = 0
    for param in module.parameters():
        if param.requires_grad:
            # torch.count_nonzero efficiently counts non-zero elements
            total_nonzero += torch.count_nonzero(param).item()
    return total_nonzero

def print_gpu_mem():
    allocated = torch.cuda.memory_allocated() / 1024**2  # MB
    reserved = torch.cuda.memory_reserved() / 1024**2
    print(f"Allocated: {allocated:.2f}MB, Reserved: {reserved:.2f}MB")

@torch.compile
def are_spikes(x: torch.Tensor) -> bool:
    not_zeros = x!=0
    not_ones = x!=1
    invalid = not_zeros & not_ones
    judgement = invalid.sum() == 0
    return judgement

def generate_spike_train(channel: int, length: int, ratio: float) -> torch.Tensor:
    '''
    return: a 2-D tensor, [channel, timestep]
    '''
    spike_train = torch.zeros((channel, length), dtype=torch.float32)
    spike_amount = int(channel*length*ratio)
    total = channel*length
    spike_idx = torch.randperm(total)[:spike_amount]
    spike_train = spike_train.view(-1)
    spike_train[spike_idx] = 1.0
    spike_train = spike_train.view(channel, length)
    return spike_train

def generate_add_task_pair(half_length, batchsize, bit=3, gating_period = None, silent_period=0):
    '''
    return: data [batchsize, channel=2, length], label [batchsize,]
    '''
    if gating_period is None:
        gating_perod = 2 * half_length

    # effective sequence length after removing silent_period
    L = 2 * half_length - silent_period
    assert L >= 2 * bit + 1, "Effective sequence too short for two spike segments and minimum gap."

    # generate full-length random spike train [batchsize, L]
    p = 0.5
    data_random = generate_spike_train(batchsize, L, p)
    data_gating = torch.zeros_like(data_random)

    # randomly sample start positions of two spike segments
    # constraint: s1 must leave room for both segments + at least 1 step gap
    max_s1 = L - 2 * bit
    min_s1 = L - gating_period
    if min_s1 < 0:
        min_s1 = 0
    elif min_s1 > max_s1:
        min_s1 = max_s1 - 1
    s1 = torch.randint(min_s1, max_s1, (batchsize,))
    
    # s2 valid range: [s1 + bit + 1, L - bit]
    # dynamic offset sampling (compatible with older PyTorch)
    offset_range = L - 2 * bit - s1
    offset = torch.floor(torch.rand(batchsize, device=data_random.device) * offset_range).long()
    s2 = s1 + bit + 1 + offset

    # build gating mask and binary decoding
    row_idx = torch.arange(batchsize).unsqueeze(1)
    idx1 = s1.unsqueeze(1) + torch.arange(bit)  # segment 1 indices [batchsize, bit]
    idx2 = s2.unsqueeze(1) + torch.arange(bit)  # segment 2 indices [batchsize, bit]

    # vectorised gating signal
    data_gating[row_idx, idx1] = 1
    data_gating[row_idx, idx2] = 1
    
    # extract spike values at corresponding positions
    spikes1 = torch.gather(data_random, 1, idx1)
    spikes2 = torch.gather(data_random, 1, idx2)

    # binary weighted decoding (MSB first: 2^(bit-1) ~ 2^0)
    weight = (2 ** torch.arange(bit-1, -1, -1, device=data_random.device)).unsqueeze(0)
    num1 = (spikes1 * weight).sum(dim=1)
    num2 = (spikes2 * weight).sum(dim=1)

    labels = (num1 + num2).long()

    # assemble data
    data = torch.stack([data_random, data_gating], dim=1)
    
    if silent_period > 0:
        silent = torch.zeros(batchsize, 2, silent_period, device=data.device)
        data = torch.cat([data, silent], dim=-1)
        
    return data.to(dtype=torch.float32), labels.to(dtype=torch.long)

def generate_add_task_pair_v2(length, batchsize, bit=3, gating_period = None, silent_period=0):
    '''
    return: data [batchsize, channel=2, length], label [batchsize,]
    '''
    # effective sequence length after removing silent_period
    L = length - silent_period

    # generate full-length random spike train [batchsize, L]
    data_random = generate_spike_train(batchsize, L, 0.5)
    data_gating = torch.zeros(batchsize, L)

    # randomly sample start positions of two spike segments
    # constraint: 0 <= s1 < s2, s2 >= s1 + bit + 1 (at least 1 step gap)
    max_s1 = L - 2 * bit
    min_s1 = L - gating_period
    if min_s1 < 0:
        min_s1 = 0
    elif min_s1 > max_s1:
        min_s1 = max_s1 - 1
    s1 = torch.randint(min_s1, max_s1, (batchsize,))

    # s2 valid range: [s1 + bit + 1, L - bit]
    # uniform sampling within dynamic range (compatible with older PyTorch)
    offset_range = (L - bit) - (s1 + bit + 1) + 1
    offset = torch.floor(torch.rand(batchsize) * offset_range).long()
    s2 = s1 + bit + 1 + offset

    # build gating mask and extract values
    row_idx = torch.arange(batchsize)
    idx1 = s1.unsqueeze(1) + torch.arange(bit)  # contiguous bit-length indices [batchsize, bit]
    idx2 = s2.unsqueeze(1) + torch.arange(bit)

    # vectorised gating signal
    data_gating[row_idx.unsqueeze(1), idx1] = 1
    data_gating[row_idx.unsqueeze(1), idx2] = 1

    # extract and sum spike values at positions
    num1 = torch.gather(data_random, 1, idx1).sum(dim=1)
    num2 = torch.gather(data_random, 1, idx2).sum(dim=1)

    labels = (num1 + num2).long()

    # assemble data: concatenate two channels [batchsize, 2, L]
    data = torch.stack([data_random, data_gating], dim=1)

    # append silent period
    if silent_period > 0:
        silent = torch.zeros(batchsize, 2, silent_period)
        data = torch.cat([data, silent], dim=-1)
        
    return data.to(dtype=torch.float32), labels.to(dtype=torch.long)

import torch

def generate_evidence_collection(
        batchsize: int,
        resting_length: int = 200,
        cue_block_length: int = 100,
        cue_block_num: int = 7,
        channel_per_band: int = 25) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    return: (data, labels)
        data: [batchsize, channel=4*channel_per_band (left, right, recall, noise), length]
        labels: [batchsize,]
    '''
    
    # ensure cue_block_num is odd to avoid equal Left/Right counts causing ambiguous labels
    if cue_block_num % 2 == 0:
        raise ValueError(f"cue_block_num must be an odd number (got {cue_block_num}).")

    # total length: cue_blocks + resting segment + recall_block
    recall_block_length = cue_block_length
    total_length = cue_block_num * cue_block_length + resting_length + recall_block_length
    total_channels = 4 * channel_per_band

    # initialise data and labels
    data = torch.zeros((batchsize, total_channels, total_length), dtype=torch.float32)
    labels = torch.zeros(batchsize, dtype=torch.long)

    # preset parameters
    cue_ratio = 0.5    # Cue and Recall firing rate
    noise_ratio = 0.1  # Noise firing rate
    rest_ratio_in_block = 0.2  # first 10% of each block is resting

    # internal length division of each block
    rest_part_len = int(cue_block_length * rest_ratio_in_block)
    burst_part_len = cue_block_length - rest_part_len

    # channel index partitioning [cite: 1]
    left_idx = (0, channel_per_band)
    right_idx = (channel_per_band, 2 * channel_per_band)
    recall_idx = (2 * channel_per_band, 3 * channel_per_band)
    noise_idx = (3 * channel_per_band, 4 * channel_per_band)

    for b in range(batchsize):
        # 1. randomly decide Left and Right counts summing to cue_block_num
        num_left = torch.randint(0, cue_block_num + 1, (1,)).item()
        num_right = cue_block_num - num_left

        # label: Left > Right is 0, otherwise 1
        labels[b] = 0 if num_left > num_right else 1

        # 2. randomly permute Cue Block order (0 = Left, 1 = Right)
        cue_sequence = [0] * num_left + [1] * num_right
        cue_sequence = [cue_sequence[i] for i in torch.randperm(cue_block_num)]

        # 3. fill Cue Blocks
        for i, cue_type in enumerate(cue_sequence):
            start_ts = i * cue_block_length
            burst_start_ts = start_ts + rest_part_len

            band_start, band_end = left_idx if cue_type == 0 else right_idx

            # generate spikes during burst period [cite: 1]
            data[b, band_start:band_end, burst_start_ts : start_ts + cue_block_length] = \
                generate_spike_train(channel_per_band, burst_part_len, cue_ratio)

        # 4. fill Recall Block (at the end of the sequence)
        recall_start_ts = cue_block_num * cue_block_length + resting_length
        recall_burst_start = recall_start_ts + rest_part_len

        data[b, recall_idx[0]:recall_idx[1], recall_burst_start : recall_start_ts + recall_block_length] = \
            generate_spike_train(channel_per_band, burst_part_len, cue_ratio)

        # 5. fill Noise channel (full duration) [cite: 1]
        data[b, noise_idx[0]:noise_idx[1], :] = \
            generate_spike_train(channel_per_band, total_length, noise_ratio)

    return data.to(dtype=torch.float32), labels.to(dtype=torch.long)

@torch.jit.script
def heaviside(x: torch.Tensor) -> torch.Tensor:
    return (x>0).to(dtype=torch.float32)

class FilterBase(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.state = torch.zeros((1,), dtype=torch.float32)
        if torch.cuda.is_available():
            self.state = self.state.cuda()
        self.reset_done = True
        self.default_value = 0
        self._memory_shapes_set = False

    @torch.jit.export
    def reset(self):
        self.state = self.state.detach()
        self.state.fill_(self.default_value)
        self.reset_done = True
        self._memory_shapes_set = False

    @torch.jit.export
    def get_state(self):
        return self.state

    @torch.jit.export
    def get_valid_memory(self, new_input):
        if new_input.shape != self.state.shape:
            if self.reset_done:
                new_valid_memory = torch.ones_like(new_input, device=new_input.device)
                new_valid_memory *= self.default_value
                self._memory_shapes_set = True
                return new_valid_memory
            raise ValueError("Reset first.")
        else:
            return self.state

class LowPassFilter(FilterBase):

    def __init__(self, decay):
        super().__init__()
        self.decay = decay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = self.get_valid_memory(x)
        self.state = self.decay * state + x
        self.reset_done = False
        return self.state

class IterativeMovingAverage(FilterBase):

    def __init__(self, bias_ratio, default_value = 0.5):
        super().__init__()
        self.bias_ratio = bias_ratio
        self.default_value = default_value
        self.reset()

    def forward(self, x):
        state = self.get_valid_memory(x)
        self.state = state + self.bias_ratio * (x - state)
        self.reset_done = False
        return self.state

class ALIF(torch.nn.Module):
    '''
    Adaptive LIF, see LSNN.
    '''

    def __init__(self, 
            voltage_decay = 0.1,
            basic_threshold = 2.0,
            threshold_gain_decay = 0.995,
            threshold_gain_ratio = 0.07,
            noise_inject_ratio = 0.0):
        '''
        kwargs:
        1. voltage_decay: default 0.1
        2. basic_threshold: default 2.0
        3. threshold_gain_decay: default 0.995
        4. threshold_gain_ratio: default 0.07
        5. noise_inject_ratio: white noise injection to the membrane potential.
            See details in: Zheng et al. Weight-Dependent Spike-Timing-Dependent Plasticity (2018).

        '''
        super().__init__()
        self.voltage_decay = voltage_decay
        self.basic_threshold = basic_threshold
        self.threshold_gain_decay = threshold_gain_decay
        self.threshold_gain_ratio = threshold_gain_ratio
        self.noise_inject_ratio = noise_inject_ratio

        self.threshold_gain_decay_cache = 0.0

        self.membrane_voltage = torch.zeros((1,), dtype=torch.float32)
        self.threshold_gain = torch.zeros((1,), dtype=torch.float32)
        self.spike_cache = torch.zeros((1,), dtype=torch.float32)
        if torch.cuda.is_available():
            self.membrane_voltage = self.membrane_voltage.cuda()
            self.threshold_gain = self.threshold_gain.cuda()
            self.spike_cache = self.spike_cache.cuda()

        self.freeze_threshold() # Normal LIF if no unfreezing

        self.reset_done = True
        self._memories_suited = False

    @torch.jit.export
    def reset(self):
        self.membrane_voltage = self.membrane_voltage.detach()
        self.spike_cache = self.spike_cache.detach()
        self.membrane_voltage.fill_(0)
        self.spike_cache.fill_(0)
        self.reset_threshold()
        self.reset_done = True
        self._memories_suited = False

    def reset_threshold(self):
        self.threshold_gain.fill_(0)
    
    @torch.jit.export
    def freeze_threshold(self):
        self.threshold_gain_decay_cache = self.threshold_gain_decay
        self.threshold_gain_decay = 0.0

    @torch.jit.export
    def unfreeze_threshold(self):
        self.threshold_gain_decay = self.threshold_gain_decay_cache

    @torch.jit.export
    def get_previous_spike(self, ref_tensor: torch.Tensor):
        '''
        Tackle the initial entrance of LIF. Use ref_tensor to infer the shape of output spikes
        '''
        if self.spike_cache.shape != ref_tensor.shape:
            if self.reset_done:
                new_spike = torch.zeros_like(ref_tensor, device=ref_tensor.device)
                return new_spike
            else:
                raise ValueError("Reset first")
        else:
            return self.spike_cache

    def suit_memories(self, new_input):
        if new_input.shape != self.membrane_voltage.shape:
            if self.reset_done:
                self.membrane_voltage = torch.zeros_like(new_input, device=new_input.device)
                self.threshold_gain = torch.zeros_like(new_input, device=new_input.device)
                self.spike_cache = torch.zeros_like(new_input, device=new_input.device)
                self._memories_suited = True
            else:
                raise ValueError("Reset First. While reset_done={}, new_input.shape={} and membrane_voltage.shape={}".format(self.reset_done, new_input.shape, self.membrane_voltage.shape))

    def forward(self, voltage_gain):
        if not self._memories_suited:
            self.suit_memories(voltage_gain)
        self.membrane_voltage = self.voltage_decay * self.membrane_voltage\
                + voltage_gain\
                - self.spike_cache * self.basic_threshold # Soft-reset
        if self.noise_inject_ratio > 0:
            spike_mask = (self.spike_cache==1)
            if(spike_mask.sum() != 0):
                noise = torch.rand_like(self.membrane_voltage[spike_mask]) * self.basic_threshold * self.noise_inject_ratio
                self.membrane_voltage[spike_mask] -= noise
        self.threshold_gain = self.threshold_gain_decay * self.threshold_gain + self.spike_cache
        threshold = self.basic_threshold + self.threshold_gain_ratio * self.threshold_gain
        self.spike_cache = heaviside(self.membrane_voltage - threshold)
        self.reset_done = False
        return self.spike_cache

    @torch.jit.export
    def is_reset_done(self):
        return self.reset_done

def _time_traverse(length, batch_x, system_ref, final_shape):
    output = torch.empty(final_shape, dtype=torch.float32, device=batch_x.device)

    # Pre-check output shape once outside the hot loop
    test_out = system_ref(batch_x[..., 0].transpose(0, 1))
    if test_out is not None:
        expected_out_shape = test_out.shape
        if expected_out_shape != (final_shape[-2], final_shape[0]):
            raise RuntimeError(
                f"Output dimensions defined in the reservoir ({final_shape[:-1]}) "
                f"do not match the real output ({test_out.T.shape}). "
                "Check the definition of reservoir.get_output_dim()"
            )
        # Manually handle the first step
        output[..., 0] = test_out.T
        # Remaining steps — no shape checking overhead
        for timestep in range(1, length):
            step_input = batch_x[..., timestep].transpose(0, 1)
            step_output = system_ref(step_input)
            output[..., timestep] = step_output.transpose(0, 1)
    else:
        # system_ref returns None for skip_monitor-only runs
        for timestep in range(length):
            step_input = batch_x[..., timestep].transpose(0, 1)
            step_output = system_ref(step_input)
            if step_output is not None:
                output[..., timestep] = step_output.transpose(0, 1)
    return output

class OfflineWrapperBase(torch.nn.Module):

    def __init__(self, system: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()
        self.system = system

    def get_output_shape(self, input_channel: int) -> Tuple[int, ...]:
        '''
        Virtual
        '''
        raise NotImplementedError()

    def reset_system(self) -> None:
        self.system.reset()

    def forward_step(self, batch_x_t: torch.Tensor) -> torch.Tensor:
        '''
        Advance the wrapped system by one timestep.

        batch_x_t: [batch_size, input_channel].
        return: [batch_size, *(output_shape)].
        '''
        if batch_x_t.dim() != 2:
            raise ValueError(
                    "Single-step input must have shape "
                    f"[batch_size, input_channel], got {batch_x_t.shape}."
                    )
        output = self.system(batch_x_t.transpose(0, 1))
        if output is None:
            return None
        return output.transpose(0, 1)

    def forward(self, batch_x: torch.Tensor) -> torch.Tensor:
        '''
        batch_x: [batch_size, input_channel, timestep].
        return: [batch_size, *(output_shape), timestep]
        '''
        batch_size, channel, length = batch_x.shape
        output_shape = self.get_output_shape(channel) # Depend solely on the input shape
        final_shape = (batch_size, *output_shape, length)

        self.reset_system()
        output = _time_traverse(length, batch_x, self.system, final_shape)
        # After traversing, the system will not be reset immediately.
        # Thus the state of the system can still be acquired.
        return output

class ALIFOfflineWrapper(OfflineWrapperBase):
    
    def __init__(self, alif):
        super().__init__(alif)

    def get_output_shape(self, input_channel: int):
        return (input_channel,)

# Following: Spike Encoding

def PSP_FIR(points = 20, time_const_1 = 4, time_const_2 = 3):
    '''
    PSP denotes Post-Synaptic Potential, see Gutig 2006.
    '''
    t = np.arange(points)
    psp = np.exp(-t/time_const_1) - np.exp(-t/time_const_2)
    sum_psp = np.sum(psp)
    norm_psp = psp / sum_psp
    return norm_psp

def compute_SNR(info: np.ndarray, noise: np.ndarray) -> float:
    '''
    info and noise: [bs, timestep]
    return: average SNR
    '''
    info_power = np.sum(info**2, axis=-1)
    noise_power = np.sum(noise**2, axis=-1)
    sn_ratio = info_power / noise_power
    snr = 20 * np.log(sn_ratio)
    avg_snr = np.mean(snr)
    return avg_snr

def compute_firing_ratio(spike_train: np.ndarray) -> float:
    '''
    spike_train: [..., timestep]
    '''
    spike_train = np.abs(spike_train).reshape(-1)
    spike_sum = np.sum(spike_train)
    ratio = spike_sum / len(spike_train)
    return ratio

def plot_reconstructed_signal(original, reconstructed) -> 'Figure':
    '''
    original and reconstructed: [timestep,]
    '''
    original = original[...,2:-2]
    reconstructed = reconstructed[..., 2:-2]
    fig = plt.figure(figsize=(16,12))
    plt.plot(original, label="original")
    plt.plot(reconstructed, label="reconstructed")
    plt.xlabel("time")
    plt.ylabel("magnitude")
    return fig

class SpikeEncoderBase():
    '''
    Convert analog signals into spike trains.
    '''

    def encode(self, signals: np.ndarray):
        raise NotImplementedError()

    def reconstruct(self, spike_trains: np.ndarray):
        raise NotImplementedError()

    def test(self, test_signals: np.ndarray) -> Tuple[float, float, Figure]:
        '''
        test_signal: [batch_size, timestep]
        '''
        spike_train = self.encode(test_signals)
        reconstructed = self.reconstruct(spike_train)
        noise = reconstructed - test_signals
        SNR = compute_SNR(test_signals, noise)

        firing_ratio = compute_firing_ratio(spike_train)

        plot_idx = np.random.randint(0, len(test_signals))
        plot_sig, plot_reconst = test_signals[plot_idx], reconstructed[plot_idx]
        fig = plot_reconstructed_signal(plot_sig, plot_reconst)
        plt.legend()

        return SNR, firing_ratio, fig

def squeeze(x):
    result = np.tanh(x)
    result = result * 0.45 + 0.55
    return result

def unsqueeze(x):
    x = (x - 0.5) * 2
    x = np.clip(x, -1+1e-6, 1-1e-6)
    result = np.arctanh(x)
    return result

class BSAEncoder(SpikeEncoderBase):
    '''
    Ben's Spike Algorithm. See Schrauwen 2003, and Petro 2020.
    '''

    def __init__(self, threshold: float = 0.95):
        '''
        Padding: padding the begining and ending wihle encoding and reconstruction
        '''
        self.threshold = threshold
        self.FIR = PSP_FIR(20, 3, 2) * 1 # determin the maximum of the reconstructable signal

    def encode(self, signals: np.ndarray):
        '''
        signal: [batch_size, timesteps], should be shifted into [0, 1] first
        '''
        signals = signals.copy()
        signals = squeeze(signals)

        batch_size, length = signals.shape
        FIRs = np.repeat(self.FIR[np.newaxis], batch_size, axis=0)
        spike_trains = np.zeros_like(signals)
        for timestep in range(length):
            window_len = FIRs.shape[-1]
            end = timestep + window_len # not included, i.e. open interval
            if end > length:
                window_len = length - timestep
                end = length
                FIRs = FIRs[..., :window_len]
            signals_window = signals[..., timestep: end]
            err_1 = np.sum(np.abs(signals_window - FIRs), axis=-1) # [bs,]
            err_2 = np.sum(np.abs(signals_window), axis=-1)
            spike_mask = err_1 < err_2 - self.threshold
            if np.sum(spike_mask) > 0:
                spike_trains[spike_mask, timestep] = 1
                masked_FIRs = np.zeros_like(FIRs)
                masked_FIRs[spike_mask] = FIRs[spike_mask] # else are zeros
                signals[..., timestep: end] -= masked_FIRs

        return spike_trains

    def reconstruct(self, spike_trains: np.ndarray):
        batch_size, length = spike_trains.shape
        FIRs = np.repeat(self.FIR[np.newaxis], batch_size, axis=0)
        reconstructed = np.zeros_like(spike_trains) + 1e-9
        for timestep in range(length):
            window_len = FIRs.shape[-1]
            end = timestep + window_len # not included, i.e. open interval
            if end > length:
                window_len = length - timestep
                end = length
                FIRs = FIRs[..., :window_len]
            spike_mask = spike_trains[..., timestep]==1
            if np.sum(spike_mask) > 0:
                masked_FIRs = np.zeros_like(FIRs)
                masked_FIRs[spike_mask] = FIRs[spike_mask] # else are zeros
                reconstructed[..., timestep: end] += masked_FIRs # For the un-spiked signal, add zeros
        
        reconstructed = unsqueeze(reconstructed)
        return reconstructed

# Following: Gradient tunneling tools.

@torch.jit.script
def compute_stdp(pre, pre_last, post_causal, post_anti_causal):
    '''
    Inputs follow column vector convention.
    Output follows numerator layout (A column for the same numerator origin).
        pre and pre_last: [pre_dim, batch_size]
        post_causal and post_anti_causal: [post_dim, batch_size]
        return: [batch_size, post_dim, pre_dim]
    See Zheng 2018 Equation (3)
    '''
    pre_comb = pre * (1 - pre_last)
    post_comb = post_causal - post_anti_causal
    pre_comb = torch.transpose(pre_comb, 0, 1).unsqueeze(1) # [batch_size, 1, pre_dim]
    post_comb = torch.transpose(post_comb, 0, 1).unsqueeze(2) # [batch_size, post_dim, 1]
    stdp = post_comb * pre_comb # [batch_size, post_dim, pre_dim] via broadcasting
    return stdp

class MonitorBase():

    def update(self, module, input_spike, output_spike):
        raise NotImplementedError()

    def get(self):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

class CausalQueue():

    def __init__(self, window):
        self.window = window

        self.causal_queue = None # New part
        self.anti_causal_queue = None # Old part
        self.ptr = 0
        self.full = False
        self._append_count = 0

    def clear(self):
        self.causal_queue = None
        self.anti_causal_queue = None
        self.ptr = 0
        self.full = False
        self._append_count = 0

    def append(self, x: torch.Tensor) -> None:
        if self.causal_queue is None:
            self.causal_queue = torch.empty(
                    self.window, *x.shape, dtype=x.dtype, device=x.device)
            self.anti_causal_queue = torch.empty(
                    self.window, *x.shape, dtype=x.dtype, device=x.device)

        # Direct copy without redundant .clone()
        self.anti_causal_queue[self.ptr].copy_(self.causal_queue[self.ptr])
        self.causal_queue[self.ptr].copy_(x)

        self._append_count += 1
        if not self.full and self._append_count >= 2 * self.window:
            self.full = True

        self.ptr += 1
        if self.ptr == self.window:
            self.ptr = 0

    def is_full(self):
        return self.full

    def get_anticausal_causal(self):
        anticausal = self.anti_causal_queue.sum(0)
        causal = self.causal_queue.sum(0)
        return anticausal, causal

class StdpMonitor(MonitorBase):

    def __init__(self,
            window = 3, filter_coef = 0.05):
        '''
        See Zheng 2018 Equation (10), but with iterative moving average.
        Suppose the delay is 0.
        '''
        self.window = window

        self.filter = torch.jit.script(IterativeMovingAverage(filter_coef, 0))

        # Pre-allocated circular buffers instead of deque
        self._pre_queue = None  # [2*window, batchsize, channel]
        self._pre_idx = 0
        self._pre_full = False
        self.post_queue = CausalQueue(window)

    def update(self, input_spike, output_spike):
        '''
        input_spike and output_spike: [batchsize, channel]
        '''
        # Circular buffer for pre_queue
        if self._pre_queue is None:
            self._pre_queue = torch.empty(
                    2 * self.window, *input_spike.shape,
                    dtype=input_spike.dtype, device=input_spike.device)
        self._pre_queue[self._pre_idx].copy_(input_spike)
        self._pre_idx += 1
        if self._pre_idx >= 2 * self.window:
            self._pre_idx = 0
            self._pre_full = True

        self.post_queue.append(output_spike)
        if self.post_queue.is_full():
            # Circular buffer indexing: pre[w] and pre[w-1] from most recent 2w entries
            idx_w = (self._pre_idx + self.window) % (2 * self.window)
            idx_wm1 = (self._pre_idx + self.window - 1) % (2 * self.window)
            pre = self._pre_queue[idx_w]
            pre_last = self._pre_queue[idx_wm1]
            post_anti_causal, post_causal = self.post_queue.get_anticausal_causal()
            stdp = compute_stdp(pre, pre_last, post_causal, post_anti_causal)
            self.filter(stdp)

    def get(self):
        '''
        Return: [batchsize, pre_dim, post_dim]
        '''
        return self.filter.get_state()

    def reset(self):
        self.filter.reset()
        self._pre_queue = None
        self._pre_idx = 0
        self._pre_full = False
        self.post_queue.clear()

class SpikeRateMonitor(MonitorBase):
    '''
    This class should be realized as the last hook.
    Because it may change the output of the module.
    '''

    def __init__(self,
            filter_coef = 0.05):

        self.input_filter = torch.jit.script(IterativeMovingAverage(filter_coef, 0.5))
        self.output_filter = torch.jit.script(IterativeMovingAverage(filter_coef, 0.5))

        self.inout_format_checked = False
        self.output_filter_hooked = False

    def cascade_upstream(self, rate_monitor):
        rate_monitor.output_filter_hooked = True
        rate_monitor.output_filter = self.input_filter

    def update(self, input_spike, output_spike) -> None:
        if not self.inout_format_checked:
            if not are_spikes(input_spike):
                raise ValueError(
                        "The inputs are not spikes.")
            if not are_spikes(output_spike):
                raise ValueError(
                        "The outputs are not spikes.")
            self.inout_format_checked = True

        with torch.no_grad():
            input_rate = self.input_filter(input_spike)
        input_rate.requires_grad_().retain_grad()

        if self.output_filter_hooked:
            return None
        with torch.no_grad():
            output_rate = self.output_filter(output_spike)
        output_rate.requires_grad_().retain_grad()

    def get(self):
        '''
        Return: [input_dim, batch_size], [output_dim, batch_size]
            with retains_grad = True
        NOTICE: if the return values are not used for forward propagation,
            detach them first, or CUDA OOM error will happen,
            due to the unlimited record of the computational graph.
        '''
        input_rate = self.input_filter.get_state()
        output_rate = self.output_filter.get_state()
        return input_rate, output_rate

    def reset(self):
        self.input_filter.reset()
        if not self.output_filter_hooked:
            self.output_filter.reset()

class ReservoirBase(torch.nn.Module):
    '''
    Abstract Base Class. Should not be instantiated.
    Class that can do the gradient tunneling.
    '''

    def __init__(self,
            stdp_monitor: StdpMonitor = None,
            spike_rate_monitor: SpikeRateMonitor = None,
            monitor_filter_coef: float = 0.01):
        '''
        This base class holds gradient-tunneler related properties.
        If gradient tunneler is not permitted by the sub-class,
        no parameter afferent is needed in __init__().
        '''
        super().__init__()
        self.stdp_monitor = stdp_monitor
        self.spike_rate_monitor = spike_rate_monitor
        self.monitor_filter_coef = monitor_filter_coef

    def reset(self) -> None:
        raise NotImplementedError()

    def get_trainable_weights(self) -> torch.nn.Parameter:
        '''
        Input weights or feedback weights, but not recursive weights.
        '''
        raise NotImplementedError()

    def get_eligibility_jacobian_traces(self) -> Tuple[torch.Tensor]:
        '''
        Method for gradient tunneling.
        '''
        if self.spike_rate_monitor is None or self.stdp_monitor is None:
            raise PermissionError("Calling tunneling-related method without monitors is prohibited.")

        input_rate, output_rate = self.spike_rate_monitor.get()
        input_rate, output_rate = input_rate.detach(), output_rate.detach()

        causality_trace = self.stdp_monitor.get()
        if input_rate.dim() != 2:
            raise ValueError(f"Invalid input_rate shape: {input_rate.shape}")

        input_rate = torch.transpose(input_rate, 0, 1).unsqueeze(1) # [bs, 1, d_{l-1}]
        # Broadcasting eliminates the need for explicit expand; causality_trace is [bs, d_l, d_{l-1}]
        eligibility = causality_trace / (1 - input_rate)  # broadcasts along dim=1
        jacobian = eligibility / input_rate           # broadcasts along dim=1

        return eligibility, jacobian

    def get_post_learning_signal(self) -> torch.Tensor:
        '''
        Actually the gradient of post-synaptic neurons.
        In the synapsis form, i.e. [batch_size, d_l, d_{l-1}]
        '''
        if self.stdp_monitor is None or self.spike_rate_monitor is None:
            raise PermissionError("Calling tunneling-related method without monitors is prohibited.")

        input_rate, output_rate = self.spike_rate_monitor.get()
        learning_signal = output_rate.grad # [d_l, batch_size]
        if learning_signal is None:
            raise ValueError(
                    ("Learning signal is None, "
                    "i.e. back-propagation failed before this module. "
                    "Perhaps the sequence of gradient tunnnelers in tunneler list "
                    "didn't follow the cascading sequence?")
                    )
        if learning_signal.dim() != 2:
            raise ValueError(f"Invalid gradient shape: {learning_signal.shape}")
        learning_signal = torch.transpose(learning_signal, 0, 1).unsqueeze(-1)
        learning_signal = learning_signal.expand(-1, -1, input_rate.shape[0])
        output_rate.grad = None
        return learning_signal

    def set_monitor_filter_coef(self, coef) -> None:
        '''
        Set iterative moving average filter coefficient of both STDP monitor and spike rate monitor.
        '''
        if self.stdp_monitor is None or self.spike_rate_monitor is None:
            raise PermissionError("Calling tunneling-related method without monitors is prohibited.")
        self.monitor_filter_coef = coef
        self.stdp_monitor = StdpMonitor(filter_coef=coef)
        self.spike_rate_monitor = SpikeRateMonitor(filter_coef=coef)

    def set_pre_learning_signal(self, learning_signal) -> None:
        '''
        Actually the gradient of pre-synaptic neurons.
        learning_signal should be in the neuron form,
        i.e. learning_signal: [d_{l-1}, batch_size]
        '''
        if self.stdp_monitor is None or self.spike_rate_monitor is None:
            raise PermissionError("Calling tunneling-related method without monitors is prohibited.")
        input_rate, _ = self.spike_rate_monitor.get()
        if input_rate.grad is None:
            input_rate.grad = learning_signal
        else:
            input_rate.grad += learning_signal

    def cascade_upstream(self, upstream_reservoir) -> None:
        '''
        Actually cascate rate monitors.
        This method is used as the definition of 
        computational graph.
        For the best practice,
        this should be called by the gradient tunneler automatically.
        '''
        if self.spike_rate_monitor is None:
            raise PermissionError("Calling tunneling-related method without monitors is prohibited.")
        self.spike_rate_monitor.cascade_upstream(
                upstream_reservoir.spike_rate_monitor
                )

@torch.compile
def _regul_comput_with_decrease(output_rate, target_firing_rate, regul_coef, decrease = 1e-9, decrease_margin_ratio = 0.2):
    rate_mean = output_rate.mean(0, keepdim=True)
    deviation = rate_mean - target_firing_rate #[output_dim, batch_size]
    margin = target_firing_rate * decrease_margin_ratio
    in_margin = torch.abs(deviation) < margin
    deviation[in_margin] *= decrease
    reg = deviation / output_rate.shape[0]
    reg *= regul_coef/reg.shape[1]
    return reg # [output_dim, batch_size]

class GradientTunneler():

    def __init__(self, reservoir: ReservoirBase,
            masked_amount = 0,
            regularization_coef = 1e-2,
            target_firing_rate = 0.25,
            regularized_amount = None,
            compute_weight_grad = True,
            ls_operator: Callable = None):
        '''
        masked_amount: musk the first {masked_amount} input channels while training.
        ls_operator: post learning signal operator.
            Using for additional operation when backpropagating, if additional forward operations exist.
        '''
        self.reservoir = reservoir
        reservoir.liquid_filter = None
        self.masked_amount = masked_amount
        self.regularization_coef = regularization_coef
        self.target_firing_rate = target_firing_rate
        self.compute_weight_grad = compute_weight_grad
        self.ls_operator = ls_operator
        if not self.compute_weight_grad:
            warnings.warn("A reservoir is gradient tunneled, but weight gradient is not computed.")

        self.learning_signal_cache = None
        self.regularized_amount = regularized_amount
        if regularized_amount is None:
            self.regularized_amount = reservoir.get_input_dim() - masked_amount

        self.upstream_exists = False

    def cascade_upstream(self, upstream_tunneler):
        self.reservoir.cascade_upstream(
                upstream_tunneler.reservoir
                )
        self.upstream_exists = True

    def get_firing_rates(self):
        '''
        Return: [input_dim, batch_size], [output_dim, batch_size]
        '''
        input_rate, output_rate = self.reservoir.spike_rate_monitor.get()
        return input_rate.detach(), output_rate.detach()

    def get_learning_signal_cache(self):
        '''
        Used for monitor post learning signal only.
        '''
        if self.learning_signal_cache is None:
            raise ValueError("Learning signal is not exhausted from the reservoir. Call tunnel() method first.")
        return self.learning_signal_cache

    def tunnel(
            self,
            accumulate_weight_grad = False,
            grad_scale = 1.0,
            regularization_scale = 1.0):
        '''
        Gradient Tunneling.

        When ``accumulate_weight_grad`` is true, add the current tunneled
        gradient to ``weight.grad`` instead of replacing it. This supports
        accumulating one GT contribution per timestep before a single
        optimizer update. ``regularization_scale`` independently scales the
        firing-rate regularization contribution.
        '''
        eligibility, jacobian = self.reservoir.get_eligibility_jacobian_traces()
        post_learning_signal = self.reservoir.get_post_learning_signal()
        if self.ls_operator is not None:
            post_learning_signal = self.ls_operator(post_learning_signal)
        self.learning_signal_cache = post_learning_signal

        if self.compute_weight_grad:
            # A cascaded upstream module may share the downstream input-rate
            # monitor. Its learning signal then contains all downstream input
            # channels, while its eligibility only covers the channels emitted
            # by this module. Those emitted channels are appended at the end,
            # so align the post-synaptic/output axis (dim=1), not the synaptic
            # input axis (dim=2).
            regularized_learning_signal = post_learning_signal[
                    :, -eligibility.shape[1]:, :
                    ].clone()
            if regularized_learning_signal.shape != eligibility.shape:
                raise RuntimeError(
                        "Learning signal and eligibility shapes are incompatible "
                        "after output-channel alignment: "
                        f"{regularized_learning_signal.shape} vs "
                        f"{eligibility.shape}."
                        )
            if self.regularization_coef > 0:
                _, output_rate = self.get_firing_rates()
                reg = _regul_comput_with_decrease(
                        output_rate,
                        self.target_firing_rate,
                        self.regularization_coef
                        ) # Neuron format
                reg.mul_(regularization_scale)
                reg = torch.transpose(reg, 0, 1).unsqueeze(-1).expand(-1, -1, self.regularized_amount) # Synapses format
                regularized_learning_signal[..., self.masked_amount: self.masked_amount+self.regularized_amount] += reg

            weight = self.reservoir.get_trainable_weights()
            # Selective division — only divide non-zero weights to avoid NaN
            grad = (regularized_learning_signal * eligibility).sum(0)
            weight_data = weight.data
            nonzero_mask = weight_data != 0
            grad[nonzero_mask] /= weight_data[nonzero_mask]
            grad[~nonzero_mask] = 0
            if self.masked_amount > 0:
                grad[..., :self.masked_amount] = 0
            grad.mul_(grad_scale)
            if accumulate_weight_grad and weight.grad is not None:
                weight.grad.add_(grad)
            else:
                weight.grad = grad

        if self.upstream_exists:
            neuron_ls = post_learning_signal[..., 0].unsqueeze(1) # [bs, 1, d_l]
            pre_ls = torch.matmul(neuron_ls, jacobian)
            pre_ls = pre_ls.view(pre_ls.shape[0], pre_ls.shape[-1])
            pre_ls = torch.transpose(pre_ls, 0, 1) # [d_{l-1}, bs]
            self.reservoir.set_pre_learning_signal(pre_ls)

# Following: analysis tools

def trace2spiketrain(trace: np.ndarray, initial_value = 0.5) -> np.ndarray:
    '''
    trace: [..., timesteps], output of a low-pass filter.
    return: the same shape.
    '''
    padding = np.ones([*trace.shape[:-1], 1]) * initial_value
    extended_trace = np.concatenate([padding, trace[..., :-1]], axis=-1)
    increment = trace - extended_trace
    spike_train = np.zeros_like(increment)
    spike_train[increment>0] = 1
    return spike_train

def spiketrain2LFP(spike_train: np.ndarray) -> np.ndarray:
    '''
    LFP: Local Field Potential
    spike_train: [..., neuron_num, timesteps]
    return: [..., timesteps]
    '''
    local_spike_train = spike_train.sum(-2)
    lfp = scipy.signal.wiener(local_spike_train, mysize=9)
    return lfp

def frequency_analysis(time_sequence: np.ndarray, fig_dir: str = None, ignore_dc = False, freq_max = None) -> np.ndarray:
    '''
    Compute the unilateral amplitude spectrum.
    time_sequence: [timesteps]
    return: [frequency points = timesteps]
    '''
    if ignore_dc:
        time_sequence -= time_sequence.mean(-1, keepdims=True)
    n = len(time_sequence)  # 信号长度
    fft_result = np.fft.fft(time_sequence * np.hanning(n))
    spectrum = (np.abs(fft_result) / n)**2
    spectrum = spectrum[:n//2] * 2
    if fig_dir is not None:
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))
        axs[0].plot(time_sequence)
        axs[0].set_title("In time domain")
        axs[0].set_ylabel("time")
        if freq_max is None:
            freq_max = len(spectrum)
        axs[1].stem(spectrum[:freq_max])
        axs[1].set_title("In frequency domain")
        axs[1].set_ylabel("frequency")
        fig.savefig(fig_dir)
    return spectrum
