import random
import warnings
from collections import deque

import numpy as np
import scipy
import torch
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from typing import Callable, Tuple

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

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
    if torch.cuda.is_available():
        spike_train = spike_train.cuda()
    return spike_train

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

    @torch.jit.export
    def reset(self):
        self.state = self.state.detach()
        self.state.fill_(self.default_value)
        self.reset_done = True

    @torch.jit.export
    def get_state(self):
        return self.state

    @torch.jit.export
    def get_valid_memory(self, new_input):
        if new_input.shape != self.state.shape:
            if self.reset_done:
                new_valid_memory = torch.ones_like(new_input, device=new_input.device)
                new_valid_memory *= self.default_value
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

    @torch.jit.export
    def reset(self):
        self.membrane_voltage = self.membrane_voltage.detach()
        self.spike_cache = self.spike_cache.detach()
        self.membrane_voltage.fill_(0)
        self.spike_cache.fill_(0)
        self.reset_threshold()
        self.reset_done = True

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
            else:
                raise ValueError("Reset First. While reset_done={}, new_input.shape={} and membrane_voltage.shape={}".format(self.reset_done, new_input.shape, self.membrane_voltage.shape))

    def forward(self, voltage_gain):
        self.suit_memories(voltage_gain)
        self.membrane_voltage = self.voltage_decay * self.membrane_voltage\
                + voltage_gain\
                - self.spike_cache * self.basic_threshold # Soft-reset
        if self.noise_inject_ratio > 0:
            spike_mask = (self.spike_cache==1)  # Not nessesary, but easy to understand
            if(spike_mask.sum() != 0):
                noise = torch.rand_like(self.membrane_voltage[spike_mask]) * self.basic_threshold * self.noise_inject_ratio
                self.membrane_voltage[spike_mask] -= noise
        self.threshold_gain = self.threshold_gain_decay * self.threshold_gain + self.spike_cache
        threshold = self.basic_threshold + self.threshold_gain_ratio * self.threshold_gain
        self.spike_cache = heaviside(self.membrane_voltage - threshold)
        self.reset_done = False
        return self.spike_cache.clone()

    @torch.jit.export
    def is_reset_done(self):
        return self.reset_done

def _time_traverse(length, batch_x, system_ref, final_shape):
    output = torch.zeros(final_shape, dtype=torch.float32, device=batch_x.device)
    for timestep in range(length):
        step_input = batch_x[..., timestep].transpose(0, 1) # Consider column vector group
        step_output = system_ref(step_input) # Input should be column vectors
        step_output = step_output.transpose(0, 1) # [batch_size, channel]
        try:
            output[..., timestep] = step_output
        except RuntimeError:
            raise RuntimeError(
                    (
                        "Output dimentions defined in the reservoir do not match the real output. "
                        "Check the definition of reservoir.get_output_dim()"
                        )
                    )
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
    stdp = torch.matmul(post_comb, pre_comb) # [batch_size, post_dim, pre_dim]
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
        self.max_times = 0

    def clear(self):
        self.causal_queue = None
        self.anti_causal_queue = None
        self.ptr = 0
        self.full = False
        self.max_times = 0

    def append(self, x: torch.Tensor) -> None:
        if self.causal_queue is None:
            self.causal_queue = torch.empty(
                    self.window, *x.shape, dtype=x.dtype, device=x.device)
            self.anti_causal_queue = self.causal_queue.clone()

        self.anti_causal_queue[self.ptr] = self.causal_queue[self.ptr]
        self.causal_queue[self.ptr] = x
    
        if not self.full:
            if self.ptr == self.window-1:
                self.max_times += 1
                if self.max_times == 2:
                    self.full = True

        self.ptr += 1
        self.ptr %= self.window

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

        self.pre_queue = deque()
        self.post_queue = CausalQueue(window)

    def update(self, input_spike, output_spike):
        '''
        input_spike and output_spike: [batchsize, channel]
        '''
        self.pre_queue.append(input_spike)
        if len(self.pre_queue) > 2*self.window:
            self.pre_queue.popleft()
        self.post_queue.append(output_spike)
        if self.post_queue.is_full():
            # Compute stdp
            pre = self.pre_queue[self.window]
            pre_last = self.pre_queue[self.window-1]
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
        self.pre_queue.clear()
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

        stdp_trace = self.stdp_monitor.get()
        if input_rate.dim() != 2:
            raise ValueError(f"Invalid input_rate shape: {input_rate.shape}")

        input_rate = torch.transpose(input_rate, 0, 1).unsqueeze(1) # [bs, 1, d_{l-1}]
        input_rate = input_rate.expand(-1, stdp_trace.shape[1], -1) # [bs, d_l, d_{l-1}]
        eligibility = stdp_trace / (1-input_rate)
        jacobian = eligibility / input_rate

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
        input_rate.grad = learning_signal

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
            compute_weight_grad = True):
        '''
        masked_amount: musk the first {masked_amount} input channels while training.
        '''
        self.reservoir = reservoir
        reservoir.liquid_filter = None
        self.masked_amount = masked_amount
        self.regularization_coef = regularization_coef
        self.target_firing_rate = target_firing_rate
        self.compute_weight_grad = compute_weight_grad
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

    def tunnel(self):
        '''
        Gradient Tunneling
        '''
        eligibility, jacobian = self.reservoir.get_eligibility_jacobian_traces()
        post_learning_signal = self.reservoir.get_post_learning_signal()
        self.learning_signal_cache = post_learning_signal

        if self.compute_weight_grad:
            regularized_learning_signal = post_learning_signal.clone()
            regularized_learning_signal = regularized_learning_signal[:, -1*eligibility.shape[1]:]
            if self.regularization_coef > 0:
                _, output_rate = self.get_firing_rates()
                reg = _regul_comput_with_decrease(
                        output_rate,
                        self.target_firing_rate,
                        self.regularization_coef
                        ) # Neuron format
                reg = torch.transpose(reg, 0, 1).unsqueeze(-1).expand(-1, -1, self.regularized_amount) # Synapses format
                regularized_learning_signal[..., self.masked_amount: self.masked_amount+self.regularized_amount] += reg

            weight = self.reservoir.get_trainable_weights()
            weight_masked = weight.data.clone()
            weight_masked[weight.data==0] = torch.inf # Zero weight synapses will not be updated.
            grad = (regularized_learning_signal * eligibility).sum(0) / weight_masked
            if self.masked_amount > 0:
                grad[..., :self.masked_amount] = 0
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
