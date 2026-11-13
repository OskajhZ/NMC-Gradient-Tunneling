'''
By Xiangnan Zhang, 2025
    School of Future Technologies, Beijing Institute of Technology
'''
import math
import random
import warnings
from typing import List

import numpy as np
import torch

from . import _base

torch.manual_seed(168)
np.random.seed(168)
random.seed(168)

# Following: kernel reservoir with STDP monitor

def generate_input_weights(input_dim, neuron_num, threshold, **kwargs):
    r'''
    kwargs:
    1. density: default 1-n^{\frac{1}{input_dim}}, where n=1e-2 is no-input probability.
    2. mean: default somas.basic_threshold/(input_dim*density)
    3. std: default mean/2 (2 \sigma criterion)
    '''
    disconnect_prob = kwargs.get("disconnect_prob", 1/neuron_num)
    density = kwargs.get("density",
            1 - (disconnect_prob)**(1/input_dim)
            )
    mean = kwargs.get("mean",
            threshold/(input_dim*density)
            )
    std = kwargs.get("std", mean/2)

    input_gaussian_weights = torch.randn(neuron_num, input_dim)*std + mean
    input_gaussian_weights[input_gaussian_weights<0] = 0
    input_gaussian_weights = input_gaussian_weights.view(-1)
    drop_indices = random.sample(
            range(
                len(input_gaussian_weights)), k=int((1-density)*len(input_gaussian_weights)
                    )
            )
    input_gaussian_weights[drop_indices] = 0
    input_weights = input_gaussian_weights.view(neuron_num, input_dim)
    return input_weights

@torch._dynamo.disable()
def _monitor_update(training, stdp_m, rate_m, in_spike, out_spike):
    in_spike = torch.abs(in_spike)
    out_spike = torch.abs(out_spike)
    if training:
        stdp_m.update(in_spike, out_spike)
    rate_m.update(in_spike, out_spike)
    _, out_rate = rate_m.get()
    return out_rate

class LIFLayer(_base.ReservoirBase):
    def __init__(self, input_dim, neuron_num,
            bias = True,
            spike_output = True,
            voltage_decay = 0.7,
            basic_threshold = 10,
            threshold_gain_decay = 0.995,
            threshold_gain_ratio = 0.07,
            noise_inject_ratio = 0.0,
            monitor_filter_coef = 0.05
            ):
        super().__init__(
                stdp_monitor = _base.StdpMonitor(
                    filter_coef = monitor_filter_coef),
                spike_rate_monitor = _base.SpikeRateMonitor(
                    filter_coef = monitor_filter_coef),
                monitor_filter_coef = monitor_filter_coef
                )

        self.input_dim = input_dim
        self.neuron_num = neuron_num
        self.bias = bias
        self.spike_output = spike_output

        ALIF_kwargs = {}
        ALIF_kwargs["voltage_decay"] = voltage_decay
        ALIF_kwargs["basic_threshold"] = basic_threshold
        ALIF_kwargs["threshold_gain_decay"] = threshold_gain_decay
        ALIF_kwargs["threshold_gain_ratio"] = threshold_gain_ratio
        ALIF_kwargs["noise_inject_ratio"] = noise_inject_ratio
        self.somas = torch.jit.script(_base.ALIF(**ALIF_kwargs))

        weight = None
        if bias:
            weight = torch.rand((neuron_num, input_dim+1), dtype=torch.float32)
            weight = weight * (2*basic_threshold/(input_dim+1)) # E[w] = threshold/input_dim
        else:
            weight = torch.rand((neuron_num, input_dim), dtype=torch.float32)
            weight = weight * (2*basic_threshold/(input_dim)) 
        self.input_weights = torch.nn.Parameter(
                weight, requires_grad = True
                )

        self.spike_cache = None

    def clone(self):
        new_lif = LIFLayer(
                self.input_dim, self.neuron_num,
                bias = self.bias,
                spike_output = self.spike_output,
                voltage_decay = self.somas.voltage_decay,
                basic_threshold = self.somas.basic_threshold,
                threshold_gain_decay = self.somas.threshold_gain_decay,
                threshold_gain_ratio = self.somas.threshold_gain_ratio,
                noise_inject_ratio = self.somas.noise_inject_ratio,
                monitor_filter_coef = self.monitor_filter_coef
                )
        new_lif.input_weights.data = self.input_weights.data.clone()
        return new_lif

    def forward(self, input_spikes: torch.Tensor) -> torch.Tensor:
        '''
        input_spike: [input_dim, batch_size], i.e. using column vector convention
        return: [neuron_num, batch_size]
        NOTICE: the weight updating of input weights will not rely on the Autograd graph.
        '''
        with torch.no_grad():
            if self.bias:
                bias_spike = torch.ones(
                    (1, input_spikes.shape[1]),
                    dtype=torch.float32, device=input_spikes.device)
                bias_spike = bias_spike * random.getrandbits(1)
                input_spikes = torch.cat([input_spikes, bias_spike], dim=0)
            input_voltage = torch.matmul(self.input_weights, input_spikes)
            output_spike = self.somas(input_voltage)
            self.spike_cache = output_spike

        output_rate = None
        output_rate = _monitor_update(
                self.training,
                self.stdp_monitor,
                self.spike_rate_monitor,
                input_spikes, output_spike
                )
        if self.spike_output:
            return output_spike
        else:
            return output_rate

    def reset(self):
        self.somas.reset()
        self.stdp_monitor.reset()
        self.spike_rate_monitor.reset()
        self.spike_cache = None

    def get_output_dim(self):
        return self.neuron_num

    def get_input_dim(self):
        return self.input_dim

    def get_trainable_weights(self) -> torch.nn.Parameter:
        return self.input_weights

class Reservoir(_base.ReservoirBase):
    '''
    Basic reservoir class for Liquid State Machine.
    '''

    def __init__(self, input_dim, edge_length,
            neuron_num: int = None,
            liquid_filter: _base.FilterBase = None,
            spike_output = False,
            skip_monitor = False,
            voltage_decay = 0.7,
            basic_threshold = 10,
            threshold_gain_decay = 0.995,
            threshold_gain_ratio = 0.07,
            noise_inject_ratio = 0.0,
            monitor_filter_coef = 0.05
            ):
        '''
        neuron_num: <= edge_length**3, default edge_length**3
        For neurons:
        1. voltage_decay: default 0.7
        2. basic_threshold: default 10
        3. threshold_gain_decay: default 0.995
        4. threshold_gain_ratio: default 0.07
        '''
        super().__init__(
                stdp_monitor = _base.StdpMonitor(
                    filter_coef = monitor_filter_coef),
                spike_rate_monitor = _base.SpikeRateMonitor(
                    filter_coef = monitor_filter_coef),
                monitor_filter_coef = monitor_filter_coef
                )

        self.input_dim = input_dim
        self.edge_length = edge_length
        self.neuron_num = neuron_num
        if neuron_num is None:
            self.neuron_num = int(self.edge_length**3)
        elif neuron_num > edge_length**3:
            raise ValueError(
                    (
                        "Reservoir initialization failed: "
                        f"neuron_num ({neuron_num}) cannot "
                        f"be greater than edge_length**3 ({edge_length**3})."
                        )
                    )
        self.liquid_filter: _base.FilterBase = liquid_filter
        self.spike_output = spike_output
        self.skip_monitor = skip_monitor

        ALIF_kwargs = {}
        ALIF_kwargs["voltage_decay"] = voltage_decay
        ALIF_kwargs["basic_threshold"] = basic_threshold
        ALIF_kwargs["threshold_gain_decay"] = threshold_gain_decay
        ALIF_kwargs["threshold_gain_ratio"] = threshold_gain_ratio
        ALIF_kwargs["noise_inject_ratio"] = noise_inject_ratio
        self.somas = torch.jit.script(_base.ALIF(**ALIF_kwargs))

        self.input_weights = torch.nn.Parameter(
                torch.zeros((self.neuron_num, input_dim), dtype=torch.float32),
                requires_grad = True
                )
        self.recursive_weights = torch.nn.Parameter(
                torch.zeros((self.neuron_num, self.neuron_num), dtype=torch.float32),
                requires_grad = False
                )
        self.inhibitory_indices = torch.nn.Parameter(
                torch.ones((1,), dtype=torch.int), # Now only a placeholder
                requires_grad = False
                )

        self.generate_recursive_weights()
        self.generate_input_weights()

        self.liquid_state = None

    def clone(self):
        new_reservoir = Reservoir(
                self.input_dim, self.edge_length,
                neuron_num = self.neuron_num,
                liquid_filter = self.liquid_filter,
                spike_output = self.spike_output,
                skip_monitor = self.skip_monitor,
                voltage_decay = self.somas.voltage_decay,
                basic_threshold = self.somas.basic_threshold,
                threshold_gain_decay = self.somas.threshold_gain_decay,
                threshold_gain_ratio = self.somas.threshold_gain_ratio,
                noise_inject_ratio = self.somas.noise_inject_ratio,
                monitor_filter_coef = self.monitor_filter_coef
                )
        new_reservoir.input_weights.data = self.input_weights.data.clone()
        new_reservoir.recursive_weights.data = self.recursive_weights.data
        new_reservoir.inhibitory_indices.data = self.inhibitory_indices.data
        return new_reservoir

    def idx2coordinate(self, idx):
        x = idx // (self.edge_length**2)
        idx -= x * self.edge_length**2
        y = idx // self.edge_length
        z = idx - y * self.edge_length
        return (x,y,z)

    def indices2distance(self, idx_1, idx_2):
        x_1, y_1, z_1 = self.idx2coordinate(idx_1)
        x_2, y_2, z_2 = self.idx2coordinate(idx_2)
        distance = math.sqrt((x_1-x_2)**2 + (y_1-y_2)**2 + (z_1-z_2)**2)
        return distance

    def generate_recursive_weights(self,
            C = 0.2,
            sigma = 3, # C and sigma refer to Ivanov 2021
            mean = 0.1,
            std = None, # according to mean (mean/2)
            inhibitory_ratio = 0.2 # inline with Ivanov 2021
            ):
        r'''
        $$
        P(i, j) = C \cdot \exp(-\frac{D(i,j)^2}{\sigma^2})
        $$
        where $std=\sigma$
        '''
        if std is None:
            std = mean / 2
        gaussian_weights = torch.randn(*self.recursive_weights.shape)*std + mean
        gaussian_weights[gaussian_weights<0] = 0
        gaussian_weights = gaussian_weights.to(device = self.recursive_weights.device)

        distance_mat = torch.zeros_like(self.recursive_weights)
        m = 0
        while m < self.neuron_num:
            n = m + 1
            while n < self.neuron_num:
                distance = self.indices2distance(m, n)
                distance_mat[m, n] = distance
                distance_mat[n, m] = distance
                n += 1
            m += 1

        probability_mat = C * torch.exp(-distance_mat**2/sigma**2)
        connect_standard = torch.rand(*self.recursive_weights.shape, device=self.recursive_weights.device)
        disconnect_mask = (connect_standard > probability_mat) # Is 1 if no connection
        for i in range(self.neuron_num):
            disconnect_mask[i, i] = 1 # Cancel the self-loop
        gaussian_weights[disconnect_mask] = 0
        recursive_weights = gaussian_weights

        inhibitory_indices = []
        if inhibitory_ratio > 0:
            inhibitory_num = int(self.neuron_num * inhibitory_ratio)
            inhibitory_indices = random.sample(range(self.neuron_num), k=inhibitory_num)
        self.inhibitory_indices.data = torch.tensor(inhibitory_indices, dtype=torch.int)

        self.recursive_weights.data = recursive_weights

    def generate_input_weights(self, **kwargs):
        r'''
        kwargs:
        1. density: default 1-n^{\frac{1}{input_dim}}, where n=1e-2 is no-input probability.
        2. mean: default somas.basic_threshold/(input_dim*density)
        3. std: default mean/2 (2 \sigma criterion)
        '''
        now_device = self.input_weights.data.device
        self.input_weights.data = generate_input_weights(
                self.input_dim,
                self.neuron_num,
                self.somas.basic_threshold,
                **kwargs).to(device = now_device)

    def get_liquid_state(self) -> torch.Tensor:
        '''
        Column convention: return [output_dim, batchsize]
        '''
        return self.liquid_state

    def forward(self, input_spikes: torch.Tensor) -> torch.Tensor:
        '''
        input_spike: [n, batch_size], i.e. using column vector convention
        return: [edge**3, batch_size]
        NOTICE: the weight updating of input weights will not rely on the Autograd graph.
        '''
        with torch.no_grad():
            input_voltage = torch.matmul(self.input_weights, input_spikes)
            batchsize = input_spikes.shape[1]

            if self.liquid_state is None:
                self.liquid_state = torch.zeros(
                        (self.neuron_num, batchsize),
                        dtype = torch.float32,
                        device = input_spikes.device)

            recursive_voltage = torch.matmul(
                    self.recursive_weights,
                    self.liquid_state)
            voltage_gain = recursive_voltage + input_voltage
            self.liquid_state = self.somas(voltage_gain)
            self.liquid_state[self.inhibitory_indices] *= -1

        output_rate = None
        if not self.skip_monitor:
            output_rate = _monitor_update(
                    self.training,
                    self.stdp_monitor,
                    self.spike_rate_monitor,
                    input_spikes, self.liquid_state
                    )

        liquid_state = None
        if self.liquid_filter is not None:
            liquid_state = self.liquid_filter(torch.abs(self.liquid_state))
        elif self.spike_output:
            liquid_state = self.liquid_state
        else:
            liquid_state = output_rate
        return liquid_state

    def reset(self):
        self.somas.reset()
        if self.liquid_filter is not None:
            self.liquid_filter.reset()
        self.stdp_monitor.reset()
        self.spike_rate_monitor.reset()
        self.liquid_state = None

    def get_output_dim(self):
        return int(self.neuron_num)

    def get_input_dim(self):
        return self.input_dim

    def get_trainable_weights(self) -> torch.nn.Parameter:
        return self.input_weights

def evenly_select(
        x: torch.Tensor, select_num: int, offset = 0) -> torch.Tensor:
    '''
    Using to select neuron output evenly
    x: [neuron_num, batch_size]
    offset: x - offset > select_num. Default 0.
    return: [select_num, batch_size]
    '''
    if offset > 0:
        x_head = x[:-offset]
        x_tail = x[-offset:]
        x = torch.cat([x_tail, x_head], dim=0)
    neuron_num = x.shape[0]
    interval = neuron_num // (select_num -1)
    selected = x[: select_num * interval : interval]
    return selected

@torch.compile
def evenly_embed(
        selected: torch.Tensor, expected_num: int, offset = 0) -> torch.Tensor:
    '''
    Inverse operation of evenly_select.
    x: [selected_num, batch_size]
    return: [expected_num, batch_size]
    '''
    x = torch.zeros((expected_num, *selected.shape[1:]), device=selected.device)
    interval = expected_num // (selected.shape[0] - 1)
    num_selected = selected.shape[0]
    selected = x[: num_selected * interval : interval]
    if offset > 0:
        x_tail = x[:offset]
        x_head = x[offset:]
        x = torch.cat([x_head, x_tail], dim=0)
    return x

class TrainableMicrocircuitBase(torch.nn.Module):
    '''
    Using following methods to define gradient tunnelers, which is equivalent to defining computational graph of a microcircuit:
    1. config_tunneler()
    Other following methods should be overriden:
    2. clone()
    3. reset()
    4. get_output_dim()
    '''

    def __init__(self):
        super().__init__()
        self.tunneler_list: List[_base.GradientTunneler] = [] # Gradient tunneler list
        self.tunneler_config_dict = {}

    def _append_tunneler_list(self, tunneler: _base.GradientTunneler):
        '''
        Well-defined method.
        When defining config_tunneler() method of the sub-class, using this method to append gradient tunnelers in sequence.
        '''
        self.tunneler_list.append(tunneler)

    def config_tunnelers(self, **tunneler_config_dict) -> None:
        '''
        When defining this method:
        1. call super().config_tunnelers() first.
        2. use well-defined self._append_tunneler_list() to append gradient tunnelers in sequence.
        tunneler_config_dict: configuration despite the related reservoir.
        '''
        self.clear_tunneler_list()
        self.tunneler_config_dict = tunneler_config_dict

    def get_tunneler_list(self) -> List[_base.GradientTunneler]:
        '''
        Well-defined method.
        '''
        return self.tunneler_list

    def clear_tunneler_list(self) -> None:
        '''
        Well-defined method.
        '''
        self.tunneler_list = []

    def clone(self):
        '''
        At the end of clone() definition, remember calling config_tunnelers() method of the new object to copy self.tunneler_config_dict
        '''
        raise NotImplementedError()

    def reset(self) -> None:
        raise NotImplementedError()

    def get_output_dim(self) -> int:
        raise NotImplementedError()

class OutputLimiter(TrainableMicrocircuitBase):

    def __init__(self, reservoir: Reservoir, output_dim: int):
        super().__init__()
        self.reservoir = reservoir
        self.output_dim = output_dim

    def clone(self):
        return OutputLimiter(
                self.reservoir.clone(),
                self.output_dim)

    def forward(self, x):
        y = self.reservoir(x)
        y = evenly_select(y, self.output_dim)
        return y

    def reset(self):
        self.reservoir.reset()

    def get_output_dim(self):
        return self.output_dim

@torch.compile
def _form_kernel_input(
        input_spikes: torch.Tensor,
        feedback: torch.Tensor,
        kernel_input_dim: int) -> torch.Tensor:
    '''
    Private function.
    '''
    with torch.no_grad():
        total_input = torch.cat([input_spikes, feedback], dim=0)
        if total_input.shape[0] < kernel_input_dim:
            lack_dim = kernel_input_dim - total_input.shape[0]
            padding = torch.zeros(
                    (lack_dim, input_spikes.shape[1]),
                    device = input_spikes.device,
                    dtype = input_spikes.dtype)
            total_input = torch.cat([total_input, padding], dim=0)
        elif total_input.shape[0] > kernel_input_dim:
            raise ValueError(f"The channel amount of the kernel reservoir is not enough. Sum of input and feedback: {total_input.shape[0]}. While kernel reservoir: {kernel_input_dim}.")
    return total_input

class FeedbackReservoir(TrainableMicrocircuitBase): # The most simple and effective way

    def __init__(self, kernel_reservoir: Reservoir,
            input_channel: int, feedback_channel: int,
            bias_channel: int = None,
            weight_coef = 1,
            output_dim: int = None,
            output_feedback_channels = False,
            output_excitatory_only = False,
            input2excitatory_only = False):
        super().__init__()
        self.kernel_reservoir = kernel_reservoir
        self.input_channel = input_channel
        self.feedback_channel = feedback_channel
        self.bias_channel = bias_channel
        self.weight_coef = weight_coef
        self.output_dim = output_dim
        self.output_feedback_channels = output_feedback_channels
        self.output_excitatory_only = output_excitatory_only
        self.input2excitatory_only = input2excitatory_only

        self.interval = self.kernel_reservoir.get_output_dim() // (self.feedback_channel - 1) # Since the first channel of the output will always be included

        trainable = (self.kernel_reservoir.input_weights.data[..., input_channel:]!=0)
        self.kernel_reservoir.input_weights.data[..., input_channel:][trainable] *= weight_coef
        if input2excitatory_only:
            self.kernel_reservoir.input_weights.data[self.kernel_reservoir.inhibitory_indices] = 0 # No input-inhibitory connection

        if bias_channel is not None:
            self.bias_spikes = torch.zeros(
                (bias_channel,), dtype=torch.float32,
                device=self.kernel_reservoir.get_trainable_weights().device)

    def clone(self):
        new_kernel = self.kernel_reservoir.clone()
        new_fb_reservoir = FeedbackReservoir(
                new_kernel,
                self.input_channel,
                self.feedback_channel,
                self.bias_channel,
                self.weight_coef,
                self.output_dim,
                self.output_feedback_channels,
                self.output_excitatory_only,
                self.input2excitatory_only)
        new_fb_reservoir.config_tunnelers(**self.tunneler_config_dict)
        return new_fb_reservoir

    def forward(self, input_spikes: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if self.bias_channel is not None:
                batch_size = input_spikes.shape[1]
                if self.bias_spikes.device != input_spikes.device:
                    self.bias_spikes = self.bias_spikes.to(device=input_spikes.device)
                self.bias_spikes = self.bias_spikes!=1
                self.bias_spikes = self.bias_spikes.to(dtype=torch.float32)
                bias_spikes = self.bias_spikes.unsqueeze(-1).expand(-1, batch_size)
                input_spikes = torch.cat([input_spikes, bias_spikes], dim=0)

            feedback = None # [feedback_channel, batchsize]
            previous_output = self.kernel_reservoir.get_liquid_state()
            if previous_output is None:
                feedback = torch.zeros(
                        (self.feedback_channel, input_spikes.shape[1]),
                        device = input_spikes.device,
                        dtype = input_spikes.dtype)
            else:
                feedback = previous_output[: self.interval * self.feedback_channel: self.interval]

            kernel_input_dim = self.kernel_reservoir.get_input_dim()
            kernel_input = _form_kernel_input(
                    input_spikes, feedback, kernel_input_dim)
        liquid_state = self.kernel_reservoir(kernel_input)
        if self.output_excitatory_only:
            mask = torch.ones_like(liquid_state)
            mask[self.kernel_reservoir.inhibitory_indices] = 0
            liquid_state = liquid_state * mask
        if self.output_feedback_channels:
            liquid_state = liquid_state[::self.interval][:self.feedback_channel]
        if self.output_dim is not None:
            liquid_state = evenly_select(liquid_state, self.output_dim)
        return liquid_state

    def reset(self) -> None:
        self.kernel_reservoir.reset()
        if self.bias_channel is not None:
            self.bias_spikes.fill_(0)

    def get_output_dim(self) -> int:
        output_dim = self.kernel_reservoir.get_output_dim()
        if self.output_feedback_channels:
            output_dim = self.feedback_channel
        if self.output_dim is not None:
            output_dim = self.output_dim
        return output_dim

    def config_tunnelers(self, **tunneler_config_dict) -> None:
        super().config_tunnelers(**tunneler_config_dict)
        if len(tunneler_config_dict) == 0:
            return # May happen when clone() called, if tunneler not configurated yet.
        tunneler = _base.GradientTunneler(
                self.kernel_reservoir,
                **tunneler_config_dict
                )
        self._append_tunneler_list(tunneler)

    def get_kernel_reservoir(self):
        return self.kernel_reservoir

class FeedbackReservoirDefault(TrainableMicrocircuitBase): 

    def __init__(self, input_dim: int, feedback_dim: int, neuron_num: int,
            output_dim: int = None,
            output_offset: int = 0,
            evidence_collection_length = 200,
            train_input = False,
            spike_output = False,
            regenerate_weights = False):
        super().__init__()
        self.kernel_reservoir = Reservoir(
                input_dim+feedback_dim, math.ceil(neuron_num**(1/3)), neuron_num, spike_output = spike_output
                )
        if regenerate_weights:
            with torch.no_grad():
                input_weights = generate_input_weights(
                        input_dim,
                        neuron_num,
                        self.kernel_reservoir.somas.basic_threshold
                        ) * 0.5
                feedback_weights = generate_input_weights(
                        feedback_dim,
                        neuron_num,
                        self.kernel_reservoir.somas.basic_threshold
                        ) * 0.5
                now_device = self.kernel_reservoir.input_weights.data.device
                self.kernel_reservoir.input_weights.data = torch.cat(
                        [input_weights, feedback_weights],
                        dim = 1).to(device = now_device)
        ERR = 0.05
        self.kernel_reservoir.set_monitor_filter_coef(
                1 - (ERR)**(1/(evidence_collection_length))
                )
        self.feedback_dim = feedback_dim
        self.output_dim = output_dim
        self.output_offset = output_offset
        self.train_input = train_input
        self.spike_output = spike_output
        self.regenerate_weights = regenerate_weights

        self.interval = self.kernel_reservoir.get_output_dim() // (self.feedback_dim - 1) # Since the first channel of the output will always be included

    def clone(self):
        new_fb_reservoir = FeedbackReservoirDefault(
                self.kernel_reservoir.get_input_dim()-self.feedback_dim,
                self.feedback_dim,
                self.kernel_reservoir.neuron_num,
                self.output_dim,
                self.output_offset,
                self.kernel_reservoir.monitor_filter_coef,
                self.train_input,
                self.spike_output,
                self.regenerate_weights
                )
        new_kernel = self.kernel_reservoir.clone()
        new_fb_reservoir.kernel_reservoir = new_kernel
        new_fb_reservoir.config_tunnelers(**self.tunneler_config_dict)
        return new_fb_reservoir

    def forward(self, input_spikes: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feedback = None # [feedback_dim, batchsize]
            previous_output = self.kernel_reservoir.get_liquid_state()
            if previous_output is None:
                feedback = torch.zeros(
                        (self.feedback_dim, input_spikes.shape[1]),
                        device = input_spikes.device,
                        dtype = input_spikes.dtype)
            else:
                feedback = previous_output[::self.interval][:self.feedback_dim]

            kernel_input_dim = self.kernel_reservoir.get_input_dim()
            kernel_input = _form_kernel_input(
                    input_spikes, feedback, kernel_input_dim)
        liquid_state = self.kernel_reservoir(kernel_input)
        if self.output_dim is not None:
            liquid_state = evenly_select(liquid_state, self.output_dim, self.output_offset)
        return liquid_state

    def reset(self) -> None:
        self.kernel_reservoir.reset()

    def get_output_dim(self) -> int:
        output_dim = self.kernel_reservoir.get_output_dim()
        if self.output_dim is not None:
            output_dim = self.output_dim
        return output_dim

    def config_tunnelers(self, **tunneler_config_dict) -> None:
        super().config_tunnelers(**tunneler_config_dict)
        if len(tunneler_config_dict) == 0:
            return # May happen when clone() called, if tunneler not configurated yet.
        masked_amount = self.kernel_reservoir.get_input_dim()-self.feedback_dim
        regularized_amount = self.feedback_dim
        if self.train_input:
            masked_amount = 0
            regularized_amount = self.kernel_reservoir.get_input_dim()
        tunneler = _base.GradientTunneler(
                self.kernel_reservoir,
                masked_amount = masked_amount,
                regularized_amount = regularized_amount,
                **tunneler_config_dict
                )
        self._append_tunneler_list(tunneler)

class HierarchicalFeedbackCircuit(TrainableMicrocircuitBase):

    def __init__(self, input_dim: int, evidence_collection_length = 200):
        super().__init__()
        self.input_dim = input_dim
        self.evidence_collection_length = evidence_collection_length

        NEURON_NUM = 6**3
        feedback_dim = int(NEURON_NUM * 0.05)
        self.supra_lamina = FeedbackReservoirDefault(
                input_dim, feedback_dim, NEURON_NUM,
                output_dim = None,
                evidence_collection_length = evidence_collection_length,
                train_input = False,
                spike_output = True
                )
        self.GAIN = 2
        self.infra_lamina = FeedbackReservoirDefault(
                self.GAIN*feedback_dim, feedback_dim, NEURON_NUM,
                output_dim = None,
                evidence_collection_length = evidence_collection_length,
                train_input = True,
                spike_output = False
                )
        self.feedback_dim = feedback_dim
        self.NEURON_NUM = NEURON_NUM

    def clone(self):
        new_DFC = HierarchicalFeedbackCircuit(self.input_dim, self.evidence_collection_length)
        new_DFC.supra_lamina = self.supra_lamina.clone()
        new_DFC.infra_lamina = self.infra_lamina.clone()
        new_DFC.config_tunnelers(**self.tunneler_config_dict)
        return new_DFC

    def forward(self, x):
        y = self.supra_lamina(x)
        interval = self.NEURON_NUM // (self.feedback_dim - 1)
        offset_list = [int(interval*i/self.GAIN) for i in range(self.GAIN)]
        y_selected_list = [evenly_select(y, self.feedback_dim, offset_n) for offset_n in offset_list]
        infra_input = torch.cat(y_selected_list, dim=0)
        z = self.infra_lamina(infra_input)
        final_state = torch.cat([y,z], dim=0)
        return final_state

    def reset(self):
        self.supra_lamina.reset()
        self.infra_lamina.reset()

    def get_output_dim(self):
        return self.infra_lamina.get_output_dim() + self.supra_lamina.get_output_dim()

    def config_tunnelers(self, **tunneler_config_dict):
        super().config_tunnelers(**tunneler_config_dict)
        if len(tunneler_config_dict) == 0:
            return
        def backward_ls(learning_signal):
            learning_signal = torch.permute(learning_signal, (1,2,0)) #[d_l, d_l-1, bs]

            interval = self.NEURON_NUM // (self.feedback_dim - 1)
            offset_list = [int(interval*i/self.GAIN) for i in range(self.GAIN)]
            embeded = torch.zeros((self.NEURON_NUM, *learning_signal.shape[1:]), device = learning_signal.device)
            for i in range(self.GAIN):
                ls = learning_signal[self.feedback_dim*i : self.feedback_dim*(i+1)]
                offset_n = offset_list[i]
                embeded += evenly_embed(ls, self.NEURON_NUM, offset_n)
            embeded = torch.permute(embeded, (2, 0, 1))
            return embeded
        self.infra_lamina.config_tunnelers(compute_weight_grad = True, **tunneler_config_dict)
        self.supra_lamina.config_tunnelers(
                ls_operator = backward_ls,
                **tunneler_config_dict
                )
        infra_tunneler = self.infra_lamina.get_tunneler_list()[0]
        supra_tunneler = self.supra_lamina.get_tunneler_list()[0]
        infra_tunneler.cascade_upstream(supra_tunneler)
        self._append_tunneler_list(infra_tunneler)
        self._append_tunneler_list(supra_tunneler)

class ReservoirOfflineWrapper(_base.OfflineWrapperBase):

    def __init__(self, reservoir: _base.ReservoirBase | TrainableMicrocircuitBase):
        super().__init__(reservoir)

    def get_output_shape(self, input_channel: int):
        dim = self.system.get_output_dim() # For LSM, the output shape is fixed after initialization
        return (dim,)

def compute_lyapunov_exponent(reservoir: Reservoir, test_num=10):
    lyapunovs = []

    for _ in range(test_num):
        spike_train = _base.generate_spike_train(reservoir.input_dim, 20, 0.4)
        spike_indices = [] # (timestep, channel)
        for timestep in range(spike_train.shape[-1]):
            for channel in range(spike_train.shape[0]):
                value = spike_train[channel, timestep]
                if value == 1:
                    spike_indices.append((timestep, channel))
                    if len(spike_indices) == 2:
                        break
            if len(spike_indices) == 2:
                break

        spike_train_var = spike_train.clone()
        var_timestep, var_channel = spike_indices[0]
        if var_timestep != 0:
            spike_train_var[var_channel, 0] = 1
        spike_train_var[var_channel, var_timestep] = 0
        batch = torch.cat([spike_train.unsqueeze(0), spike_train_var.unsqueeze(0)], dim=0)

        wrapper = ReservoirOfflineWrapper(reservoir)
        liquid_states = wrapper(batch)
        truncate_time, _ = spike_indices[1]
        truncated_liquid_states = liquid_states[..., truncate_time+1:]

        variation = (truncated_liquid_states[0] - truncated_liquid_states[1])**2
        variation = torch.sqrt(variation.sum(dim=0)) # [timestep,]
        if variation[0].item() < 1e-9:
            warnings.warn("(For Lyapunov exponent) The reservoir is not sensitive enough to the single-spike variation.")
        var_ratio = variation / variation[0]
        epsilon = 1e-10
        log_var = torch.log(var_ratio+epsilon).cpu().numpy()

        timesteps = np.arange(len(log_var))
        exponent, _ = np.polyfit(timesteps, log_var, 1)
        lyapunovs.append(exponent)

    lyapunovs_arr = np.array(lyapunovs)
    lyapunov_exponent = lyapunovs_arr.mean()

    return lyapunov_exponent

class StdLSM(torch.nn.Module):

    def __init__(self, reservoir: Reservoir, output_dim = 2, random_ratio = 0.1):
        super().__init__()
        self.random_ratio = random_ratio
        self.wrapped_reservoir = ReservoirOfflineWrapper(reservoir)
        self.wrapped_reservoir = torch.jit.script(self.wrapped_reservoir)
        dim_mid = reservoir.get_output_dim()
        self.regression = torch.nn.Linear(dim_mid, output_dim)

    def forward(self, x):
        with torch.no_grad():
            x = self.wrapped_reservoir(x)
            if self.training:
                end = x.shape[-1]
                start = int(self.random_ratio * end)
                idx = np.random.randint(start, end)
                x = x[..., idx]
            else:
                idx = (1-self.random_ratio/2) * x.shape[-1]
                x = x[..., idx]
        x = self.regression(x)
        return x

# Following: Some validated realization of reservoirs

def get_in14_edge10_avgfilter_reservoir(lyap_min=-0.0006, lyap_max=0, generate_iter=10):
    '''
    The weight generation distribution of these parameters is
    with an average Lyapunov exponent as -0.0003.
    '''
    channel = 14
    edge_length = 10
    avg_filter = _base.IterativeMovingAverage(1 - (0.05)**(1/(200+1)))
    reservoir = Reservoir(channel, edge_length, liquid_filter=avg_filter)
    lyap = 0
    for _ in range(generate_iter):
        reservoir.generate_input_weights(
                density = 0.3)
        lyap = compute_lyapunov_exponent(reservoir)
        print(f"Generating with Lyapunov={lyap}")
        if lyap is torch.nan:
            continue

        if lyap > lyap_min and lyap < lyap_max:
            break
    print("Reservoir generated with Lyapunov exponent: {}".format(lyap))
    return reservoir

def get_in14_edge9_avgfilter_reservoir(lyap_min=-0.0001, lyap_max=0, generate_iter=10):
    '''
    The weight generation distribution of these parameters is
    with an average Lyapunov exponent as -0.0004.
    '''
    channel = 14
    edge_length = 9
    avg_filter = _base.IterativeMovingAverage(1 - (0.05)**(1/(200+1)))
    reservoir = Reservoir(channel, edge_length, liquid_filter=avg_filter)
    lyap = 0
    for _ in range(generate_iter):
        reservoir.generate_input_weights(
                density = 0.29)
        lyap = compute_lyapunov_exponent(reservoir)
        print(f"Generating with Lyapunov={lyap}")
        if lyap is torch.nan:
            continue

        if lyap > lyap_min and lyap < lyap_max:
            break
    print("Reservoir generated with Lyapunov exponent: {}".format(lyap))
    return reservoir

def get_in14_edge8_avgfilter_reservoir(lyap_min=-0.0005, lyap_max=0, generate_iter=10):
    '''
    The weight generation distribution of these parameters is
    with an average Lyapunov exponent as -0.0004.
    '''
    channel = 14
    edge_length = 8
    avg_filter = _base.IterativeMovingAverage(1 - (0.05)**(1/(200+1)))
    reservoir = Reservoir(channel, edge_length, liquid_filter=avg_filter)
    lyap = 0
    for _ in range(generate_iter):
        reservoir.generate_input_weights(
                density = 0.32)
        lyap = compute_lyapunov_exponent(reservoir)
        print(f"Generating with Lyapunov={lyap}")
        if lyap is torch.nan:
            continue

        if lyap > lyap_min and lyap < lyap_max:
            break
    print("Reservoir generated with Lyapunov exponent: {}".format(lyap))
    return reservoir

def get_in14_edge7_avgfilter_reservoir(lyap_min=-0.0005, lyap_max=0, generate_iter=10):
    '''
    The weight generation distribution of these parameters is
    with an average Lyapunov exponent as -0.0005.
    '''
    channel = 14
    edge_length = 7
    avg_filter = _base.IterativeMovingAverage(1 - (0.05)**(1/(200+1)))
    reservoir = Reservoir(channel, edge_length, liquid_filter=avg_filter)
    lyap = 0
    for _ in range(generate_iter):
        reservoir.generate_input_weights(
                density = 0.38)
        lyap = compute_lyapunov_exponent(reservoir)
        print(f"Generating with Lyapunov={lyap}")
        if lyap is torch.nan:
            continue

        if lyap > lyap_min and lyap < lyap_max:
            break
    print("Reservoir generated with Lyapunov exponent: {}".format(lyap))
    return reservoir
    
def get_in14_edge6_avgfilter_reservoir(lyap_min=-0.0005, lyap_max=0, generate_iter=10):
    '''
    The weight generation distribution of these parameters is
    with an average Lyapunov exponent as .
    '''
    channel = 14
    edge_length = 6
    avg_filter = _base.IterativeMovingAverage(1 - (0.05)**(1/(200+1)))
    reservoir = Reservoir(channel, edge_length, liquid_filter=avg_filter)
    lyap = 0
    for _ in range(generate_iter):
        reservoir.generate_input_weights(
                density = 0.48)
        lyap = compute_lyapunov_exponent(reservoir)
        print(f"Generating with Lyapunov={lyap}")
        if lyap is torch.nan:
            continue

        if lyap > lyap_min and lyap < lyap_max:
            break
    print("Reservoir generated with Lyapunov exponent: {}".format(lyap))
    return reservoir
    
def get_in2_edge8_avgfilter_reservoir(lyap_min=-0.001, lyap_max=0, generate_iter=10):
    '''
    The weight generation distribution of these parameters is
    with an average Lyapunov exponent as .
    '''
    channel = 2
    edge_length = 8
    avg_filter = _base.IterativeMovingAverage(1 - (0.05)**(1/9))
    reservoir = Reservoir(channel, edge_length, liquid_filter=avg_filter)
    lyap = 0
    for _ in range(generate_iter):
        reservoir.generate_recursive_weights(mean=1.1)
        reservoir.generate_input_weights(density = 0.14)
        lyap = compute_lyapunov_exponent(reservoir)
        print(f"Generating with Lyapunov={lyap}")
        if lyap is torch.nan:
            continue

        if lyap > lyap_min and lyap < lyap_max:
            break
    print("Reservoir generated with Lyapunov exponent: {}".format(lyap))
    return reservoir
