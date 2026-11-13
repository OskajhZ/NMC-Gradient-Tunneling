'''
By Xiangnan Zhang, 2025,
    School of Future Technologies, Beijing Institute of Technology.

High-level readout layers and multi-band reservoir ensembles.
Implements: DenseReadout / ResidualReadout (readout heads), ParallelReservoirs
(multi-band wrapper around TrainableMicrocircuitBase), BandsDenseLSM (full
multi-band LSM with dense readout), AddTaskNMC (add-task model with residual
readout). Also provides the LSMBase abstract interface.
'''
import warnings
import random
from typing import List

import torch

from . import _base
from . import reservoir

random.seed(42)
torch.torch.manual_seed(42)

class LSMBase(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.reservoir: torch.nn.Module = None
        self.readout: torch.nn.Module = None

    def register(self, reservoir: torch.nn.Module, readout: torch.nn.Module):
        self.reservoir = reservoir
        self.readout = readout

    def get_reservoir(self):
        return self.reservoir

    def get_readout(self):
        return self.readout

    def get_tunneler_list(self) -> List[_base.GradientTunneler]:
        '''
        This method should be overriden by the sub-class, if it is needed.
        '''
        return []

    def forward_evolution(self) -> torch.Tensor:
        raise NotImplementedError()

class ParallelReservoirs(torch.nn.Module):
    '''
    Dealing with multi-band signals like filtered EEGs.
    '''

    def __init__(self, reservoir_prototype: reservoir.TrainableMicrocircuitBase, amount: int):
        super().__init__()
        self.amount = amount

        self.wrapped_reservoirs = torch.nn.ModuleList()
        for _ in range(amount):
            sub_reservoir = reservoir_prototype.clone()
            wrapped = reservoir.ReservoirOfflineWrapper(sub_reservoir)
            self.wrapped_reservoirs.append(wrapped)

    def forward(self, x, retain_history=False):
        '''
        x: [batch_size, band_amount, channel_amount, timesteps]
            where band_amount == reservoir_amount,
            channel_amount == reservoir_input.
        return:
            If retain_history: [batch_size, reservoir_amount, reservoir_dim, time]
            else: [batch_size, reservoir_amount, reservoir_dim]
        '''
        reservoir_out = [None for _ in range(self.amount)]
        for idx in range(self.amount):
            out = self.wrapped_reservoirs[idx](x[:, idx]) # [batchsize, reservoir_dim, timestep]
            out = out.unsqueeze(1) # [batchsize, 1, reservoir_dim, timestep]
            reservoir_out[idx] = out
        x = torch.cat(reservoir_out, dim = 1)
        if not retain_history:
            if self.training:
                x = x[..., -1 * random.randint(1, int(0.1*x.shape[-1]))]
            else:
                x = x[..., -1]
        return x

    def reset_system(self):
        for wrapped_reservoir in self.wrapped_reservoirs:
            wrapped_reservoir.reset_system()

    def forward_step(self, x):
        '''
        Advance every reservoir branch by one timestep.

        x: [batch_size, reservoir_amount, channel_amount].
        return: [batch_size, reservoir_amount, reservoir_dim].
        '''
        if x.dim() != 3:
            raise ValueError(
                    "ParallelReservoirs single-step input must have shape "
                    "[batch_size, reservoir_amount, channel_amount], "
                    f"got {x.shape}."
                    )
        if x.shape[1] < self.amount:
            raise ValueError(
                    f"Expected at least {self.amount} reservoir branches, "
                    f"got {x.shape[1]}."
                    )

        reservoir_out = []
        for idx, wrapped_reservoir in enumerate(self.wrapped_reservoirs):
            out = wrapped_reservoir.forward_step(x[:, idx])
            reservoir_out.append(out.unsqueeze(1))
        return torch.cat(reservoir_out, dim=1)

class DenseReadout(torch.nn.Module):

    def __init__(self, reservoir_num, reservoir_output_dim, category_num):
        super().__init__()
        self.reservoir_output_dim = reservoir_output_dim
        self.reservoir_num = reservoir_num

        self.norm = torch.nn.BatchNorm1d(reservoir_output_dim*reservoir_num) 
        self.out = torch.nn.Linear(reservoir_output_dim*reservoir_num, category_num)

        self.liquid_state_checked = False

    def forward(self, x):
        '''
        x: [batch_size, reservoir_amount, length_of_liquid_states]
            where length_of_liquid_states == edge**3
        '''
        if not self.liquid_state_checked:
            if _base.are_spikes(x):
                warnings.warn("Liquid states should not be spike trains.")
            self.liquid_state_checked = True
       
        x = x.reshape(-1, self.reservoir_output_dim*self.reservoir_num)
        x_normed = self.norm(x)
        y = self.out(x_normed)
        return y

class BandsDenseLSM(LSMBase):
    '''
    Using ParallelReservoirs to deal with multi-band EEGs,
    and using ConvReadout for classification.
    '''

    def __init__(self,
            prototype_reservoir: reservoir.Reservoir,
            band_amount = 4, # For Theta, Alpha, Beta, Gamma
            category_num = 3 # For SEED
            ):
        '''
        category_num <= 3
        '''
        super().__init__()
        self.band_amount = band_amount

        self.reservoir = ParallelReservoirs(
                prototype_reservoir,
                band_amount
                )

        self.readout = DenseReadout(
                band_amount,
                prototype_reservoir.get_output_dim(),
                category_num
                )

    def forward(self, x):
        '''
        x: [batch_size, band_amount, channel_amount, timesteps]
        return: [batch_size, category_num]
        '''
        x = x[..., :self.band_amount, :, :]
        x = self.reservoir(x, retain_history=False) # [batchsize, reservoir_amount, liquid_state_length]
        x = self.readout(x)
        return x

    def get_tunneler_list(self) -> List[_base.GradientTunneler]:
        tunneler_list = []
        for wrapped_reservoir in self.reservoir.wrapped_reservoirs:
            per_reservoir_tunnelers = wrapped_reservoir.system.get_tunneler_list()
            for tunneler in per_reservoir_tunnelers:
                tunneler_list.append(tunneler)
        return tunneler_list

# Following: Add Task Model

class ResidualReadout(torch.nn.Module):

    def __init__(self, reservoir_output_dim, out_dim = 1, intermediate_ratio=0.1):
        super().__init__()
        self.norm = torch.nn.BatchNorm1d(reservoir_output_dim)
        self.res_comput = torch.nn.Sequential(
                torch.nn.Linear(reservoir_output_dim, int(intermediate_ratio*reservoir_output_dim)),
                torch.nn.GELU(),
                torch.nn.Linear(int(intermediate_ratio*reservoir_output_dim), reservoir_output_dim)
                )
        self.out = torch.nn.Sequential(
                torch.nn.BatchNorm1d(reservoir_output_dim),
                torch.nn.Linear(reservoir_output_dim, out_dim)
                )

    def forward(self, x):
        '''
        x: [batch_size, reservoir_output_dim]
        '''
        normed_x = self.norm(x)
        res = self.res_comput(normed_x)
        y = normed_x + res
        y = self.out(y)
        return y

class AddTaskNMC(LSMBase):

    def __init__(self, microcircuit, output_dim = 1):
        super().__init__()
        self.reservoir = reservoir.ReservoirOfflineWrapper(microcircuit)
        self.readout = ResidualReadout(
                microcircuit.get_output_dim(), output_dim
                )

    def forward(self, x):
        state = self.reservoir(x)[..., -1]
        y = self.readout(state)
        return y

    def get_tunneler_list(self):
        return self.reservoir.system.get_tunneler_list()
