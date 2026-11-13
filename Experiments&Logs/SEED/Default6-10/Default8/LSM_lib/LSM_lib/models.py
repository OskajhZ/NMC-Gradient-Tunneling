'''
By Xiangnan Zhang, 2025,
    School of Future Technologies, Beijing Institute of Technology.

This file contains Readouts and LSM models.
'''
import warnings
from typing import List

import torch

from . import _base
from . import reservoir

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

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.parallel_streams = [torch.cuda.Stream(device=device) for _ in range(amount)]

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
            with torch.cuda.stream(self.parallel_streams[idx]):
                out = self.wrapped_reservoirs[idx](x[:, idx]) # [batchsize, reservoir_dim, timestep]
                out = out.unsqueeze(1) # [batchsize, 1, reservoir_dim, timestep]
                reservoir_out[idx] = out
        torch.cuda.synchronize()
        x = torch.cat(reservoir_out, dim = 1)
        if not retain_history:
            x = x[..., -1]
        return x

class DenseReadout(torch.nn.Module):

    def __init__(self, reservoir_num, reservoir_output_dim, category_num, feature_dim = 3):
        super().__init__()
        self.reservoir_output_dim = reservoir_output_dim

        self.decrease_list = torch.nn.ModuleList()
        for _ in range(reservoir_num):
            dec = torch.nn.Sequential(
                    torch.nn.BatchNorm1d(reservoir_output_dim),
                    torch.nn.Linear(reservoir_output_dim, feature_dim)
                    )
            self.decrease_list.append(dec)
        
        self.out = torch.nn.Linear(feature_dim*reservoir_num, category_num)

        self.liquid_state_checked = False

    def project_latent(self, x):
        '''
        Project liquid states linearly into latent space.
        x: [batch_size, reservoir_amount, length_of_liquid_states]
        return: [batchsize, reservoir_amount, feature_dim]
        '''
        feature_list = []
        for i in range(x.shape[1]):
            feature = self.decrease_list[i](x[:, i])
            feature_list.append(feature.unsqueeze(1))
        all_feature = torch.cat(feature_list, dim=1)
        return all_feature

    def forward(self, x):
        '''
        x: [batch_size, reservoir_amount, length_of_liquid_states]
            where length_of_liquid_states == edge**3
        '''
        if not self.liquid_state_checked:
            if _base.are_spikes(x):
                warnings.warn("Liquid states should not be spike trains.")
            self.liquid_state_checked = True
        
        latent_feature = self.project_latent(x)
        latent_feature = latent_feature.view(
                latent_feature.shape[0],
                latent_feature.shape[1]*latent_feature.shape[2]) # = cat along dim=-1
        y = self.out(latent_feature)
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
                category_num,
                feature_dim = 3
                )

    def forward_evolution(self, x):
        '''
        x: [batch_size, band_amount, channel_amount, timesteps]
        return: (trajectory in latent space, trajectory of liquid states)
            [batch_size, band_amount, space_dim (=3 if latent, =liquid_state_length if liquid), timesteps]
        '''
        batchsize, band_amount, _, timesteps = x.shape
        x = x[..., :self.band_amount, :, :]
        states = self.reservoir(x, retain_history=True) # [batchsize, reservoir_amount, liquid_state_length, timestep]
        y = states.permute(0, 3, 1, 2) # [batchsize, timesteps, reservoir_amount, liquid_state_length]
        y = y.reshape(batchsize*timesteps, band_amount, y.shape[-1]) # [batchsize*timesteps, reservoir_amount, liquid_state_length]
        y = self.readout.project_latent(y) # [batchsize*timesteps, reservoir_amount, latent_space_dim=3]
        y = y.reshape(batchsize, timesteps, band_amount, y.shape[-1])
        y = y.permute(0, 2, 3, 1)
        return y, states

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
