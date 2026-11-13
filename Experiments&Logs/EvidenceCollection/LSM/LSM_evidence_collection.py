import numpy as np
import torch
import NMC_lib

if __name__ == "__main__":

    block_length = 20
    CHANNEL_PER_BAND = 25

    reservoir = NMC_lib.reservoir.get_in100_edge10_avgfilter_reservoir()
    reservoir.skip_monitor = True
    model = NMC_lib.models.AddTaskNMC(reservoir, 2)

    block_length = 40
    generate_func = lambda : NMC_lib.generate_evidence_collection(512, cue_block_length=block_length, resting_length=block_length, channel_per_band=CHANNEL_PER_BAND)

    executor = NMC_lib.training_utils.EvidenceCollectionExecutor(
            model, generate_func,
            reservoir_lr = 0,
            readout_lr = 1e-4,
            log_folder = "logs"
            )

#    executor.load()

    executor.train(iteration = 4000, save_point = [500, 1500, 2500, 3500])
