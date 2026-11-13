import numpy as np
import torch
import LSM_lib

if __name__ == "__main__":

    block_length = 20
    CHANNEL_PER_BAND = 25

    INPUT_CHANNEL = 4*CHANNEL_PER_BAND
    NEURON_NUM = 10**3
    feedback_channel = int(NEURON_NUM * 0.05)
    fb_NMC = LSM_lib.reservoir.FeedbackReservoirDefault(
            INPUT_CHANNEL, feedback_channel, NEURON_NUM,
            output_dim=None,  evidence_collection_length=block_length, regenerate_weights=True)
    fb_NMC.config_tunnelers(
            regularization_coef = 1,
            target_firing_rate = 0.2,
            )
    model = LSM_lib.models.AddTaskNMC(fb_NMC, 2)

    tunneler_list = model.get_tunneler_list()

    block_length = 40
    generate_func = lambda : LSM_lib.generate_evidence_collection(512, cue_block_length=block_length, resting_length=block_length, channel_per_band=CHANNEL_PER_BAND)

    executor = LSM_lib.training_utils.EvidenceCollectionExecutor(
            model, generate_func,
            reservoir_lr = 1e-1,
            readout_lr = 1e-4,
            grad_tunneler_list = tunneler_list,
            log_folder = "logs"
            )

#    executor.load()

    executor.train(iteration = 4000, save_point = [500, 1500, 2500, 3500])
