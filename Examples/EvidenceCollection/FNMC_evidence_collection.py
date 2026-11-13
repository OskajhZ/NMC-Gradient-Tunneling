import numpy as np
import torch
import NMC_lib

if __name__ == "__main__":

    CHANNEL_PER_BAND = 25

    INPUT_CHANNEL = 4*CHANNEL_PER_BAND
    NEURON_NUM = 10**3
    feedback_channel = int(NEURON_NUM * 0.05)
    fb_NMC = NMC_lib.reservoir.FeedbackReservoirDefault(
            INPUT_CHANNEL, feedback_channel, NEURON_NUM,
            output_dim=None,  stationary_window_length=15, regenerate_weights=True)
    fb_NMC.config_tunnelers(
            regularization_coef = 1,
            target_firing_rate = 0.2,
            )
    model = NMC_lib.models.AddTaskNMC(fb_NMC, 2)

    tunneler_list = model.get_tunneler_list()

    generate_func = lambda : NMC_lib.generate_evidence_collection(512, 20, channel_per_band=CHANNEL_PER_BAND)

    executor = NMC_lib.training_utils.EvidenceCollectionExecutor(
            model, generate_func,
            reservoir_lr = 1e-1,
            readout_lr = 1e-4,
            grad_tunneler_list = tunneler_list,
            log_folder = "logs"
            )

#    executor.load()

    executor.train(iteration = 4000, save_point = [500, 1500, 2500, 3500])
