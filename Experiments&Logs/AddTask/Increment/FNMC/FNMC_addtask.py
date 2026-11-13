import numpy as np
import torch
import LSM_lib


if __name__ == "__main__":

    length = 90
    bit = 4

    INPUT_CHANNEL = 2
    NEURON_NUM = 8**3
    feedback_channel = int(NEURON_NUM * 0.1)
    fb_NMC = LSM_lib.reservoir.FeedbackReservoirDefault(
            INPUT_CHANNEL, feedback_channel, NEURON_NUM,
            output_dim=None,  evidence_collection_length=15, regenerate_weights=True)
    fb_NMC.config_tunnelers(
            regularization_coef = 1,
            target_firing_rate = 0.3,
            )
    model = LSM_lib.models.AddTaskNMC(fb_NMC)
    tunneler_list = model.get_tunneler_list()

    generate_func = lambda gating_period: LSM_lib.generate_add_task_pair_v2(length, 512, bit, gating_period)

    executor = LSM_lib.training_utils.AddTaskExecutor(
            model, generate_func, 2*bit,
            reservoir_lr = 1,
            readout_lr = 1e-4,
            grad_tunneler_list = tunneler_list,
            log_folder = "logs"
            )

    executor.train(iteration = 20000)

