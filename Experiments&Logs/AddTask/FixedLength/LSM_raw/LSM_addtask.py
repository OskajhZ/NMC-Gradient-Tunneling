import numpy as np
import torch
import NMC_lib

if __name__ == "__main__":

    length = 90
    bit = 4

    reservoir = NMC_lib.reservoir.get_in2_edge8_avgfilter_reservoir()
    reservoir.skip_monitor = True
    model = NMC_lib.models.AddTaskNMC(reservoir)

    generate_func = lambda gating_period: NMC_lib.generate_add_task_pair_v2(length, 512, bit, gating_period)

    executor = NMC_lib.training_utils.AddTaskExecutor(
            model, generate_func, 2*bit,
            reservoir_lr = 0,
            readout_lr = 1e-4,
            log_folder = "logs"
            )

    executor.train(iteration = 20000, incremental = False)

