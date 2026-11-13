'''
Classical Reservoir tuned according to Lyapunov exponent.
'''

import numpy as np
import NMC_lib

if __name__ == "__main__":
    fseed_file = "/home/featurize/datasets/dc4fcfee-422a-46e3-8d03-8cc53593d27b/fSEED.h5"

    EEG_list, labels_list = [], []
    for session in ["session_3"]:
        EEG, labels = NMC_lib.EEG.load_SEEDh5(
                fseed_file,
                session = session,
                selected_electrodes = NMC_lib.EEG.SEED_14_electrode_indices
                )
        EEG_list.append(EEG)
        labels_list.append(labels)
    fEEG = np.concatenate(EEG_list, axis=1)
    labels = np.concatenate(labels_list, axis=1)

    is_iter = NMC_lib.training_utils.ISIter(
            fEEG, labels,
            NMC_lib.EEG.preprocess_as_FES,
            train_test_val_ratio = (0.6, 0.2, 0.2),
            segment_window = 200,
            segment_stride = None,
            encode = True
            )

    OUTPUT_DIM = 25
    reservoir_prototype = NMC_lib.reservoir.get_in14_edge8_avgfilter_reservoir()
    reservoir_prototype.skip_monitor = True
    reservoir_prototype = NMC_lib.reservoir.OutputLimiter(
            reservoir_prototype,
            OUTPUT_DIM)
    model = NMC_lib.models.BandsDenseLSM(reservoir_prototype,
                                        band_amount=4,
                                        category_num=3)

    _, record, _ = NMC_lib.training_utils.top_testbench(
            is_iter,
            model,
            train_batch_size = 512,
            epochs = 300,
            early_stop_start = 100,
            reservoir_lr = 0,
            readout_lr = 1e-3,
            ensemble = False
            )

    print("Full Record:")
    print(record)

    NMC_lib.training_utils.output_avg_se(record)
