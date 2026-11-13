import numpy as np
import NMC_lib

if __name__ == "__main__":
    fdeap_file = "/home/featurize/datasets/174af60d-ebd8-4669-a5e7-1eb771bc7c24/fDEAP.h5"

    fEEG, labels = NMC_lib.EEG.load_DEAPh5(
            fdeap_file,
            "arousal",
            selected_electrodes = NMC_lib.EEG.DEAP_14_electrode_indices
            )
    
    LENGTH = 128

    is_iter = NMC_lib.training_utils.ISIter(
            fEEG, labels,
            NMC_lib.EEG.preprocess_as_FES,
            train_test_val_ratio = (0.6, 0.2, 0.2),
            segment_window = LENGTH,
            segment_stride = None,
            encode = True
            )

    reservoir_prototype = NMC_lib.reservoir.get_in14_edge8_avgfilter_reservoir()
    reservoir_prototype.skip_monitor = True
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
