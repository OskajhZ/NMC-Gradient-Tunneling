import numpy as np
import torch
import NMC_lib

def do_test(model, session_idx: int):
    fseed_file = "/home/featurize/datasets/dc4fcfee-422a-46e3-8d03-8cc53593d27b/fSEED.h5"
    EEG_list, labels_list = [], []
    for session in [f"session_{session_idx}"]:
        EEG, labels = NMC_lib.EEG.load_SEEDh5(
                fseed_file,
                session = session,
                selected_electrodes = NMC_lib.EEG.SEED_14_electrode_indices
                )
        EEG_list.append(EEG)
        labels_list.append(labels)
    fEEG = np.concatenate(EEG_list, axis=1)
    labels = np.concatenate(labels_list, axis=1)

    length = 200
    is_iter = NMC_lib.training_utils.ISIter(
            fEEG, labels,
            NMC_lib.EEG.preprocess_as_FES,
            train_test_val_ratio = (0.6, 0.2, 0.2),
            segment_window = length,
            encode = False,
            mean = 0,
            std = 1,
            avg_through_time = False
            )


    for i in range(5):
        print(f"Test with length={is_iter.segment_window}:")
        _, record, _ = NMC_lib.training_utils.testonly_bench(
                is_iter,
                model
                )
        print("Full Record:")
        print(record)
        NMC_lib.training_utils.output_avg_se(record)
        print("")
        is_iter.segment_window += 10


if __name__ == "__main__":

    # Configure model start

    # Configure model end

    do_test(model, 1)

