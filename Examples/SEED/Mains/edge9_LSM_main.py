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

    LENGTH = 200

    is_iter = NMC_lib.training_utils.ISIter(
            fEEG, labels,
            NMC_lib.EEG.preprocess_as_FES,
            train_test_val_ratio = (0.6, 0.2, 0.2),
            segment_window = LENGTH,
            segment_stride = None,
            encode = True
            )

    INPUT_CHANNEL = 14
    FEEDBACK_CHANNEL = 48
    REGUL_CHANNEL = 10
    kernel_reservoir = NMC_lib.reservoir.get_in62_edge9_avgfilter_reservoir()

    ERR = 0.05
    kernel_reservoir.set_monitor_filter_coef(
            1 - (ERR)**(1/(LENGTH+1))
            )
    fb_reservoir_prototype = NMC_lib.reservoir.FeedbackReservoir(
            kernel_reservoir, INPUT_CHANNEL, FEEDBACK_CHANNEL, weight_coef=1.7)
    fb_reservoir_prototype.config_tunnelers(
            masked_amount = INPUT_CHANNEL,
            regularization_coef = 10,
            target_firing_rate = 0.25,
            regularized_amount = REGUL_CHANNEL
            )
    model = NMC_lib.models.BandsDenseLSM(fb_reservoir_prototype,
                                        band_amount=4,
                                        category_num=3)

    tunneler_list = model.get_tunneler_list()

    _, record, _ = NMC_lib.training_utils.top_testbench(
            is_iter,
            model,
            train_batch_size = 512,
            epochs = 300,
            early_stop_start = 100,
            reservoir_lr = 1e-1,
            readout_lr = 1e-3,
            ensemble = False,
            grad_tunneler_list = tunneler_list
            )

    print("Full Record:")
    print(record)

    NMC_lib.training_utils.output_avg_se(record)
