import numpy as np
import NMC_lib


if __name__ == "__main__":
    fseed_file = "/home/featurize/datasets/dc4fcfee-422a-46e3-8d03-8cc53593d27b/fSEED.h5"

    subject_idx = 14

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

    LENGTH = 400

    is_iter = NMC_lib.training_utils.ISIter(
            fEEG, labels,
            NMC_lib.EEG.preprocess_as_FES,
            train_test_val_ratio = (0.6, 0.2, 0.2),
            segment_window = LENGTH,
            segment_stride = None,
            encode = True,
            skip_to = subject_idx
            )

    INPUT_CHANNEL = 14
    FEEDBACK_CHANNEL = 48
    ERR = 0.05
    kernel_reservoir = NMC_lib.reservoir.get_in62_edge11_avgfilter_reservoir()
    kernel_reservoir.liquid_filter = None

    kernel_reservoir.set_monitor_filter_coef(
            1 - (ERR)**(1/(LENGTH+1))
            )
    fb_reservoir_prototype = NMC_lib.reservoir.FeedbackReservoir(kernel_reservoir, INPUT_CHANNEL, FEEDBACK_CHANNEL, weight_coef=2)
    model = NMC_lib.models.BandsDenseLSM(fb_reservoir_prototype,
                                        band_amount=4,
                                        category_num=3)
    model.to("cuda")

    NMC_lib.training_utils.get_trajectory(
            is_iter,
            model,
            criterion_input_length = 200,
            subject_idx = subject_idx,
            segment_amount = 10,
            trajectory_dump_file = "traj_subject14.h5",
            select_correct = True
            )
