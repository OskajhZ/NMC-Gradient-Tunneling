import numpy as np
import LSM_lib

if __name__ == "__main__":
    fseed_file = "/home/featurize/datasets/dc4fcfee-422a-46e3-8d03-8cc53593d27b/fSEED.h5"

    EEG_list, labels_list = [], []
    for session in ["session_2"]:
        EEG, labels = LSM_lib.EEG.load_SEEDh5(
                fseed_file,
                session = session,
                selected_electrodes = LSM_lib.EEG.SEED_14_electrode_indices
                )
        EEG_list.append(EEG)
        labels_list.append(labels)
    fEEG = np.concatenate(EEG_list, axis=1)
    labels = np.concatenate(labels_list, axis=1)

    LENGTH = 200

    is_iter = LSM_lib.training_utils.ISIter(
            fEEG, labels,
            LSM_lib.EEG.preprocess_as_FES,
            train_test_val_ratio = (0.6, 0.2, 0.2),
            segment_window = LENGTH,
            segment_stride = None,
            encode = True
            )

    INPUT_CHANNEL = 14
    NEURON_NUM = 7**3
    feedback_channel = int(NEURON_NUM * 0.05)
    fb_reservoir_prototype = LSM_lib.reservoir.FeedbackReservoirDefault(
            INPUT_CHANNEL, feedback_channel, NEURON_NUM,
            output_dim=feedback_channel,  evidence_collection_length=LENGTH)
    print(f"Output dim of feedback reservoir: {fb_reservoir_prototype.get_output_dim()}")
    fb_reservoir_prototype.config_tunnelers(
            regularization_coef = 1,
            target_firing_rate = 0.40,
            )
    model = LSM_lib.models.BandsDenseLSM(fb_reservoir_prototype,
                                        band_amount=4,
                                        category_num=3)

    tunneler_list = model.get_tunneler_list()

    _, record, _ = LSM_lib.training_utils.top_testbench(
            is_iter,
            model,
            train_batch_size = 512,
            epochs = 300,
            early_stop_start = 100,
            reservoir_lr = 1,
            readout_lr = 1e-4,
            ensemble = False,
            grad_tunneler_list = tunneler_list
            )

    print("Full Record:")
    print(record)

    LSM_lib.training_utils.output_avg_se(record)
