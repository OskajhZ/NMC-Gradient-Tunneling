import random

import LSM_lib
import torch
import numpy as np
import h5py

random.seed(42)

def prepare_model(model_dir: str) -> torch.nn.Module:
    INPUT_CHANNEL = 14
    NEURON_NUM = 8**3
    feedback_channel = int(NEURON_NUM * 0.05)
    fb_reservoir_prototype = LSM_lib.reservoir.FeedbackReservoirDefault(
            INPUT_CHANNEL, feedback_channel, NEURON_NUM,
            output_dim=None,  evidence_collection_length=200, regenerate_weights=True)
    print(f"Output dim of feedback reservoir: {fb_reservoir_prototype.get_output_dim()}")
    fb_reservoir_prototype.config_tunnelers(
            regularization_coef = 1e-3,
            target_firing_rate = 0.40,
            )
    model = LSM_lib.models.BandsDenseLSM(fb_reservoir_prototype,
                                        band_amount=4,
                                        category_num=3)
    
    state_dict = torch.load(model_dir)
    model.load_state_dict(state_dict)
    return model

def forward_recurrent(
        model: torch.nn.Module,
        x: torch.Tensor
        ):
    '''
    x: [batch_size, band_amount, channel_amount, timesteps]
    return : [batch_size, band_amount, neuron_num, timesteps]
    '''
    model.eval()

    wrapped_reservoir_list = model.reservoir.wrapped_reservoirs
    reservoir_list = [reservoir.system for reservoir in wrapped_reservoir_list]

    neuron_amount = reservoir_list[0].get_output_dim()
    
    recurrent_parts = torch.zeros(
            x.shape[0],
            x.shape[1],
            neuron_amount,
            x.shape[3]
            ).to(device=x.device)

    for band_idx in range(x.shape[1]):
        x_band = x[:, band_idx, ...].permute(2,1,0) # [timesteps, channel_amount, batch_size]
        reservoir_band = reservoir_list[band_idx]
        reservoir_band.reset()
        for timestep in range(x_band.shape[0]):
            _, recurrent = reservoir_band(x_band[timestep]) # recurrent: [neuron_num, batch_size]
            recurrent = recurrent.T # [batch_size, neuron_num]
            recurrent_parts[:, band_idx, :, timestep] = recurrent

    return recurrent_parts

def get_input(fseed_file: str):
    '''
    return: [1, band_idx, channel_idx, timesteps]
    '''

    EEG, labels = LSM_lib.EEG.load_SEEDh5(
                fseed_file,
                session = "session_1",
                selected_electrodes = LSM_lib.EEG.SEED_14_electrode_indices
                )

    is_iter = LSM_lib.training_utils.ISIter(
            EEG, labels,
            LSM_lib.EEG.preprocess_as_FES,
            train_test_val_ratio = (0.6, 0.2, 0.2),
            segment_window = 200,
            segment_stride = None,
            encode = True,
            skip_to = 14
            )

    for i, (train_set, _, _) in enumerate(is_iter):
        if i == 14:
            train_data = torch.from_numpy(train_set["data"]).to(dtype=torch.float32)
            select_idx = random.randint(0, train_data.shape[0]-1)
            selected_data = train_data[select_idx].unsqueeze(0)
            return selected_data

def save_recurrent(
        recurrent_traj: torch.Tensor,
        save_name: str):
    recurrent_traj = recurrent_traj.cpu().numpy()
    with h5py.File(save_name, "w") as root:
        root.create_dataset("recurrent_traj", data=recurrent_traj)

if __name__ == "__main__":
    model_dir = "/home/featurize/work/SEED_exp/Noise/Default8_gradient_42/session1_AdamW/models/model_state_dict_15.pt"
    fseed_dir = "/home/featurize/datasets/dc4fcfee-422a-46e3-8d03-8cc53593d27b/fSEED.h5"
    save_dir = "recurrent_traj.h5"

    model = prepare_model(model_dir)
    data = get_input(fseed_dir)
    if torch.cuda.is_available():
        model = model.to("cuda")
        data = data.cuda()

    with torch.no_grad():
        recurrent_traj = forward_recurrent(model, data)

    save_recurrent(recurrent_traj, save_dir)
