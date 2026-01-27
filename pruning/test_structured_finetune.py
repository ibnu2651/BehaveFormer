import sys
sys.path.append("..")

import torch
import numpy as np
from model.behaveformer import BehaveFormer
from model.dataset import HUMITestDataset
import os
from torch.utils.data import DataLoader
from torch import nn
import json
from evaluation.metrics import Metric

def get_periods(user_id, num_enroll_sess, num_verify_sess=None, user_session_count=None, dataset='humi'):
    def get_window_time_humi(seqs):
        seq = seqs[0][0]
        start = seq[0][0]
        end = seq[-1][0]
        
        i = -1
        while (end == 0):  # handle zero padding at the end
            end = seq[i-1][0]
            i = i - 1
        return (end - start) / 1000

    def get_window_time_feta(seqs):
        seq = seqs[0].numpy()
        start = seq[0][0]
        end = seq[-1][0]
        
        i = -1
        while (end == 0):  # handle zero padding at the end
            end = seq[i-1][0]
            i = i - 1
        return (end - start) / 1000

    # Get 2 period from the same user and 2 from different user
    periods = []
    for j in range(num_verify_sess):
        periods.append(get_window_time_humi(test_dataset.data[user_id][num_enroll_sess + j]))
    for i in range(len(test_dataset.data)):
        if (i != user_id):
            for j in range(num_verify_sess):
                periods.append(get_window_time_humi(test_dataset.data[i][num_enroll_sess + j]))
    return periods

@torch.no_grad()
def prune_two_linear_mlp(seq: nn.Sequential, new_hidden: int):
    fc1: nn.Linear = seq[0]
    fc2: nn.Linear = seq[3]
    hidden = fc1.out_features

    importance = fc2.weight.abs().sum(dim=0)  # [hidden]
    keep = torch.topk(importance, k=new_hidden, largest=True).indices.sort().values

    new_fc1 = nn.Linear(fc1.in_features, new_hidden, bias=(fc1.bias is not None))
    new_fc2 = nn.Linear(new_hidden, fc2.out_features, bias=(fc2.bias is not None))

    new_fc1.weight.copy_(fc1.weight[keep, :])
    if fc1.bias is not None:
        new_fc1.bias.copy_(fc1.bias[keep])

    new_fc2.weight.copy_(fc2.weight[:, keep])
    if fc2.bias is not None:
        new_fc2.bias.copy_(fc2.bias)

    seq[0] = new_fc1
    seq[3] = new_fc2


def keep_first_n_encoder_layers(transformer_module, n: int):
    transformer_module.encoder.layers = nn.ModuleList(list(transformer_module.encoder.layers[:n]))


model_path = "/home/i/ibnu2651/BehaveFormer/pruning/prune_structured_finetuned_last.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BehaveFormer(8, 36, 50, 100, 64, 20, 4, 10, 6, 10, "acc_gyr_mag")
prune_two_linear_mlp(model.linear_imu, new_hidden=1200)
prune_two_linear_mlp(model.linear_behave, new_hidden=160)
keep_first_n_encoder_layers(model.behave_transformer, 3)
keep_first_n_encoder_layers(model.imu_transformer, 3)
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu"), weights_only=True))
model.to(device)

# Force any "hidden" tensors in the positional encoding to the correct device
for m in model.modules():
    if hasattr(m, 'positions'):
        m.positions = m.positions.to(device)
    if hasattr(m, 'mu'):
        m.mu = m.mu.to(device)
    if hasattr(m, 'sigma'):
        m.sigma = m.sigma.to(device)

model.eval()

test_dataset = HUMITestDataset(action='down', 
                                validation_file=os.path.join('/home/i/ibnu2651/BehaveFormer/Humidb/scroll50downup_imu100all', 'testing_scroll_imu_data_all.pickle'),
                                imu_type='acc_gyr_mag')
test_dataloader = DataLoader(test_dataset, batch_size=16)

pt_outputs = []

with torch.no_grad():
    for batch in test_dataloader:
        behave_inputs, imu_inputs = batch  # adjust if your dataset returns differently
        print(behave_inputs.shape, imu_inputs.shape)

        # move to device
        behave_inputs = behave_inputs.to(device).float()
        imu_inputs = imu_inputs.to(device).float()

        # forward pass
        outputs = model((behave_inputs, imu_inputs))

        pt_outputs.append(outputs.cpu())


pt_outputs = torch.cat(pt_outputs, dim=0)

# Evaluate
pt_outputs = pt_outputs.view(test_dataset.num_users, test_dataset.num_sessions, test_dataset.num_seqs, 64)

num_enroll_sessions = 3
num_verify_sessions = 2

pt_acc = []
pt_usability = []
pt_tcr = []
pt_fawi = []
pt_frwi = []

for i in range(pt_outputs.shape[0]):
    all_ver_embeddings = torch.cat([pt_outputs[i,num_enroll_sessions:], torch.flatten(pt_outputs[:i,num_enroll_sessions:], start_dim=0, end_dim=1), torch.flatten(pt_outputs[i+1:,num_enroll_sessions:], start_dim=0, end_dim=1)], dim=0)
    scores = Metric.cal_session_distance_fixed_sessions(all_ver_embeddings, pt_outputs[i,:num_enroll_sessions])

    periods = get_periods(i, num_enroll_sessions, num_verify_sessions)   #### use num_verify_sessions for period & also skip num_enroll_sessions
    labels = torch.tensor([1] * num_verify_sessions + [0] * (pt_outputs.shape[0] - 1) * num_verify_sessions)

    acc, threshold = Metric.eer_compute(scores[:num_verify_sessions], scores[num_verify_sessions:])

    usability = Metric.calculate_usability(scores, threshold, periods, labels)
    tcr = Metric.calculate_TCR(scores, threshold, periods, labels)
    frwi = Metric.calculate_FRWI(scores, threshold, periods, labels)
    fawi = Metric.calculate_FAWI(scores, threshold, periods, labels)
    pt_acc.append(acc)    
    pt_usability.append(usability)
    pt_tcr.append(tcr)
    pt_fawi.append(fawi)
    pt_frwi.append(frwi)


print("Pytorch\nEER:", 100 - np.mean(pt_acc, axis=0), "Usability:", np.mean(pt_usability, axis=0), "TCR:", np.mean(pt_tcr, axis=0), "FRWI:", np.mean(pt_frwi, axis=0) , "FAWI:", np.mean(pt_fawi, axis=0))