import sys
sys.path.append("..")

import torch
import numpy as np
import onnxruntime as ort
from model.behaveformer import BehaveFormer
from model.dataset import HUMITestDataset
import os
from torch.utils.data import DataLoader
from torch import nn
from evaluation.metrics import Metric
import argparse

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


# =========================
# Utils
# =========================

def get_periods(user_id, num_enroll_sess, num_verify_sess=None):
    def get_window_time_humi(seqs):
        seq = seqs[0][0]
        start = seq[0][0]
        end = seq[-1][0]

        i = -1
        while end == 0:
            end = seq[i - 1][0]
            i -= 1

        return (end - start) / 1000

    periods = []

    for j in range(num_verify_sess):
        periods.append(get_window_time_humi(test_dataset.data[user_id][num_enroll_sess + j]))

    for i in range(len(test_dataset.data)):
        if i != user_id:
            for j in range(num_verify_sess):
                periods.append(get_window_time_humi(test_dataset.data[i][num_enroll_sess + j]))

    return periods


@torch.no_grad()
def prune_two_linear_mlp(seq: nn.Sequential, new_hidden: int):
    fc1: nn.Linear = seq[0]
    fc2: nn.Linear = seq[3]

    importance = fc2.weight.abs().sum(dim=0)
    keep = torch.topk(importance, k=new_hidden, largest=True).indices.sort().values

    new_fc1 = nn.Linear(fc1.in_features, new_hidden)
    new_fc2 = nn.Linear(new_hidden, fc2.out_features)

    new_fc1.weight.copy_(fc1.weight[keep, :])
    new_fc1.bias.copy_(fc1.bias[keep])
    new_fc2.weight.copy_(fc2.weight[:, keep])
    new_fc2.bias.copy_(fc2.bias)

    seq[0] = new_fc1
    seq[3] = new_fc2


def keep_first_n_encoder_layers(transformer_module, n: int):
    transformer_module.encoder.layers = nn.ModuleList(
        list(transformer_module.encoder.layers[:n])
    )


# =========================
# Args
# =========================

parser = argparse.ArgumentParser()
parser.add_argument("config", type=str)
args = parser.parse_args()


# =========================
# Paths
# =========================

pruned_path = f"/home/i/ibnu2651/BehaveFormer/pruning/prune_structured_{args.config}_last.pt"
onnx_path = f"/home/i/ibnu2651/BehaveFormer/pruning/prune_structured_{args.config}_last_int8.onnx"

baseline_path = "/home/i/ibnu2651/BehaveFormer/work_dirs/humi_scroll50down_imu100all_epoch500_enroll3_b128/20231026_155303/best_models/epoch_210_eer_2.60817307692308.pt"


# =========================
# Models
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- Baseline -----
baseline_model = BehaveFormer(8, 36, 50, 100, 64, 20, 4, 10, 6, 10, "acc_gyr_mag")
baseline_model.load_state_dict(torch.load(baseline_path, map_location="cpu", weights_only=True))
baseline_model.to(device).eval()

for m in baseline_model.modules():
    if hasattr(m, "positions"):
        m.positions = m.positions.to(device)
    if hasattr(m, "mu"):
        m.mu = m.mu.to(device)
    if hasattr(m, "sigma"):
        m.sigma = m.sigma.to(device)

# ----- Pruned -----
model = BehaveFormer(8, 36, 50, 100, 64, 20, 4, 10, 6, 10, "acc_gyr_mag")

prune_two_linear_mlp(model.linear_imu, 200)
prune_two_linear_mlp(model.linear_behave, 40)

keep_first_n_encoder_layers(model.behave_transformer, 3)
keep_first_n_encoder_layers(model.imu_transformer, 3)

model.load_state_dict(torch.load(pruned_path, map_location="cpu", weights_only=True))
model.to(device).eval()

# Fix buffers
for m in model.modules():
    if hasattr(m, 'positions'):
        m.positions = m.positions.to(device)
    if hasattr(m, 'mu'):
        m.mu = m.mu.to(device)
    if hasattr(m, 'sigma'):
        m.sigma = m.sigma.to(device)


# ----- ONNX -----
ort_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])


# =========================
# Dataset
# =========================

test_dataset = HUMITestDataset(
    action="down",
    validation_file=os.path.join(
        "/home/i/ibnu2651/BehaveFormer/Humidb/scroll50downup_imu100all",
        "testing_scroll_imu_data_all.pickle"
    ),
    imu_type="acc_gyr_mag"
)

test_dataloader = DataLoader(test_dataset, batch_size=16)


# =========================
# Inference
# =========================

baseline_outputs = []
pt_outputs = []
onnx_outputs = []

with torch.no_grad():
    for behave_inputs, imu_inputs in test_dataloader:

        behave_inputs = behave_inputs.to(device).float()
        imu_inputs = imu_inputs.to(device).float()

        # Baseline
        baseline_outputs.append(
            baseline_model((behave_inputs, imu_inputs)).cpu()
        )

        # Pruned
        pt_outputs.append(
            model((behave_inputs, imu_inputs)).cpu()
        )

        # ONNX
        behave_np = behave_inputs.cpu().numpy().astype(np.float32)
        imu_np = imu_inputs.cpu().numpy().astype(np.float32)

        ort_inputs = {
            ort_session.get_inputs()[0].name: behave_np,
            ort_session.get_inputs()[1].name: imu_np,
        }

        onnx_outputs.append(ort_session.run(None, ort_inputs)[0])


baseline_outputs = torch.cat(baseline_outputs, dim=0)
pt_outputs = torch.cat(pt_outputs, dim=0)
onnx_outputs = torch.from_numpy(np.concatenate(onnx_outputs, axis=0))


# reshape
baseline_outputs = baseline_outputs.view(test_dataset.num_users, test_dataset.num_sessions, test_dataset.num_seqs, 64)
pt_outputs = pt_outputs.view(test_dataset.num_users, test_dataset.num_sessions, test_dataset.num_seqs, 64)
onnx_outputs = onnx_outputs.view(test_dataset.num_users, test_dataset.num_sessions, test_dataset.num_seqs, 64)


# =========================
# Evaluation
# =========================

num_enroll_sessions = 3
num_verify_sessions = 2

baseline_scores_all, pt_scores_all, onnx_scores_all = [], [], []
labels_all = []

for i in range(baseline_outputs.shape[0]):

    labels = torch.tensor(
        [1] * num_verify_sessions +
        [0] * (baseline_outputs.shape[0] - 1) * num_verify_sessions
    )

    # baseline
    base_emb = torch.cat([
        baseline_outputs[i, num_enroll_sessions:],
        torch.flatten(baseline_outputs[:i, num_enroll_sessions:], 0, 1),
        torch.flatten(baseline_outputs[i+1:, num_enroll_sessions:], 0, 1)
    ], dim=0)

    base_scores = Metric.cal_session_distance_fixed_sessions(base_emb, baseline_outputs[i, :num_enroll_sessions])
    baseline_scores_all.extend((-base_scores).cpu().numpy())

    # pruned
    pt_emb = torch.cat([
        pt_outputs[i, num_enroll_sessions:],
        torch.flatten(pt_outputs[:i, num_enroll_sessions:], 0, 1),
        torch.flatten(pt_outputs[i+1:, num_enroll_sessions:], 0, 1)
    ], dim=0)

    pt_scores = Metric.cal_session_distance_fixed_sessions(pt_emb, pt_outputs[i, :num_enroll_sessions])
    pt_scores_all.extend((-pt_scores).cpu().numpy())

    # onnx
    onnx_emb = torch.cat([
        onnx_outputs[i, num_enroll_sessions:],
        torch.flatten(onnx_outputs[:i, num_enroll_sessions:], 0, 1),
        torch.flatten(onnx_outputs[i+1:, num_enroll_sessions:], 0, 1)
    ], dim=0)

    onnx_scores = Metric.cal_session_distance_fixed_sessions(onnx_emb, onnx_outputs[i, :num_enroll_sessions])
    onnx_scores_all.extend((-onnx_scores).cpu().numpy())

    labels_all.extend(labels.cpu().numpy())


# =========================
# ROC Plot
# =========================

labels_all = np.array(labels_all)

fpr_b, tpr_b, _ = roc_curve(labels_all, np.array(baseline_scores_all))
fpr_p, tpr_p, _ = roc_curve(labels_all, np.array(pt_scores_all))
fpr_o, tpr_o, _ = roc_curve(labels_all, np.array(onnx_scores_all))

auc_b = auc(fpr_b, tpr_b)
auc_p = auc(fpr_p, tpr_p)
auc_o = auc(fpr_o, tpr_o)

plt.figure(figsize=(6,6))

plt.plot(fpr_b, tpr_b, color="red", label=f"Baseline (AUC={auc_b:.4f})")
plt.plot(fpr_p, tpr_p, color="blue", label=f"Pareto-optimal (AUC={auc_p:.4f})")
plt.plot(fpr_o, tpr_o, color="green", label=f"Pareto-optimal with\nquantisation (AUC={auc_o:.4f})")

plt.plot([0,1],[0,1],"k--", label="Random")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Comparison")
plt.legend(loc="lower right")
plt.grid(True)

plt.savefig(f"roc_all_{args.config}.png", dpi=300)
plt.show()

print("AUC Baseline:", auc_b)
print("AUC Pruned:", auc_p)
print("AUC ONNX:", auc_o)