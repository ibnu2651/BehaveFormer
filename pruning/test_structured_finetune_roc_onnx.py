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
    transformer_module.encoder.layers = nn.ModuleList(
        list(transformer_module.encoder.layers[:n])
    )


parser = argparse.ArgumentParser()
parser.add_argument("config", type=str, help="config")
args = parser.parse_args()


# =========================
# Paths
# =========================

pt_model_path = f"/home/i/ibnu2651/BehaveFormer/pruning/prune_structured_{args.config}_last.pt"
onnx_model_path = f"/home/i/ibnu2651/BehaveFormer/pruning/prune_structured_{args.config}_last_int8.onnx"


# =========================
# Load PyTorch model
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

new_imu_hidden = 200
new_behave_hidden = 40
new_num_layers_behave = 3
new_num_layers_imu = 3

model = BehaveFormer(8, 36, 50, 100, 64, 20, 4, 10, 6, 10, "acc_gyr_mag")

prune_two_linear_mlp(model.linear_imu, new_hidden=new_imu_hidden)
prune_two_linear_mlp(model.linear_behave, new_hidden=new_behave_hidden)

keep_first_n_encoder_layers(model.behave_transformer, new_num_layers_behave)
keep_first_n_encoder_layers(model.imu_transformer, new_num_layers_imu)

model.load_state_dict(
    torch.load(pt_model_path, map_location=torch.device("cpu"), weights_only=True)
)
model.to(device)

for m in model.modules():
    if hasattr(m, "positions"):
        m.positions = m.positions.to(device)
    if hasattr(m, "mu"):
        m.mu = m.mu.to(device)
    if hasattr(m, "sigma"):
        m.sigma = m.sigma.to(device)

model.eval()


# =========================
# Load ONNX model
# =========================

ort_session = ort.InferenceSession(
    onnx_model_path,
    providers=["CPUExecutionProvider"]
)


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

pt_outputs = []
onnx_outputs = []

with torch.no_grad():
    for batch in test_dataloader:
        behave_inputs, imu_inputs = batch

        behave_inputs = behave_inputs.to(device).float()
        imu_inputs = imu_inputs.to(device).float()

        pt_out = model((behave_inputs, imu_inputs))
        pt_outputs.append(pt_out.cpu())

        behave_np = behave_inputs.detach().cpu().numpy().astype(np.float32)
        imu_np = imu_inputs.detach().cpu().numpy().astype(np.float32)

        ort_inputs = {
            ort_session.get_inputs()[0].name: behave_np,
            ort_session.get_inputs()[1].name: imu_np,
        }

        onnx_out = ort_session.run(None, ort_inputs)[0]
        onnx_outputs.append(onnx_out)


pt_outputs = torch.cat(pt_outputs, dim=0)
onnx_outputs = torch.from_numpy(np.concatenate(onnx_outputs, axis=0))

pt_outputs = pt_outputs.view(
    test_dataset.num_users,
    test_dataset.num_sessions,
    test_dataset.num_seqs,
    64
)

onnx_outputs = onnx_outputs.view(
    test_dataset.num_users,
    test_dataset.num_sessions,
    test_dataset.num_seqs,
    64
)


# =========================
# Evaluation + ROC data
# =========================

num_enroll_sessions = 3
num_verify_sessions = 2

pt_acc = []
onnx_acc = []

pt_scores_all = []
pt_labels_all = []

onnx_scores_all = []
onnx_labels_all = []

for i in range(pt_outputs.shape[0]):
    labels = torch.tensor(
        [1] * num_verify_sessions +
        [0] * (pt_outputs.shape[0] - 1) * num_verify_sessions
    )

    # ----- PyTorch -----
    pt_ver_embeddings = torch.cat([
        pt_outputs[i, num_enroll_sessions:],
        torch.flatten(pt_outputs[:i, num_enroll_sessions:], start_dim=0, end_dim=1),
        torch.flatten(pt_outputs[i + 1:, num_enroll_sessions:], start_dim=0, end_dim=1)
    ], dim=0)

    pt_scores = Metric.cal_session_distance_fixed_sessions(
        pt_ver_embeddings,
        pt_outputs[i, :num_enroll_sessions]
    )

    pt_scores_all.extend((-pt_scores).detach().cpu().numpy())
    pt_labels_all.extend(labels.detach().cpu().numpy())

    pt_acc_i, _ = Metric.eer_compute(
        pt_scores[:num_verify_sessions],
        pt_scores[num_verify_sessions:]
    )
    pt_acc.append(pt_acc_i)

    # ----- ONNX -----
    onnx_ver_embeddings = torch.cat([
        onnx_outputs[i, num_enroll_sessions:],
        torch.flatten(onnx_outputs[:i, num_enroll_sessions:], start_dim=0, end_dim=1),
        torch.flatten(onnx_outputs[i + 1:, num_enroll_sessions:], start_dim=0, end_dim=1)
    ], dim=0)

    onnx_scores = Metric.cal_session_distance_fixed_sessions(
        onnx_ver_embeddings,
        onnx_outputs[i, :num_enroll_sessions]
    )

    onnx_scores_all.extend((-onnx_scores).detach().cpu().numpy())
    onnx_labels_all.extend(labels.detach().cpu().numpy())

    onnx_acc_i, _ = Metric.eer_compute(
        onnx_scores[:num_verify_sessions],
        onnx_scores[num_verify_sessions:]
    )
    onnx_acc.append(onnx_acc_i)


print("PyTorch EER:", 100 - np.mean(pt_acc, axis=0))
print("ONNX EER:", 100 - np.mean(onnx_acc, axis=0))


# =========================
# Combined ROC Plot
# =========================

pt_fpr, pt_tpr, _ = roc_curve(np.array(pt_labels_all), np.array(pt_scores_all))
pt_auc = auc(pt_fpr, pt_tpr)

onnx_fpr, onnx_tpr, _ = roc_curve(np.array(onnx_labels_all), np.array(onnx_scores_all))
onnx_auc = auc(onnx_fpr, onnx_tpr)

plt.figure(figsize=(6, 6))

plt.plot(pt_fpr, pt_tpr, color="blue", label=f"Without quantisation (AUC = {pt_auc:.4f})")
plt.plot(onnx_fpr, onnx_tpr, color="red", label=f"With quantisation (AUC = {onnx_auc:.4f})")

plt.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve (Pareto-optimal)")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()

plt.savefig(f"roc_comparison_{args.config}.png", dpi=300)
plt.show()

print("PyTorch ROC AUC:", pt_auc)
print("ONNX ROC AUC:", onnx_auc)