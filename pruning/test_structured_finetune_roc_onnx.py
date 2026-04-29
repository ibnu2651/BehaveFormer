import sys
sys.path.append("..")

import torch
import numpy as np
import onnxruntime as ort
from model.dataset import HUMITestDataset
import os
from torch.utils.data import DataLoader
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
        periods.append(
            get_window_time_humi(test_dataset.data[user_id][num_enroll_sess + j])
        )

    for i in range(len(test_dataset.data)):
        if i != user_id:
            for j in range(num_verify_sess):
                periods.append(
                    get_window_time_humi(test_dataset.data[i][num_enroll_sess + j])
                )

    return periods


# =========================
# Argument parser (MATCHES YOUR FIRST SCRIPT)
# =========================

parser = argparse.ArgumentParser()
parser.add_argument("config", type=str, help="config")
args = parser.parse_args()


# =========================
# Model path using config
# =========================

model_int8_path = f"/home/i/ibnu2651/BehaveFormer/pruning/prune_structured_{args.config}_last_int8.onnx"


# =========================
# ONNX Runtime
# =========================

ort_session = ort.InferenceSession(
    model_int8_path,
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
# Run ONNX inference
# =========================

onnx_outputs = []

for batch in test_dataloader:
    behave_inputs, imu_inputs = batch

    behave_np = behave_inputs.detach().cpu().numpy().astype(np.float32)
    imu_np = imu_inputs.detach().cpu().numpy().astype(np.float32)

    ort_inputs = {
        ort_session.get_inputs()[0].name: behave_np,
        ort_session.get_inputs()[1].name: imu_np,
    }

    out_onnx = ort_session.run(None, ort_inputs)[0]
    onnx_outputs.append(out_onnx)


onnx_outputs = np.concatenate(onnx_outputs, axis=0)
onnx_outputs = torch.from_numpy(onnx_outputs)

print("ONNX outputs:", onnx_outputs.shape)


# =========================
# Reshape outputs
# =========================

onnx_outputs = onnx_outputs.view(
    test_dataset.num_users,
    test_dataset.num_sessions,
    test_dataset.num_seqs,
    64
)


# =========================
# Evaluation
# =========================

num_enroll_sessions = 3
num_verify_sessions = 2

onnx_acc = []
onnx_usability = []
onnx_tcr = []
onnx_fawi = []
onnx_frwi = []

all_scores = []
all_labels = []

for i in range(onnx_outputs.shape[0]):
    all_ver_embeddings = torch.cat([
        onnx_outputs[i, num_enroll_sessions:],
        torch.flatten(onnx_outputs[:i, num_enroll_sessions:], start_dim=0, end_dim=1),
        torch.flatten(onnx_outputs[i + 1:, num_enroll_sessions:], start_dim=0, end_dim=1)
    ], dim=0)

    scores = Metric.cal_session_distance_fixed_sessions(
        all_ver_embeddings,
        onnx_outputs[i, :num_enroll_sessions]
    )

    periods = get_periods(i, num_enroll_sessions, num_verify_sessions)

    labels = torch.tensor(
        [1] * num_verify_sessions +
        [0] * (onnx_outputs.shape[0] - 1) * num_verify_sessions
    )

    # IMPORTANT: invert distance for ROC
    all_scores.extend((-scores).detach().cpu().numpy())
    all_labels.extend(labels.detach().cpu().numpy())

    acc, threshold = Metric.eer_compute(
        scores[:num_verify_sessions],
        scores[num_verify_sessions:]
    )

    usability = Metric.calculate_usability(scores, threshold, periods, labels)
    tcr = Metric.calculate_TCR(scores, threshold, periods, labels)
    frwi = Metric.calculate_FRWI(scores, threshold, periods, labels)
    fawi = Metric.calculate_FAWI(scores, threshold, periods, labels)

    onnx_acc.append(acc)
    onnx_usability.append(usability)
    onnx_tcr.append(tcr)
    onnx_fawi.append(fawi)
    onnx_frwi.append(frwi)


print(
    "ONNX\nEER:",
    100 - np.mean(onnx_acc, axis=0),
    "Usability:", np.mean(onnx_usability, axis=0),
    "TCR:", np.mean(onnx_tcr, axis=0),
    "FRWI:", np.mean(onnx_frwi, axis=0),
    "FAWI:", np.mean(onnx_fawi, axis=0)
)


# =========================
# ROC Curve
# =========================

all_scores = np.array(all_scores)
all_labels = np.array(all_labels)

fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ONNX ROC ({args.config})")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()

plt.savefig(f"onnx_roc_{args.config}.png", dpi=300)
plt.show()

print("ROC AUC:", roc_auc)