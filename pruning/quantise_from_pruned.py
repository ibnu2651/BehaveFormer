import sys
sys.path.append("..")

import os
import numpy as np
import torch
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from torch.utils.data import DataLoader

from model.dataset import HUMITestDataset
from evaluation.metrics import Metric


def get_periods(user_id, num_enroll_sess, num_verify_sess=None):
    def get_window_time_humi(seqs):
        seq = seqs[0][0]
        start = seq[0][0]
        end = seq[-1][0]

        i = -1
        while end == 0:  # handle zero padding at the end
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


# =========================
# Paths
# =========================
model_fp32_path = "/home/i/ibnu2651/BehaveFormer/pruning/prune_structured_200_40_3.onnx"
model_int8_path = "/home/i/ibnu2651/BehaveFormer/pruning/prune_structured_200_40_3_int8.onnx"

test_data_path = "/home/i/ibnu2651/BehaveFormer/Humidb/scroll50downup_imu100all/testing_scroll_imu_data_all.pickle"


# =========================
# Quantise existing ONNX
# =========================
print("Quantising ONNX model...")
quantize_dynamic(
    model_input=model_fp32_path,
    model_output=model_int8_path,
    weight_type=QuantType.QUInt8,
)
print(f"Quantised model saved to: {model_int8_path}")


# =========================
# Load quantised ONNX model
# =========================
ort_session = ort.InferenceSession(
    model_int8_path,
    providers=["CPUExecutionProvider"]
)

print("ONNX inputs:")
for inp in ort_session.get_inputs():
    print(f"  {inp.name}: shape={inp.shape}, type={inp.type}")

print("ONNX outputs:")
for out in ort_session.get_outputs():
    print(f"  {out.name}: shape={out.shape}, type={out.type}")


# =========================
# Test dataset
# =========================
test_dataset = HUMITestDataset(
    action="down",
    validation_file=test_data_path,
    imu_type="acc_gyr_mag"
)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

onnx_outputs = []

print("Running inference on quantised ONNX model...")
for batch in test_dataloader:
    behave_inputs, imu_inputs = batch

    behave_np = behave_inputs.cpu().numpy().astype(np.float32)
    imu_np = imu_inputs.cpu().numpy().astype(np.float32)

    ort_inputs = {
        ort_session.get_inputs()[0].name: behave_np,
        ort_session.get_inputs()[1].name: imu_np,
    }

    out_onnx = ort_session.run(None, ort_inputs)[0]
    onnx_outputs.append(out_onnx)

onnx_outputs = np.concatenate(onnx_outputs, axis=0)
onnx_outputs = torch.from_numpy(onnx_outputs)

torch.set_printoptions(precision=8)
print("Quantised ONNX outputs:", onnx_outputs.shape)
print(onnx_outputs)


# =========================
# Evaluation
# =========================
embedding_dim = onnx_outputs.shape[-1]
onnx_quantised_outputs = onnx_outputs.view(
    test_dataset.num_users,
    test_dataset.num_sessions,
    test_dataset.num_seqs,
    embedding_dim
)

num_enroll_sessions = 3
num_verify_sessions = 2

onnx_quantised_acc = []
onnx_quantised_usability = []
onnx_quantised_tcr = []
onnx_quantised_fawi = []
onnx_quantised_frwi = []

for i in range(onnx_quantised_outputs.shape[0]):
    all_ver_embeddings = torch.cat(
        [
            onnx_quantised_outputs[i, num_enroll_sessions:],
            torch.flatten(
                onnx_quantised_outputs[:i, num_enroll_sessions:],
                start_dim=0,
                end_dim=1
            ),
            torch.flatten(
                onnx_quantised_outputs[i + 1:, num_enroll_sessions:],
                start_dim=0,
                end_dim=1
            ),
        ],
        dim=0
    )

    scores = Metric.cal_session_distance_fixed_sessions(
        all_ver_embeddings,
        onnx_quantised_outputs[i, :num_enroll_sessions]
    )

    periods = get_periods(i, num_enroll_sessions, num_verify_sessions)
    labels = torch.tensor(
        [1] * num_verify_sessions +
        [0] * (onnx_quantised_outputs.shape[0] - 1) * num_verify_sessions
    )

    acc, threshold = Metric.eer_compute(
        scores[:num_verify_sessions],
        scores[num_verify_sessions:]
    )

    usability = Metric.calculate_usability(scores, threshold, periods, labels)
    tcr = Metric.calculate_TCR(scores, threshold, periods, labels)
    frwi = Metric.calculate_FRWI(scores, threshold, periods, labels)
    fawi = Metric.calculate_FAWI(scores, threshold, periods, labels)

    onnx_quantised_acc.append(acc)
    onnx_quantised_usability.append(usability)
    onnx_quantised_tcr.append(tcr)
    onnx_quantised_fawi.append(fawi)
    onnx_quantised_frwi.append(frwi)

print(
    "ONNX quantised\n"
    f"EER: {100 - np.mean(onnx_quantised_acc, axis=0)} "
    f"Usability: {np.mean(onnx_quantised_usability, axis=0)} "
    f"TCR: {np.mean(onnx_quantised_tcr, axis=0)} "
    f"FRWI: {np.mean(onnx_quantised_frwi, axis=0)} "
    f"FAWI: {np.mean(onnx_quantised_fawi, axis=0)}"
)