import torch
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import shape_inference, quantize_dynamic, QuantType, QuantFormat
from model.behaveformer import BehaveFormer
from model.dataset import HUMITestDataset
import os
from torch.utils.data import DataLoader
import json
from evaluation.metrics import Metric

model_fp32_path = "/home/i/ibnu2651/BehaveFormer/work_dirs/humi_scroll50down_imu100all_epoch500_enroll3_b128/20231026_155303/best_models/model_fp32.onnx"
model_preprocess_path = "/home/i/ibnu2651/BehaveFormer/work_dirs/humi_scroll50down_imu100all_epoch500_enroll3_b128/20231026_155303/best_models/model_preprocess.onnx"
model_int8_path = "/home/i/ibnu2651/BehaveFormer/work_dirs/humi_scroll50down_imu100all_epoch500_enroll3_b128/20231026_155303/best_models/model_int8.onnx"

# Preprocess
# shape_inference.quant_pre_process(input_model=model_fp32_path, output_model_path=model_int8_path, skip_symbolic_shape=False)

# Quantise
# quantised_model = quantize_dynamic(model_fp32_path, model_int8_path, weight_type=QuantType.QUInt8)

# Test

# Setup pytorhc model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BehaveFormer(8, 36, 50, 100, 64, 20, 4, 10, 6, 10, "acc_gyr_mag")
model.load_state_dict(torch.load("/home/i/ibnu2651/BehaveFormer/work_dirs/humi_scroll50down_imu100all_epoch500_enroll3_b128/20231026_155303/best_models/epoch_210_eer_2.60817307692308.pt", map_location=torch.device("cpu"), weights_only=True))
model.to(device).eval()
# onnx_model_test = onnx.load(model_fp32_path)
# print({n.op_type for n in onnx_model_test.graph.node})

# onnx_model_test = onnx.load(model_int8_path)
# print({n.op_type for n in onnx_model_test.graph.node})

ort_session = ort.InferenceSession(model_int8_path, providers=["CPUExecutionProvider"])

test_dataset = HUMITestDataset(action='down', 
                                validation_file=os.path.join('/home/i/ibnu2651/BehaveFormer/Humidb/scroll50downup_imu100all', 'testing_scroll_imu_data_all.pickle'),
                                imu_type='acc_gyr_mag')
test_dataloader = DataLoader(test_dataset, batch_size=16)


pt_outputs = []
onnx_outputs = []

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

        # onnx
        behave_np = behave_inputs.numpy().astype(np.float32)
        imu_np = imu_inputs.numpy().astype(np.float32)

        # behave_np.tofile("behave_input_tensor.bin")
        # imu_np.tofile("imu_input_tensor.bin")
        # with open("input_tensors.json", "w") as f:
        #     json.dump({"behave_shape": list(behave_np.shape), "imu_shape": list(imu_np.shape), "dtype": "float32"}, f)

        # exit()

        ort_inputs = {
            ort_session.get_inputs()[0].name: behave_np,
            ort_session.get_inputs()[1].name: imu_np,
        }
        out_onnx = ort_session.run(None, ort_inputs)[0]
        onnx_outputs.append(out_onnx)

        # print(outputs.cpu())


pt_outputs = torch.cat(pt_outputs, dim=0)
onnx_outputs = np.concatenate(onnx_outputs, axis=0)
onnx_outputs = torch.from_numpy(onnx_outputs)

torch.set_printoptions(precision=8)
print("PyTorch outputs:", pt_outputs.shape)
print(pt_outputs)

print("ONNX outputs:", onnx_outputs.shape)
print(onnx_outputs)

# Evaluate
pt_outputs = pt_outputs.view(test_dataset.num_users, test_dataset.num_sessions, test_dataset.num_seqs, 64)
onnx_quantised_outputs = onnx_outputs.view(test_dataset.num_users, test_dataset.num_sessions, test_dataset.num_seqs, 64)

num_enroll_sessions = 3
num_verify_sessions = 2

pt_acc = []
onnx_quantised_acc = []

for i in range(pt_outputs.shape[0]):
    all_ver_embeddings = torch.cat([pt_outputs[i,num_enroll_sessions:], torch.flatten(pt_outputs[:i,num_enroll_sessions:], start_dim=0, end_dim=1), torch.flatten(pt_outputs[i+1:,num_enroll_sessions:], start_dim=0, end_dim=1)], dim=0)
    scores = Metric.cal_session_distance_fixed_sessions(all_ver_embeddings, pt_outputs[i,:num_enroll_sessions])

    acc, threshold = Metric.eer_compute(scores[:num_verify_sessions], scores[num_verify_sessions:])

    pt_acc.append(acc)

    all_ver_embeddings = torch.cat([onnx_quantised_outputs[i,num_enroll_sessions:], torch.flatten(onnx_quantised_outputs[:i,num_enroll_sessions:], start_dim=0, end_dim=1), torch.flatten(onnx_quantised_outputs[i+1:,num_enroll_sessions:], start_dim=0, end_dim=1)], dim=0)
    scores = Metric.cal_session_distance_fixed_sessions(all_ver_embeddings, onnx_quantised_outputs[i,:num_enroll_sessions])

    acc, threshold = Metric.eer_compute(scores[:num_verify_sessions], scores[num_verify_sessions:])

    onnx_quantised_acc.append(acc)


print("Pytorch acc: ", 100 - np.mean(pt_acc, axis=0))
print(pt_acc)
print("ONNX quantised acc: ", 100 - np.mean(onnx_quantised_acc, axis=0))
print(onnx_quantised_acc)


# outs = pt_outputs[:5]
# outs.numpy().astype(np.int8).tofile("output_tensor.bin")
# print(onnx_outputs[-1])

# onnx_inputs = [tensor.numpy(force=True) for tensor in test_input]


# onnxruntime_input = {input_arg.name: input_value for input_arg, input_value in zip(ort_session.get_inputs(), onnx_inputs)}

# # ONNX Runtime returns a list of outputs
# onnxruntime_outputs = ort_session.run(None, onnxruntime_input)[0]