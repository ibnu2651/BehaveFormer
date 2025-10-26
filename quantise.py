import torch
import numpy as np
import onnx
import onnxruntime as ort
from model.behaveformer import BehaveFormer
from model.dataset import HUMITestDataset
import os
from torch.utils.data import DataLoader
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BehaveFormer(8, 36, 50, 100, 64, 20, 4, 10, 6, 10, "acc_gyr_mag")
model.load_state_dict(torch.load("/home/i/ibnu2651/BehaveFormer/work_dirs/humi_scroll50down_imu100all_epoch500_enroll3_b128/20231026_155303/best_models/epoch_210_eer_2.60817307692308.pt", map_location=torch.device("cpu"), weights_only=True))
model.to(device).eval()

onnx_path = "/home/i/ibnu2651/BehaveFormer/work_dirs/humi_scroll50down_imu100all_epoch500_enroll3_b128/20231026_155303/best_models/model.onnx"
ort_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])


dummy_input = ([torch.rand(64, 50, 8), torch.rand(64, 100, 36)],)

# onnx_model = torch.onnx.export(
#     model,
#     dummy_input,
#     onnx_path,
#     input_names=["behave_inputs", "imu_inputs"],
#     output_names=["output"],
#     dynamic_axes={
#         "behave_inputs": {0: "batch"},
#         "imu_inputs":    {0: "batch"},
#         "output":        {0: "batch"},
#     },
# )


# onnx_model_test = onnx.load(onnx_path)
# onnx.checker.check_model(onnx_model_test)

test_dataset = HUMITestDataset(action='down', 
                                validation_file=os.path.join('/home/i/ibnu2651/BehaveFormer/Humidb/scroll50downup_imu100all', 'testing_scroll_imu_data_all.pickle'),
                                imu_type='acc_gyr_mag')
test_dataloader = DataLoader(test_dataset, batch_size=5)


pt_outputs = []
onnx_outputs = []

with torch.no_grad():
    for batch in test_dataloader:
        behave_inputs, imu_inputs = batch  # adjust if your dataset returns differently

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

torch.set_printoptions(precision=8)
print("PyTorch outputs:", pt_outputs.shape)
print("ONNX outputs:", onnx_outputs.shape)

outs = pt_outputs[:5]
outs.numpy().astype(np.float32).tofile("output_tensor.bin")
# print(onnx_outputs[-1])

# onnx_inputs = [tensor.numpy(force=True) for tensor in test_input]


# onnxruntime_input = {input_arg.name: input_value for input_arg, input_value in zip(ort_session.get_inputs(), onnx_inputs)}

# # ONNX Runtime returns a list of outputs
# onnxruntime_outputs = ort_session.run(None, onnxruntime_input)[0]