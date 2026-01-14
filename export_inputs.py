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

test_dataset = HUMITestDataset(action='down', 
                                validation_file=os.path.join('/home/i/ibnu2651/BehaveFormer/Humidb/scroll50downup_imu100all', 'testing_scroll_imu_data_all.pickle'),
                                imu_type='acc_gyr_mag')
test_dataloader = DataLoader(test_dataset, batch_size=25)

# pt_outputs = []
# onnx_outputs = []
all_touch = []
all_imu = []

with torch.no_grad():
    for batch in test_dataloader:
        touch_inputs, imu_inputs = batch  # adjust if your dataset returns differently

        touch_np = touch_inputs.detach().cpu().numpy().astype(np.float32)
        imu_np = imu_inputs.detach().cpu().numpy().astype(np.float32)

        all_touch.append(touch_np)
        all_imu.append(imu_np)

touch_all = np.concatenate(all_touch, axis=0)
imu_all = np.concatenate(all_imu, axis=0)

touch_all.tofile("touch_input_tensor.bin")
imu_all.tofile("imu_input_tensor.bin")
with open("input_tensors.json", "w") as f:
    json.dump({"touch_shape": list(touch_all.shape), "imu_shape": list(imu_all.shape), "dtype": "float32"}, f)
