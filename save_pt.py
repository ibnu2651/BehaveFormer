import torch
import onnx
import onnxruntime
from model.behaveformer import BehaveFormer
from model.dataset import HUMITestDataset
import os

model = BehaveFormer(8, 36, 50, 100, 64, 20, 4, 10, 6, 10, "acc_gyr_mag")
model.load_state_dict(torch.load("/home/i/ibnu2651/BehaveFormer/work_dirs/humi_scroll50down_imu100all_epoch500_enroll3_b128/20231026_155303/best_models/epoch_210_eer_2.60817307692308.pt",weights_only=True))
model.eval()

dummy_input = ([torch.rand(64, 50, 8), torch.rand(64, 100, 36)],)

# torch.save(model, '/home/i/ibnu2651/BehaveFormer/work_dirs/humi_scroll50down_imu100all_epoch500_enroll3_b128/20231026_155303/best_models/model.pt')

exported_model = torch.export.export(model, dummy_input)
torch.export.save(exported_model, '/home/i/ibnu2651/BehaveFormer/work_dirs/humi_scroll50down_imu100all_epoch500_enroll3_b128/20231026_155303/best_models/model.pt2')

