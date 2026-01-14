import torch
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import shape_inference, quantize_dynamic, QuantType, QuantFormat, quantize_static, CalibrationDataReader
from model.behaveformer import BehaveFormer
from model.dataset import HUMITestDataset
import os
from torch.utils.data import DataLoader
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

model_fp32_path = "/home/i/ibnu2651/BehaveFormer/work_dirs/humi_scroll50down_imu100all_epoch500_enroll3_b128/20231026_155303/best_models/model_fp32.onnx"
model_preprocess_path = "/home/i/ibnu2651/BehaveFormer/work_dirs/humi_scroll50down_imu100all_epoch500_enroll3_b128/20231026_155303/best_models/model_preprocess.onnx"
model_int8_path = "/home/i/ibnu2651/BehaveFormer/work_dirs/humi_scroll50down_imu100all_epoch500_enroll3_b128/20231026_155303/best_models/model_int8_static.onnx"

# Preprocess
# shape_inference.quant_pre_process(input_model=model_fp32_path, output_model_path=model_int8_path, skip_symbolic_shape=False)

# Quantise
calibration_dataset = HUMITestDataset(action='down', 
                                validation_file=os.path.join('/home/i/ibnu2651/BehaveFormer/Humidb/scroll50downup_imu100all', 'validation_scroll_imu_data_all.pickle'),
                                imu_type='acc_gyr_mag')

class QuantisedDataReader(CalibrationDataReader):
    def __init__(self, dataset, batch_size, touch_input_name, imu_input_name):
        self.dataloader = DataLoader(dataset, batch_size)

        self.touch_input_name = touch_input_name
        self.imu_input_name = imu_input_name
        self.datasize = len(self.dataloader)

        self.iterator = iter(self.dataloader)

    def to_numpy(self, tensor):
        return tensor.detach().cpu().numpy().astype(np.float32) if tensor.requires_grad else tensor.cpu().numpy().astype(np.float32)

    def get_next(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            return None
        
        return {
            self.touch_input_name: self.to_numpy(batch[0]),
            self.imu_input_name: self.to_numpy(batch[1])}

datareader = QuantisedDataReader(calibration_dataset, batch_size=32, touch_input_name="behave_inputs", imu_input_name="imu_inputs")
q_static_opts = {
    "ActivationSymmetric": False,
    "WeightSymmetric": True
}
# quantised_model = quantize_static(model_fp32_path, model_int8_path, calibration_data_reader=datareader, extra_options=q_static_opts)

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
        # print(behave_inputs.shape, imu_inputs.shape)

        # move to device
        behave_inputs = behave_inputs.to(device).float()
        imu_inputs = imu_inputs.to(device).float()

        # forward pass
        outputs = model((behave_inputs, imu_inputs))

        pt_outputs.append(outputs.cpu())

        # onnx
        behave_np = behave_inputs.numpy().astype(np.float32)
        imu_np = imu_inputs.numpy().astype(np.float32)

        behave_np.tofile("behave_input_tensor.bin")
        imu_np.tofile("imu_input_tensor.bin")
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
pt_usability = []
pt_tcr = []
pt_fawi = []
pt_frwi = []
onnx_quantised_acc = []
onnx_quantised_usability = []
onnx_quantised_tcr = []
onnx_quantised_fawi = []
onnx_quantised_frwi = []

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
    

    all_ver_embeddings = torch.cat([onnx_quantised_outputs[i,num_enroll_sessions:], torch.flatten(onnx_quantised_outputs[:i,num_enroll_sessions:], start_dim=0, end_dim=1), torch.flatten(onnx_quantised_outputs[i+1:,num_enroll_sessions:], start_dim=0, end_dim=1)], dim=0)
    scores = Metric.cal_session_distance_fixed_sessions(all_ver_embeddings, onnx_quantised_outputs[i,:num_enroll_sessions])

    acc, threshold = Metric.eer_compute(scores[:num_verify_sessions], scores[num_verify_sessions:])

    usability = Metric.calculate_usability(scores, threshold, periods, labels)
    tcr = Metric.calculate_TCR(scores, threshold, periods, labels)
    frwi = Metric.calculate_FRWI(scores, threshold, periods, labels)
    fawi = Metric.calculate_FAWI(scores, threshold, periods, labels)
    onnx_quantised_acc.append(acc)
    onnx_quantised_usability.append(usability)
    onnx_quantised_tcr.append(tcr)
    onnx_quantised_fawi.append(fawi)
    onnx_quantised_frwi.append(frwi)


print("Pytorch\nEER:", 100 - np.mean(pt_acc, axis=0), "Usability:", np.mean(pt_usability, axis=0), "TCR:", np.mean(pt_tcr, axis=0), "FRWI:", np.mean(pt_frwi, axis=0) , "FAWI:", np.mean(pt_fawi, axis=0))
# print(pt_acc)
print("ONNX quantised\nEER:", 100 - np.mean(onnx_quantised_acc, axis=0), "Usability:", np.mean(onnx_quantised_usability, axis=0), "TCR:", np.mean(onnx_quantised_tcr, axis=0), "FRWI:", np.mean(onnx_quantised_frwi, axis=0) , "FAWI:", np.mean(onnx_quantised_fawi, axis=0))
# print(onnx_quantised_acc)


# outs = pt_outputs[:5]
# outs.numpy().astype(np.int8).tofile("output_tensor.bin")
# print(onnx_outputs[-1])

# onnx_inputs = [tensor.numpy(force=True) for tensor in test_input]


# onnxruntime_input = {input_arg.name: input_value for input_arg, input_value in zip(ort_session.get_inputs(), onnx_inputs)}

# # ONNX Runtime returns a list of outputs
# onnxruntime_outputs = ort_session.run(None, onnxruntime_input)[0]