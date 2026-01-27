import sys
sys.path.append("..")

import torch
from torch import nn
from model.behaveformer import BehaveFormer

import onnxruntime as ort
import numpy as np

# ===== rebuild PRUNED architecture =====
imu_type = "acc_gyr_mag"

model = BehaveFormer(8, 36, 50, 100, 64, 20, 4, 10, 6, 10, imu_type)

@torch.no_grad()
def prune_two_linear_mlp(seq: nn.Sequential, new_hidden: int, importance_from: str = "fc2_l1"):
    """
    Structured prune of the hidden dimension for:
      [Linear(in, hidden), ReLU, Dropout, Linear(hidden, out), ReLU]
    """
    assert isinstance(seq, nn.Sequential)
    assert isinstance(seq[0], nn.Linear) and isinstance(seq[3], nn.Linear)

    fc1: nn.Linear = seq[0]
    fc2: nn.Linear = seq[3]
    old_hidden = fc1.out_features
    if not (0 < new_hidden < old_hidden):
        raise ValueError(f"new_hidden must be in (0, {old_hidden}), got {new_hidden}")

    if importance_from == "fc2_l1":
        importance = fc2.weight.abs().sum(dim=0)  # [hidden]
    elif importance_from == "fc1_l1":
        importance = fc1.weight.abs().sum(dim=1)  # [hidden]
    else:
        raise ValueError("importance_from must be 'fc2_l1' or 'fc1_l1'")

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
    return keep


def keep_first_n_encoder_layers(transformer_module, n: int):
    transformer_module.encoder.layers = nn.ModuleList(list(transformer_module.encoder.layers[:n]))


prune_two_linear_mlp(model.linear_imu, new_hidden=1200)
prune_two_linear_mlp(model.linear_behave, new_hidden=160)
keep_first_n_encoder_layers(model.behave_transformer, 3)
keep_first_n_encoder_layers(model.imu_transformer, 3)

# load fine-tuned weights
state = torch.load("prune_structured_finetuned_last.pt", map_location="cpu")
model.load_state_dict(state, strict=True)

model.eval()

# ===== dummy inputs =====
dummy_scroll = torch.randn(1, 50, 8)
dummy_imu = torch.randn(1, 100, 36)
dummy_input = ([dummy_scroll, dummy_imu])

# ===== export =====
torch.onnx.export(
    model,
    dummy_input,
    "prune_structured_finetuned_last.onnx",
    opset_version=17,
    input_names=["scroll_inputs", "imu_inputs"],
    output_names=["output"],
    dynamic_axes={
        "scroll_inputs": {0: "batch"},
        "imu_inputs": {0: "batch"},
        "output": {0: "batch"}
    }
)

print("ONNX export successful")


sess = ort.InferenceSession("prune_structured_finetuned_last.onnx")

out = sess.run(
    None,
    {
        "scroll_inputs": np.random.randn(1,50,8).astype(np.float32),
        "imu_inputs": np.random.randn(1,100,36).astype(np.float32)
    }
)

print(out[0].shape)