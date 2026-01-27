import sys
sys.path.append("..")

import torch
from torch import nn
from model.behaveformer import BehaveFormer


@torch.no_grad()
def prune_two_linear_mlp(seq: nn.Sequential, new_hidden: int, importance_from: str = "fc2_l1"):
    """
    Prunes the hidden dimension of an MLP shaped like:
      [Linear(in, hidden), ReLU, Dropout, Linear(hidden, out), ReLU]

    importance_from:
      - "fc2_l1": rank hidden units by L1 norm of fc2 columns (common + stable)
      - "fc1_l1": rank hidden units by L1 norm of fc1 rows
    """
    assert isinstance(seq, nn.Sequential), "Expected nn.Sequential"
    assert isinstance(seq[0], nn.Linear) and isinstance(seq[3], nn.Linear), "Unexpected MLP structure"

    fc1: nn.Linear = seq[0]
    fc2: nn.Linear = seq[3]

    hidden = fc1.out_features
    if not (0 < new_hidden < hidden):
        raise ValueError(f"new_hidden must be in (0, {hidden}), got {new_hidden}")

    # Importance per hidden neuron
    if importance_from == "fc2_l1":
        # fc2.weight: [out, hidden] -> importance per hidden is column norm
        importance = fc2.weight.abs().sum(dim=0)  # [hidden]
    elif importance_from == "fc1_l1":
        # fc1.weight: [hidden, in] -> importance per hidden is row norm
        importance = fc1.weight.abs().sum(dim=1)  # [hidden]
    else:
        raise ValueError("importance_from must be 'fc2_l1' or 'fc1_l1'")

    keep = torch.topk(importance, k=new_hidden, largest=True).indices.sort().values

    # Rebuild with smaller hidden dim
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


def keep_first_n_encoder_layers(transformer_module: nn.Module, n: int):
    """
    transformer_module should be your `Transformer` class instance:
      transformer_module.encoder.layers is a ModuleList
    """
    layers = transformer_module.encoder.layers
    if n < 1 or n > len(layers):
        raise ValueError(f"n must be in [1, {len(layers)}], got {n}")
    transformer_module.encoder.layers = nn.ModuleList(list(layers[:n]))


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def main():
    # ---- Load model (your config) ----
    model = BehaveFormer(
        behave_feature_dim=8,
        imu_feature_dim=36,
        behave_len=50,
        imu_len=100,
        target_len=64,
        gre_k=20,
        behave_temporal_heads=4,
        behave_channel_heads=10,
        imu_temporal_heads=6,
        imu_channel_heads=10,
        imu_type="acc_gyr_mag",
        # num_layer is default=5 in your class unless you changed it elsewhere
    )

    ckpt_path = "/home/i/ibnu2651/BehaveFormer/work_dirs/humi_scroll50down_imu100all_epoch500_enroll3_b128/20231026_155303/best_models/epoch_210_eer_2.60817307692308.pt"
    state = torch.load(ckpt_path, map_location=torch.device("cpu"), weights_only=True)
    model.load_state_dict(state)
    model.eval()

    params_before = count_params(model)
    print(f"Original parameter count: {params_before}")

    # ---- My pruning plan knobs (Balanced defaults) ----
    # linear_imu: in=3600, hidden=1800  -> try 1200 for balanced
    new_imu_hidden = 1200

    # linear_behave: in=400, hidden=200 -> try 160 for balanced
    new_behave_hidden = 160

    # layer dropping: 5 -> 3 for balanced
    new_num_layers_behave = 3
    new_num_layers_imu = 3

    # ---- Apply structured pruning (MLP hidden width) ----
    if model.imu_type != "none":
        prune_two_linear_mlp(model.linear_imu, new_hidden=new_imu_hidden, importance_from="fc2_l1")
        print(f"Pruned linear_imu hidden -> {new_imu_hidden}")

    prune_two_linear_mlp(model.linear_behave, new_hidden=new_behave_hidden, importance_from="fc2_l1")
    print(f"Pruned linear_behave hidden -> {new_behave_hidden}")

    # ---- Drop transformer layers ----
    keep_first_n_encoder_layers(model.behave_transformer, new_num_layers_behave)
    print(f"Dropped behave_transformer layers -> {new_num_layers_behave}")

    if model.imu_type != "none":
        keep_first_n_encoder_layers(model.imu_transformer, new_num_layers_imu)
        print(f"Dropped imu_transformer layers -> {new_num_layers_imu}")

    # ---- Sanity forward pass (shape check) ----
    with torch.no_grad():
        behave_x = torch.rand(1, 50, 8)
        if model.imu_type != "none":
            imu_x = torch.rand(1, 100, 36)
            y = model([behave_x, imu_x])
        else:
            y = model(behave_x)

    print(f"Output shape: {tuple(y.shape)} (expected: (1, 64))")

    params_after = count_params(model)
    print(f"Pruning complete!")
    print(f"Parameters: {params_before} -> {params_after} ({100*(1-params_after/params_before):.2f}% reduction)")

    # ---- Save ----
    output_path_sd = "prune_structured_state_dict.pt"
    output_path_full = "prune_structured_full_model.pt"

    torch.save(model.state_dict(), output_path_sd)
    torch.save(model, output_path_full)

    print(f"Saved state_dict to: {output_path_sd}")
    print(f"Saved full model to: {output_path_full}")


if __name__ == "__main__":
    main()