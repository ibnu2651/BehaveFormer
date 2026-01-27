import sys
sys.path.append("..")

import os
import torch
from torch import nn
from torch.utils.data import DataLoader

from model.behaveformer import BehaveFormer
from model.loss import TripletLoss
from model.dataset import HUMITrainDataset, HUMITestDataset
from evaluation.metrics import Metric
from utils.utils import read_pickle


def move_positional_encoding_tensors_to_device(model, device):
    """
    Your PositionalEncoding stores tensors like `positions` that are not registered buffers,
    so we move them manually to avoid device mismatch errors.
    """
    for m in model.modules():
        if hasattr(m, "positions"):
            m.positions = m.positions.to(device)
        if hasattr(m, "mu"):
            m.mu = m.mu.to(device)
        if hasattr(m, "sigma"):
            m.sigma = m.sigma.to(device)


# @torch.no_grad()
# def evaluate(model, test_dataset, test_dataloader,
#              target_len, number_of_enrollment_sessions, number_of_verify_sessions,
#              imu_type, device):
#     model.eval()

#     feature_embeddings = []
#     for item in test_dataloader:
#         if imu_type != "none":
#             out = model([item[0].to(device).float(), item[1].to(device).float()])
#         else:
#             out = model(item[0].to(device).float())
#         feature_embeddings.append(out)

#     feats = torch.cat(feature_embeddings, dim=0)
#     feats = feats.view(test_dataset.num_users, test_dataset.num_sessions, test_dataset.num_seqs, target_len)

#     eer = Metric.cal_user_eer_fixed_sessions(
#         feats, number_of_enrollment_sessions, number_of_verify_sessions
#     )[0]
#     return eer

def evaluate(model, test_dataset, test_dataloader, target_len, number_of_enrollment_sessions, number_of_verify_sessions, imu_type, device):
    model.train(False)

    with torch.no_grad():
        feature_embeddings = []
        for batch_idx, item in enumerate(test_dataloader):
            if imu_type != 'none':
                feature_embeddings.append(model([item[0].to(device).float(), item[1].to(device).float()]))
            else:
                feature_embeddings.append(model(item[0].to(device).float()))
    
    eer = Metric.cal_user_eer_fixed_sessions(torch.cat(feature_embeddings, dim=0).view(test_dataset.num_users, test_dataset.num_sessions, test_dataset.num_seqs, target_len), number_of_enrollment_sessions, number_of_verify_sessions)[0]
       
    return eer


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


def main():
    imu_type = "acc_gyr_mag"

    new_imu_hidden = 1200
    new_behave_hidden = 160
    new_num_layers_behave = 3
    new_num_layers_imu = 3

    target_len = 64
    enroll_sessions = 3
    verify_sessions = 2

    # Files
    base_dir = "/home/i/ibnu2651/BehaveFormer/Humidb2/scroll50downup_imu100all"
    # val_base_dir = "/home/i/ibnu2651/BehaveFormer/Humidb/scroll50downup_imu100all"
    ckpt_structured_pruned = "prune_structured_state_dict.pt"
    out_best = "prune_structured_finetuned_best.pt"
    out_last = "prune_structured_finetuned_last.pt"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load datasets
    print("Loading datasets...")
    splits = read_pickle(os.path.join(base_dir, "splits.pickle"))

    train_dataset = HUMITrainDataset(
        batch_size=128,
        epoch_batch_count=100,
        action="down",
        training_file=os.path.join(base_dir, "training_scroll_imu_data_all.pickle"),
        user_list=splits["training"],
        imu_type=imu_type,
    )

    val_dataset = HUMITestDataset(
        action="down",
        validation_file=os.path.join(base_dir, "validation_scroll_imu_data_all.pickle"),
        imu_type=imu_type,
    )

    train_dataloader = DataLoader(train_dataset, batch_size=128)
    val_dataloader = DataLoader(val_dataset, batch_size=128)

    print("Datasets + dataloaders ready.\n")


    # Build a base model then apply the SAME structured pruning transforms
    # (so the architecture matches the pruned state_dict)
    print("Building pruned architecture...")

    model = BehaveFormer(8, 36, 50, 100, 64, 20, 4, 10, 6, 10, imu_type)
    model.eval()

    # Apply same structural changes as the pruning plan
    prune_two_linear_mlp(model.linear_imu, new_hidden=new_imu_hidden, importance_from="fc2_l1")
    prune_two_linear_mlp(model.linear_behave, new_hidden=new_behave_hidden, importance_from="fc2_l1")

    keep_first_n_encoder_layers(model.behave_transformer, new_num_layers_behave)
    keep_first_n_encoder_layers(model.imu_transformer, new_num_layers_imu)

    # Now load weights from the already-pruned checkpoint
    pruned_sd = torch.load(ckpt_structured_pruned, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(pruned_sd, strict=False)

    print(f"Loaded pruned weights. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

    model.to(device)
    move_positional_encoding_tensors_to_device(model, device)

    # Sanity forward
    with torch.no_grad():
        y = model([torch.rand(1, 50, 8).to(device), torch.rand(1, 100, 36).to(device)])
    print(f"Sanity output shape: {tuple(y.shape)} (expected (1, {target_len}))\n")

    # Finetune
    loss_fn = TripletLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    epochs = 10

    best_eer = float("inf")

    for epoch in range(epochs):
        model.train()
        t_loss = 0.0

        for batch_idx, (anchor, positive, negative) in enumerate(train_dataloader):
            optimizer.zero_grad()

            # Forward (tuple inputs)
            anchor_out = model([anchor[0].to(device).float(), anchor[1].to(device).float()])
            positive_out = model([positive[0].to(device).float(), positive[1].to(device).float()])
            negative_out = model([negative[0].to(device).float(), negative[1].to(device).float()])

            loss = loss_fn(anchor_out, positive_out, negative_out)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()

        eer = evaluate(
            model, val_dataset, val_dataloader,
            target_len, enroll_sessions, verify_sessions,
            imu_type, device
        )

        avg_loss = t_loss / max(1, len(train_dataloader))
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f} | Val EER: {eer:.4f}")

        if eer < best_eer:
            best_eer = eer
            torch.save(model.state_dict(), out_best)

    torch.save(model.state_dict(), out_last)
    print(f"\nDone. Best Val EER: {best_eer:.4f}")
    print(f"Saved best: {out_best}")
    print(f"Saved last: {out_last}")


if __name__ == "__main__":
    main()