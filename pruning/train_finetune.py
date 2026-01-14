import sys
sys.path.append("..")

import os
import torch
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader
from model.behaveformer import BehaveFormer
from model.loss import TripletLoss
from model.dataset import HUMITrainDataset, HUMITestDataset
from evaluation.metrics import Metric
from utils.utils import read_pickle

def get_masks(model):
    """Creates a binary mask for all pruned layers (1 for non-zero, 0 for zero)."""
    masks = {}
    device = next(model.parameters()).device
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            # Weights that are NOT zero are the ones we want to keep training
            masks[name] = (module.weight != 0).float().to(device)
    return masks

def evaluate(model, test_dataset, test_dataloader, target_len, number_of_enrollment_sessions, number_of_verify_sessions, imu_type, device, dataname):
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

print("Start loading datasets\n")

splits = read_pickle(os.path.join("/home/i/ibnu2651/BehaveFormer/Humidb2/scroll50downup_imu100all", 'splits.pickle'))

print("Finished pickling\n")
imu_type = "acc_gyr_mag"
train_dataset = HUMITrainDataset(batch_size=128,
                                    epoch_batch_count=100,
                                    action="down",
                                    training_file='/home/i/ibnu2651/BehaveFormer/Humidb2/scroll50downup_imu100all/training_scroll_imu_data_all.pickle',
                                    user_list=splits['training'],
                                    imu_type=imu_type)

val_dataset = HUMITestDataset(action="down", 
                                validation_file='/home/i/ibnu2651/BehaveFormer/Humidb2/scroll50downup_imu100all/validation_scroll_imu_data_all.pickle',
                                imu_type=imu_type)

print("Loaded datasets\n")

# Create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=128)
val_dataloader = DataLoader(val_dataset, batch_size=128)

print("Loaded dataloaders\n")

pruned_weights_path = "prune_unstructured.pt"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 1. Initialize Model and Load Pruned Weights
# (Use the same hyperparams from your config)
model = BehaveFormer(8, 36, 50, 100, 64, 20, 4, 10, 6, 10, "acc_gyr_mag") 
model.load_state_dict(torch.load(pruned_weights_path))
model.to(device)

# Force any "hidden" tensors in the positional encoding to the correct device
for m in model.modules():
    if hasattr(m, 'positions'):
        m.positions = m.positions.to(device)
    if hasattr(m, 'mu'):
        m.mu = m.mu.to(device)
    if hasattr(m, 'sigma'):
        m.sigma = m.sigma.to(device)

print("Initialised model\n")

# 2. Extract Masks
# This is vital: it tells us which weights were pruned
masks = get_masks(model)

print("Extracted masks\n")

# 3. Setup Optimizer (Use a lower learning rate for fine-tuning)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.05)
loss_fn = TripletLoss()

epochs = 5  # Fine-tuning usually requires very few epochs

for epoch in range(epochs):
    model.train()
    t_loss = 0.0
    
    for batch_idx, (anchor, positive, negative) in enumerate(train_dataloader):
        optimizer.zero_grad()

        # Forward pass (Handling your specific BehaveFormer tuple input)
        if imu_type != 'none':
            anchor_out = model([anchor[0].to(device).float(), anchor[1].to(device).float()])
            positive_out = model([positive[0].to(device).float(), positive[1].to(device).float()])
            negative_out = model([negative[0].to(device).float(), negative[1].to(device).float()])
        else:
            anchor_out = model(anchor[0].to(device).float())
            positive_out = model(positive[0].to(device).float())
            negative_out = model(negative[0].to(device).float())

        loss = loss_fn(anchor_out, positive_out, negative_out)
        loss.backward()

        with torch.no_grad():
            for name, module in model.named_modules():
                if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                    if module.weight.grad is not None:
                        # Create mask on the fly based on the current weight's device
                        # This is slightly slower but 100% prevents device errors
                        mask = (module.weight != 0).float()
                        module.weight.grad.mul_(mask)
        optimizer.step()
        t_loss += loss.item()

    # Evaluation (using your evaluate function logic)
    eer = evaluate(model, val_dataset, val_dataloader, 64, 3, 2, "acc_gyr_mag", device, 'humi') 
    print(f"Epoch {epoch+1} | Loss: {t_loss/len(train_dataloader):.6f} | EER: {eer:.4f}")

# Save finalized fine-tuned model
torch.save(model.state_dict(), "prune_unstructured_finetuned.pt")