import sys
sys.path.append("..")

import torch
import torch.nn.utils.prune as prune
from model.behaveformer import BehaveFormer

model = BehaveFormer(8, 36, 50, 100, 64, 20, 4, 10, 6, 10, "acc_gyr_mag")
model.load_state_dict(torch.load("/home/i/ibnu2651/BehaveFormer/work_dirs/humi_scroll50down_imu100all_epoch500_enroll3_b128/20231026_155303/best_models/epoch_210_eer_2.60817307692308.pt", map_location=torch.device("cpu"), weights_only=True))

params_to_prune = []
for name, module in model.named_modules():
    if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
        params_to_prune.append((module, 'weight'))

prune.global_unstructured(params_to_prune, pruning_method=prune.L1Unstructured, amount=0.20)

def check_sparsity(model):
    total_zeros = 0
    total_elements = 0
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            zeros = float(torch.sum(module.weight == 0))
            elements = float(module.weight.nelement())
            total_zeros += zeros
            total_elements += elements
            print(f"Sparsity in {name}: {100. * zeros / elements:.2f}%")
    
    print(f"\nGlobal Sparsity: {100. * total_zeros / total_elements:.2f}%")

check_sparsity(model)

for module, name in params_to_prune:
    prune.remove(module, name)

print("\nPruning finalized. Model is ready for export.")

output_path = "prune_unstructured.pt"
torch.save(model.state_dict(), output_path)

print("\nModel saved.")

