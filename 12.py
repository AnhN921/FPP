import torch
import os

# Tải tensor từ file
tensor = torch.load('/Users/ngocanh/Downloads/DynamicFL2/mymodel.pt')
print(tensor)
import torch

# Tải nội dung từ file .pt
content = torch.load('/Users/ngocanh/Downloads/DynamicFL2/mymodel.pt')
state_dict = {
    'state': {
        0: {'step': 1400, 'exp_avg': torch.tensor([0.1, 0.2]), 'exp_avg_sq': torch.tensor([0.01, 0.04])},
        1: {'step': 1400, 'exp_avg': torch.tensor([0.3, 0.4]), 'exp_avg_sq': torch.tensor([0.09, 0.16])},
        # Các giá trị khác
    },
    # Các phần khác của state_dict
}

for key, value in state_dict['state'].items():
    if isinstance(value, torch.Tensor):
        print(f"Layer: {key}, Weights: {value.size()}")
    elif isinstance(value, dict):
        print(f"Layer: {key}, Weights: { {k: v.size() for k, v in value.items() if isinstance(v, torch.Tensor)} }")
    else:
        print(f"Layer: {key}, Weights: {type(value)}")

