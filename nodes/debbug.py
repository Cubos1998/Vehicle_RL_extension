import torch
import torch.nn.functional as F

# Original tensor of shape [196, 196]
tensor = torch.rand(196, 196).unsqueeze(0)

# Add batch and channel dimensions: [1, 1, 196, 196]
print("Original tensor shape:", tensor.shape)

# Apply adaptive_avg_pool2d to downsample to [1, 1, 14, 14]
pooled_tensor = F.adaptive_avg_pool2d(tensor, (14, 14))

print("Shape of input tensor:", tensor.shape)
print("Shape of pooled tensor:", pooled_tensor.shape)
