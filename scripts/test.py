import torch

# Define some sample data
data = [1,2,3,4,5]

# Create a tensor from the data
tensor = torch.tensor(data)

# Print the tensor
print(tensor)

# Get some information about the tensor
print(f"Tensor shape: {tensor.shape}")
print(f"Tensor data type: {tensor.dtype}")