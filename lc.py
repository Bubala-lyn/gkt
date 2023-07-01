import torch
import numpy as np
from time import time
import torch
import torch.nn as nn


class YourModel(nn.Module):
    def __init__(self, hidden_dim, bias):
        super(YourModel, self).__init__()
        self.predict = nn.Linear(hidden_dim, concept_num, bias=bias)

    def forward(self, adj, qt):
        """
        Parameters:
            adj: adjacency tensor representing the connection between knowledge points (batch_size, concept_num, 1)
            qt: question indices for all students in a batch at the current timestamp (batch_size)
        Shape:
            adj: [batch_size, concept_num, 1]
            qt: [batch_size]
            y: [batch_size, concept_num]
        Return:
            y: predicted correct probability of all concepts at the next timestamp
        """
        qt_mask = torch.ne(qt, -1)  # [batch_size], qt != -1
        y = self.predict(adj.squeeze(dim=-1)).squeeze(dim=-1)  # [batch_size, concept_num]
        y[qt_mask] = torch.sigmoid(y[qt_mask])  # [batch_size, concept_num]
        return y




def construct_positions_dict(adj_tensor, max_order):
    mask_num, concept_num, _ = adj_tensor.size()
    subsets = {}
    coefficients = {j: 2.0 / (j * (j + 1) * (j + 2)) for j in range(1, max_order + 1)}

    for i in range(concept_num):
        neighbor_indices = torch.nonzero(adj_tensor[:, i]).squeeze(1).reshape(-1)
        subsets[i] = []
        for k in range(max_order):
            combinations = torch.combinations(neighbor_indices, k + 1)
            subsets[i].extend(combinations.tolist())

    positions_dict = {(i, l, fill): [] for i in range(concept_num) for l in range(1, max_order + 1) for fill in [0, 1]}
    key_to_idx = {}
    positions = []
    count = 0

    for i in subsets:
        for l in range(1, max_order + 1):
            for fill in [0, 1]:
                for subset in subsets[i]:
                    if len(subset) <= l:
                        positions_dict[(i, l, fill)].append(torch.tensor(subset))
                        pos = torch.zeros((concept_num, 1), dtype=torch.float64)
                        pos[subset, :] = fill
                        positions.append(pos)
                        key_to_idx[(i, l, fill)] = count
                        count += 1

    positions = torch.cat(positions, dim=0)

    positions_tensor = positions.clone().detach().to(torch.float64)
    unique_positions, unique_inverse = torch.unique(positions_tensor, dim=0, return_inverse=True)

    return positions_dict, key_to_idx, positions, coefficients, unique_inverse, unique_positions

# Example usage
hidden_dim = 10
concept_num = 10
batch_size = 5
model = YourModel(hidden_dim, concept_num)
adj_tensor = adj = torch.randn(batch_size, concept_num, 1)
qt = torch.randint(-1, concept_num, (batch_size,))
# Assuming adj_tensor and qt are tensors of appropriate shapes

y = model(adj_tensor, qt)
print("Predicted probabilities:", y.shape)
print("pred",y)



mask_num,concept_num,_ =adj_tensor.size()
max_order = 4
print("adj_tensor_shape",adj_tensor.shape)
print("adj_tensor:",adj_tensor)

positions_dict, key_to_idx, positions, coefficients, unique_inverse, unique_positions = construct_positions_dict(adj_tensor, max_order)

print("Positions Dictionary:")
for key, value in positions_dict.items():
    print(key, ":", value)

print("\nKey to Index Mapping:")
print(key_to_idx)

print("\nPositions Tensor:")
print(positions)
print("\nKey_to_val:",key_to_idx)
print("\nCoefficients:")
print(coefficients)
print("Unique Positions:")
print(unique_positions)
print("Unique Inverse:")
print(unique_inverse)

