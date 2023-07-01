from typing import Dict, Tuple
from time import time
import torch
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as f
import itertools
import math
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from layers import MLP, EraseAddGate, MLPEncoder, MLPDecoder, ScaledDotProductAttention
from utils import gumbel_softmax
from ShGKT import GKT
# 模型初始化
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


concept_num = 10
hidden_dim = 32
embedding_dim = 64
edge_type_num = 2
graph_type = "Dense"
graph = np.zeros((concept_num, concept_num))
graph_model = None
dropout = 0.5
bias = True
binary = False
has_cuda = False
model = GKT(concept_num, hidden_dim, embedding_dim, edge_type_num, graph_type, graph, graph_model, dropout, bias, binary, has_cuda)

# 模拟输入数据
batch_size = 5
seq_len = 20
features = torch.randint(0, 2, (batch_size, seq_len))  # 二值化输入
questions = torch.randint(-1, concept_num, (batch_size, seq_len))  # 随机问题

# 模型前向运行

# 打印中间变量的形状和值
# 注意: 以下只是示例，并未覆盖模型的所有变量和函数，您需要根据需要自行添加打印语句

# 打印 _aggregate 函数中的变量
print("In _aggregate function:")
tmp_ht = model._aggregate(features[:, 0], questions[:, 0], torch.zeros((batch_size, model.concept_num, model.hidden_dim)), batch_size)
print("tmp_ht shape: ", tmp_ht.shape)
print("tmp_ht value: ", tmp_ht)
# 打印 _agg_neighbors 函数中的变量
print("In _agg_neighbors function:")
m_next  = model._agg_neighbors(tmp_ht, questions[:,0])
for i, item in enumerate(m_next):
    print("m_next[{i}] value: ", item)

# 打印 _update 函数中的变量
print("In _update function:")
h_next = model._update(tmp_ht, torch.zeros((batch_size, model.concept_num, model.hidden_dim)), questions[:, 0])
print("h_next shape: ", h_next.shape)
print("h_next value: ", h_next)

# 打印 _predict 函数中的变量
print("In _predict function:")
yt = model._predict(h_next, questions[:, 0])
print("yt shape: ", yt.shape)
print("yt value: ", yt)

# 打印 _get_next_pred 函数中的变量
print("In _get_next_pred function:")
pred = model._get_next_pred(yt, questions[:, 1])

print("pred shape: ", pred.shape)
print("pred value: ", pred)

log_probs = torch.log(yt + torch.finfo(torch.float32).resolution)
discrete_probs = torch.eye(pred.shape[1])[torch.argmax(pred, dim=-1)]
vals = torch.sum(discrete_probs * log_probs, dim=0)
print("vals_shape:",vals.shape)
print("vals:",vals)
# 这意味着对应的知识点在预测结果中具有负面的重要性，即对于正确预测结果而言，该知识点的存在可能会对预测结果产生负面影响。
# 这种情况可能发生在模型对于某些知识点的预测结果较差，导致其对整体预测结果产生负面影响。负值的重要性分数可能需要进一步分析，以确定该知识点在模型中的影响。
adj_tensor = model._adj(questions[:, 0])
print("adj_tensor_shape",adj_tensor.shape)
print("adj_tensor:",adj_tensor)


positions_dict, key_to_idx, positions, coefficients, unique_inverse, unique_positions = construct_positions_dict(adj_tensor, max_order=4)
# key_to_val = {key: torch.tensor([vals[unique_inverse[idx]] for idx in key_to_idx[key]], dtype=torch.float64) for key in key_to_idx}
# key_to_val = {key: torch.tensor(vals[unique_inverse[key_to_idx[key]]], dtype=torch.float64) for key in key_to_idx}
key_to_val = {key: vals[unique_inverse[key_to_idx[key]]].clone().detach().to(torch.float64) for key in key_to_idx}

# Compute importance scores.
phis = torch.zeros(concept_num, dtype=torch.float64)
for i in range(concept_num):
    phis[i] = torch.sum(torch.stack([(coefficients[j] * (key_to_val[(i, j, 1)] - key_to_val[(i, j, 0)])) for j in coefficients]))

# Compute importance scores.

print("phis_shape:",phis.shape)
print("phis:",phis)
adj_tensor = adj_tensor.to(torch.float64)
result = torch.squeeze(torch.matmul(adj_tensor, phis.unsqueeze(0)), dim=-1)

#result = torch.squeeze(torch.matmul(adj_tensor.unsqueeze(-1), phis.unsqueeze(0)), dim=-1)

print("result_shape",result.shape)
print("result",result)
reshape = result = torch.sum(result, dim=2)
print("reshape",reshape.shape)
print("result",reshape)