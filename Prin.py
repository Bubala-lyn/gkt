import numpy as np
import torch
from  ShGKT import GKT
# 模型初始化
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

# 打印 cpd 函数中的变量
print("construct_positions_dict:")
adj_tensor = model._adj(questions[:, 0])
positions_dict, key_to_idx, positions, coefficients, unique_inverse, unique_positions = model.construct_positions_dict(adj_tensor,max_order=4)

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

# 打印 cal_sha_adj函数中的变量
print("cal_sha_adj(:")
new_adj = model.cal_sha_adj(yt,pred, questions[:, 1])
print("new_adj shape: ", new_adj.shape)
print("new_adj value: ", new_adj)


# 打印 forward函数中的变量
print("forward")
pred_res = model.forward(features, questions)
print("pred_res shape: ", pred_res.shape)
print("pred_res value: ", pred_res)