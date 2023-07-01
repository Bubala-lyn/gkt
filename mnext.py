import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import math
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from layers import MLP, EraseAddGate, MLPEncoder, MLPDecoder, ScaledDotProductAttention
from utils import gumbel_softmax
from SGKT import GKT
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

qt = torch.tensor([0, 1, -1, 3, -1])
tmp_ht = torch.randn(5, 10, 96)
self_graph = torch.randint(0, 2, (10, 10))


print(m_next)
print(m_next.shape)
