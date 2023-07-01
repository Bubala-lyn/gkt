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





class GKT(nn.Module):

    def __init__(self, concept_num, hidden_dim, embedding_dim, edge_type_num, graph_type, graph=None, graph_model=None,
                 dropout=0.5, bias=True, binary=False, has_cuda=False):
        super(GKT, self).__init__()
        self.concept_num = concept_num
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.edge_type_num = edge_type_num
        self.res_len = 2 if binary else 12
        self.has_cuda = has_cuda
        self.graph_type = graph_type
        self.graph = torch.Tensor(concept_num, concept_num)  # [concept_num, concept_num]
        self.graph.requires_grad = False  # fix parameter
        self.graph_model = graph_model

        # one-hot feature and question
        one_hot_feat = torch.eye(self.res_len * self.concept_num)
        self.one_hot_feat = one_hot_feat.cuda() if self.has_cuda else one_hot_feat
        self.one_hot_q = torch.eye(self.concept_num, device=self.one_hot_feat.device)
        zero_padding = torch.zeros(1, self.concept_num, device=self.one_hot_feat.device)
        self.one_hot_q = torch.cat((self.one_hot_q, zero_padding), dim=0)
        # concept and concept & response embeddings
        self.emb_x = nn.Embedding(self.res_len * concept_num, embedding_dim)
        # last embedding is used for padding, so dim + 1
        self.emb_c = nn.Embedding(concept_num + 1, embedding_dim, padding_idx=-1)

        # f_self function and f_neighbor functions
        mlp_input_dim = hidden_dim + embedding_dim
        self.f_self = MLP(mlp_input_dim, hidden_dim, hidden_dim, dropout=dropout, bias=bias)
        self.f_neighbor_list = nn.ModuleList()
        # f_in and f_out functions
        self.f_neighbor_list.append(MLP(2 * mlp_input_dim, hidden_dim, hidden_dim, dropout=dropout, bias=bias))
        self.f_neighbor_list.append(MLP(2 * mlp_input_dim, hidden_dim, hidden_dim, dropout=dropout, bias=bias))
        # Erase & Add Gate
        self.erase_add_gate = EraseAddGate(hidden_dim, concept_num)
        # Gate Recurrent Unit
        self.gru = nn.GRUCell(hidden_dim, hidden_dim, bias=bias)
        # prediction layer
        self.predict = nn.Linear(hidden_dim, 1, bias=bias)

    # Aggregate step, as shown in Section 3.2.1 of the paper
    def _aggregate(self, xt, qt, ht, batch_size):
        r"""
        Parameters:
            xt: input one-hot question answering features at the current timestamp
            qt: question indices for all students in a batch at the current timestamp
            ht: hidden representations of all concepts at the current timestamp
            batch_size: the size of a student batch
        Shape:
            xt: [batch_size]
            qt: [batch_size]
            ht: [batch_size, concept_num, hidden_dim]
            tmp_ht: [batch_size, concept_num, hidden_dim + embedding_dim]
        Return:
            tmp_ht: aggregation results of concept hidden knowledge state and concept(& response) embedding
        """
        qt_mask = torch.ne(qt, -1)  # [batch_size], qt != -1
        x_idx_mat = torch.arange(self.res_len * self.concept_num, device=xt.device)
        x_embedding = self.emb_x(x_idx_mat)  # [res_len * concept_num, embedding_dim]
        masked_feat = F.embedding(xt[qt_mask], self.one_hot_feat)  # [mask_num, res_len * concept_num]
        res_embedding = masked_feat.mm(x_embedding)  # [mask_num, embedding_dim]
        mask_num = res_embedding.shape[0]

        concept_idx_mat = self.concept_num * torch.ones((batch_size, self.concept_num), device=xt.device).long()
        concept_idx_mat[qt_mask, :] = torch.arange(self.concept_num, device=xt.device)
        concept_embedding = self.emb_c(concept_idx_mat)  # [batch_size, concept_num, embedding_dim]

        index_tuple = (torch.arange(mask_num, device=xt.device), qt[qt_mask].long())
        concept_embedding[qt_mask] = concept_embedding[qt_mask].index_put(index_tuple, res_embedding)
        tmp_ht = torch.cat((ht, concept_embedding), dim=-1)  # [batch_size, concept_num, hidden_dim + embedding_dim]
        return tmp_ht

    # GNN aggregation step, as shown in 3.3.2 Equation 1 of the paper
    def _agg_neighbors(self, tmp_ht, qt,adj,reverse_adj):
        r"""
        Parameters:
            tmp_ht: temporal hidden representations of all concepts after the aggregate step
            qt: question indices for all students in a batch at the current timestamp
        Shape:
            tmp_ht: [batch_size, concept_num, hidden_dim + embedding_dim]
            qt: [batch_size]
            m_next: [batch_size, concept_num, hidden_dim]
        Return:
            m_next: hidden representations of all concepts aggregating neighboring representations at the next timestamp
            concept_embedding: input of VAE (optional)
            rec_embedding: reconstructed input of VAE (optional)
            z_prob: probability distribution of latent variable z in VAE (optional)
        """
        qt_mask = torch.ne(qt, -1)  # [batch_size], qt != -1
        print("masked_qt shape", qt_mask.shape)
        print("masked_qt", qt_mask)
        masked_qt = qt[qt_mask]  # [mask_num, ]
        print("masked_qt shape", masked_qt.shape)
        print("masked_qt", masked_qt)
        print("tmp_ht", tmp_ht)
        print("tmp_ht.shape", tmp_ht.shape)
        masked_tmp_ht = tmp_ht[qt_mask]  # [mask_num, concept_num, hidden_dim + embedding_dim]
        print("masked_tmp_ht.shape", masked_tmp_ht.shape)
        mask_num = masked_tmp_ht.shape[0]
        self_index_tuple = (torch.arange(mask_num, device=qt.device), masked_qt.long())
        self_ht = masked_tmp_ht[self_index_tuple]  # [mask_num, hidden_dim + embedding_dim]
        self_features = self.f_self(self_ht)  # [mask_num, hidden_dim]
        expanded_self_ht = self_ht.unsqueeze(dim=1).repeat(1, self.concept_num,1)  # [mask_num, concept_num, hidden_dim + embedding_dim]
        neigh_ht = torch.cat((expanded_self_ht, masked_tmp_ht),dim=-1)  # [mask_num, concept_num, 2 * (hidden_dim + embedding_dim)]
        # concept_embedding, rec_embedding, z_prob = None, None, None
        # self.f_neighbor_list[0](neigh_ht) shape: [mask_num, concept_num, hidden_dim]
        neigh_features = (adj * self.f_neighbor_list[0](neigh_ht) + reverse_adj * self.f_neighbor_list[1](neigh_ht))
        print("adj_shape", adj.shape)
        print("adj_value", adj)
        print("reverse_adj_shape", reverse_adj.shape)
        print("reverse_adj_value", reverse_adj)
        print("neigh_features_shape", neigh_features.shape)
        print("neigh_features_value", neigh_features)  # neigh_features: [mask_num, concept_num, hidden_dim]
        m_next = tmp_ht[:, :, :self.hidden_dim]
        neigh_features = neigh_features.to(m_next.dtype)
        m_next[qt_mask] = neigh_features
        m_next[qt_mask] = m_next[qt_mask].index_put(self_index_tuple, self_features)
        return m_next

    def _adj(self, qt):
        qt_mask = torch.ne(qt, -1)  # [batch_size], qt != -1
        print("masked_qt shape", qt_mask.shape)
        print("masked_qt", qt_mask)
        masked_qt = qt[qt_mask]  # [mask_num, ]
        adj = self.graph[masked_qt.long(), :].unsqueeze(dim=-1)  # [mask_num, concept_num, 1]
        reverse_adj = self.graph[:, masked_qt.long()].transpose(0, 1).unsqueeze(dim=-1)  # [mask_num, concept_num, 1]
        # self.f_neighbor_list[0](neigh_ht) shape: [mask_num, concept_num, hidden_dim]
        return adj,reverse_adj

    # Update step, as shown in Section 3.3.2 of the paper
    def _update(self, tmp_ht, ht, qt,adj,reverse_adj):
        r"""
        Parameters:
            tmp_ht: temporal hidden representations of all concepts after the aggregate step
            ht: hidden representations of all concepts at the current timestamp
            qt: question indices for all students in a batch at the current timestamp
        Shape:
            tmp_ht: [batch_size, concept_num, hidden_dim + embedding_dim]
            ht: [batch_size, concept_num, hidden_dim]
            qt: [batch_size]
            h_next: [batch_size, concept_num, hidden_dim]
        Return:
            h_next: hidden representations of all concepts at the next timestamp
            concept_embedding: input of VAE (optional)
            rec_embedding: reconstructed input of VAE (optional)
            z_prob: probability distribution of latent variable z in VAE (optional)
        """
        qt_mask = torch.ne(qt, -1)  # [batch_size], qt != -1
        mask_num = qt_mask.nonzero().shape[0]
        # GNN Aggregation
        m_next = self._agg_neighbors(tmp_ht, qt,adj,reverse_adj)  # [batch_size, concept_num, hidden_dim]
        # Erase & Add Gate
        m_next[qt_mask] = self.erase_add_gate(m_next[qt_mask])  # [mask_num, concept_num, hidden_dim]
        # GRU
        h_next = m_next
        res = self.gru(m_next[qt_mask].reshape(-1, self.hidden_dim),
                       ht[qt_mask].reshape(-1, self.hidden_dim))  # [mask_num * concept_num, hidden_num]
        index_tuple = (torch.arange(mask_num, device=qt_mask.device),)
        h_next[qt_mask] = h_next[qt_mask].index_put(index_tuple, res.reshape(-1, self.concept_num, self.hidden_dim))
        return h_next

    # Predict step, as shown in Section 3.3.3 of the paper
    def _predict(self, h_next, qt):
        r"""
        Parameters:
            h_next: hidden representations of all concepts at the next timestamp after the update step
            qt: question indices for all students in a batch at the current timestamp
        Shape:
            h_next: [batch_size, concept_num, hidden_dim]
            qt: [batch_size]
            y: [batch_size, concept_num]
        Return:
            y: predicted correct probability of all concepts at the next timestamp
        """
        qt_mask = torch.ne(qt, -1)  # [batch_size], qt != -1
        y = self.predict(h_next).squeeze(dim=-1)  # [batch_size, concept_num]
        y[qt_mask] = torch.sigmoid(y[qt_mask])  # [batch_size, concept_num]
        return y

    def _get_next_pred(self, yt, q_next):
        r"""
        Parameters:
            yt: predicted correct probability of all concepts at the next timestamp
            q_next: question index matrix at the next timestamp
            batch_size: the size of a student batch
        Shape:
            y: [batch_size, concept_num]
            questions: [batch_size, seq_len]
            pred: [batch_size, ]
        Return:
            pred: predicted correct probability of the question answered at the next timestamp
        """
        next_qt = q_next
        next_qt = torch.where(next_qt != -1, next_qt, self.concept_num * torch.ones_like(next_qt, device=yt.device))
        one_hot_qt = F.embedding(next_qt.long(), self.one_hot_q)  # [batch_size, concept_num]
        # dot product between yt and one_hot_qt
        pred = yt * one_hot_qt  # [batch_size, ]
        return pred

    def cal_sha_adj(self, yt, pred, qt):
        log_probs = torch.log(yt + torch.finfo(torch.float32).resolution)
        discrete_probs = torch.eye(pred.shape[1])[torch.argmax(pred, dim=-1)]
        vals = torch.sum(discrete_probs * log_probs, dim=0)
        print("vals_shape:", vals.shape)
        print("vals:", vals)
        # 这意味着对应的知识点在预测结果中具有负面的重要性，即对于正确预测结果而言，该知识点的存在可能会对预测结果产生负面影响。
        # 这种情况可能发生在模型对于某些知识点的预测结果较差，导致其对整体预测结果产生负面影响。负值的重要性分数可能需要进一步分析，以确定该知识点在模型中的影响。
        adj,reverse_adj = self._adj(qt)
        concept_num = adj[0]
        print("adj_shape", adj.shape)
        print("adj_value:", adj)
        print("reverse_adj_shape", reverse_adj.shape)
        print("reverse_adj_value:", reverse_adj)

        positions_dict, key_to_idx, positions, coefficients, unique_inverse, unique_positions = self.construct_positions_dict(adj, max_order=4)
        rpositions_dict, rkey_to_idx, rpositions, rcoefficients, runique_inverse, runique_positions = self.construct_positions_dict(reverse_adj, max_order=4)
        # key_to_val = {key: torch.tensor([vals[unique_inverse[idx]] for idx in key_to_idx[key]], dtype=torch.float64) for key in key_to_idx}
        # key_to_val = {key: torch.tensor(vals[unique_inverse[key_to_idx[key]]], dtype=torch.float64) for key in key_to_idx}
        key_to_val = {key: vals[unique_inverse[key_to_idx[key]]].clone().detach().to(torch.float64) for key in key_to_idx}
        rkey_to_val = {key: vals[runique_inverse[rkey_to_idx[key]]].clone().detach().to(torch.float64) for key in rkey_to_idx}
        # Compute importance scores.

        phis = torch.zeros(self.concept_num, dtype=torch.float64)
        for i in range(self.concept_num):
            phis[i] = torch.sum(torch.stack(
                [(coefficients[j] * (key_to_val[(i, j, 1)] - key_to_val[(i, j, 0)])) for j in coefficients]))

        rphis = torch.zeros(self.concept_num, dtype=torch.float64)
        for i in range(self.concept_num):
            rphis[i] = torch.sum(torch.stack(
                [(coefficients[j] * (rkey_to_val[(i, j, 1)] - rkey_to_val[(i, j, 0)])) for j in coefficients]))


        # Compute importance scores.

        print("phis_shape:", phis.shape)
        print("phis:", phis)
        print("rphis_shape:", rphis.shape)
        print("rphis:", rphis)
        adj = adj.to(torch.float64)
        result = torch.squeeze(torch.matmul(adj, phis.unsqueeze(0)), dim=-1)
        reverse_adj = reverse_adj.to(torch.float64)
        r_result = torch.squeeze(torch.matmul(reverse_adj, rphis.unsqueeze(0)), dim=-1)
        # result = torch.squeeze(torch.matmul(adj_tensor.unsqueeze(-1), phis.unsqueeze(0)), dim=-1)

        print("result_shape", result.shape)
        print("result", result)
        print("r_result_shape", r_result.shape)
        print("r_result", r_result)
        new_adj = torch.sum(result, dim=2).unsqueeze(-1)
        new_reverse_adj = torch.sum(r_result,dim=2).unsqueeze(-1)
        print("new_adj", new_adj.shape)
        print("new_adj", new_adj)
        print("new_reverse_adj", new_reverse_adj.shape)
        print("new_reverse_adj", new_reverse_adj)
        return new_adj,new_reverse_adj

    def construct_positions_dict(self,adj, max_order):
        mask_num, concept_num, _ = adj.size()
        subsets = {}
        coefficients = {j: 2.0 / (j * (j + 1) * (j + 2)) for j in range(1, max_order + 1)}

        for i in range(concept_num):
            neighbor_indices = torch.nonzero(adj[:, i]).squeeze(1).reshape(-1)
            subsets[i] = []
            for k in range(max_order):
                combinations = torch.combinations(neighbor_indices, k + 1)
                subsets[i].extend(combinations.tolist())

        positions_dict = {(i, l, fill): [] for i in range(concept_num) for l in range(1, max_order + 1) for fill in
                          [0, 1]}
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

    def forward(self, features, questions):
        r"""
        Parameters:
            features: input one-hot matrix
            questions: question index matrix
        seq_len dimension needs padding, because different students may have learning sequences with different lengths.
        Shape:
            features: [batch_size, seq_len]
            questions: [batch_size, seq_len]
            pred_res: [batch_size, seq_len - 1]
        Return:
            pred_res: the correct probability of questions answered at the next timestamp
            concept_embedding: input of VAE (optional)
            rec_embedding: reconstructed input of VAE (optional)
            z_prob: probability distribution of latent variable z in VAE (optional)
        """
        batch_size, seq_len = features.shape
        ht = Variable(torch.zeros((batch_size, self.concept_num, self.hidden_dim), device=features.device))
        pred_list = []
        for i in range(seq_len):
            xt = features[:, i]  # [batch_size]
            qt = questions[:, i]  # [batch_size]
            qt_mask = torch.ne(qt, -1)  # [batch_size], next_qt != -1
            tmp_ht = self._aggregate(xt, qt, ht, batch_size)  # [batch_size, concept_num, hidden_dim + embedding_dim]
            if i == 0:
                adj, reverse_adj = self._adj(qt)
            else:
                adj,reverse_adj = self.cal_sha_adj(yt,pred,qt)
            h_next = self._update(tmp_ht, ht, qt,adj,reverse_adj)  # [batch_size, concept_num, hidden_dim]
            ht[qt_mask] = h_next[qt_mask]  # update new ht
            yt = self._predict(h_next, qt)  # [batch_size, concept_num]
            if i < seq_len - 1:
                pred = self._get_next_pred(yt, questions[:, i + 1])
                pred_list.append(pred)
        pred_res = torch.stack(pred_list, dim=1)  # [batch_size, seq_len - 1]
        return pred_res

    """
   
  pred_list = []
for i in range(seq_len - 1):
    if i < seq_len - 1:
        pred = self._get_next_pred(yt, questions[:, i + 1])
        pred_list.append(pred)

vals_list = []
for pred in pred_list:
    log_probs = torch.log(yt + torch.finfo(torch.float32).resolution)
    discrete_probs = torch.eye(pred.shape[1])[torch.argmax(pred, dim=-1)]
    vals = torch.sum(discrete_probs * log_probs, dim=0)
    vals_list.append(vals)

pred_res = torch.stack(pred_list, dim=1)  # [batch_size, seq_len - 1]
vals_res = torch.stack(vals_list, dim=1)  # [batch_size, seq_len - 1]

print("vals_res shape:", vals_res.shape)
print("vals_res:", vals_res)


    """
