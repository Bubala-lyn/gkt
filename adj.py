import torch
import numpy as np
from time import time
import torch
import torch.nn as nn

def explain_shapley(qt,predict, x, d, k, adj_tensor, positions_dict, key_to_idx, coefficients, unique_inverse):
    """
    Compute the importance score of each feature of x for the predict.
    Inputs:
    predict: a function that takes in inputs of shape (n,d), and
    outputs the distribution of response variable, of shape (n,c),
    where n is the number of samples, d is the input dimension, and
    c is the number of classes.
    x: input vector (d,)
    k: number of neighbors taken into account for each feature.
    adj_tensor: adjacency tensor representing the connection between knowledge points (mask_num, concept_num, 1)
    positions_dict: dictionary containing positions for each concept and neighbor size
    key_to_idx: mapping of positions dictionary keys to indices
    inputs: input tensor containing positions for each concept (n, concept_num)
    coefficients: dictionary of Shapley coefficients
    unique_inverse: mapping from current position to original position indices
    Outputs:
    phis: importance scores of shape (d,)
    """
    while torch.all(k >= d) :
        k -= 2
    st1 = time()
    # Evaluate predict at inputs
    f_vals = predict(adj_tensor,qt)  # [n, c]

    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
    probs = predict(x_tensor,qt)  # [1, c]

    st2 = time()
    log_probs = torch.log(f_vals + np.finfo(float).resolution)

    discrete_probs = torch.eye(len(probs[0]))[torch.argmax(probs, dim=-1)]
    vals = torch.sum(discrete_probs * log_probs, dim=1)

    # Compute importance scores
    phis = torch.zeros(d)
    for i in range(d):
        key = (i, k + 1, 1)
        if key in key_to_idx:
            idx = key_to_idx[key]
            phis[i] = torch.sum(coefficients[j] * (vals[unique_inverse[idx]] - vals[unique_inverse[idx+1]]))
    st3 = time()
    print('func evaluation: {}s, post processing {}s'.format(st2 - st1, st3 - st2))
    return phis.numpy()


