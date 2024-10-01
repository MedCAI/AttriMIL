import numpy as np
import torch
from utils import *
import os
import queue
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc


def spatial_constraint(A, n_classes, nearest, ks=3):
    loss_spatial = torch.tensor(0.0).to(device)
    N = A.shape[-1]
    for c in range(1, n_classes):
        score = A[:, c] # N
        nearest_score = score[nearest] # N ks^2-1
        abs_nearest = torch.abs(nearest_score)
        max_indices = torch.argmax(abs_nearest, dim=1)
        local_prototype = nearest_score.gather(1, max_indices.view(-1, 1)).squeeze()
        # print(local_prototype[:10])
        loss_spatial += torch.mean(torch.abs(torch.tanh(score - local_prototype)))
    return loss_spatial


def rank_constraint(data, label, model, A, n_classes, label_positive_list, label_negative_list):
    loss_rank = torch.tensor(0.0).to(device)
    for c in range(n_classes):
        if label == c:
            value, indice = torch.topk(A[0, c], k=1)
            h = data[indice.item(): indice.item() + 1] # top feature
            if label_positive_list[c].full():
                _ = label_positive_list[c].get()
            label_positive_list[c].put(h)
            if label_negative_list[c].empty():
                loss_rank = loss_rank + torch.tensor(0.0).to(device)
            else:
                h = label_negative_list[c].get()
                label_negative_list[c].put(h)
                Ah, _, _, _ = model(h.detach())
                if c != 0:
                    loss_rank = loss_rank + torch.clamp(torch.mean(Ah[0, c] - value), min=0.0) + torch.clamp(torch.mean(-value), min=0.0) + torch.clamp(torch.mean(Ah[0, c]), min=0.0)
                else:
                    loss_rank = loss_rank + torch.clamp(torch.mean(-value), min=0.0) + torch.clamp(torch.mean(Ah[0, c]), min=0.0)
        else:
            value, indice = torch.topk(A[0, c], k=1)
            h = data[indice.item(): indice.item() + 1] # top feature
            if label_negative_list[c].full():
                _ = label_negative_list[c].get()
            label_negative_list[c].put(h)
            if label_positive_list[c].empty():
                loss_rank = loss_rank + torch.tensor(0.0).to(device)
            else:
                h = label_positive_list[c].get()
                label_positive_list[c].put(h)
                Ah, _, _, _ = model(h.detach())
                if c != 0:
                    loss_rank = loss_rank + torch.clamp(torch.mean(value - Ah[0, c]), min=0.0) + torch.clamp(torch.mean(value), min=0.0)
                else:
                    loss_rank = loss_rank + torch.clamp(torch.mean(value), min=0.0) + torch.clamp(torch.mean(-Ah[0, c]), min=0.0)

    loss_rank = loss_rank / n_classes
    return loss_rank, label_positive_list, label_negative_list