import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x
    

class AttriMIL(nn.Module): 
    '''
    Multi-Branch ABMIL with constraints
    '''
    def __init__(self, n_classes=2, dim=512):
        super().__init__()
        self.adaptor = nn.Sequential(nn.Linear(dim, dim//2),
                                     nn.ReLU(),
                                     nn.Linear(dim // 2 , dim))
        
        attention = []
        classifer = [nn.Linear(dim, 1) for i in range(n_classes)]
        for i in range(n_classes):
            attention.append(Attn_Net_Gated(L = dim, D = dim // 2,))
        self.attention_nets = nn.ModuleList(attention)
        self.classifiers = nn.ModuleList(classifer)
        self.n_classes = n_classes
        self.bias = nn.Parameter(torch.zeros(n_classes), requires_grad=True)
    
    def forward(self, h):
        h = h + self.adaptor(h)
        A_raw = torch.empty(self.n_classes, h.size(0), ) # N x 1
        instance_score = torch.empty(1, self.n_classes, h.size(0)).float().to(h.device)
        for c in range(self.n_classes):
            A, h = self.attention_nets[c](h)
            A = torch.transpose(A, 1, 0)  # 1 x N
            A_raw[c] = A
            instance_score[0, c] = self.classifiers[c](h)[:, 0]
        attribute_score = torch.empty(1, self.n_classes, h.size(0)).float().to(h.device)
        for c in range(self.n_classes):
            attribute_score[0, c] = instance_score[0, c] * torch.exp(A_raw[c])
            
        logits = torch.empty(1, self.n_classes).float().to(h.device)
        for c in range(self.n_classes):
            logits[0, c] = torch.sum(attribute_score[0, c], keepdim=True, dim=-1) / torch.sum(torch.exp(A_raw[c]), dim=-1) + self.bias[c]
            
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        results_dict = {}
        return logits, Y_prob, Y_hat, attribute_score, results_dict