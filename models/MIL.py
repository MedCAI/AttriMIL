import torch
import torch.nn as nn
import torch.nn.functional as F

class MIL_MeanPooling(nn.Module):
    def __init__(self, 
                 n_classes = 2, 
                 top_k=1,
                 embed_dim=512):
        super().__init__()
        fc = [nn.Linear(embed_dim, embed_dim // 2), nn.ReLU(), nn.Dropout(0.25)]
        self.fc = nn.Sequential(*fc)
        self.classifier=  nn.Linear(embed_dim // 2, n_classes)
        self.top_k=top_k

    def forward(self, h, return_features=False):
        h = self.fc(h)
        h = torch.mean(h, dim=0, keepdim=True)
        logits  = self.classifier(h)
        
        y_probs = F.softmax(logits, dim = 1)
        top_instance_idx = torch.topk(y_probs[:, 1], self.top_k, dim=0)[1].view(1,)
        top_instance = torch.index_select(logits, dim=0, index=top_instance_idx)
        Y_hat = torch.topk(top_instance, 1, dim = 1)[1]
        Y_prob = F.softmax(top_instance, dim = 1) 
        results_dict = {}

        if return_features:
            top_features = torch.index_select(h, dim=0, index=top_instance_idx)
            results_dict.update({'features': top_features})
        return top_instance, Y_prob, Y_hat, y_probs, results_dict

class MIL_MaxPooling(nn.Module):
    def __init__(self, 
                 n_classes = 2, 
                 top_k=1, 
                 embed_dim=512):
        super().__init__()
        fc = [nn.Linear(embed_dim, embed_dim//2), nn.ReLU(), nn.Dropout(0.25)]
        self.fc = nn.Sequential(*fc)
        self.classifiers = nn.Linear(embed_dim//2, n_classes)
        self.top_k=top_k
        self.n_classes = n_classes
        assert self.top_k == 1
    
    def forward(self, h, return_features=False):       
        h = self.fc(h)
        logits = self.classifiers(h)

        y_probs = F.softmax(logits, dim = 1)
        m = y_probs.view(1, -1).argmax(1)
        top_indices = torch.cat(((m // self.n_classes).view(-1, 1), (m % self.n_classes).view(-1, 1)), dim=1).view(-1, 1)
        top_instance = logits[top_indices[0]]

        Y_hat = top_indices[1]
        Y_prob = y_probs[top_indices[0]]

        results_dict = {}

        if return_features:
            top_features = torch.index_select(h, dim=0, index=top_indices[0])
            results_dict.update({'features': top_features})
        return top_instance, Y_prob, Y_hat, y_probs, results_dict


class MIL_RNN(nn.Module):
    def __init__(self, n_classes = 2, embed_dim = 512):
        super().__init__()
        self.dropout = nn.Dropout(0.25)
        self.rnn = nn.RNN(embed_dim, 128, 3, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(256, n_classes)
        self.top_k = 1

    def forward(self, h, return_features=False):
        h0 = torch.zeros(6, 1, 128).to(h.device)
        h = self.dropout(h)
        h = h.unsqueeze(0)
        if return_features:
            h, _ = self.rnn(h, h0)
            h = h.squeeze(1)
            logits = self.classifier(h)
        else:
            h, _ = self.rnn(h, h0)
            h = h.squeeze(0)
            logits = self.classifier(h)
        y_probs = F.softmax(logits, dim = 1)
        top_instance_idx = torch.topk(y_probs[:, 1], self.top_k, dim=0)[1].view(1,)
        top_instance = torch.index_select(logits, dim=0, index=top_instance_idx)
        Y_hat = torch.topk(top_instance, 1, dim = 1)[1]
        Y_prob = F.softmax(top_instance, dim = 1)
        results_dict = {}

        if return_features:
            top_features = torch.index_select(h, dim=0, index=top_instance_idx)
            results_dict.update({'features': top_features})
        return top_instance, Y_prob, Y_hat, y_probs, results_dict