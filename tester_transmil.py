from dataloader import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset

import os
import torch
from torch import nn
import torch.optim as optim
import pdb
import torch.nn.functional as F

import pandas as pd
import numpy as np

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc

from models.TransMIL import TransMIL
from utils import *


class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count

def summary(model, loader, n_classes):
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.inference_mode():
            logits, Y_prob, Y_hat, h0 = model(data)
                
        acc_logger.log(Y_hat, label)
        
        probs = Y_prob.cpu().numpy()

        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        all_preds[batch_idx] = Y_hat.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        
        error = calculate_error(Y_hat, label)
        test_error += error

    del data
    test_error /= len(loader)

    aucs = []
    if len(np.unique(all_labels)) == 1:
        auc_score = -1

    else: 
        if n_classes == 2:
            auc_score = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
            for class_idx in range(n_classes):
                if class_idx in all_labels:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                    aucs.append(calc_auc(fpr, tpr))
                else:
                    aucs.append(float('nan'))
            binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
            fpr, tpr, _ = roc_curve(binary_labels.ravel(), all_probs.ravel())
            auc_score = calc_auc(fpr, tpr)

    results_dict = {'slide_id': slide_ids, 'Y': all_labels, 'Y_hat': all_preds}
    for c in range(n_classes):
        results_dict.update({'p_{}'.format(c): all_probs[:,c]})
    df = pd.DataFrame(results_dict)
    return patient_results, test_error, auc_score, df, acc_logger

if __name__ == "__main__":
    save_dir = './results/camelyon16to17unseen_transmil_simclr_100/'
    csv_path = '/data1/ceiling/workspace/AttriMIL_v2/dataset_csv/camelyon17_unseen.csv'
    data_dir = '/data2/clh/camelyon17/resnet18_simclr/'
    weight_dir = '/data1/ceiling/workspace/MIL/AttriMIL/save_weights/camelyon16_transmil_simclr_100/'
    split_path = '/data1/ceiling/workspace/AttriMIL_v2/splits/unitopatho/'
    dataset = Generic_MIL_Dataset(csv_path = csv_path,
                            data_dir = data_dir,
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'normal_tissue':0, 'tumor_tissue':1},
                            patient_strat=False,
                            ignore=[])
    os.makedirs(save_dir, exist_ok=True)
    model = TransMIL(dim=512, n_classes=2).cuda()
    folds = [0, 1, 2, 3, 4]
    ckpt_paths = [os.path.join(weight_dir, 's_{}_checkpoint.pt'.format(fold)) for fold in folds]
    all_results = []
    all_auc = []
    all_acc = []
    csv_path = [split_path + 'splits_{}.csv'.format(i) for i in range(4)]
    for ckpt_idx in range(len(ckpt_paths)):
        # train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False, csv_path=csv_path[ckpt_idx])
        # loader = get_split_loader(test_dataset, testing = False)
        loader = get_simple_loader(dataset)
        model.load_state_dict(torch.load(ckpt_paths[ckpt_idx]))
        patient_results, test_error, auc, df, acc_logger = summary(model, loader, n_classes=2)
        all_results.append(all_results)
        all_auc.append(auc)
        all_acc.append(1-test_error)
        df.to_csv(os.path.join(save_dir, 'fold_{}.csv'.format(folds[ckpt_idx])), index=False)

    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_auc, 'test_acc': all_acc})
    save_name = 'summary.csv'
    final_df.to_csv(os.path.join(save_dir, save_name))
