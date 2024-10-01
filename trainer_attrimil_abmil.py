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

from models.AttriMIL import AttriMIL
from utils import *

from constraints import spatial_constraint, rank_constraint
import queue

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
        
    for batch_idx, (data, label, coords, nearest) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.inference_mode():
            ins_prediction, bag_prediction, _, _ = model(data)
        Y_hat = torch.topk(bag_prediction.view(1, -1), 1, dim = 1)[1]
        acc_logger.log(Y_hat, label)

        max_prediction, _ = torch.max(ins_prediction, 0)
        Y_prob = F.softmax(bag_prediction, dim=-1)

        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    return patient_results, test_error, auc, acc_logger

def train_abmil(datasets,
                save_path='./save_weights/camelyon16_abmil_imagenet/',
                feature_dim = 512,
                n_classes = 2,
                fold = 0,
                writer_flag = True,
                max_epoch = 200,
                early_stopping = True,):
    
    writer_dir = os.path.join(save_path, str(fold))
    if not os.path.isdir(writer_dir):
        os.makedirs(writer_dir)
    if writer_flag:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)
    else:
        writer = None
    
    print("\nInit train/val/test splits...")
    train_split, val_split, test_split = datasets
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))
    
    print("\nInit loss function...")
    loss_fn = nn.CrossEntropyLoss()
    
    model = AttriMIL(dim=feature_dim, 
                     n_classes=n_classes)
    _ = model.to(device)
    
    print("\nInit optimizer")
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4, momentum=0.9, weight_decay=1e-5)
    
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = False, weighted = True)
    val_loader = get_split_loader(val_split,  testing = False)
    test_loader = get_split_loader(test_split, testing = False)
    print('Done!')
    
    mini_loss = 10000
    retain = 0
    
    for epoch in range(max_epoch):
        train_loop(epoch, model, train_loader, optimizer, n_classes, writer, loss_fn)
        loss = validate(epoch, model, val_loader, n_classes, writer, loss_fn)
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(save_path, 's_{}_checkpoint_{}.pt'.format(fold, epoch)))
        if loss < mini_loss:
            print("loss decrease from:{} to {}".format(mini_loss, loss))
            torch.save(model.state_dict(), os.path.join(save_path, 's_{}_checkpoint.pt'.format(fold)))
            mini_loss = loss
            retain = 0
        else:
            retain += 1
            print("Retain of early stopping: {} / {}".format(retain, 20))
        if early_stopping:
            if retain > 20 and epoch > 50:
                print("Early stopping")
                break
                
    model.load_state_dict(torch.load(os.path.join(save_path, 's_{}_checkpoint.pt'.format(fold))))
    summary(model, test_loader, n_classes)
    
def train_loop(epoch, model, loader, optimizer, n_classes, writer, loss_fn):
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    bag_loss = 0.
    ins_loss = 0.
    
    print('\n')
    
        
    label_positive_list = []
    label_negative_list = []
    for i in range(n_classes):
        label_positive_list.append(queue.Queue(maxsize=4))
        label_negative_list.append(queue.Queue(maxsize=4))
        
    for batch_idx, (data, label, coords, nearest) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        
        logits, Y_prob, Y_hat, attribute_score, results_dict = model(data)
        acc_logger.log(Y_hat, label)
        loss_bag = loss_fn(logits, label)
        loss_spa = spatial_constraint(attribute_score, n_classes, nearest, ks=3)
        loss_rank, label_positive_list, label_negative_list = rank_constraint(data, label, model, attribute_score, n_classes, label_positive_list, label_negative_list)

        loss = loss_bag + 1.0 * loss_spa + 5.0 * loss_rank
        
        loss_bag_value = loss_bag.item()
        loss_ins_value = loss_ins.item()
        loss_spa_value = loss_spa.item()
        loss_rank_value = loss_rank.item()
        loss_value = loss.item()
        
        train_loss += loss_value
        bag_loss += loss_bag
        
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, loss_bag: {:.4f}, loss_spa: {:.4f}, bag_size: {}'.format(batch_idx, loss_value, loss_bag_value, loss_spa_value, label.item(), data.size(0)))
            
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    train_loss /= len(loader)
    bag_loss /= len(loader)
    ins_loss /= len(loader)
    
    print('Epoch: {}, train_loss: {:.4f}, bag_loss: {:.4f}'.format(epoch, train_loss, bag_loss))
    
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)
    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/loss_bag', loss_bag, epoch)


def validate(epoch, model, loader, n_classes, writer, loss_fn):
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    
    with torch.no_grad():
        for batch_idx, (data, label, coords, nearest) in enumerate(loader):
            data, label = data.to(device, non_blocking=True), label.to(device, non_blocking=True)

            logits, Y_prob, Y_hat, attribute_score, results_dict = model(data)
            acc_logger.log(Y_hat, label)
            loss_bag = loss_fn(logits, label)

            loss = loss_bag

            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            val_loss += loss.item()
        
    val_loss /= len(loader)
    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
    
    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
    print('\nVal Set, val_loss: {:.4f}, auc: {:.4f}'.format(val_loss, auc))
    
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))     

    return val_loss

if __name__ == "__main__":
    csv_path = '/data1/ceiling/workspace/AttriMIL_v2/dataset_csv/camelyon16_total.csv'
    data_dir = '/data2/clh/camelyon16/resnet18_imagenet/'
    split_path = '/data1/ceiling/workspace/AttriMIL_v2/splits/camelyon16_100/'
    save_dir = './save_weights/camelyon16_attrimil_imagenet_100/'
    # {'normal_tissue':0, 'tumor_tissue':1}
    
    dataset = Generic_MIL_Dataset(csv_path = csv_path,
                                  data_dir = data_dir,
                                  shuffle = False, 
                                  seed = 1, 
                                  print_info = True,
                                  label_dict = {'normal_tissue':0, 'tumor_tissue':1},
                                  patient_strat=False,
                                  ignore=[])
    csv_path = [split_path + 'splits_{}.csv'.format(i) for i in range(5)]
    for step, name in enumerate(csv_path):
        train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False, csv_path=name)
        train_abmil((train_dataset, val_dataset, test_dataset),
                     save_path=save_dir,
                     feature_dim = 512,
                     n_classes = 2,
                     fold = step,
                     writer_flag = True,
                     max_epoch = 200,
                     early_stopping = False,)