import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR

import json
import os

import pandas as pd
import numpy as np

import pickle
import operator
from collections import defaultdict

from model import UNet
from preprocessing import adjust_landmarks, Landmarks_Dataset, Rescale, ToTensor

########################################################################################################################
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
########################################################################################################################

with open('data/openmonkeychallenge/train_annotation.json', 'r') as f:
    train_ann = json.load(f)
    
train_ann = train_ann['data']
train_ann = adjust_landmarks(train_ann)

train_d = pd.DataFrame(train_ann)
train_d['file'] = 'data/openmonkeychallenge/train/train/'+train_d['file']


########################################################################################################################

with open('data/openmonkeychallenge/val_annotation.json', 'r') as f:
    val_ann = json.load(f)
    
val_ann = val_ann['data']
val_ann = adjust_landmarks(val_ann)

val_d = pd.DataFrame(val_ann)
val_d['file'] = 'data/openmonkeychallenge/val/val/'+val_d['file']

########################################################################################################################
train_data = Landmarks_Dataset(train_d, transform=transforms.Compose([Rescale(200),
                                                                      ToTensor()]))
val_data = Landmarks_Dataset(val_d, transform=transforms.Compose([Rescale(200),
                                                                      ToTensor()]))
num_workers = 2
dataset_train = DataLoader(train_data, batch_size=256, num_workers=num_workers)

dataset_val = DataLoader(val_data, batch_size=256, num_workers=num_workers)

########################################################################################################################
# Unet configuration
config = {
    'channels': {
        'c1': 3, 
        'c2': 64,
        'c3': 128,
        'c4': 256,
        'c5': 512,
        'c6': 2
    },
    'FFN': {
        'in': 23328,
        'hl1': 5000,
        'hl2': 500,
        'hl3': 34
    },
    'out':200
}

U = UNet(config).to(device = DEVICE)
with open('Uw.pt', 'rb') as f:
    U3  = torch.load(f)
U.load_state_dict(U3)

########################################################################################################################


def eval_(model, datatrain, device):
    loss1 = nn.L1Loss(reduction='none')
    loss2 = nn.MSELoss(reduction='none')

    model.eval()
    loss1_by_species = defaultdict(list)
    loss2_by_species = defaultdict(list)
    for batch in datatrain:
        spc_input = batch['species']
        imgs_input = batch['image']
        imgs_input = imgs_input.to(device=device, dtype=torch.float32)
        ldm_target = batch['landmarks']
        ldm_target = ldm_target.to(device=device, dtype=torch.float32)

        prediction = model(imgs_input)

        loss_1 = loss1(prediction, ldm_target).sum(dim=-1)
        loss_2 = loss2(prediction, ldm_target).sum(dim=-1)

        for sp, l1, l2 in zip(spc_input, loss_1, loss_2):
            loss1_by_species[sp].append(l1.item())
            loss2_by_species[sp].append(l2.item())

    return {'l1':loss1_by_species, 'l2': loss2_by_species}


with torch.no_grad():
    loss_train = eval_(U, dataset_train, DEVICE)
    loss_val = eval_(U, dataset_val, DEVICE)

pickle.dump(loss_train, open('eval_train.pickle', 'wb'))
pickle.dump(loss_val, open('eval_val.pickle', 'wb'))
