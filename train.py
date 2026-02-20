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


## validation
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
dataset_train = DataLoader(train_data, batch_size=160,
                           shuffle=True, num_workers=num_workers)
dataset_val = DataLoader(val_data, batch_size=160,
                           shuffle=False, num_workers=num_workers)

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

def avg(listt):
    return sum(listt)/len(listt)

########################################################################################################################
warmup_epochs = 5
linear_warmup = lambda epoch: min(1.0, epoch / warmup_epochs)

def train(model, datatrain, dataval, epochs, lr, device):
    loss_function = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=linear_warmup)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=epochs-warmup_epochs)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])


    train_loss = []
    val_loss = []

    for j in range(epochs):
        model.train()
        av = []
        for batch in datatrain:

            imgs_input = batch['image']
            imgs_input = imgs_input.to(device=device, dtype=torch.float32)
            ldm_target = batch['landmarks']
            ldm_target = ldm_target.to(device=device, dtype=torch.float32)

            optimizer.zero_grad()

            prediction = model(imgs_input)
            loss = loss_function(prediction, ldm_target)

            loss.backward()
            optimizer.step()

            av.append(loss.item())

        train_loss.append(avg(av))
        print(f"lr: {scheduler.get_last_lr()}\nLoss: {avg(av)}")
        scheduler.step() 
        

        model.eval()
        av = []
        with torch.no_grad():
            for batch in dataval:

                imgs_input = batch['image']
                imgs_input = imgs_input.to(device=device, dtype=torch.float32)
                ldm_target = batch['landmarks']
                ldm_target = ldm_target.to(device=device, dtype=torch.float32)

                prediction = model(imgs_input)

                loss = loss_function(prediction, ldm_target)

                av.append(loss.item())

        val_loss.append(avg(av))


    return train_loss, val_loss


tr, vl = train(U, dataset_train, dataset_val, 15, 2e-4, DEVICE)

torch.save(U.state_dict(), f'Uw.pt')
pickle.dump(tr, open('train_loss.pickle', 'wb'))
pickle.dump(vl, open('val_loss.pickle', 'wb'))
