import os
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage import io, transform

class Landmarks_Dataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            idx = idx.tolist()
            
        bbox = self.dataset.loc[index, 'bbox']
        bb_x1 = int(bbox[0])
        bb_y1 = int(bbox[1])
        bb_x2 = int(bbox[0] + bbox[2])
        bb_y2 = int(bbox[1] + bbox[3])
        
        img_name = os.path.join(self.dataset.loc[index, 'file'])
        image = io.imread(img_name)
        image = image[bb_y1:bb_y2,bb_x1:bb_x2]
        specie, fil = self.dataset.loc[index,'species'], self.dataset.loc[index,'file']
        landmarks = self.dataset.loc[index, 'landmarks']
        landmarks = landmarks.astype('float').flatten()
        sample = {'image': image, 'landmarks': landmarks}
        
        sample.update({'species': specie, 'file':fil})

        if self.transform:
            sample = self.transform(sample)

        return sample

class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks, specie, fil = sample.values()
        h, w = image.shape[:2]

        if isinstance(self.output_size, int):
            new_h, new_w = self.output_size, self.output_size
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image,(new_h,new_w))
        landmarks = (landmarks.reshape((-1,2))*[new_w/w, new_h/h]).flatten()

        return {'image':img,
                'landmarks':landmarks,
                'species':specie,
                'file':fil}

class ToTensor(object):
    def __call__(self, sample):
        image, landmarks, specie, fil = sample.values()

        img = image.transpose((2,0,1))
        return {'image':torch.from_numpy(img),
                'landmarks':torch.from_numpy(landmarks),
                'species':specie,
                'file':fil}