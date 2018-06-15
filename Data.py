import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
import glob
import cv2
import json
import random

IMAGE_ROOT_DIR = "../lfw/images/"


def get_image(file_name):
    imgData = cv2.cvtColor(cv2.imread(
        IMAGE_ROOT_DIR+file_name), cv2.COLOR_BGR2RGB)
    # random flip
    if np.random.rand() > 0.5:
        imgData = np.flip(imgData, axis=1).copy()
    imgData = cv2.resize(imgData, (224, 224))
    # data
    imgData = np.transpose(imgData, (2, 0, 1))
    display_image = imgData.copy()
    imgData = torch.from_numpy(imgData).float()
    return imgData


class RawData(Dataset):

    def __init__(self, img_json, transform=None):
        # self.transform = transform
        json_data = open(IMAGE_ROOT_DIR+"../"+img_json).read()
        data = json.loads(json_data)
        self._data = []
        for key, value in data.items():
            for i in value:
                self._data += i

    def __getitem__(self, index):
        imgData = get_image(self._data[index])
        return imgData, self._data[index]

    def __len__(self):
        return len(self._data)


class TripletData(Dataset):
    def __init__(self, transform=None):
        json_data = open(IMAGE_ROOT_DIR+"../pairs_dev_train.json").read()
        data = json.loads(json_data)
        self._train_data = {}
        self._triplet_pairs = []
        self._images = []
        for key, value in data.items():
            for images in value:
                for i in images:
                    self._images.append(i)
                    key = i.split("/")[0]
                    if key not in self._train_data:
                        self._train_data[key] = [i]
                    else:
                        self._train_data[key].append(i)
                        if key not in self._triplet_pairs:
                            self._triplet_pairs.append(key)

    def __getitem__(self, index):
        anchor_name, pos_name, neg_name = self.get_random_triplet(index)
        anchor = get_image(anchor_name)
        pos = get_image(pos_name)
        neg = get_image(neg_name)
        return anchor, pos, neg

    def get_random_triplet(self, index):
        anchor_name, pos_name = random.sample(
            self._train_data[self._triplet_pairs[index]], 2)
        # 30% chance to get a hard negative
        if np.random.rand() > .3:
            anchor_class = neg_class = ""
            while anchor_class == neg_class:
                neg_name = random.choice(self._images)
                anchor_class = anchor_name.split("/")[0]
                neg_class = neg_name.split("/")[0]
        else:
            i = self.labels.index(anchor_name)
            key = anchor_name.split("/")[0]
            dist, ind = self.tree.query([self.embeddings[i]], k=550)
            for j in ind[0]:
                if key != self.labels[j].split("/")[0]:
                    neg_name = self.labels[j]
                    break
        return anchor_name, pos_name, neg_name

    def set_tree(self, tree, embeddings, labels):
        self.tree = tree
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self._triplet_pairs)


class TestData(Dataset):
    def __init__(self, img_json, transform=None):
        json_data = open(img_json).read()
        data = json.loads(json_data)

        for i in data["diff"]:
            i += [False]
        for i in data["same"]:
            i += [True]

        self._data = data["diff"] + data["same"]

    def __getitem__(self, index):
        img1 = get_image(self._data[index][0])
        img2 = get_image(self._data[index][1])
        label = self._data[index][2]
        return img1, img2, label

    def __len__(self):
        return len(self._data)
