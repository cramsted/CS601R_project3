import torch.nn as nn
import torchvision.models as models
from Data import *
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import time
import pickle
import sys
import numpy as np
from FaceNet import FaceNet
from sklearn.neighbors import KDTree

TRAIN_DATA = "../lfw/pairs_dev_train.json"
TEST_DATA = "../lfw/images/pairs_dev_test.json"

MODEL_FILENAME = 'single_Linear_embedded_layer.json'
LOSS_FILENAME = 'single_Linear_embedded_layer.pkl'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
face = FaceNet()
face.to(device)
# load model
try:
    face.load_state_dict(torch.load(MODEL_FILENAME))
except:
    pass

# data for tracking accuracies
try:
    with open(LOSS_FILENAME, 'rb') as f:
        accuracies = pickle.load(f)
        train_accuracy = accuracies[0]
        test_accuracy = accuracies[1]
        num_epochs = accuracies[2]
except:
    train_accuracy = []
    test_accuracy = []
    num_epochs = 0

# Datasets
raw_dataset = RawData(TRAIN_DATA)
raw_loader = DataLoader(raw_dataset, batch_size=50,
                        shuffle=False, num_workers=10)

triplet_dataset = TripletData()
triplet_loader = DataLoader(triplet_dataset, batch_size=50,
                            shuffle=False, num_workers=10)


def create_tree():
    for i, data in enumerate(raw_loader):
        images, labels = data[0].to(device), data[1]

        scores = face(images).cpu().detach().numpy()
        if i == 0:
            output_labels = labels
            output = scores
        else:
            output_labels += labels
            output = np.vstack((output, scores))
    return KDTree(output), output, output_labels



# get number of epochs to run
try:
    epochs = int(sys.argv[1])
except:
    raise ValueError("No epoch value given")

for epoch in range(1, epochs+1):
    start = time.time()
    tree, embeddings, labels = create_tree()
    triplet_dataset.set_tree(tree, embeddings, labels)
    triplet_dataset.__getitem__(0)
    for data in triplet_loader:
        anchor, pos, neg = data
        anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)
        import pdb
        pdb.set_trace()

    print('[Epoch:', epoch, '] [Time:',
          (time.time()-start)/60, " minutes]")
