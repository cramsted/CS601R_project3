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
TEST_DATA = "../lfw/pairs_dev_test.json"

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
        avg_loss = accuracies[0]
        avg_acc = accuracies[1]
except:
    avg_loss = []
    avg_acc = []

# Datasets
raw_dataset = RawData(TRAIN_DATA)
raw_loader = DataLoader(raw_dataset, batch_size=50,
                        shuffle=False, num_workers=10)

triplet_dataset = TripletData()
triplet_loader = DataLoader(triplet_dataset, batch_size=30,
                            shuffle=False, num_workers=15)

test_dataset = TestData(TEST_DATA)
train_dataset = TestData(TRAIN_DATA)
test_loader = DataLoader(test_dataset, batch_size=50,
                         shuffle=True, num_workers=15)
train_loader = DataLoader(train_dataset, batch_size=50,
                          shuffle=True, num_workers=15)

# loss functions
criterion = nn.TripletMarginLoss(margin=0.2)
optimizer = optim.Adam(
    [{"params": face.parameters(), 'initial_lr': 5e-4}], lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 25, gamma=0.1)


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


def train():
    losses = []
    for i, data in enumerate(triplet_loader, 1):
        anchor, pos, neg = data
        anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)

        optimizer.zero_grad()

        out_anchor = face(anchor)
        out_pos = face(pos)
        out_neg = face(neg)

        loss = criterion(out_anchor, out_pos, out_neg)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.average(losses)


def test():
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            img1, img2, label = data[0].to(device), data[1].to(device), data[2]
            label = label.type(torch.ByteTensor).cuda()

            out_img1 = face(img1)
            out_img2 = face(img2)

            dist = (((out_img1 - out_img2)**2).sum(1)**(1/2)) <= 1.0

            acc = dist == label if i == 0 else torch.cat((acc, dist == label))
    acc = acc.cpu().detach().numpy()
    return acc.sum()/acc.shape[0]


# get number of epochs to run
try:
    epochs = int(sys.argv[1])
except:
    raise ValueError("No epoch value given")

for epoch in range(1, epochs+1):
    start = time.time()
    tree, embeddings, labels = create_tree()
    triplet_dataset.set_tree(tree, embeddings, labels)

    scheduler.step()
    loss = train()
    acc = test()

    avg_loss.append(loss)
    avg_acc.append(acc)

    torch.save(face.state_dict(), MODEL_FILENAME)
    with open(LOSS_FILENAME, 'wb') as f:
        pickle.dump((avg_loss, avg_acc), f)

    print('[Epoch:', len(avg_acc), '] [Time:',
          (time.time()-start)/60, " minutes] [Accuracy:", acc, "%] [Average Loss:", loss, "]")
