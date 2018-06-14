import torch.nn as nn
import torchvision.models as models
import torch
import torch.optim as optim
import numpy as np
import cv2


class FaceNet(nn.Module):
    def __init__(self):
        super(FaceNet, self).__init__()
        self.resnet = models.resnet18(pretrained=True)

        self.embedded_layer = torch.nn.Linear(
            in_features=25088, out_features=128, bias=True)

    def forward(self, x):
        # resnet
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        # custom code
        x = x.view(x.size(0), -1)
        x = self.embedded_layer(x)
        # normalization
        xn = torch.norm(x, p=2, dim=0).detach()
        x = x.div(xn.expand_as(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


if __name__ == '__main__':
    fc = FaceNet()
    img = cv2.imread("../lfw/images/Aaron_Eckhart/Aaron_Eckhart_0001.jpg")
    img = np.array([np.transpose(img, (2, 0, 1))])
    img = torch.from_numpy(img).float()
    fc(img)
