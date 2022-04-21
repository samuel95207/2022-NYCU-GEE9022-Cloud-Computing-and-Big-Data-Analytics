import torch.nn as nn
import torchvision.models as models


class ResNetModel(nn.Module):
    def __init__(self, out_dim):
        super(ResNetModel, self).__init__()

        self.model =  models.resnet18(pretrained=False, num_classes=out_dim)
        # self.model =  models.resnet50(pretrained=False, num_classes=out_dim)

        in_dim = self.model.fc.in_features

        self.model.fc = nn.Sequential(nn.Linear(in_dim, in_dim), nn.ReLU(), self.model.fc)

    def forward(self, x):
        return self.model(x)