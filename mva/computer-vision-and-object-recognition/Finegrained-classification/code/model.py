import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses = 20 


class Pretrained_model(nn.Module):
    def __init__(self, model_name, n_frozen_layers):
        super(Pretrained_model, self).__init__()

        if model_name == "resnet50":
            from torchvision.models import resnet50

            self.pretrained = resnet50(pretrained=True)
        elif model_name == "resnet18":
            from torchvision.models import resnet18
            self.pretrained = resnet18(pretrained = True)
        elif model_name == "resnext101":
            from torchvision.models import resnext101_32x8d
            self.pretrained = resnext101_32x8d(pretrained=True)
        #freeze the first n_frozen_layers
        ct = 0
        for child in self.pretrained.children():
            ct += 1
            if ct <= n_frozen_layers:
                for param in child.parameters():
                    param.requires_grad = False
      
        #the last fc layers n_classes outputs 
        self.fc = nn.Linear(self.pretrained.fc.out_features, nclasses)

    def forward(self, x):
        x = self.pretrained(x)
        return self.fc(x)