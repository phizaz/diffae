import torch
from torchvision import models
from torch import nn
from torchvision import transforms


class ResNetEncoderModel(nn.Module):
    def __init__(self):
        super().__init__()
        resnet18 = models.video.r3d_18(weights='DEFAULT')
        modules = list(resnet18.children())[:-1]
        resnet18 = nn.Sequential(*modules)
        for param in resnet18.parameters():
            param.requires_grad = True
        self.model = resnet18

        self.transform_resnet = transforms.Compose([
            transforms.CenterCrop(112),
            transforms.Normalize((0.43216, 0.394666, 0.37645), (0.22803, 0.22145, 0.216989))
        ])

    def forward(self, x: torch.Tensor):
        x = x.permute((0,2,1,3,4))
        x = self.transform_resnet(x)
        x = x.permute((0,2,1,3,4))
        out = self.model(x)
        out = out.reshape(out.shape[0], out.shape[1])
        return out
