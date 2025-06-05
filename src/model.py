"""
Contains PyTorch model code to instantiate a TinyVGG model.
"""

import torchvision
from torch import nn 

class VGG16(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.vgg16(weights='DEFAULT')

        # Freeze all layers except the classifier
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Modify the classifier to output 2 classes
        self.model.classifier[6] = nn.Linear(self.model.classifier[6].in_features, 1)
        

    def forward(self, x):
        return self.model(x)
    
class VGG19(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.vgg19(weights='DEFAULT')

        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.classifier[6] = nn.Linear(self.model.classifier[6].in_features, 1)
        
    def forward(self, x):
        return self.model(x)
    
class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet50(weights='DEFAULT')

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.fc= nn.Linear(self.model.fc.in_features, 1)
        
    def forward(self, x):
        return self.model(x)

class InceptionV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.inception_v3(weights='DEFAULT')

        for param in self.model.parameters():
            param.requires_grad = False
        

        self.model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.model.fc.in_features, 1)
        )

    def forward(self, x):
        return self.model(x)
    
class MobileNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.mobilenet_v2(weights='DEFAULT')

        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, 1)

    def forward(self, x):
        return self.model(x)


class EfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.efficientnet_b0(weights='DEFAULT')

        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, 1)
        
    def forward(self, x):
        return self.model(x)
    

class ViTAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.vit_b_16(weights='DEFAULT')

        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.heads = nn.Linear(in_features= self.model.heads.head.in_features, out_features=1)

    def forward(self, x):
        return self.model(x)