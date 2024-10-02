import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
import torch
import torchvision.models as models

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # model = timm.create_model('resnet50_gn', num_classes=1000, pretrained=False)
        # path = './weights/resnet50_gn_a1h2-8fe6c4d0.pth'
        # model.load_state_dict(torch.load(path))
        path = './weights/simclr_resnet50.safetensors'
        model = timm.create_model('resnet50', pretrained=False, pretrained_cfg_overlay=dict(file=path))
        self.resnet = model
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.act1(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        # x = rearrange(h, 'b c h w -> b h w c')
        # x = self.adapter(x)
        # x = rearrange(x, 'b h w c -> b c h w')
        # x = x + h
        x = self.resnet.layer2(x)
        # x = rearrange(h, 'b c h w -> b h w c')
        # x = self.adapter(x)
        # x = rearrange(x, 'b h w c -> b c h w')
        # x = x + h
        x = self.resnet.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        return x


class ResNet18(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # model = timm.create_model('resnet50_gn', num_classes=1000, pretrained=False)
        # path = './weights/resnet50_gn_a1h2-8fe6c4d0.pth'
        # model.load_state_dict(torch.load(path))
        # path = './weights/resnet18_imagenet.safetensors'
        # model = timm.create_model('resnet18', pretrained=False, pretrained_cfg_overlay=dict(file=path))
        resnet18 = models.resnet18(pretrained=True)
        self.resnet = resnet18
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        x = F.interpolate(x, (256, 256), mode='bilinear')
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        # x = rearrange(h, 'b c h w -> b h w c')
        # x = self.adapter(x)
        # x = rearrange(x, 'b h w c -> b c h w')
        # x = x + h
        x = self.resnet.layer2(x)
        # x = rearrange(h, 'b c h w -> b h w c')
        # x = self.adapter(x)
        # x = rearrange(x, 'b h w c -> b c h w')
        # x = x + h
        x = self.resnet.layer3(x)
        # x = rearrange(h, 'b c h w -> b h w c')
        # x = self.adapter(x)
        # x = rearrange(x, 'b h w c -> b c h w')
        # x = x + h
        x = self.resnet.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        return x

class IClassifier(nn.Module):
    def __init__(self, feature_extractor,):
        super(IClassifier, self).__init__()
        
        self.feature_extractor = feature_extractor      
        
    def forward(self, x):
        device = x.device 
        feats = self.feature_extractor(x) # N x K
        return feats.view(feats.shape[0], -1)
    
