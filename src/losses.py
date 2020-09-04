import torch
from torch import nn as nn
import torchvision.models as models

class PerceptualLoss(nn.Module):
    """
    Perceptual loss implemented with a VGG-19 model
    """

    def __init__(self, model_path, feature_idx, bn=True, loss_criterion='l1'):
        # Instantiate model
        model = None
        if bn:
            model = models.vgg19_bn()
        else:
            model = models.vgg19()
        
        # Load weight
        print('Loading VGG-19 model weights')
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model'])

        # Specifying layer to extract from
        self.model = nn.Sequential(*list(model.features.children())[:feature_idx])
        self.model.eval()
        
        # Specifying loss criterion
        self.loss = None
        if loss_criterion == 'l1':
            self.loss = nn.L1Loss()
        elif loss_criterion == 'l2':
            self.loss = nn.MSELoss()
    
    def forward(self, fake, real):
        with torch.no_grad():            
            fake_features = self.model(fake)
            real_features = self.model(real)
        
        return self.loss(fake_features, real_features)
    
