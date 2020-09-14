import torch
from torch import nn as nn
from torch.nn import L1Loss
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
    

class TVLoss(L1Loss):
    """ Total Variation loss.
        Args:
            loss_weight (float): Loss weight. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0):
        super(TVLoss, self).__init__()

    def forward(self, pred):
        y_diff = super(WeightedTVLoss, self).forward(
            pred[:, :, :-1, :], pred[:, :, 1:, :])
        x_diff = super(WeightedTVLoss, self).forward(
            pred[:, :, :, :-1], pred[:, :, :, 1:])

        loss = x_diff + y_diff

        return loss

class DecoderLoss(nn.Module):
    """ Loss for decoder that encompasses MSE (pixel-wise) + Perceptual (VGG-19) + Total Variation Loss
    """

    def __init__(self, model_path, feature_idx, bn=True, loss_criterion='l1'):
        self.px_loss = nn.MSELoss()
        self.pr_loss = PerceptualLoss(model_path, feature_idx, bn, loss_criterion)
        self.tv_loss = TVLoss()
    
    def forward(self, fake_mask_patches, real_mask_patches, fake_img, real_img):
        # TODO; weighting of each losses
        return self.px_loss(fake_mask_patches, real_mask_patches) + self.pr_loss(fake_img, real_img) + self.tv_loss(fake_img, real_img)
        