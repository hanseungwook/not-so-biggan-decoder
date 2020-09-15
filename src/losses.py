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
        pretrained = False if model_path else True
        if bn:
            model = models.vgg19_bn(pretrained=pretrained)
        else:
            model = models.vgg19(pretrained=pretrained)
        
        # Load weight
        if model_path:
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
        
        # Input normalization
        if self.use_input_norm:
            # the mean is for image with range [0, 1]
            self.register_buffer(
                'mean',
                torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            # the std is for image with range [0, 1]
            self.register_buffer(
                'std',
                torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def forward(self, fake, real):
        # Clipping into (0, 1) range for fake
        # fake = np.clip(fake, 0, 1)

        # Input normalization
        if self.use_input_norm:
            fake = (fake - self.mean) / self.std
            real = (real - self.mean) / self.std

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
    # Feature idx = 34, for no bn
    def __init__(self, model_path=None, feature_idx=49, bn=True, loss_criterion='l1', use_input_norm=False):
        self.pr_loss = PerceptualLoss(model_path, feature_idx, bn, loss_criterion, use_input_norm)
    
    def forward(self, fake_img, real_img):
        # TODO: weighting of each losses
        return self.pr_loss(fake_img, real_img)
        