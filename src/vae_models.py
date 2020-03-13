import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision.utils import save_image
from utils.utils import zero_mask
import numpy as np
import pywt

def truncated_normal_(tensor, mean=0, std=0.02):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        with torch.no_grad():
            truncated_normal_(m.weight.data, mean=0, std=0.02)
        nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, mean=0, std=0.02)
        nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.Sequential):
        for sub_m in m:
            nn.init.normal_(sub_m.weight.data, mean=0, std=0.02)
            nn.init.constant_(sub_m.bias.data, 0.0)


def iwt(vres, inv_filters, levels=1):
    h = vres.size(2)
    w = vres.size(3)
    res = vres.contiguous().view(-1, h//2, 2, w//2).transpose(1, 2).contiguous().view(-1, 4, h//2, w//2).clone()
    if levels > 1:
        res[:,:1] = iwt(res[:,:1], inv_filters, levels=levels-1)
    res = torch.nn.functional.conv_transpose2d(res, Variable(inv_filters[:,None]),stride=2)
    res = res[:,:,2:-2,2:-2] #removing padding

    return res

def get_upsampling_layer(name, res, bottleneck_dim=100):
    layer = None
    if name == 'linear':
        layer = nn.Linear(3 * res * res, 3 * res * res)
    elif name == 'conv1d':
        layer = nn.Conv1d(res*res, res*res, kernel_size=1, stride=1)
    elif name == 'conv2d':
        layer = nn.Conv2d(3, 3, kernel_size=1, stride=1)
    elif name == 'bottleneck':
        layer = nn.Sequential(nn.Linear(res*res*3, 1024),
                                nn.Linear(1024, bottleneck_dim),
                                nn.Linear(bottleneck_dim, 1024),
                                nn.Linear(1024, res*res*3)
                             )  
    
    return layer

def get_upsampling_dims(name, res):
    sizes = None
    if name == 'linear' or name == 'bottleneck':
        sizes = (-1, 3 * res * res)
    elif name == 'conv1d':
        sizes = (-1, res*res, 3)
    elif name == 'conv2d':
        sizes = (-1, 3, res, res)
    
    return sizes

class WT_layer(nn.Module):    
    def forward(self, input, filters):
        batch_size = input.shape[0]
        h = input.shape[2]
        w = input.shape[3]
        input = input.reshape(-1,1,h,w)
        padded = torch.nn.functional.pad(input,(2,2,2,2))
        res = torch.nn.functional.conv2d(padded, Variable(filters[:,None]),stride=2)
        res = res.view(-1,2,h//2,w//2).transpose(1,2).contiguous().view(-1,1,h,w)

        return res.reshape(batch_size, -1, h, w)

class WT(nn.Module):
    def __init__(self, num_wt=2, device='cpu'):
        super(WT, self).__init__()

        self.num_wt = num_wt
        self.device = device

        w = pywt.Wavelet('bior2.2')
        dec_hi = torch.Tensor(w.dec_hi[::-1]).to(self.device)
        dec_lo = torch.Tensor(w.dec_lo[::-1]).to(self.device)
        rec_hi = torch.Tensor(w.rec_hi).to(self.device)
        rec_lo = torch.Tensor(w.rec_lo).to(self.device)

        self.filters = torch.stack([dec_lo.unsqueeze(0)*dec_lo.unsqueeze(1),
                       dec_lo.unsqueeze(0)*dec_hi.unsqueeze(1),
                       dec_hi.unsqueeze(0)*dec_lo.unsqueeze(1),
                       dec_hi.unsqueeze(0)*dec_hi.unsqueeze(1)], dim=0)
        
        self.wt = nn.Sequential()
        for i in range(self.num_wt):
            self.wt.add_module('wt_{}'.format(i), WT_layer())
        
    def forward(self, input):
        for m in self.wt:
            input = m(input, self.filters)

        return (input, )

class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()

        #self.device = 'cpu'
        w = pywt.Wavelet('bior2.2')
        rec_hi = torch.Tensor(w.rec_hi).cuda()
        rec_lo = torch.Tensor(w.rec_lo).cuda()

        self.inv_filters = torch.stack([rec_lo.unsqueeze(0)*rec_lo.unsqueeze(1),
                                    rec_lo.unsqueeze(0)*rec_hi.unsqueeze(1),
                                    rec_hi.unsqueeze(0)*rec_lo.unsqueeze(1),
                                    rec_hi.unsqueeze(0)*rec_hi.unsqueeze(1)], dim=0)

    def forward(self, input):
        batch_size = input.shape[0]
        h = input.shape[2]
        w = input.shape[3]
        input = input.view(-1,1,h,w).contiguous().view(-1,h//2,2,w//2).transpose(1,2).contiguous().view(-1,4,h//2,w//2).clone()
        input = torch.nn.functional.conv_transpose2d(input, Variable(self.inv_filters[:,None]),stride=2)
        input = input[:,:,2:-2,2:-2] # Remove padding
        
        return input.reshape(batch_size, -1, h, w)

    # def set_device(device):
    #     self.device = device

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
class UnFlatten(nn.Module):
    def forward(self, input, size=2048):
        return input.view(input.size(0), size, 1, 1)

# Mask substracting from image
class Mask_Sub(nn.Module):
    def forward(self, img, mask):
        return img - mask.unsqueeze(1)

# Mask adding to image
class Mask_Add(nn.Module):
    def forward(self, img, mask):
        return img + mask.unsqueeze(1)

class UnFlatten1(nn.Module):
    def forward(self, input, size=512):
        return input.view(input.size(0), size, 2, 2) 

# WTVAE for 64 x 64 images
# num_wt of WT layers (default: 2)
# Using Unflatten for decoder (N * 2048 * 1 * 1), instead of Unflatten1
class WTVAE_64(nn.Module):
    def __init__(self, image_channels=3, h_dim=2048, z_dim=100, unflatten=0, num_wt=2):
        super(WTVAE_64, self).__init__()

        self.inv_filters = None
        self.cuda = False
        self.device = None
        self.num_wt = num_wt
        
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2), # N * 32 * 31 * 31
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # N * 64 * 14 * 14,
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2), # N * 128 * 6 * 6
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2), # N * 256 * 2 * 2
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=1, stride=1), # N * 512 * 2 * 2
            nn.BatchNorm2d(512),
            nn.ReLU(),
            Flatten()
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        # Decoder with Flatten (N * 2048 * 1 * 1)
        if unflatten == 0:
            
            self.fct_decode_1 = nn.Sequential(
                UnFlatten(),
                nn.ConvTranspose2d(h_dim, 128, kernel_size=5, stride=2), # N * 128 * 5 * 5
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2), # N * 64 * 13 * 13
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2), # N * 64 * 30 * 30
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.ConvTranspose2d(32, image_channels, kernel_size=6, stride=2), # N * 3 * 64 * 64
                nn.BatchNorm2d(3),
                nn.Sigmoid(),
            )

        # Decoder with Flatten1 (N * 512 * 2 * 2) and 1 more layer of convolutions
        elif unflatten == 1:
            self.fct_decode_1 = nn.Sequential(
                UnFlatten1(),                                          # N * 512 * 2 * 2
                nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2), # N * 256 * 6 * 6
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2), # N * 128 * 14 * 14
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2), # N * 64 * 30 * 30
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2), # N * 32 * 62 * 62
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.ConvTranspose2d(32, image_channels, kernel_size=3, stride=1), # N * 3 * 64 * 64 
                nn.BatchNorm2d(3),
                nn.ReLU(),
                nn.Sigmoid(),
            )
        
        self.wt = nn.Sequential()
        for i in range(self.num_wt):
            self.wt.add_module('wt{}_conv2d'.format(i), nn.Conv2d(image_channels, image_channels, kernel_size=5, stride=1, padding=2)) # N * 3 * 64 * 64
            self.wt.add_module('wt{}_bn'.format(i), nn.BatchNorm2d(image_channels))
        

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            # return torch.normal(mu, std)
            esp = torch.randn(*mu.size()).to(self.device)
            z = mu + std * esp
            return z
        else:
            return mu
    
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.fct_decode_1(z)
        z = self.wt(z)
        
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar

    def loss_function(self, wt_x, x, mu, logvar) -> Variable:
        
        wt_x = wt_x.view(-1,1,64,64)
        x_recon = iwt(wt_x, self.inv_filters, levels=3)
        x_recon = x_recon.view(-1,3,64,64)
        x_recon = x_recon.contiguous()
        
        # Loss btw reconstructed img and original img
        BCE = F.l1_loss(x_recon.view(-1, 3 * 64 * 64), x.view(-1, 3 * 64 * 64))

        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * 0.0001
        KLD /= x.shape[0] * 3 * 64 * 64

        return BCE + KLD
    
    def set_inv_filters(self, inv_filters):
        self.inv_filters = inv_filters
    
    def set_device(self, device):
        if device != 'cpu':
            self.cuda = True
        
        self.device = device


# WTVAE for 128 x 128 images
# 2 WT layers
# Using Unflatten1 for decoder (N * 512 * 2 * 2), instead of Flatten (N * 2048 * 1 * 1)
class WTVAE_128(nn.Module):
    def __init__(self, image_channels=3, h_dim=2048, z_dim=100, num_wt=2):
        super(WTVAE_128, self).__init__()

        self.inv_filters = None
        self.cuda = False
        self.device = None
        self.num_wt = num_wt
        
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2), # N * 32 * 63 * 63
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # N * 64 * 30 * 30,
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2), # N * 128 * 14 * 14
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2), # N * 256 * 6 * 6
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2), # N * 512 * 2 * 2
            nn.BatchNorm2d(512),
            nn.ReLU(),
            Flatten()
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        # Decoder
        self.fct_decode_1 = nn.Sequential(
            UnFlatten1(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2), # N * 256 * 6 * 6
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2), # N * 128 * 14 * 14
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2), # N * 64 * 30 * 30
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2), # N * 64 * 62 * 62
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=5, stride=2, output_padding=1), # N * 3 * 128 * 128
            nn.BatchNorm2d(3),
            nn.Sigmoid(),
        )
        
        self.wt1 = nn.Sequential(
            nn.Conv2d(image_channels, image_channels, kernel_size=5, stride=1, padding=2), # N * 3 * 128 * 128
            nn.BatchNorm2d(image_channels)
        )
        
        self.wt2 = nn.Sequential(
            nn.Conv2d(image_channels, image_channels, kernel_size=5, stride=1, padding=2), # N * 3 * 128 * 128
            nn.BatchNorm2d(image_channels)
        )
        
        self.wt = nn.Sequential()
        for i in range(self.num_wt):
            self.wt.add_module('wt{}_conv2d'.format(i), nn.Conv2d(image_channels, image_channels, kernel_size=5, stride=1, padding=2)) # N * 3 * 64 * 64
            self.wt.add_module('wt{}_bn'.format(i), nn.BatchNorm2d(image_channels))
        
    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            # return torch.normal(mu, std)
            esp = torch.randn(*mu.size()).to(self.device)
            z = mu + std * esp
            return z
        else:
            return mu


    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.fct_decode_1(z)
        z = self.wt(z)
        
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar
       

    def loss_function(self, wt_x, x, mu, logvar) -> Variable:
        
        wt_x = wt_x.view(-1,1,64,64)
        x_recon = iwt(wt_x, self.inv_filters, levels=3)
        x_recon = x_recon.view(-1,3,64,64)
        x_recon = x_recon.contiguous()
        
        # Loss btw reconstructed img and original img
        BCE = F.l1_loss(x_recon.view(-1, 3 * 64 * 64), x.view(-1, 3 * 64 * 64))

        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * 0.0001
        KLD /= x.shape[0] * 3 * 64 * 64

        return BCE + KLD
    def loss_function(self, wt_x, x, mu, logvar) -> Variable:
        
        wt_x = wt_x.view(-1,1,128,128)
        x_recon = iwt(wt_x, levels=2)
        x_recon = x_recon.view(-1,3,128,128)
        x_recon = x_recon.contiguous()
        
        # Loss btw reconstructed img and original img
        BCE = F.l1_loss(x_recon.view(-1, 3 * 128 * 128), x.view(-1, 3 * 128 * 128))

        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * 0.0001
        KLD /= BATCH_SIZE * 3 * 128 * 128

        return BCE + KLD

    def set_inv_filters(self, inv_filters):
        self.inv_filters = inv_filters
    
    def set_device(self, device):
        if device != 'cpu':
            self.cuda = True
        
        self.device = device

# IWT VAE for 64 x 64 images
# Assumes that 2 GPUs available
class IWTVAE_64(nn.Module):
    def __init__(self, image_channels=3, z_dim=100, bottleneck_dim=100, upsampling='linear', num_upsampling=2, reuse=False):
        super(IWTVAE_64, self).__init__()
        # Resolution of images (64 x 64)
        self.res = 64
        self.upsampling = upsampling
        self.reuse = reuse
        self.num_upsampling = num_upsampling
        self.devices = None
        self.cuda = False
        
        self.z_dim = z_dim
        self.bottleneck_dim = bottleneck_dim
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # X - Y Residual Encoder
        self.e1 = nn.Conv2d(3, 64, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 64, 32, 32]
        weights_init(self.e1)
        self.instance_norm_e1 = nn.InstanceNorm2d(num_features=64, affine=False)

        self.e2 = nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 128, 16, 16]
        weights_init(self.e2)
        self.instance_norm_e2 = nn.InstanceNorm2d(num_features=128, affine=False)

        self.e3 = nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 256, 8, 8]
        weights_init(self.e3)
        self.instance_norm_e3 = nn.InstanceNorm2d(num_features=256, affine=False)

        self.e4 = nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 512, 4, 4]
        weights_init(self.e4)
        self.instance_norm_e4 = nn.InstanceNorm2d(num_features=512, affine=False)
        
        self.fc_enc = nn.Linear(512 * 4 * 4, 256)
        weights_init(self.fc_enc)
        
        self.fc_mean = nn.Linear(256, z_dim)
        weights_init(self.fc_mean)
        
        self.fc_var = nn.Linear(256, z_dim)
        weights_init(self.fc_var)
        
        # IWT Decoder        
        self.d1 = get_upsampling_layer(self.upsampling, self.res, self.bottleneck_dim)
        weights_init(self.d1)
        self.mu1 = nn.Linear(z_dim, 3 * 64 * 64)
        self.var1 = nn.Linear(z_dim, 3 * 64 * 64)
        self.instance_norm_d1 = nn.InstanceNorm2d(num_features=3, affine=False)
        self.iwt1 = nn.ConvTranspose2d(image_channels, image_channels, kernel_size=5, stride=1, padding=2)
        
        # Only instantiate if # of upsampling > 1, and set d2 to d1 if re-using upsampling layer
        if self.num_upsampling > 1:
            if self.reuse:
                self.d2 = self.d1
            else:
                self.d2 = get_upsampling_layer(self.upsampling, self.res, self.bottleneck_dim)
                weights_init(self.d2)

            self.mu2 = nn.Linear(z_dim, 3 * 64 * 64)
            self.var2 = nn.Linear(z_dim, 3 * 64 * 64)
            self.instance_norm_d2 = nn.InstanceNorm2d(num_features=3, affine=False)
            self.iwt2 = nn.ConvTranspose2d(image_channels, image_channels, kernel_size=5, stride=1, padding=2)
      
    def encode(self, x, y):
        h = self.leakyrelu(self.instance_norm_e1(self.e1(x)))   #[b, 64, 32, 32]
        h = self.leakyrelu(self.instance_norm_e2(self.e2(h)))     #[b, 128, 16, 16]
        h = self.leakyrelu(self.instance_norm_e3(self.e3(h)))     #[b, 256, 8, 8]
        h = self.leakyrelu(self.instance_norm_e4(self.e4(h)))     #[b, 512, 4, 4]
        h = self.leakyrelu(self.fc_enc(h.view(-1,512*4*4)))       #[b, 512 * 4 * 4]
        
        return self.fc_mean(h), F.softplus(self.fc_var(h))        #[b, z_dim]

    def reparameterize(self, mu, var):
        std = torch.sqrt(var)
        if self.cuda:
            eps = torch.FloatTensor(std.size()).normal_().to(self.devices[0])
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu) 
    
    def decode(self, y, z):
        upsampling_sizes = get_upsampling_dims(self.upsampling, self.res)
        mu = self.mu1(z).reshape(-1, 3, 64, 64)
        var = self.var1(z).reshape(-1, 3, 64, 64)
        h = self.leakyrelu(var*self.instance_norm_d1(self.d1(y.view(upsampling_sizes)).reshape(-1, 3, 64, 64) + mu)) #[b, 3, 64, 64]
        h = self.leakyrelu(self.iwt1(h))                               #[b, 3, 64, 64]
        
        if self.num_upsampling > 1:
            mu = self.mu2(z).reshape(-1, 3, 64, 64)
            var = self.var2(z).reshape(-1, 3, 64, 64)
            h = self.leakyrelu(var*self.instance_norm_d2(self.d2(h.view(upsampling_sizes)).reshape(-1, 3, 64, 64) + mu)) #[b, 3, 64, 64]
            h = self.leakyrelu(self.iwt2(h))                               #[b, 3, 64, 64]
        
        return self.sigmoid(h)
        
        
    def forward(self, x, y):
        mu, var = self.encode(x, y)
        if self.training:
            z = self.reparameterize(mu, var)
        else:
            z = mu
        x_hat = self.decode(y, z)
        
        return x_hat, mu, var
        
    def loss_function(self, x, x_hat, mu, var) -> Variable:
        
        # Loss btw reconstructed img and original img
        BCE = F.mse_loss(x_hat.view(-1), x.view(-1))
        
        logvar = torch.log(var)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * 0.01
#         KLD /= x.shape[0] * 3 * 64 * 64

        return BCE + KLD

    def set_devices(self, devices):
        self.devices = devices
        if 'cuda' in self.devices[0] and 'cuda' in self.devices[1]:
            self.cuda = True

class IWTVAE_64_FreezeIWT(nn.Module):
    def __init__(self, image_channels=3, z_dim=100, bottleneck_dim=0, upsampling='linear', num_upsampling=2, reuse=False):
        super(IWTVAE_64_FreezeIWT, self).__init__()
        # Resolution of images (64 x 64)
        self.res = 64
        self.upsampling = upsampling
        self.reuse = reuse
        self.num_upsampling = num_upsampling
        self.devices = None
        self.cuda = False
        
        self.z_dim = z_dim
        self.bottleneck_dim = bottleneck_dim
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # X - Y Residual Encoder
        self.e1 = nn.Conv2d(3, 64, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 64, 32, 32]
        weights_init(self.e1)
        self.instance_norm_e1 = nn.InstanceNorm2d(num_features=64, affine=False)

        self.e2 = nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 128, 16, 16]
        weights_init(self.e2)
        self.instance_norm_e2 = nn.InstanceNorm2d(num_features=128, affine=False)

        self.e3 = nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 256, 8, 8]
        weights_init(self.e3)
        self.instance_norm_e3 = nn.InstanceNorm2d(num_features=256, affine=False)

        self.e4 = nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 512, 4, 4]
        weights_init(self.e4)
        self.instance_norm_e4 = nn.InstanceNorm2d(num_features=512, affine=False)
        
        self.fc_enc = nn.Linear(512 * 4 * 4, 256)
        weights_init(self.fc_enc)
        
        self.fc_mean = nn.Linear(256, z_dim)
        weights_init(self.fc_mean)
        
        self.fc_var = nn.Linear(256, z_dim)
        weights_init(self.fc_var)
        
        # IWT Decoder        
        self.d1 = get_upsampling_layer(self.upsampling, self.res, self.bottleneck_dim)
        weights_init(self.d1)
        self.mu1 = nn.Linear(z_dim, 3 * 64 * 64)
        self.var1 = nn.Linear(z_dim, 3 * 64 * 64)
        self.instance_norm_d1 = nn.InstanceNorm2d(num_features=3, affine=False)
        self.iwt1 = IWT()
        
        # Only instantiate if # of upsampling > 1, and set d2 to d1 if re-using upsampling layer
        if self.num_upsampling > 1:
            if self.reuse:
                self.d2 = self.d1
            else:
                self.d2 = get_upsampling_layer(self.upsampling, self.res, self.bottleneck_dim)
                weights_init(self.d2)

            self.mu2 = nn.Linear(z_dim, 3 * 64 * 64)
            self.var2 = nn.Linear(z_dim, 3 * 64 * 64)
            self.instance_norm_d2 = nn.InstanceNorm2d(num_features=3, affine=False)
            self.iwt2 = self.iwt1
      
    def encode(self, x, y):
        h = self.leakyrelu(self.instance_norm_e1(self.e1(x)))   #[b, 64, 32, 32]
        h = self.leakyrelu(self.instance_norm_e2(self.e2(h)))     #[b, 128, 16, 16]
        h = self.leakyrelu(self.instance_norm_e3(self.e3(h)))     #[b, 256, 8, 8]
        h = self.leakyrelu(self.instance_norm_e4(self.e4(h)))     #[b, 512, 4, 4]
        h = self.leakyrelu(self.fc_enc(h.view(-1,512*4*4)))       #[b, 512 * 4 * 4]
        
        return self.fc_mean(h), F.softplus(self.fc_var(h))        #[b, z_dim]

    def reparameterize(self, mu, var):
        std = torch.sqrt(var)
        if self.cuda:
            eps = torch.FloatTensor(std.size()).normal_().to(self.devices[0])
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu) 
    
    def decode(self, y, z):
        upsampling_sizes = get_upsampling_dims(self.upsampling, self.res)
        mu = self.mu1(z).reshape(-1, 3, 64, 64)
        var = self.var1(z).reshape(-1, 3, 64, 64)
        h = self.leakyrelu(var*self.instance_norm_d1(self.d1(y.reshape(upsampling_sizes)).reshape(-1, 3, 64, 64) + mu)) #[b, 3, 64, 64]
        h = self.iwt1(h)                               #[b, 3, 64, 64]
        
        if self.num_upsampling > 1:
            mu = self.mu2(z).reshape(-1, 3, 64, 64)
            var = self.var2(z).reshape(-1, 3, 64, 64)
            h = self.leakyrelu(var*self.instance_norm_d2(self.d2(h.reshape(upsampling_sizes)).reshape(-1, 3, 64, 64) + mu)) #[b, 3, 64, 64]
            h = self.iwt2(h)                               #[b, 3, 64, 64]
        
        return h
        
    def forward(self, x, y):
        mu, var = self.encode(x, y)
        if self.training:
            z = self.reparameterize(mu, var)
        else:
            z = mu
        x_hat = self.decode(y, z)
        
        return x_hat, mu, var
        
    def loss_function(self, x, x_hat, mu, var) -> Variable:
        
        # Loss btw reconstructed img and original img
        BCE = F.mse_loss(x_hat.reshape(-1), x.reshape(-1))
        
        logvar = torch.log(var)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * 0.01
#         KLD /= x.shape[0] * 3 * 64 * 64

        return BCE + KLD

    def set_devices(self, devices):
        self.devices = devices
        if 'cuda' in self.devices[0] and 'cuda' in self.devices[1]:
            self.cuda = True

# IWT VAE for 64 x 64 images
# Assumes that 2 GPUs available
class IWTVAE_64_Bottleneck(nn.Module):
    def __init__(self, image_channels=3, z_dim=100, bottleneck_dim=100):
        super(IWTVAE_64_Bottleneck, self).__init__()
        # Resolution of images (64 x 64)
        self.res = 64
        self.devices = None
        self.cuda = False
        
        self.z_dim = z_dim
        self.bottleneck_dim = bottleneck_dim
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # X - Y Residual Encoder
        self.e1 = nn.Conv2d(3, 64, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 64, 32, 32]
        weights_init(self.e1)
        self.instance_norm_e1 = nn.InstanceNorm2d(num_features=64, affine=False)

        self.e2 = nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 128, 16, 16]
        weights_init(self.e2)
        self.instance_norm_e2 = nn.InstanceNorm2d(num_features=128, affine=False)

        self.e3 = nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 256, 8, 8]
        weights_init(self.e3)
        self.instance_norm_e3 = nn.InstanceNorm2d(num_features=256, affine=False)

        self.e4 = nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 512, 4, 4]
        weights_init(self.e4)
        self.instance_norm_e4 = nn.InstanceNorm2d(num_features=512, affine=False)
        
        self.fc_enc = nn.Linear(512 * 4 * 4, 256)
        weights_init(self.fc_enc)
        
        self.fc_mean = nn.Linear(256, z_dim)
        weights_init(self.fc_mean)
        
        self.fc_var = nn.Linear(256, z_dim)
        weights_init(self.fc_var)
        
        # IWT Decoder        
        self.d1 = get_upsampling_layer('bottleneck', self.res)
        weights_init(self.d1)
        self.mu1 = nn.Linear(z_dim, 3 * 64 * 64)
        self.var1 = nn.Linear(z_dim, 3 * 64 * 64)
        self.instance_norm_d1 = nn.InstanceNorm2d(num_features=3, affine=False)
        self.iwt1 = nn.ConvTranspose2d(image_channels, image_channels, kernel_size=5, stride=1, padding=2)
        
        # Only instantiate if # of upsampling > 1, and set d2 to d1 if re-using upsampling layer
        # if self.num_upsampling > 1:
        #     if self.reuse:
        #         self.d2 = self.d1
        #     else:
        #         self.d2 = get_upsampling_layer(self.upsampling, self.res, self.bottleneck_dim)
        #         weights_init(self.d2)

        # self.mu2 = nn.Linear(z_dim, 3 * 64 * 64)
        # self.var2 = nn.Linear(z_dim, 3 * 64 * 64)
        # self.instance_norm_d2 = nn.InstanceNorm2d(num_features=3, affine=False)
        self.iwt2 = nn.ConvTranspose2d(image_channels, image_channels, kernel_size=5, stride=1, padding=2)
      
    def encode(self, x, y):
        h = self.leakyrelu(self.instance_norm_e1(self.e1(x)))   #[b, 64, 32, 32]
        h = self.leakyrelu(self.instance_norm_e2(self.e2(h)))     #[b, 128, 16, 16]
        h = self.leakyrelu(self.instance_norm_e3(self.e3(h)))     #[b, 256, 8, 8]
        h = self.leakyrelu(self.instance_norm_e4(self.e4(h)))     #[b, 512, 4, 4]
        h = self.leakyrelu(self.fc_enc(h.view(-1,512*4*4)))       #[b, 512 * 4 * 4]
        
        return self.fc_mean(h), F.softplus(self.fc_var(h))        #[b, z_dim]

    def reparameterize(self, mu, var):
        std = torch.sqrt(var)
        if self.cuda:
            eps = torch.FloatTensor(std.size()).normal_().to(self.devices[0])
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu) 
    
    def decode(self, y, z):
        mu = self.mu1(z).reshape(-1, 3, 64, 64)
        var = self.var1(z).reshape(-1, 3, 64, 64)
        h = self.leakyrelu(var*self.instance_norm_d1(y + mu)) #[b, 3, 64, 64]
        h = self.d1(h.reshape(-1, 3*64*64))                                                 #[b, 3*64*64]
        h = self.leakyrelu(self.iwt1(h.reshape(-1, 3, 64, 64)))                               #[b, 3, 64, 64]
        
        # mu = self.mu2(z).reshape(-1, 3, 64, 64)
        # var = self.var2(z).reshape(-1, 3, 64, 64)
        # h = self.leakyrelu(var*self.instance_norm_d2(self.d2(h.view(upsampling_sizes)).reshape(-1, 3, 64, 64) + mu)) #[b, 3, 64, 64]
        h = self.leakyrelu(self.iwt2(h))                               #[b, 3, 64, 64]
        
        return self.sigmoid(h)
        
        
    def forward(self, x, y):
        mu, var = self.encode(x, y)
        if self.training:
            z = self.reparameterize(mu, var)
        else:
            z = mu
        x_hat = self.decode(y, z)
        
        return x_hat, mu, var
        
    def loss_function(self, x, x_hat, mu, var) -> Variable:
        
        # Loss btw reconstructed img and original img
        BCE = F.mse_loss(x_hat.view(-1), x.view(-1))
        
        logvar = torch.log(var)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * 0.01
#         KLD /= x.shape[0] * 3 * 64 * 64

        return BCE + KLD

    def set_devices(self, devices):
        self.devices = devices
        if 'cuda' in self.devices[0] and 'cuda' in self.devices[1]:
            self.cuda = True


# IWT VAE for 64 x 64 images
# Assumes that 2 GPUs available
class IWTVAE_64_Mask(nn.Module):
    def __init__(self, image_channels=3, z_dim=100, num_upsampling=2, reuse=False):
        super(IWTVAE_64_Mask, self).__init__()
        # Resolution of images (64 x 64)
        self.res = 64
        self.reuse = reuse
        self.num_upsampling = num_upsampling
        self.devices = None
        self.cuda = False
        
        self.z_dim = z_dim
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # X - Y Residual Encoder
        self.e1 = nn.Conv2d(3, 64, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 64, 32, 32]
        weights_init(self.e1)
        self.instance_norm_e1 = nn.InstanceNorm2d(num_features=64, affine=False)

        self.e2 = nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 128, 16, 16]
        weights_init(self.e2)
        self.instance_norm_e2 = nn.InstanceNorm2d(num_features=128, affine=False)

        self.e3 = nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 256, 8, 8]
        weights_init(self.e3)
        self.instance_norm_e3 = nn.InstanceNorm2d(num_features=256, affine=False)

        self.e4 = nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 512, 4, 4]
        weights_init(self.e4)
        self.instance_norm_e4 = nn.InstanceNorm2d(num_features=512, affine=False)
        
        self.fc_enc = nn.Linear(512 * 4 * 4, 256)
        weights_init(self.fc_enc)
        
        self.fc_mean = nn.Linear(256, z_dim)
        weights_init(self.fc_mean)
        
        self.fc_var = nn.Linear(256, z_dim)
        weights_init(self.fc_var)
        
        # IWT Decoder
        self.d1 = Mask()    
        self.mu1 = nn.Linear(z_dim, 64 * 64)
        self.var1 = nn.Linear(z_dim, 3 * 64 * 64)
        self.instance_norm_d1 = nn.InstanceNorm2d(num_features=3, affine=False)
        self.iwt1 = nn.ConvTranspose2d(image_channels, image_channels, kernel_size=5, stride=1, padding=2)
        
        # Only instantiate if # of upsampling > 1, and set d2 to d1 if re-using upsampling layer
        if self.num_upsampling > 1:
            if self.reuse:
                self.d2 = self.d1
            else:
                self.d2 = Mask()

            self.mu2 = nn.Linear(z_dim, 64 * 64)
            self.var2 = nn.Linear(z_dim, 3 * 64 * 64)
            self.instance_norm_d2 = nn.InstanceNorm2d(num_features=3, affine=False)
            self.iwt2 = nn.ConvTranspose2d(image_channels, image_channels, kernel_size=5, stride=1, padding=2)
      
    def encode(self, x, y):
        h = self.leakyrelu(self.instance_norm_e1(self.e1(x-y)))   #[b, 64, 32, 32]
        h = self.leakyrelu(self.instance_norm_e2(self.e2(h)))     #[b, 128, 16, 16]
        h = self.leakyrelu(self.instance_norm_e3(self.e3(h)))     #[b, 256, 8, 8]
        h = self.leakyrelu(self.instance_norm_e4(self.e4(h)))     #[b, 512, 4, 4]
        h = self.leakyrelu(self.fc_enc(h.view(-1,512*4*4)))       #[b, 512 * 4 * 4]
        
        return self.fc_mean(h), F.softplus(self.fc_var(h))        #[b, z_dim]

    def reparameterize(self, mu, var):
        std = torch.sqrt(var)
        if self.cuda:
            eps = torch.FloatTensor(std.size()).normal_().to(self.devices[0])
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu) 
    
    def decode(self, y, z):
        mu = self.mu1(z).reshape(-1, 64, 64)
        var = self.var1(z).reshape(-1, 3, 64, 64)
        h = self.leakyrelu(var*self.instance_norm_d1(self.d1(y, mu))) #[b, 3, 64, 64]
        h = self.leakyrelu(self.iwt1(h))                               #[b, 3, 64, 64]
        
        if self.num_upsampling > 1:
            mu = self.mu2(z).reshape(-1, 64, 64)
            var = self.var2(z).reshape(-1, 3, 64, 64)
            h = self.leakyrelu(var*self.instance_norm_d2(self.d2(y, mu))) #[b, 3, 64, 64]
            h = self.leakyrelu(self.iwt2(h))                               #[b, 3, 64, 64]
        
        return self.sigmoid(h)
        
        
    def forward(self, x, y):
        mu, var = self.encode(x, y)
        if self.training:
            z = self.reparameterize(mu, var)
        else:
            z = mu
        x_hat = self.decode(y, z)
        
        return x_hat, mu, var
        
    def loss_function(self, x, x_hat, mu, var) -> Variable:
        
        # Loss btw reconstructed img and original img
        BCE = F.mse_loss(x_hat.view(-1), x.view(-1))
        
        logvar = torch.log(var)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * 0.01
#         KLD /= x.shape[0] * 3 * 64 * 64

        return BCE + KLD

    def set_devices(self, devices):
        self.devices = devices
        if 'cuda' in self.devices[0] and 'cuda' in self.devices[1]:
            self.cuda = True

class IWTVAE_512_Mask(nn.Module):
    def __init__(self, image_channels=3, z_dim=500, num_iwt=2):
        super(IWTVAE_512_Mask, self).__init__()
        # Resolution of images (512 x 512)
        self.res = 512
        self.devices = None
        self.cuda = False
        
        self.z_dim = z_dim
        self.num_iwt = num_iwt
        self.leakyrelu = nn.LeakyReLU(0.2)

        # Z Encoder - Decoder                                                                [b, 3, 512, 512]
        self.e1 = nn.Conv2d(3, 64, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 64, 256, 256]
        weights_init(self.e1)
        self.instance_norm_e1 = nn.InstanceNorm2d(num_features=64, affine=False)

        self.e2 = nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 128, 128, 128]
        weights_init(self.e2)
        self.instance_norm_e2 = nn.InstanceNorm2d(num_features=128, affine=False)

        self.m1 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1, return_indices=True) #[b, 128, 64, 64]

        self.e3 = nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 256, 32, 32]
        weights_init(self.e3)
        self.instance_norm_e3 = nn.InstanceNorm2d(num_features=256, affine=False)

        self.e4 = nn.Conv2d(256, 512, 4, stride=2, padding=1, bias=True, padding_mode='zeros') #[b, 512, 16, 16]
        weights_init(self.e4)
        self.instance_norm_e4 = nn.InstanceNorm2d(num_features=512, affine=False)

        self.m2 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1, return_indices=True) #[b, 512, 8, 8]
        
        self.fc_enc = nn.Linear(512 * 8 * 8, 1024)
        weights_init(self.fc_enc)
        
        self.fc_mean = nn.Linear(1024, z_dim)
        weights_init(self.fc_mean)
        
        self.fc_var = nn.Linear(1024, z_dim)
        weights_init(self.fc_var)
        
        self.fc_dec = nn.Linear(z_dim, 512 * 8 * 8)
        weights_init(self.fc_dec)

        self.u1 = nn.MaxUnpool2d(kernel_size=4, stride=2, padding=1) #[b, 512, 16, 16]

        self.d1 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=True) #[b, 256, 32, 32]
        weights_init(self.d1)
        self.instance_norm_d1 = nn.InstanceNorm2d(num_features=256, affine=False)

        self.d2= nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=True) #[b, 128, 64, 64]
        weights_init(self.d2)
        self.instance_norm_d2 = nn.InstanceNorm2d(num_features=128, affine=False)
    
        self.u2 = nn.MaxUnpool2d(kernel_size=4, stride=2, padding=1) #[b, 128, 128, 128]

        self.d3 = nn.ConvTranspose2d(128, 32, 4, stride=2, padding=1, bias=True) #[b, 32, 256, 256]
        weights_init(self.d3)
        self.instance_norm_d3 = nn.InstanceNorm2d(num_features=32, affine=False)

        self.d4 = nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1, bias=True) #[b, 1, 512, 512]
        weights_init(self.d4)
        self.instance_norm_d4 = nn.InstanceNorm2d(num_features=3, affine=False)
        
        self.iwt = IWT()
    
      
    def encode(self, x, y):
        h = self.leakyrelu(self.instance_norm_e1(self.e1(x)))                       #[b, 64, 256, 256]
        h = self.leakyrelu(self.instance_norm_e2(self.e2(h)))                       #[b, 128, 128, 128]
        h, m1_idx = self.m1(h)                                                      #[b, 128, 64, 64]
        h = self.leakyrelu(h)                                                       
        h = self.leakyrelu(self.instance_norm_e3(self.e3(h)))                       #[b, 256, 32, 32]
        h = self.leakyrelu(self.instance_norm_e4(self.e4(h)))                       #[b, 512, 16, 16]

        h, m2_idx = self.m2(h)                                                      #[b, 512, 8, 8]
        h = self.leakyrelu(h)
        h = self.leakyrelu(self.fc_enc(h.reshape(-1,512*8*8)))                      #[b, z_dim]

        return self.fc_mean(h), F.softplus(self.fc_var(h)), m1_idx, m2_idx          #[b, z_dim]
    
    def reparameterize(self, mu, var):
        std = torch.sqrt(var)
        if self.cuda:
            eps = torch.FloatTensor(std.size()).normal_().to(self.devices[0])
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)

        return eps.mul(std).add_(mu) 
    
    def decode(self, y, z, m1_idx, m2_idx):
        h = self.leakyrelu(self.fc_dec(z))                                      #[b, 512*8*8]
        h = self.leakyrelu(self.u1(h.reshape(-1, 512, 8, 8), indices=m2_idx))   #[b, 512, 16, 16]
        h = self.leakyrelu(self.instance_norm_d1(self.d1(h)))                   #[b, 256, 32, 32]
        h = self.leakyrelu(self.instance_norm_d2(self.d2(h)))                   #[b, 128, 64, 64]
        h = self.leakyrelu(self.u2(h, indices=m1_idx))                          #[b, 128, 128, 128]
        h = self.leakyrelu(self.instance_norm_d3(self.d3(h)))                   #[b, 32, 256, 512]
        h = self.leakyrelu(self.instance_norm_d4(self.d4(h)))                   #[b, 1, 256, 512]

        for i in range(self.num_iwt):
            with torch.no_grad():
                mask = h.clone().detach()
                mask = zero_mask(mask.squeeze(1), self.num_iwt, i+1)
            h = y - mask.unsqueeze(1)
            h = self.iwt(h)
        
        return h
        
    def forward(self, x, y):
        mu, var, m1_idx, m2_idx = self.encode(x, y)
        if self.training:
            z = self.reparameterize(mu, var)
        else:
            z = mu
        x_hat = self.decode(y, z, m1_idx, m2_idx)
        
        return x_hat, mu, var
        
    def loss_function(self, x, x_hat, mu, var) -> Variable:
        
        # Loss btw reconstructed img and original img
        BCE = F.l1_loss(x_hat.reshape(-1), x.reshape(-1))
        
        logvar = torch.log(var)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * 0.01
        # KLD /= x.shape[0] * 3 * 64 * 64

        return BCE + KLD, BCE, KLD

    def set_devices(self, devices):
        self.devices = devices
        if 'cuda' in self.devices[0] and 'cuda' in self.devices[1]:
            self.cuda = True