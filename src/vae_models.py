import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision.utils import save_image
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


def iwt(vres, inv_filters, levels=1):
    h = vres.size(2)
    w = vres.size(3)
    res = vres.contiguous().view(-1, h//2, 2, w//2).transpose(1, 2).contiguous().view(-1, 4, h//2, w//2).clone()
    if levels > 1:
        res[:,:1] = iwt(res[:,:1], levels=levels-1)
    res = torch.nn.functional.conv_transpose2d(res, Variable(inv_filters[:,None]),stride=2)
    res = res[:,:,2:-2,2:-2] #removing padding

    return res

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
class UnFlatten(nn.Module):
    def forward(self, input, size=2048):
        return input.view(input.size(0), size, 1, 1)

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
            esp = torch.randn(*mu.size()).cuda()
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
    def __init__(self, image_channels=3, h_dim=2048, z_dim=100, unflatten=0):
        super(WTVAE_128, self).__init__()
        
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
        
#         self.wt3 = nn.Sequential(
#             nn.Conv2d(image_channels, image_channels, kernel_size=5, stride=1, padding=2), # N * 3 * 64 * 64
#             nn.BatchNorm2d(image_channels)
#         )
        

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            # return torch.normal(mu, std)
            esp = torch.randn(*mu.size()).cuda()
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
        z = self.wt1(z)
        z = self.wt2(z)
#         z = self.wt3(z)
        
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar
       
#         z_final = Variable(torch.stack([z_1,z_2,z_3,z_4], dim=1))
#         z_final = z_final.view(-1,2,128//2,128//2).transpose(1,2).contiguous().view(-1,1,128,128)

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

class IWTVAE_64(nn.Module):
    def __init__(self, image_channels=3, z_dim=100, device=None):
        super(IWTVAE_64, self).__init__()
        
        if device is None:
            self.cuda = False
            self.device = None
        else:
            self.device = device
            self.cuda = True
        
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
        self.d1 = nn.Linear(3 * 64 * 64, 3 * 64 * 64)
        weights_init(self.d1)
        self.mu1 = nn.Linear(z_dim, 3 * 64 * 64)
        self.var1 = nn.Linear(z_dim, 3 * 64 * 64)
        self.instance_norm_d1 = nn.InstanceNorm2d(num_features=3, affine=False)
        self.iwt1 = nn.ConvTranspose2d(image_channels, image_channels, kernel_size=5, stride=1, padding=2)
        
        self.d2 = nn.Linear(3 * 64 * 64, 3 * 64 * 64)
        weights_init(self.d2)
        self.mu2 = nn.Linear(z_dim, 3 * 64 * 64)
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
            eps = torch.FloatTensor(std.size()).normal_().to(self.device)
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu) 
    
    def decode(self, y, z):
        mu = self.mu1(z).reshape(-1, 3, 64, 64)
        var = self.var1(z).reshape(-1, 3, 64, 64)
        h = self.leakyrelu(var*self.instance_norm_d1(self.d1(y.view(y.shape[0], -1)).reshape(-1, 3, 64, 64)) + mu) #[b, 3, 64, 64]
        h = self.leakyrelu(self.iwt1(h))                               #[b, 3, 64, 64]
        
        mu = self.mu2(z).reshape(-1, 3, 64, 64)
        var = self.var2(z).reshape(-1, 3, 64, 64)
        h = self.leakyrelu(var*self.instance_norm_d2(self.d2(h.view(h.shape[0], -1)).reshape(-1, 3, 64, 64)) + mu) #[b, 3, 64, 64]
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
        
        
    def loss_function(self, x, x_hat) -> Variable:
        
        # Loss btw reconstructed img and original img
        BCE = F.mse_loss(x_hat.view(-1), x.view(-1))
        
        logvar = torch.log(var)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * 0.01
        KLD /= x.shape[0] * 3 * 64 * 64

        return BCE + KLD
        
        
        