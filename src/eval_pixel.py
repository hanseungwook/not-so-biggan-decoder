import time
import torch.nn.functional as F
from torchvision.utils import save_image
import wandb
from tqdm import tqdm, trange
import h5py

from UNET_utils import *


# Creating HDF5 dataset of reconstructed using real 64x64 patch (input is 256x256 real image)
def eval_unet_128_256(model_128, model_256, data_loader, data_type, args):
    model_128.eval()
    model_256.eval()


    # Create hdf5 dataset
    f1 = h5py.File(args.output_dir + data_type + '/recon_img.hdf5', 'w')
    f2 = h5py.File(args.output_dir + data_type + '/real_img.hdf5', 'w')

    recon_dataset = f1.create_dataset('data', shape=(50000, 3, 256, 256), dtype=np.float32, fillvalue=0)
    real_dataset = f2.create_dataset('data', shape=(50000, 3, 256, 256), dtype=np.float32, fillvalue=0)

    counter = 0

    with torch.no_grad():
        for data, _ in tqdm(data_loader):
            if counter >= 50000:
                break
            
            #Downsample to 64, upsample to 128, and then pass through first UNET
            data = data.to(args.device)
            x = data

            x = F.interpolate(x, 64, mode='bilinear')
            y_128 = F.interpolate(x, 128, mode="bilinear")
            y_128 = y_128.to(args.device)
            x_mask_hat_128 = model_128(y_128)
            x_hat_128 = y_128 + x_mask_hat_128

            #Upsample to 256 then pass through second UNET
            y_256 = F.interpolate(x_hat_128, 256, mode="bilinear")
            y_256 = y_256.to(args.device)
            x_mask_hat_256 = model_256(y_256)
            x_hat_256 = y_256 + x_mask_hat_256

            #set final image equal to output of both UNETs
            recon_img = x_hat_256

            # Save image into hdf5
            batch_size = recon_img.shape[0]
            recon_dataset[counter: counter+batch_size] = recon_img.cpu()
            real_dataset[counter: counter+batch_size] = data.cpu()
            counter += batch_size

    f1.close()

# Creating HDF5 dataset of real and reconstructed, given real 64x64 TL patch
def eval_biggan_unet_128_256(model_128, model_256, data_loader, args):
    model_128.eval()
    model_256.eval()


    # Create hdf5 dataset
    f1 = h5py.File(args.output_dir + '/recon_img.hdf5', 'w')
    f2 = h5py.File(args.output_dir + '/real_sample.hdf5', 'w')

    recon_dataset = f1.create_dataset('data', shape=(50000, 3, 256, 256), dtype=np.float32, fillvalue=0)
    real_sample_dataset = f2.create_dataset('data', shape=(50000, 3, 64, 64), dtype=np.float32, fillvalue=0)

    counter = 0

    with torch.no_grad():
        for data in tqdm(data_loader):
            if counter >= 50000:
                break
            
            #Downsample to 64, upsample to 128, and then pass through first UNET
            data = data.to(args.device)
            x = data
            y_128 = F.interpolate(x, 128, mode="bilinear")
            y_128 = y_128.to(args.device)
            x_mask_hat_128 = model_128(y_128)
            x_hat_128 = y_128 + x_mask_hat_128

            #Upsample to 256 then pass through second UNET
            y_256 = F.interpolate(x_hat_128, 256, mode="bilinear")
            y_256 = y_256.to(args.device)
            x_mask_hat_256 = model_256(y_256)
            x_hat_256 = y_256 + x_mask_hat_256

            #set final image equal to output of both UNETs
            recon_img = x_hat_256

            # Save image into hdf5
            batch_size = recon_img.shape[0]
            recon_dataset[counter: counter+batch_size] = recon_img.cpu()
            real_sample_dataset[counter: counter+batch_size] = data.cpu()
            counter += batch_size

    f1.close()
    f2.close()

# Creating HDF5 dataset of real and reconstructed, given sample from pretrained 256x256
def eval_pretrained_biggan_unet_128_256(model_128, model_256, data_loader, args):
    model_128.eval()
    model_256.eval()


    # Create hdf5 dataset
    f1 = h5py.File(args.output_dir + '/recon_img.hdf5', 'w')
    f2 = h5py.File(args.output_dir + '/real_sample.hdf5', 'w')

    recon_dataset = f1.create_dataset('data', shape=(50000, 3, 256, 256), dtype=np.float32, fillvalue=0)
    real_sample_dataset = f2.create_dataset('data', shape=(50000, 3, 256, 256), dtype=np.float32, fillvalue=0)

    counter = 0

    with torch.no_grad():
        for data in tqdm(data_loader):
            if counter >= 50000:
                break
            
            #Downsample to 64, upsample to 128, and then pass through first UNET
            data = data.to(args.device)

            # Added step for pretrained
            x = F.interpolate(data, 64, mode="bilinear")
            y_128 = F.interpolate(x, 128, mode="bilinear")
            y_128 = y_128.to(args.device)
            x_mask_hat_128 = model_128(y_128)
            x_hat_128 = y_128 + x_mask_hat_128

            #Upsample to 256 then pass through second UNET
            y_256 = F.interpolate(x_hat_128, 256, mode="bilinear")
            y_256 = y_256.to(args.device)
            x_mask_hat_256 = model_256(y_256)
            x_hat_256 = y_256 + x_mask_hat_256

            #set final image equal to output of both UNETs
            recon_img = x_hat_256

            # Save image into hdf5
            batch_size = recon_img.shape[0]
            recon_dataset[counter: counter+batch_size] = recon_img.cpu()
            real_sample_dataset[counter: counter+batch_size] = data.cpu()
            counter += batch_size

    f1.close()
    f2.close()