import time
import torch.nn.functional as F
from torchvision.utils import save_image
import wandb
from tqdm import tqdm, trange
import h5py

from wt_utils import *

# Run model through dataloader and save all images
def eval_unet128(model, data_loader, data_type, args):
    model.eval()

    # Create filters
    filters = create_filters(device=args.device)
    inv_filters = create_inv_filters(device=args.device)

    # Create hdf5 dataset
    f1 = h5py.File(args.output_dir + data_type + '/recon_img.hdf5', 'w')
    f2 = h5py.File(args.output_dir + data_type + '/real_img.hdf5', 'w')
    f3 = h5py.File(args.output_dir + data_type + '/low_64_img.hdf5', 'w')

    recon_dataset = f1.create_dataset('data', shape=(50000, 3, 256, 256), dtype=np.float32, fillvalue=0)
    real_dataset = f2.create_dataset('data', shape=(50000, 3, 256, 256), dtype=np.float32, fillvalue=0)
    low_dataset = f3.create_dataset('data', shape=(50000, 3, 256, 256), dtype=np.float32, fillvalue=0)

    counter = 0

    for data, _ in tqdm(data_loader):
        if counter >= 50000:
            break
        data = data.to(args.device)
    
        Y = wt_128_3quads(data, filters, levels=3)

        # Get real 1st level masks
        Y_64 = Y[:, :, :64, :64]
        real_mask_64_tl, real_mask_64_tr, real_mask_64_bl, real_mask_64_br = get_4masks(Y_64, 32)
        Y_64_patches = torch.cat((real_mask_64_tl, real_mask_64_tr, real_mask_64_bl, real_mask_64_br), dim=1)

        with torch.no_grad():
            # Run through 128 mask network and get reconstructed image
            recon_mask_all = model(Y_64_patches)
            recon_mask_tr, recon_mask_bl, recon_mask_br = split_masks_from_channels(recon_mask_all)

        Y_real = wt(data, filters, levels=3)
        
        real_img_128_padded = Y_real[:, :, :128, :128]
        real_img_128_padded = zero_pad(real_img_128_padded, 256, args.device)
        real_img_128_padded = iwt(real_img_128_padded, inv_filters, levels=3)

        # Collate all masks concatenated by channel to an image (slice up and put into a square)
        recon_mask_tr_img = collate_channels_to_img(recon_mask_tr, args.device)
        recon_mask_bl_img = collate_channels_to_img(recon_mask_bl, args.device)   
        recon_mask_br_img = collate_channels_to_img(recon_mask_br, args.device)
        
        recon_mask_tr_img = iwt(recon_mask_tr_img, inv_filters, levels=1)
        recon_mask_bl_img = iwt(recon_mask_bl_img, inv_filters, levels=1)    
        recon_mask_br_img = iwt(recon_mask_br_img, inv_filters, levels=1) 

        zeros = torch.zeros(recon_mask_tr_img.shape)
        
        recon_mask_iwt = collate_patches_to_img(zeros, recon_mask_tr_img, recon_mask_bl_img, recon_mask_br_img)
        
        recon_mask_padded = zero_pad(recon_mask_iwt, 256, args.device)
        recon_mask_padded[:, :, :64, :64] = Y_64
        recon_img = iwt(recon_mask_padded, inv_filters, levels=3)
    
        # Reconstructed image with only 128x128
        Y_64_low = zero_pad(Y_64, 256, args.device)
        Y_64_low = iwt(Y_64_low, inv_filters, levels=3)    
        
        # Save image into hdf5
        batch_size = recon_img.shape[0]
        recon_dataset[counter: counter+batch_size] = recon_img.cpu()
        real_dataset[counter: counter+batch_size] = data.cpu()
        low_dataset[counter: counter+batch_size] = Y_64_low.cpu()
        counter += batch_size

        # Save images
        # for j in range(recon_img.shape[0]):
        #     save_image(recon_img.cpu(), args.output_dir + data_type + '/recon_img_{}.png'.format(counter))
        #     # save_image(real_img_128_padded.cpu(), args.output_dir + data_type + '/img_128_{}.png'.format(counter))
        #     # save_image(data.cpu(), args.output_dir + data_type + '/img_{}.png'.format(counter))

        #     counter += 1

    f1.close()
    f2.close()
    f3.close()


# Run model through dataloader and save all images
def eval_unet256(model, data_loader, data_type, args):
    model.eval()

    # Create filters
    filters = create_filters(device=args.device)
    inv_filters = create_inv_filters(device=args.device)

    # Create hdf5 dataset
    f1 = h5py.File(args.output_dir + data_type + '/recon_img.hdf5', 'w')
    f2 = h5py.File(args.output_dir + data_type + '/real_img.hdf5', 'w')
    f3 = h5py.File(args.output_dir + data_type + '/low_128_img.hdf5', 'w')

    recon_dataset = f1.create_dataset('data', shape=(50000, 3, 256, 256), dtype=np.float32, fillvalue=0)
    real_dataset = f2.create_dataset('data', shape=(50000, 3, 256, 256), dtype=np.float32, fillvalue=0)
    low_dataset = f3.create_dataset('data', shape=(50000, 3, 256, 256), dtype=np.float32, fillvalue=0)

    counter = 0

    with torch.no_grad():
        for data, _ in tqdm(data_loader):
            if counter >= 50000:
                break
            data = data.to(args.device)
        
            Y = wt_256_3quads(data, filters, levels=3)

            # Get real 1st level masks
            Y_64 = Y[:, :, :64, :64]
            real_mask_64_tl, real_mask_64_tr, real_mask_64_bl, real_mask_64_br = get_4masks(Y_64, 32)
            Y_64_patches = torch.cat((real_mask_64_tl, real_mask_64_tr, real_mask_64_bl, real_mask_64_br), dim=1)

            # Get real 2nd level masks (128 level)
            real_mask_128_tr, real_mask_128_bl, real_mask_128_br = get_3masks(Y[:, :, :128, :128], 64)

            # Divide into 32 x 32 patches
            real_mask_128_tr = collate_channels_from_grid(real_mask_128_tr)
            real_mask_128_bl = collate_channels_from_grid(real_mask_128_bl)
            real_mask_128_br = collate_channels_from_grid(real_mask_128_br)

            # Creating input with real masks up to 128 level
            Y_128_patches = torch.cat((Y_64_patches, real_mask_128_tr, real_mask_128_bl, real_mask_128_br), dim=1)

            # Run through 128 mask network and get reconstructed image
            recon_mask_256_all = model(Y_128_patches)
            recon_mask_256_tr, recon_mask_256_bl, recon_mask_256_br = split_masks_from_channels(recon_mask_256_all)

            Y_real = wt(data, filters, levels=3)
            Y_128 = Y_real[:, :, :128, :128]
            

            # Collate all masks concatenated by channel to an image (slice up and put into a square)
            recon_mask_256_tr_img = collate_16_channels_to_img(recon_mask_256_tr, args.device)
            recon_mask_256_bl_img = collate_16_channels_to_img(recon_mask_256_bl, args.device)   
            recon_mask_256_br_img = collate_16_channels_to_img(recon_mask_256_br, args.device)
            
            recon_mask_256_tr_img = apply_iwt_quads_128(recon_mask_256_tr_img, inv_filters)
            recon_mask_256_bl_img = apply_iwt_quads_128(recon_mask_256_bl_img, inv_filters)
            recon_mask_256_br_img = apply_iwt_quads_128(recon_mask_256_br_img, inv_filters)
            
            recon_mask_256_iwt = collate_patches_to_img(Y_128, recon_mask_256_tr_img, recon_mask_256_bl_img, recon_mask_256_br_img)
            
            recon_mask_padded = zero_pad(recon_mask_256_iwt, 256, args.device)
            recon_img = iwt(recon_mask_padded, inv_filters, levels=3)
        
            # Reconstructed image with only 128x128
            Y_128_low = zero_pad(Y_128, 256, args.device)
            Y_128_low = iwt(Y_128_low, inv_filters, levels=3)

            # Save image into hdf5
            batch_size = recon_img.shape[0]
            recon_dataset[counter: counter+batch_size] = recon_img.cpu()
            real_dataset[counter: counter+batch_size] = data.cpu()
            low_dataset[counter: counter+batch_size] = Y_128_low.cpu()
            counter += batch_size


    f1.close()
    f2.close()
    f3.close()

# Creating HDF5 dataset of real and reconstructed, given real 64x64 TL patch
def eval_unet_128_256(model_128, model_256, data_loader, data_type, args):
    model_128.eval()
    model_256.eval()

    # Create filters
    filters = create_filters(device=args.device)
    inv_filters = create_inv_filters(device=args.device)

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

            data = data.to(args.device)
        
            # Y = wt_256_3quads(data, filters, levels=3)

            # # Get real 1st level masks
            # Y_64 = Y[:, :, :64, :64]
            Y = F.interpolate(data, 64, mode='bicubic')
            Y = wt(Y, filters, levels=1)

            real_mask_64_tl, real_mask_64_tr, real_mask_64_bl, real_mask_64_br = get_4masks(Y, 32)
            Y_64_patches = torch.cat((real_mask_64_tl, real_mask_64_tr, real_mask_64_bl, real_mask_64_br), dim=1)

            # Run through unet 128
            recon_mask_128_all = model_128(Y_64_patches)
            recon_mask_128_tr, recon_mask_128_bl, recon_mask_128_br = split_masks_from_channels(recon_mask_128_all)

            Y_128_patches = torch.cat((Y_64_patches, recon_mask_128_tr, recon_mask_128_bl, recon_mask_128_br), dim=1)

            # Run through unet 256
            recon_mask_256_all = model_256(Y_128_patches)
            recon_mask_256_tr, recon_mask_256_bl, recon_mask_256_br = split_masks_from_channels(recon_mask_256_all)

            # Collate al masks constructed by first 128 level
            recon_mask_128_tr_img = collate_channels_to_img(recon_mask_128_tr, args.device)
            recon_mask_128_bl_img = collate_channels_to_img(recon_mask_128_bl, args.device)
            recon_mask_128_br_img = collate_channels_to_img(recon_mask_128_br, args.device)
            
            recon_mask_128_tr_img = iwt(recon_mask_128_tr_img, inv_filters, levels=1)
            recon_mask_128_bl_img = iwt(recon_mask_128_bl_img, inv_filters, levels=1)   
            recon_mask_128_br_img = iwt(recon_mask_128_br_img, inv_filters, levels=1)
            
            # Y_64 = wt(data, filters, levels=3)[:, :, :64, :64]
            recon_mask_128_iwt = collate_patches_to_img(Y, recon_mask_128_tr_img, recon_mask_128_bl_img, recon_mask_128_br_img)

            # Collate all masks concatenated by channel to an image (slice up and put into a square)
            recon_mask_256_tr_img = collate_16_channels_to_img(recon_mask_256_tr, args.device)
            recon_mask_256_bl_img = collate_16_channels_to_img(recon_mask_256_bl, args.device)   
            recon_mask_256_br_img = collate_16_channels_to_img(recon_mask_256_br, args.device)

            zeros = torch.zeros(recon_mask_256_tr_img.shape)
            
            recon_mask_256_tr_img = apply_iwt_quads_128(recon_mask_256_tr_img, inv_filters)
            recon_mask_256_bl_img = apply_iwt_quads_128(recon_mask_256_bl_img, inv_filters)
            recon_mask_256_br_img = apply_iwt_quads_128(recon_mask_256_br_img, inv_filters)
            
            recon_mask_256_iwt = collate_patches_to_img(zeros, recon_mask_256_tr_img, recon_mask_256_bl_img, recon_mask_256_br_img, device=args.device)
            
            # IWT to reconstruct iamge
            recon_mask_256_iwt[:, :, :128, :128] = recon_mask_128_iwt
            recon_img = iwt(recon_mask_256_iwt, inv_filters, levels=3)
        
            # Save image into hdf5
            batch_size = recon_img.shape[0]
            recon_dataset[counter: counter+batch_size] = recon_img.cpu()
            real_dataset[counter: counter+batch_size] = data.cpu()
            counter += batch_size

    f1.close()
    f2.close()

def eval_biggan_unet128(model, data_loader, args):
    model.eval()

    # Create filters
    filters = create_filters(device=args.device)
    inv_filters = create_inv_filters(device=args.device)

    # Create hdf5 dataset
    f1 = h5py.File(args.output_dir + '/recon_img.hdf5', 'w')
    f2 = h5py.File(args.output_dir + '/sample_img.hdf5', 'w')

    recon_dataset = f1.create_dataset('data', shape=(50000, 3, 256, 256), dtype=np.float32, fillvalue=0)
    sample_dataset = f2.create_dataset('data', shape=(50000, 3, 256, 256), dtype=np.float32, fillvalue=0)

    counter = 0

    for data in tqdm(data_loader):
        if counter >= 50000:
            break
        data = data.to(args.device)
    
        Y_64 = wt(data, filters, levels=1)
        real_mask_64_tl, real_mask_64_tr, real_mask_64_bl, real_mask_64_br = get_4masks(Y_64, 32)
        Y_64_patches = torch.cat((real_mask_64_tl, real_mask_64_tr, real_mask_64_bl, real_mask_64_br), dim=1)

        with torch.no_grad():
            # Run through 128 mask network and get reconstructed image
            recon_mask_all = model(Y_64_patches)
            recon_mask_tr, recon_mask_bl, recon_mask_br = split_masks_from_channels(recon_mask_all)

        # Collate all masks concatenated by channel to an image (slice up and put into a square)
        recon_mask_tr_img = collate_channels_to_img(recon_mask_tr, args.device)
        recon_mask_bl_img = collate_channels_to_img(recon_mask_bl, args.device)   
        recon_mask_br_img = collate_channels_to_img(recon_mask_br, args.device)
        
        recon_mask_tr_img = iwt(recon_mask_tr_img, inv_filters, levels=1)
        recon_mask_bl_img = iwt(recon_mask_bl_img, inv_filters, levels=1)    
        recon_mask_br_img = iwt(recon_mask_br_img, inv_filters, levels=1) 

        zeros = torch.zeros(recon_mask_tr_img.shape)
        
        recon_mask_iwt = collate_patches_to_img(zeros, recon_mask_tr_img, recon_mask_bl_img, recon_mask_br_img)
        
        recon_mask_padded = zero_pad(recon_mask_iwt, 256, args.device)
        recon_mask_padded[:, :, :64, :64] = Y_64
        recon_img = iwt(recon_mask_padded, inv_filters, levels=3)

        sample_padded = zero_pad(Y_64, 256, args.device)
        sample_img = iwt(sample_padded, inv_filters, levels=3)
    
        # Save image into hdf5
        batch_size = recon_img.shape[0]
        recon_dataset[counter: counter+batch_size] = recon_img.cpu()
        sample_dataset[counter: counter+batch_size] = sample_img.cpu()
        counter += batch_size

    f1.close()
    f2.close()


# Creating HDF5 dataset of real and reconstructed, given real 64x64 TL patch
def eval_biggan_unet_128_256(model_128, model_256, data_loader, args):
    model_128.eval()
    model_256.eval()

    # Create filters
    filters = create_filters(device=args.device)
    inv_filters = create_inv_filters(device=args.device)

    # Create hdf5 dataset
    f1 = h5py.File(args.output_dir + '/recon_img.hdf5', 'w')
    f2 = h5py.File(args.output_dir + '/sample_img.hdf5', 'w')

    recon_dataset = f1.create_dataset('data', shape=(50000, 3, 256, 256), dtype=np.float32, fillvalue=0)
    sample_dataset = f2.create_dataset('data', shape=(50000, 3, 256, 256), dtype=np.float32, fillvalue=0)

    counter = 0

    with torch.no_grad():
        for data in tqdm(data_loader):
            if counter >= 50000:
                break

            data = data.to(args.device)
        
            Y_64 = wt(data, filters, levels=1)
            real_mask_64_tl, real_mask_64_tr, real_mask_64_bl, real_mask_64_br = get_4masks(Y_64, 32)
            Y_64_patches = torch.cat((real_mask_64_tl, real_mask_64_tr, real_mask_64_bl, real_mask_64_br), dim=1)

            # Run through unet 128
            recon_mask_128_all = model_128(Y_64_patches)
            recon_mask_128_tr, recon_mask_128_bl, recon_mask_128_br = split_masks_from_channels(recon_mask_128_all)

            Y_128_patches = torch.cat((Y_64_patches, recon_mask_128_tr, recon_mask_128_bl, recon_mask_128_br), dim=1)

            # Run through unet 256
            recon_mask_256_all = model_256(Y_128_patches)
            recon_mask_256_tr, recon_mask_256_bl, recon_mask_256_br = split_masks_from_channels(recon_mask_256_all)

            # Collate al masks constructed by first 128 level
            recon_mask_128_tr_img = collate_channels_to_img(recon_mask_128_tr, args.device)
            recon_mask_128_bl_img = collate_channels_to_img(recon_mask_128_bl, args.device)
            recon_mask_128_br_img = collate_channels_to_img(recon_mask_128_br, args.device)
            
            recon_mask_128_tr_img = iwt(recon_mask_128_tr_img, inv_filters, levels=1)
            recon_mask_128_bl_img = iwt(recon_mask_128_bl_img, inv_filters, levels=1)   
            recon_mask_128_br_img = iwt(recon_mask_128_br_img, inv_filters, levels=1)
            
            recon_mask_128_iwt = collate_patches_to_img(Y_64, recon_mask_128_tr_img, recon_mask_128_bl_img, recon_mask_128_br_img)

            # Collate all masks concatenated by channel to an image (slice up and put into a square)
            recon_mask_256_tr_img = collate_16_channels_to_img(recon_mask_256_tr, args.device)
            recon_mask_256_bl_img = collate_16_channels_to_img(recon_mask_256_bl, args.device)   
            recon_mask_256_br_img = collate_16_channels_to_img(recon_mask_256_br, args.device)

            zeros = torch.zeros(recon_mask_256_tr_img.shape)
            
            recon_mask_256_tr_img = apply_iwt_quads_128(recon_mask_256_tr_img, inv_filters)
            recon_mask_256_bl_img = apply_iwt_quads_128(recon_mask_256_bl_img, inv_filters)
            recon_mask_256_br_img = apply_iwt_quads_128(recon_mask_256_br_img, inv_filters)
            
            recon_mask_256_iwt = collate_patches_to_img(zeros, recon_mask_256_tr_img, recon_mask_256_bl_img, recon_mask_256_br_img, device=args.device)
            
            # IWT to reconstruct iamge
            recon_mask_256_iwt[:, :, :128, :128] = recon_mask_128_iwt
            recon_img = iwt(recon_mask_256_iwt, inv_filters, levels=3)

            sample_padded = zero_pad(Y_64, 256, args.device)
            sample_img = iwt(sample_padded, inv_filters, levels=3)
        
            # Save image into hdf5
            batch_size = recon_img.shape[0]
            recon_dataset[counter: counter+batch_size] = recon_img.cpu()
            sample_dataset[counter: counter+batch_size] = sample_img.cpu()
            counter += batch_size

    f1.close()
    f2.close()


# Run model through dataloader and save all images
def eval_tl(data_loader, data_type, args):
    # Create filters
    filters = create_filters(device=args.device)
    inv_filters = create_inv_filters(device=args.device)

    # Create hdf5 dataset
    f1 = h5py.File(args.output_dir + data_type + '/recon_img.hdf5', 'w')
    f2 = h5py.File(args.output_dir + data_type + '/real_img.hdf5', 'w')

    recon_dataset = f1.create_dataset('data', shape=(50000, 3, 256, 256), dtype=np.float32, fillvalue=0)
    real_dataset = f2.create_dataset('data', shape=(50000, 3, 256, 256), dtype=np.float32, fillvalue=0)

    counter = 0

    for data, _ in tqdm(data_loader):
        if counter >= 50000:
            break
        data = data.to(args.device)
    
        Y =wt(data, filters, levels=3)

        # Get real 1st level masks
        Y_64 = Y[:, :, :64, :64]
        
        Y_64_padded = zero_pad(Y_64, 256, args.device)
        Y_64_padded = iwt(Y_64_padded, inv_filters, levels=3)
    
        # Save image into hdf5
        batch_size = Y_64_padded.shape[0]
        recon_dataset[counter: counter+batch_size] = Y_64_padded.cpu()
        real_dataset[counter: counter+batch_size] = data.cpu()
        counter += batch_size

        # Save images
        # for j in range(recon_img.shape[0]):
        #     save_image(recon_img.cpu(), args.output_dir + data_type + '/recon_img_{}.png'.format(counter))
        #     # save_image(real_img_128_padded.cpu(), args.output_dir + data_type + '/img_128_{}.png'.format(counter))
        #     # save_image(data.cpu(), args.output_dir + data_type + '/img_{}.png'.format(counter))

        #     counter += 1

    f1.close()
    f2.close()