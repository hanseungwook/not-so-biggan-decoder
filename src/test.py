# Train function for Unet 128 + Unet 256 with perceptual loss on the entire mask
def train_unet_128_256(epoch, state_dict, model_128, model_256, optimizer, train_loader, valid_loader, args, logger, loss_fn=nn.MSELoss()):
    model_128.train()
    model_256.train()

    filters = create_filters(device=args.device)
    inv_filters = create_inv_filters(device=args.device)

    for data, _ in tqdm(train_loader):
        start_time = time.time()
        optimizer.zero_grad()

        data = data.to(args.device)
    
        Y = wt_256_3quads(data, filters, levels=3)

        # Get real 1st level masks
        Y_64 = Y[:, :, :64, :64]
        real_mask_64_tl, real_mask_64_tr, real_mask_64_bl, real_mask_64_br = get_4masks(Y_64, 32)
        Y_64_patches = torch.cat((real_mask_64_tl, real_mask_64_tr, real_mask_64_bl, real_mask_64_br), dim=1)

        # Run through model 128
        recon_mask_128_all = model_128(Y_64_patches)
        recon_mask_128_tr, recon_mask_128_bl, recon_mask_128_br = split_masks_from_channels(recon_mask_128_all)

        Y_128_patches = torch.cat((Y_64_patches, recon_mask_128_tr, recon_mask_128_bl, recon_mask_128_br), dim=1)

        # Run through 128 mask network and get reconstructed image
        recon_mask_256_all = model_256(Y_128_patches)

        # Reconstruction and real
        recon_img_128, recon_img, recon_mask = mask_outputs_to_img(Y_64, recon_mask_128_all, recon_mask_256_all, args.device, mask=True)
        Y_real = wt(data, filters, levels=3)
        real_mask_iwt = iwt(zero_mask(Y_real, 3, 2).to(args.device), inv_filters, levels=3)

        # Loss
        loss = loss_fn(recon_img, data)
    
        loss.backward()
        optimizer.step()
        
        # Calculate iteration time
        end_time = time.time()
        itr_time = end_time - start_time

        # Update logger & wandb
        logger.update(state_dict['itr'], loss.cpu().item(), itr_time)
        wandb.log({'train_loss': loss.item()}, commit=False)
        wandb.log({'train_itr_time': itr_time}, commit=True)    

        # Save images, logger, weights on save_every interval
        if not state_dict['itr'] % args.save_every:
            Y_real = wt(data, filters, levels=3)
            Y_64 = Y_real[:, :, :64, :64]
            Y_128 = Y_real[:, :, :128, :128]
            zeros = torch.zeros(real_mask_tr.shape)

            # Real mask with both levels (128, 256) 
            real_mask_iwt = iwt(zero_mask(Y_real, 3, 2).to(args.device), inv_filters, levels=3)

            # Create images with outputs of both 128 and 256 models
            recon_img_128, recon_img, recon_mask = mask_outputs_to_img(Y_64, recon_mask_128_all, recon_mask_256_all, device, mask=True)
            
            # Reconstructed image with only 64x64
            Y_128_low = zero_pad(Y_128, 256, args.device)
            Y_128_low = iwt(Y_128_low, inv_filters, levels=3)
            
            # Save images
            save_image(real_mask_iwt.cpu(), args.output_dir + 'real_mask_iwt_itr{}.png'.format(state_dict['itr']))
            save_image(recon_mask.cpu(), args.output_dir + 'recon_mask_iwt_itr{}.png'.format(state_dict['itr']))
            
            save_image(recon_img_128.cpu(), args.output_dir + 'recon_img_128_itr{}.png'.format(state_dict['itr']))
            save_image(recon_img.cpu(), args.output_dir + 'recon_img_itr{}.png'.format(state_dict['itr']))

            save_image(Y_128_low.cpu(), args.output_dir + 'low_img_128_full_itr{}.png'.format(state_dict['itr']))
            save_image(data.cpu(), args.output_dir + 'img_itr{}.png'.format(state_dict['itr']))
            
            # Save model & optimizer weights
            torch.save({
                'model_128_state_dict': model_128.state_dict(),
                'model_256_state_dict': model_256.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, args.output_dir + '/iwt_model_128_256_itr{}.pth'.format(state_dict['itr']))
            
            # Save logger 
            torch.save(logger, args.output_dir + '/logger.pth')
        
        if not state_dict['itr'] % args.valid_every:
            model_128.eval()
            model_256.eval()

            val_losses = []
            with torch.no_grad(): 
                for data, _ in tqdm(valid_loader):
                    data = data.to(args.device)
                
                    Y = wt_256_3quads(data, filters, levels=3)

                    # Get real 1st level masks
                    Y_64 = Y[:, :, :64, :64]
                    real_mask_64_tl, real_mask_64_tr, real_mask_64_bl, real_mask_64_br = get_4masks(Y_64, 32)
                    Y_64_patches = torch.cat((real_mask_64_tl, real_mask_64_tr, real_mask_64_bl, real_mask_64_br), dim=1)

                    recon_mask_128_all = model_128(Y_64_patches)
                    recon_mask_128_tr, recon_mask_128_bl, recon_mask_128_br = split_masks_from_channels(recon_mask_128_all)

                    Y_128_patches = torch.cat((Y_64_patches, recon_mask_128_tr, recon_mask_128_bl, recon_mask_128_br), dim=1)

                    # Run through 128 mask network and get reconstructed image
                    recon_mask_256_all = model_256(Y_128_patches)

                    # Reconstruction and real
                    recon_img_128, recon_img, recon_mask = mask_outputs_to_img(Y_64, recon_mask_128_all, recon_mask_256_all, device, mask=True)
                    Y_real = wt(data, filters, levels=3)
                    real_mask_iwt = iwt(zero_mask(Y_real, 3, 2).to(args.device), inv_filters, levels=3)

                    # Loss
                    loss = loss_fn(recon_mask, real_mask_iwt)                    

                    val_losses.append(loss.item())

                val_losses_mean = np.mean(val_losses)
                wandb.log({'val_loss': val_losses_mean}, commit=True)
                logger.update_val_loss(state_dict['itr'], val_losses_mean)
                val_losses.clear()

            model_128.train()
            model_256.train()

            # Save validation images
            Y_real = wt(data, filters, levels=3)
            Y_64 = Y_real[:, :, :64, :64]
            Y_128 = Y_real[:, :, :128, :128]
            zeros = torch.zeros(real_mask_tr.shape)

            # Real mask with both levels (128, 256) 
            real_mask_iwt = iwt(zero_mask(Y_real, 3, 2).to(args.device), inv_filters, levels=3)

            # Create images with outputs of both 128 and 256 models
            recon_img_128, recon_img, recon_mask = mask_outputs_to_img(Y_64, recon_mask_128_all, recon_mask_256_all, device, mask=True)
            
            # Reconstructed image with only 64x64
            Y_128_low = zero_pad(Y_128, 256, args.device)
            Y_128_low = iwt(Y_128_low, inv_filters, levels=3)
            
            # Save images
            save_image(real_mask_iwt.cpu(), args.output_dir + 'val_real_mask_iwt_itr{}.png'.format(state_dict['itr']))
            save_image(recon_mask.cpu(), args.output_dir + 'val_recon_mask_iwt_itr{}.png'.format(state_dict['itr']))
            
            save_image(recon_img_128.cpu(), args.output_dir + 'val_recon_img_128_itr{}.png'.format(state_dict['itr']))
            save_image(recon_img.cpu(), args.output_dir + 'val_recon_img_itr{}.png'.format(state_dict['itr']))

            save_image(Y_128_low.cpu(), args.output_dir + 'val_low_img_128_full_itr{}.png'.format(state_dict['itr']))
            save_image(data.cpu(), args.output_dir + 'val_img_itr{}.png'.format(state_dict['itr']))

        # Increment iteration number
        state_dict['itr'] += 1