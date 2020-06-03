iwt_model.train()

# Training 64->128 mask network
for epoch in range(16, 300):
    start_time = time.time()
    for i, data in enumerate(dataloader_vanilla, 0):
        optimizer.zero_grad()

        real_cpu = data.to(DEVICE0)
#         real_mask_tr, real_mask_bl, real_mask_br = data[1]

        Y = wt_128_3quads(real_cpu, filters, levels=3)
        Y_32 = Y[:, :, :32, :32]
        
        # Get 3 real masks at 128 level (mask_dim=64)
        real_mask_tr, real_mask_bl, real_mask_br = get_3masks(Y, mask_dim)
        
        # Normalize 128 level masks and divide into 32 x 32 patches
        real_mask_tr_patches = create_patches_from_grid(real_mask_tr)
        real_mask_bl_patches = create_patches_from_grid(real_mask_bl)
        real_mask_br_patches = create_patches_from_grid(real_mask_br)
        
        # Get 3 real masks at 64 level (mask_dim=32) or reconstructed from lower model (0.5 probability)
#         if random.random() < 0.5:
#             mask_tr_32, mask_bl_32, mask_br_32 = get_3masks(Y[:, :, :64, :64], 32)
#         else:
        with torch.no_grad():
            mask_tr_32 = iwt_model_32_1(Y_32.to(DEVICE1))
            mask_bl_32 = iwt_model_32_2(Y_32.to(DEVICE1))
            mask_br_32 = iwt_model_32_3(Y_32.to(DEVICE1))
        
        # Concatenate the first level's patches for input
        patches = torch.cat((Y_32, mask_tr_32, mask_bl_32, mask_br_32), dim=1)

        # Run through 128 mask network and get reconstructed image
        recon_mask_all = iwt_model(patches.to(DEVICE0))
        recon_mask_tr, recon_mask_bl, recon_mask_br = split_masks_from_channels(recon_mask_all)
    
        # Reshape channel-wise concatenated patches to new dimension
        recon_mask_tr_patches = recon_mask_tr.reshape(batch_size, -1, 3, 32, 32)
        recon_mask_bl_patches  = recon_mask_bl.reshape(batch_size, -1, 3, 32, 32)
        recon_mask_br_patches = recon_mask_br.reshape(batch_size, -1, 3, 32, 32)
        
        # Calculate loss
        loss = 0
        for j in range(real_mask_tr_patches.shape[1]):
            loss += loss_fn(real_mask_tr_patches[:,j].to(DEVICE0), recon_mask_tr_patches[:,j])
            loss += loss_fn(real_mask_bl_patches[:,j].to(DEVICE0), recon_mask_bl_patches[:,j])
            loss += loss_fn(real_mask_br_patches[:,j].to(DEVICE0), recon_mask_br_patches[:,j])
            
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        
        if i % 50 == 0:
            print('[{}/{}][{}/{}]\t Loss: {}'.format(epoch, num_epochs, i, len(dataloader_vanilla), loss.item()))
        
        gc.collect()
        torch.cuda.empty_cache()
    
    end_time = time.time()
    wandb.log({'train_loss': np.mean(train_losses)}, commit=False)
    wandb.log({'train_epoch_time': end_time - start_time}, commit=False)
    train_losses.clear()

    Y_real = wt(real_cpu, filters, levels=3)
    zeros = torch.zeros(real_mask_tr.shape)
    
    real_mask = collate_patches_to_img(zeros, real_mask_tr, real_mask_bl, real_mask_br)
    
    real_mask_tr_iwt = iwt(real_mask_tr, inv_filters, levels=1)
    real_mask_bl_iwt = iwt(real_mask_bl, inv_filters, levels=1)
    real_mask_br_iwt = iwt(real_mask_br, inv_filters, levels=1)
    real_mask_iwt = collate_patches_to_img(zeros, real_mask_tr_iwt, real_mask_bl_iwt, real_mask_br_iwt)
    
    real_img_256_padded = Y_real[:, :, :128, :128]
    real_img_256_padded = zero_pad(real_img_256_padded, 256, DEVICE0)
    real_img_256_padded = iwt(real_img_256_padded, inv_filters, levels=3)

    # Collate all masks concatenated by channel to an image (slice up and put into a square)
    recon_mask_tr_img = collate_channels_to_img(recon_mask_tr, DEVICE0)
    recon_mask_bl_img = collate_channels_to_img(recon_mask_bl, DEVICE0)   
    recon_mask_br_img = collate_channels_to_img(recon_mask_br, DEVICE0)

    recon_mask_32 = collate_patches_to_img(Y_real[:, :, :32, :32], mask_tr_32, mask_bl_32, mask_br_32)
    recon_mask = collate_patches_to_img(zeros, recon_mask_tr_img, recon_mask_bl_img, recon_mask_br_img)
    
    recon_mask_tr_img = iwt(recon_mask_tr_img, inv_filters, levels=1)
    recon_mask_bl_img = iwt(recon_mask_bl_img, inv_filters, levels=1)    
    recon_mask_br_img = iwt(recon_mask_br_img, inv_filters, levels=1) 
    
    recon_mask_iwt = collate_patches_to_img(zeros, recon_mask_tr_img, recon_mask_bl_img, recon_mask_br_img)
    
    recon_mask_padded = zero_pad(recon_mask_iwt, 256, DEVICE0)
    recon_mask_padded[:, :, :64, :64] = recon_mask_32
    recon_img = iwt(recon_mask_padded, inv_filters, levels=3)
    
    recon_mask_32_padded = zero_pad(recon_mask_32, 256, DEVICE0)
    recon_img_32 = iwt(recon_mask_32_padded, inv_filters, levels=3)
    
    Y_64 = Y_real[:, :, :64, :64]
    Y_64_low = zero_pad(Y_64, 256, DEVICE0)
    Y_64_low = iwt(Y_64_low, inv_filters, levels=3)
    
    Y_32 = Y_real[:, :, :32, :32]
    Y_32_low = zero_pad(Y_32, 256, DEVICE0)
    Y_32_low = iwt(Y_32_low, inv_filters, levels=3)
    
    save_image(real_mask_iwt.cpu(), output_dir + 'real_mask_iwt{}.png'.format(epoch))
    save_image(real_mask.cpu(), output_dir + 'real_mask{}.png'.format(epoch))
    save_image(recon_mask_iwt.cpu(), output_dir + 'recon_mask_iwt{}.png'.format(epoch))
    save_image(recon_mask.cpu(), output_dir + 'recon_mask{}.png'.format(epoch))
    save_image(recon_img.cpu(), output_dir + 'recon_img{}.png'.format(epoch))
    save_image(recon_img_32.cpu(), output_dir + 'recon_img_32_{}.png'.format(epoch))
    save_image(Y_64_low.cpu(), output_dir + 'low_img_64_full{}.png'.format(epoch))
    save_image(Y_32_low.cpu(), output_dir + 'low_img_32_full{}.png'.format(epoch))
    save_image(real_img_256_padded.cpu(), output_dir + 'img_64_real_masks{}.png'.format(epoch))
    save_image(real_cpu.cpu(), output_dir + 'img{}.png'.format(epoch))
    
    # Progressive gan evaluation
    with torch.no_grad():
        iwt_model.eval()
        
        # Generate samples from prog-gan
        z_save = hypersphere(torch.randn(32, 4 * 32, 1, 1, device=DEVICE0))
        sample = generator(z_save).detach()
        sample_denorm = denormalize_pg(sample, abs_max_pg).to(DEVICE1)
        
        # Run through 1st level mask network
        pg_mask_tr_32 = iwt_model_32_1(sample_denorm)
        pg_mask_bl_32 = iwt_model_32_2(sample_denorm)
        pg_mask_br_32 = iwt_model_32_3(sample_denorm)
        zeros = torch.zeros(pg_mask_tr_32.shape)
        
        pg_mask_32 = collate_patches_to_img(zeros, pg_mask_tr_32, pg_mask_bl_32, pg_mask_br_32)
        
        # Run through 2nd level mask network 
        patches = torch.cat((sample_denorm, pg_mask_tr_32, pg_mask_bl_32, pg_mask_br_32), dim=1)
        
        pg_mask_64_all = iwt_model(patches.to(DEVICE0))
        pg_mask_tr_64, pg_mask_bl_64, pg_mask_br_64 = split_masks_from_channels(pg_mask_64_all)
        
        # Collate all masks concatenated by channel to an image (slice up and put into a square)
        pg_mask_tr_64_img = collate_channels_to_img(pg_mask_tr_64, DEVICE0)
        pg_mask_bl_64_img = collate_channels_to_img(pg_mask_bl_64, DEVICE0)   
        pg_mask_br_64_img = collate_channels_to_img(pg_mask_br_64, DEVICE0)
        
        # Apply IWT once to every mask (consists of 32 x 32 patches)
        pg_mask_tr_64_img = iwt(pg_mask_tr_64_img, inv_filters, levels=1)
        pg_mask_bl_64_img = iwt(pg_mask_bl_64_img, inv_filters, levels=1)    
        pg_mask_br_64_img = iwt(pg_mask_br_64_img, inv_filters, levels=1) 
        
        pg_mask_64 = collate_patches_to_img(pg_mask_32, pg_mask_tr_64_img, pg_mask_bl_64_img, pg_mask_br_64_img)
        
        pg_low_padded = zero_pad(sample_denorm, 256, DEVICE0)
        pg_low_img = iwt(pg_low_padded, inv_filters, levels=3)
        
        pg_low_32_padded = zero_pad(pg_mask_32, 256, DEVICE0)
        pg_low_32_padded[:, :, :32, :32] = sample_denorm.to(DEVICE0)
        pg_low_32_img = iwt(pg_low_32_padded, inv_filters, levels=3)
        
        pg_64_padded = zero_pad(pg_mask_64, 256, DEVICE0)
        pg_64_padded[:, :, :32, :32] = sample_denorm.to(DEVICE0)
        pg_64_img = iwt(pg_64_padded, inv_filters, levels=3)
        
        iwt_model.train()

    save_image(sample.cpu(), output_dir + 'pg_sample_norm{}.png'.format(epoch))
    save_image(pg_mask_32.cpu(), output_dir + 'pg_mask_32_{}.png'.format(epoch))
    save_image(pg_mask_64.cpu(), output_dir + 'pg_mask_64_{}.png'.format(epoch))
    save_image(pg_64_img.cpu(), output_dir + 'pg_64_img{}.png'.format(epoch))
    save_image(pg_low_img.cpu(), output_dir + 'pg_low_img_full{}.png'.format(epoch))
    save_image(pg_low_32_img.cpu(), output_dir + 'pg_low_32_img_full{}.png'.format(epoch))
    
    # Validation dataset evaluation
    with torch.no_grad():
        iwt_model.eval()
        
        for k, data in enumerate(val_dataloader):
            real_cpu = data.to(DEVICE0)

            Y = wt_128_3quads(real_cpu, filters, levels=3)
            Y_32 = Y[:, :, :32, :32]

            # Get 3 real masks at 128 level (mask_dim=64)
            real_mask_tr, real_mask_bl, real_mask_br = get_3masks(Y, mask_dim)

            # Normalize 128 level masks and divide into 32 x 32 patches
            real_mask_tr_patches = create_patches_from_grid(real_mask_tr)
            real_mask_bl_patches = create_patches_from_grid(real_mask_bl)
            real_mask_br_patches = create_patches_from_grid(real_mask_br)

            # Get 3 reconstructed masks from lower model
            with torch.no_grad():
                mask_tr_32 = iwt_model_32_1(Y_32.to(DEVICE1))
                mask_bl_32 = iwt_model_32_2(Y_32.to(DEVICE1))
                mask_br_32 = iwt_model_32_3(Y_32.to(DEVICE1))

            # Concatenate the first level's patches for input
            patches = torch.cat((Y_32, mask_tr_32, mask_bl_32, mask_br_32), dim=1)

            # Run through 128 mask network and get reconstructed image
            recon_mask_all = iwt_model(patches.to(DEVICE0))
            recon_mask_tr, recon_mask_bl, recon_mask_br = split_masks_from_channels(recon_mask_all)

            # Reshape channel-wise concatenated patches to new dimension
            recon_mask_tr_patches = recon_mask_tr.reshape(batch_size, -1, 3, 32, 32)
            recon_mask_bl_patches  = recon_mask_bl.reshape(batch_size, -1, 3, 32, 32)
            recon_mask_br_patches = recon_mask_br.reshape(batch_size, -1, 3, 32, 32)

            # Calculate loss
            loss = 0
            for j in range(real_mask_tr_patches.shape[1]):
                loss += loss_fn(real_mask_tr_patches[:,j].to(DEVICE0), recon_mask_tr_patches[:,j])
                loss += loss_fn(real_mask_bl_patches[:,j].to(DEVICE0), recon_mask_bl_patches[:,j])
                loss += loss_fn(real_mask_br_patches[:,j].to(DEVICE0), recon_mask_br_patches[:,j])
            val_losses.append(loss.item())
            
        wandb.log({'val_loss': np.mean(val_losses)}, commit=True)
        val_losses.clear()
        iwt_model.train()
    
    if epoch % 30 == 0:
        torch.save({
            'model_state_dict': iwt_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, output_dir + '/iwt_model_128_all_epoch{}.pth'.format(epoch))
        
        
        