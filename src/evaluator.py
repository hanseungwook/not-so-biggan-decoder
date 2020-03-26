import torch

def eval_ae_mask(epoch, wt_model, model, sample_loader, args, img_output_dir, model_dir, writer):
    with torch.no_grad():
        model.eval()
        
        for data in sample_loader:
            data = data.to(model.device)
            
            # Get Y
            Y = wt_model(data)
            
            # Zeroing out all other patches
            Y = zero_patches(Y, num_wt=args.num_wt)

            x_hat = model(Y)

            save_image(x_hat.cpu(), img_output_dir + '/sample_recon{}.png'.format(epoch))
            save_image(data.cpu(), img_output_dir + '/sample{}.png'.format(epoch))

    torch.save(model.state_dict(), model_dir + '/aemask512_epoch{}.pth'.format(epoch))





    def train_ae_mask(epoch, wt_model, model, criterion, optimizer, train_loader, train_losses, args, writer):
    # toggle model to train mode
    model.train()
    train_loss = 0
    
    for batch_idx, data in enumerate(train_loader):
        
        data = data.to(model.device)

        optimizer.zero_grad()
        
        # Get Y
        Y = wt_model(data)
        
        # Zeroing out all other patches
        Y = zero_patches(Y, num_wt=args.num_wt)

        x_hat = model(Y)
        loss = model.loss_function(data, x_hat, criterion)
        loss.backward()

        # Calculating and printing gradient norm
        total_norm = calc_grad_norm_2(model)

        # Calculating and printing gradient norm
        global log_idx
        writer.add_scalar('Loss', loss, log_idx)
        writer.add_scalar('Gradient_norm/before', total_norm, log_idx)
        log_idx += 1 

        # Gradient clipping
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip, norm_type=2)
            # Re-calculating total norm after gradient clipping
            total_norm = calc_grad_norm_2(model)
            writer.add_scalar('Gradient_norm/clipped', total_norm, log_idx)
        
        train_losses.append(loss.cpu().item())
        train_loss += loss

        optimizer.step()
        if batch_idx % args.log_interval == 0:
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss / len(data)))
            
            n = min(data.size(0), 8)  

    logging.info('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))