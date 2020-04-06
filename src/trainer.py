import os, sys
import torch
import torch.nn as nn
import numpy as np
from vae_models import IWT, iwt
from utils.utils import zero_patches, zero_mask, calc_grad_norm_2, preprocess_low_freq, create_inv_filters, hf_collate_to_channels, hf_collate_to_channels_wt2, hf_collate_to_img, preprocess_mask
import logging
import IPython

log_idx = 0
decoder_outputs = []
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(asctime)s %(message)s')

def get_decoder_output(self, input, output):
    global decoder_outputs
    decoder_outputs.append(output)

def train_wtvae(epoch, model, optimizer, train_loader, train_losses, args, writer):
    # toggle model to train mode
    model.train()
    train_loss = 0

    for batch_idx, data in enumerate(train_loader):
        
        if model.cuda:
            data = data.to(model.device)

        optimizer.zero_grad()
        
        wt_data, mu, logvar = model(data)
        loss, loss_bce, loss_kld = model.loss_function(data, wt_data, mu, logvar, kl_weight=args.kl_weight)
        loss.backward()

        # Calculating and printing gradient norm
        total_norm = calc_grad_norm_2(model)

        global log_idx
        writer.add_scalar('Loss/total', loss, log_idx)
        writer.add_scalar('Loss/bce', loss_bce, log_idx)
        writer.add_scalar('Loss/kld', loss_kld, log_idx)
        writer.add_scalar('Gradient_norm/before', total_norm, log_idx)
        writer.add_scalar('KL_weight', args.kl_weight, log_idx)
        log_idx += 1 

        # Gradient clipping
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip, norm_type=2)
            
            # Re-calculating and printing gradient norm
            total_norm = calc_grad_norm_2(model)
            writer.add_scalar('Gradient_norm/clipped', total_norm, log_idx)

        train_losses.append((loss.item(), loss_bce, loss_kld))
        train_loss += loss
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                            len(train_loader.dataset),
                                                                            100. * batch_idx / len(train_loader),
                                                                            loss / len(data)))
            
            n = min(data.size(0), 8)
            
    writer.flush()
    logging.info('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

# WTVAE training in which input dims are same as output dims (therefore, need higher dimensional img to use as target when computing loss)
# Order: (lower dimensional, higher dimensional)
def train_wtvae_pair(epoch, model, optimizer, train_loader, train_losses, args, writer):
    # toggle model to train mode
    model.train()
    train_loss = 0
    anneal_rate = (1.0 - args.kl_start) / (args.kl_warmup * len(train_loader))
    global decoder_outputs

    # Register hook onto decoder so that we can compute additional loss on reconstruction of original image (in adddition to patch loss)
    model.decoder.register_forward_hook(get_decoder_output)

    for batch_idx, data in enumerate(train_loader):
        data0 = data[0].to(model.device)
        data1 = data[1].to(model.device)

        optimizer.zero_grad()
        
        wt_data, mu, logvar = model(data0)
        decoder_output = decoder_outputs[-1]
        loss, loss_bce, loss_kld = model.loss_function(data1, data0, wt_data, decoder_output, mu, logvar, kl_weight=args.kl_weight)
        loss.backward()
        
        # Clearing saved outputs
        decoder_outputs.clear()

        # Calculating and printing gradient norm
        total_norm = calc_grad_norm_2(model)

        global log_idx
        writer.add_scalar('Loss/total', loss, log_idx)
        writer.add_scalar('Loss/bce', loss_bce, log_idx)
        writer.add_scalar('Loss/kld', loss_kld, log_idx)
        writer.add_scalar('Gradient_norm/before', total_norm, log_idx)
        writer.add_scalar('KL_weight', args.kl_weight, log_idx)
        log_idx += 1 

        # Gradient clipping
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip, norm_type=2)
            
            # Re-calculating and printing gradient norm
            total_norm = calc_grad_norm_2(model)
            writer.add_scalar('Gradient_norm/clipped', total_norm, log_idx)

        train_losses.append((loss.item(), loss_bce, loss_kld))
        train_loss += loss
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data0),
                                                                            len(train_loader.dataset),
                                                                            100. * batch_idx / len(train_loader),
                                                                            loss / len(data0)))
            
            n = min(data0.size(0), 8)
            
    writer.flush()
    logging.info('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))


def train_wtvae_128(epoch, model, optimizer, train_loader, train_losses, args, writer):
    # toggle model to train mode
    model.train()
    train_loss = 0
    # anneal_rate = (1.0 - args.kl_start) / (args.kl_warmup * len(train_loader))
    # global decoder_outputs

    # # Register hook onto decoder so that we can compute additional loss on reconstruction of original image (in adddition to patch loss)
    # model.decoder.register_forward_hook(get_decoder_output)

    for batch_idx, data in enumerate(train_loader):
        # args.kl_weight = min(1.0, args.kl_weight + anneal_rate)
        data128 = data[0]
        data512 = data[1]
        if model.cuda:
            data128 = data128.to(model.device)
            data512 = data512.to(model.device)

        optimizer.zero_grad()
        
        wt_data, mu, logvar = model(data128)
        decoder_output = decoder_outputs[-1]
        loss, loss_bce, loss_kld = model.loss_function(data512, data128, wt_data, decoder_output, mu, logvar, kl_weight=args.kl_weight)
        loss.backward()
        
        # Clearing saved outputs
        # decoder_outputs.clear()

        # Calculating and printing gradient norm
        total_norm = calc_grad_norm_2(model)

        global log_idx
        writer.add_scalar('Loss/total', loss, log_idx)
        writer.add_scalar('Loss/bce', loss_bce, log_idx)
        writer.add_scalar('Loss/kld', loss_kld, log_idx)
        writer.add_scalar('Gradient_norm/before', total_norm, log_idx)
        writer.add_scalar('KL_weight', args.kl_weight, log_idx)
        log_idx += 1 

        # Gradient clipping
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip, norm_type=2)
            # Re-calculating and printing gradient norm
            total_norm = calc_grad_norm_2(model)
            writer.add_scalar('Gradient_norm/clipped', total_norm, log_idx)

        train_losses.append((loss.item(), loss_bce, loss_kld))
        train_loss += loss
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data128),
                                                                            len(train_loader.dataset),
                                                                            100. * batch_idx / len(train_loader),
                                                                            loss / len(data128)))
            
            n = min(data128.size(0), 8)
            
    writer.flush()
    logging.info('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))


def train_wtvae_128_fixed_wt(epoch, model, optimizer, train_loader, train_losses, args, writer):
    # toggle model to train mode
    model.train()
    train_loss = 0
    anneal_rate = (1.0 - args.kl_start) / (args.kl_warmup * len(train_loader))

    for batch_idx, data in enumerate(train_loader):
        args.kl_weight = min(1.0, args.kl_weight + anneal_rate)
        data128 = data[0]
        data512 = data[1]
        if model.cuda:
            data128 = data128.to(model.device)
            data512 = data512.to(model.device)

        optimizer.zero_grad()
        
        wt_data, mu, logvar = model(data128)
        loss, loss_bce, loss_kld = model.loss_function(data512, wt_data, mu, logvar, kl_weight=args.kl_weight)
        loss.backward()
    
        # Calculating and printing gradient norm
        total_norm = calc_grad_norm_2(model)

        global log_idx
        writer.add_scalar('Loss/total', loss, log_idx)
        writer.add_scalar('Loss/bce', loss_bce, log_idx)
        writer.add_scalar('Loss/kld', loss_kld, log_idx)
        writer.add_scalar('Gradient_norm/before', total_norm, log_idx)
        writer.add_scalar('KL_weight', args.kl_weight, log_idx)
        log_idx += 1 

        # Gradient clipping
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip, norm_type=2)
            # Re-calculating and printing gradient norm
            total_norm = calc_grad_norm_2(model)
            writer.add_scalar('Gradient_norm/clipped', total_norm, log_idx)

        train_losses.append((loss.item(), loss_bce, loss_kld))
        train_loss += loss
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data128),
                                                                            len(train_loader.dataset),
                                                                            100. * batch_idx / len(train_loader),
                                                                            loss / len(data128)))
            
            n = min(data128.size(0), 8)
            
    writer.flush()
    logging.info('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))


def train_iwtvae(epoch, wt_model, iwt_model, optimizer, iwt_fn, train_loader, train_losses, args, writer):
    # toggle model to train mode
    iwt_model.train()
    train_loss = 0

    # iwt_fn = IWT(iwt=iwt, num_iwt=self.num_iwt)
    # iwt_fn.set_filters(filters)
    
    for batch_idx, data in enumerate(train_loader):
        
        data0 = data.to(iwt_model.device)
        data1 = data.to(wt_model.device)

        optimizer.zero_grad()
        
        # Get Y
        Y = wt_model(data1)
        
        # Zeroing out all other patches, if given zero arg
        Y_full = Y.clone()
        if args.zero:
            Y = zero_patches(Y, num_wt=args.num_iwt)

        # Run model to get mask (zero out first patch of mask) and x_wt_hat
        mask, mu, var = iwt_model(data0, Y_full.to(iwt_model.device), Y.to(iwt_model.device))
        with torch.no_grad():
            mask = zero_mask(mask, args.num_iwt, 1)
            assert (mask[:, :, :128, :128] == 0).all()

        # Y only has first patch + mask
        x_wt_hat = Y + mask
        x_hat = iwt_fn(x_wt_hat)

        # Get x_wt, assuming deterministic WT model/function, and fill 0's in first patch
        x_wt = wt_model(data0)
        x_wt = zero_mask(x_wt, args.num_iwt, 1)
        
        # Calculate loss
        img_loss = (epoch >= args.img_loss_epoch)
        loss, loss_bce, loss_kld = iwt_model.loss_function(data0, x_hat, x_wt, x_wt_hat, mu, var, img_loss, kl_weight=args.kl_weight)
        loss.backward()

        # Calculating and printing gradient norm
        total_norm = calc_grad_norm_2(iwt_model)

        # Calculating and printing gradient norm
        global log_idx
        writer.add_scalar('Loss/total', loss, log_idx)
        writer.add_scalar('Loss/bce', loss_bce, log_idx)
        writer.add_scalar('Loss/kld', loss_kld, log_idx)
        writer.add_scalar('Gradient_norm/before', total_norm, log_idx)
        writer.add_scalar('KL_weight', args.kl_weight, log_idx)
        log_idx += 1 

        # Gradient clipping
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(iwt_model.parameters(), max_norm=args.grad_clip, norm_type=2)
            total_norm = calc_grad_norm_2(iwt_model)
            writer.add_scalar('Gradient_norm/clipped', total_norm, log_idx)
        
        train_losses.append([loss.cpu().item(), loss_bce.cpu().item(), loss_kld.cpu().item()])
        train_loss += loss

        optimizer.step()

        # Logging
        if batch_idx % args.log_interval == 0:
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss / len(data)))
            
            n = min(data.size(0), 8)  

    logging.info('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

def train_iwtvae_iwtmask(epoch, wt_model, iwt_model, optimizer, iwt_fn, train_loader, train_losses, args, writer):
    # toggle model to train mode
    iwt_model.train()
    train_loss = 0
    
    for batch_idx, data in enumerate(train_loader):
        
        data0 = data.to(iwt_model.device)
        data1 = data.to(wt_model.device)

        optimizer.zero_grad()
        
        # Get Y
        Y = wt_model(data1)
        
        # Zeroing out first patch, if given zero arg
        Y = zero_mask(Y, args.num_iwt, 1)

        # IWT all the leftover high frequencies
        Y = iwt_fn(Y)

        # Run model to get mask (zero out first patch of mask) and x_wt_hat
        mask, mu, var = iwt_model(Y)

        loss, loss_bce, loss_kld = iwt_model.loss_function(Y, mask, mu, var)
        loss.backward()

        # Calculating and printing gradient norm
        total_norm = calc_grad_norm_2(iwt_model)

        # Calculating and printing gradient norm
        global log_idx
        writer.add_scalar('Loss/total', loss, log_idx)
        writer.add_scalar('Loss/bce', loss_bce, log_idx)
        writer.add_scalar('Loss/kld', loss_kld, log_idx)
        writer.add_scalar('Gradient_norm/before', total_norm, log_idx)
        log_idx += 1 

        # Gradient clipping
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(iwt_model.parameters(), max_norm=args.grad_clip, norm_type=2)
            total_norm = calc_grad_norm_2(iwt_model)
            writer.add_scalar('Gradient_norm/clipped', total_norm, log_idx)
        
        train_losses.append([loss.cpu().item(), loss_bce.cpu().item(), loss_kld.cpu().item()])
        train_loss += loss

        optimizer.step()

        # Logging
        if batch_idx % args.log_interval == 0:
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss / len(data)))
            
            n = min(data.size(0), 8)  

    logging.info('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

# Taking in low frequency IWT'ed image to produce mask in autoencoder
def train_iwtae_iwtmask(epoch, wt_model, iwt_model, optimizer, iwt_fn, train_loader, train_losses, args, writer):
    # toggle model to train mode
    iwt_model.train()
    train_loss = 0
    
    for batch_idx, data in enumerate(train_loader):
        
        data = data.to(wt_model.device)

        optimizer.zero_grad()
        
        # Get Y
        Y = wt_model(data)
        
        # Zeroing out first patch, if given zero arg
        Y_mask = zero_mask(Y, args.num_iwt, 1)
        # IWT all the leftover high frequencies
        Y_mask = iwt_fn(Y_mask)

        # Getting IWT of only first patch
        Y_low = zero_patches(Y, args.num_iwt)
        Y_low = iwt_fn(Y_low)

        # Run model to get mask (zero out first patch of mask) and x_wt_hat
        mask, mu, var = iwt_model(Y_low)

        loss, loss_bce, loss_kld = iwt_model.loss_function(Y_mask, mask, mu, var)
        loss.backward()

        # Calculating and printing gradient norm
        total_norm = calc_grad_norm_2(iwt_model)

        # Calculating and printing gradient norm
        global log_idx
        writer.add_scalar('Loss/total', loss, log_idx)
        writer.add_scalar('Loss/bce', loss_bce, log_idx)
        writer.add_scalar('Loss/kld', loss_kld, log_idx)
        writer.add_scalar('Gradient_norm/before', total_norm, log_idx)
        log_idx += 1 

        # Gradient clipping
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(iwt_model.parameters(), max_norm=args.grad_clip, norm_type=2)
            total_norm = calc_grad_norm_2(iwt_model)
            writer.add_scalar('Gradient_norm/clipped', total_norm, log_idx)
        
        train_losses.append([loss.cpu().item(), loss_bce.cpu().item(), loss_kld.cpu().item()])
        train_loss += loss

        optimizer.step()

        # Logging
        if batch_idx % args.log_interval == 0:
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss / len(data)))
            
            n = min(data.size(0), 8)  

    logging.info('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

def train_iwtvae_3masks(epoch, wt_model, iwt_model, optimizer, iwt_fn, train_loader, train_losses, args, writer):
    # toggle model to train mode
    iwt_model.train()
    train_loss = 0
    
    for batch_idx, data in enumerate(train_loader):
        
        data = data.to(iwt_model.device)

        optimizer.zero_grad()
        
        # Get Y
        Y = wt_model(data)
        mask1 = Y[:, :, :128, 128:256]
        mask2 = Y[:, :, 128:256, :128]
        mask3 = Y[:, :, 128:256, 128:256]
        masks = torch.cat((mask1, mask2, mask3), dim=1)

        # Run model to get mask (zero out first patch of mask) and x_wt_hat
        mask1_hat, mask2_hat, mask3_hat, mu, var = iwt_model(masks)

        loss, loss_bce, loss_kld = iwt_model.loss_function(mask1, mask1_hat, mask2, mask2_hat, mask3, mask3_hat, mu, var)
        loss.backward()

        # Calculating and printing gradient norm
        total_norm = calc_grad_norm_2(iwt_model)

        # Calculating and printing gradient norm
        global log_idx
        writer.add_scalar('Loss/total', loss, log_idx)
        writer.add_scalar('Loss/bce', loss_bce, log_idx)
        writer.add_scalar('Loss/kld', loss_kld, log_idx)
        writer.add_scalar('Gradient_norm/before', total_norm, log_idx)
        log_idx += 1 

        # Gradient clipping
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(iwt_model.parameters(), max_norm=args.grad_clip, norm_type=2)
            total_norm = calc_grad_norm_2(iwt_model)
            writer.add_scalar('Gradient_norm/clipped', total_norm, log_idx)
        
        train_losses.append([loss.cpu().item(), loss_bce.cpu().item(), loss_kld.cpu().item()])
        train_loss += loss

        optimizer.step()

        # Logging
        if batch_idx % args.log_interval == 0:
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss / len(data)))
            
            n = min(data.size(0), 8)  

    logging.info('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))


def train_ae_mask(epoch, wt_model, model, criterion, optimizer, train_loader, train_losses, args, writer):
    # toggle model to train mode
    model.train()
    train_loss = 0
    
    for batch_idx, data in enumerate(train_loader):
        
        data = data.to(model.device)

        optimizer.zero_grad()
        
        # Get Y
        Y = wt_model(data)
        
        # Zeroing out first patch
        Y = zero_mask(Y, num_iwt=args.num_wt, cur_iwt=1)
        Y = preprocess_mask(Y, low=-0.1, high=0.1)
        assert ((Y[(Y >= -0.1) & (Y <= 0.1)] == 0).all())

        x_hat = model(Y.to(model.device))
        loss = model.loss_function(Y, x_hat, criterion)
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

def train_ae_mask_channels(epoch, wt_model, model, criterion, optimizer, train_loader, train_losses, args, writer):
    # toggle model to train mode
    model.train()
    train_loss = 0
    
    for batch_idx, data in enumerate(train_loader):
        
        data = data.to(model.device)

        optimizer.zero_grad()
        
        # Get Y
        Y = wt_model(data)
        
        # Zeroing out first patch
        Y = zero_mask(Y, num_iwt=args.num_wt, cur_iwt=1)
        if args.num_wt == 1:
            Y = hf_collate_to_channels(Y, device=model.device)
        elif args.num_wt == 2:
            Y = hf_collate_to_channels_wt2(Y, device=model.device)

        x_hat = model(Y)
        loss = model.loss_function(Y, x_hat, criterion)
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

def train_fullvae(epoch, full_model, optimizer, train_loader, train_losses, args, writer):
    # toggle model to train mode
    full_model.train()
    full_model.wt_model.train()
    full_model.iwt_model.train()
    train_loss = 0
    
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        
        y, mu_wt, logvar_wt, x_hat, mu, var = full_model(data)
        
        loss, loss_bce, loss_kld = full_model.loss_function(data, y, mu_wt, logvar_wt, x_hat, mu, var)
        loss.backward()
        
        train_losses.append([loss.cpu().item(), loss_bce.cpu().item(), loss_kld.cpu().item()])
        train_loss += loss
        
        # Calculating and printing gradient norm
        total_norm = 0
        for p in full_model.parameters():
            # If frozen parameters, then skip
            if not p.requires_grad:
                continue
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        logging.info('Gradient Norm: {}'.format(total_norm))

        # Gradient clipping
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(full_model.parameters(), max_norm=10000, norm_type=2)

        optimizer.step()
        if batch_idx % args.log_interval == 0:
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss / len(data)))
            
            n = min(data.size(0), 8)  

    logging.info('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

def train_full_wtvae128_iwtae512(epoch, full_model, optimizer, train_loader, train_losses, args, writer):
    # toggle model to train mode, IWT model in eval b/c frozen
    full_model.train()
    full_model.wt_model.train()
    full_model.iwt_model.eval()
    train_loss = 0
    
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        
        X_128, X_512 = data

        Y_low_hat, mask_hat, X_hat, mu, logvar = full_model(X_128)

        loss, loss_bce, loss_kld = full_model.loss_function(X_512, Y_low_hat, X_hat, mu, logvar, args.kl_weight)
        loss.backward()
        
        train_losses.append([loss.cpu().item(), loss_bce.cpu().item(), loss_kld.cpu().item()])
        train_loss += loss
        
        # Calculating and printing gradient norm
        total_norm = calc_grad_norm_2(full_model)
        
        global log_idx
        writer.add_scalar('Loss/total', loss, log_idx)
        writer.add_scalar('Loss/bce', loss_bce, log_idx)
        writer.add_scalar('Loss/kld', loss_kld, log_idx)
        writer.add_scalar('Gradient_norm/before', total_norm, log_idx)
        log_idx += 1 

        # Gradient clipping
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(full_model.parameters(), max_norm=10000, norm_type=2)
            total_norm = calc_grad_norm_2(full_model)
            writer.add_scalar('Gradient_norm/clipped', total_norm, log_idx)

        optimizer.step()
        if batch_idx % args.log_interval == 0:
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(X_128),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss / len(X_128)))
            
            n = min(X_128.size(0), 8)  

    logging.info('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))


def train_wtcnn(epoch, model, optimizer, train_loader, train_losses, args, writer):
    # toggle model to train mode
    model.train()
    train_loss = 0

    for batch_idx, data in enumerate(train_loader):
        
        if model.cuda:
            data = data.to(model.device)

        optimizer.zero_grad()
        
        wt_data = model(data)
        loss = model.loss_function(data, wt_data)
        loss.backward()

        # Calculating and printing gradient norm
        total_norm = 0
        for p in model.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)

        global log_idx
        writer.add_scalar('Loss/bce', loss, log_idx)
        writer.add_scalar('Gradient_norm/before', total_norm, log_idx)
        log_idx += 1 

        # Gradient clipping
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip, norm_type=2)
            # Calculating and printing gradient norm
            total_norm = 0
            for p in model.parameters():
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
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
            
    writer.flush()
    logging.info('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))
