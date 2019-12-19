import os

import numpy as np
import torch
from torch_unet.globals import SAVE_EVERY
from torch_unet.tools.evaluation import eval_net, eval_net_full
from torch_unet.tools.losses import dice_loss
from torch_unet.utils import get_lr
from tqdm import tqdm


def train_model(epochs, criterion, optimizer, lr_scheduler, net, train_loader, val_loader, dir_checkpoint, logger, n_train,
                n_val, batch_size, writer, val_ratio, balance_classes):
    # torch.multiprocessing.set_start_method('spawn')
    # Register device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device {device}')
    # Create the Network
    
    net.to(device=device)
    
    dataset_length = n_val + n_train
    global_step = 0
    
    for epoch in range(epochs):
        net.train()  # Sets module in training mode
        epoch_loss = []
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                imgs = imgs.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)
                
                if balance_classes:
                    # Neg / pos to rectify class imbalance
                    pos_weight = torch.sum(torch.abs(true_masks - 1)) / torch.sum(true_masks)
                    criterion.pos_weight = torch.tensor([pos_weight]).to(device=device, dtype=torch.float32)
                # Optimization step
                optimizer.zero_grad()
                masks_pred = net(imgs)  # Make predictions
                loss = criterion(masks_pred, true_masks)  # Evaluate loss
                batch_loss = loss.item()
                loss.backward()
                optimizer.step()
                
                # Add data to tensorboard
                epoch_loss.append(batch_loss)  # Add loss to epoch
                writer.add_scalar('Train/BCE_loss', batch_loss, global_step)
                d_loss = dice_loss(torch.sigmoid(masks_pred), true_masks)
                writer.add_scalar('Train/Dice_loss', d_loss, global_step)
                pbar.set_postfix(**{'loss (batch)': batch_loss})
                pbar.update(imgs.shape[0])
                
                global_step += 1
                
                # Validation every 10 batches
                if global_step % (dataset_length // (10 * batch_size)) == 0 and n_val > 0:
                    net.eval()
                    val_score, val_loss = eval_net(net, val_loader, device, n_val)
                    net.train()  # Reset in training mode
                    
                    logger.info('Validation Dice Coeff: {}'.format(val_score))
                    writer.add_scalar('Validation/Dice_coef', val_score, global_step)
                    writer.add_scalar('Validation/Dice_loss', val_loss, global_step)
                    writer.add_images('images', imgs, global_step)
                    writer.add_images('masks/true', true_masks, global_step)
                    writer.add_images('masks/pred', torch.sigmoid(masks_pred), global_step)
                
                # Evaluate on full image every 300 steps
                if global_step % 300 == 0 and n_val > 0:
                    net.eval()
                    val_full_score, val_full_loss, img, true_mask, mask_pred = eval_net_full(net, val_loader, device,
                                                                                             val_ratio)
                    net.train()
                    
                    logger.info('Full Validation Dice Coeff: {}'.format(val_full_score))
                    writer.add_scalar('Full_Validation/Dice_coef', val_full_score, global_step)
                    writer.add_scalar('Full_Validation/Dice_loss', val_full_loss, global_step)
                    
                    writer.add_images('full_images', img[None, :, :, :], global_step)
                    writer.add_images('full_masks/true', true_mask[None, :, :, :], global_step)
                    writer.add_images('full_masks/pred', mask_pred[None, :, :, :], global_step)
                
                if (global_step + 1) % SAVE_EVERY == 0:
                    torch.save(net.state_dict(),
                               dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
                    logger.info(f'Checkpoint {epoch + 1} saved !')
        
        if lr_scheduler is not None:
            ep_loss = int(np.mean(epoch_loss) * 1000)
            lr_scheduler.step(ep_loss)
            writer.add_scalar("LR/epoch_loss", epoch_loss)
            writer.add_scalar("LR", get_lr(optimizer), global_step)
    
    writer.close()
    torch.save(net.state_dict(), os.path.join(dir_checkpoint, "final.pth"))
