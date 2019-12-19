import logging
import os

import click
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch_unet.globals import *
from torch_unet.tools.dataset import TrainingSet
from torch_unet.tools.losses import dice_loss_withlogits
from torch_unet.unet import UNet
from torch_unet.unet.train import train_model


def split_train_val(dataset, val_ratio, batch_size):
    """ Splits the given dataset into train/val. Returns two `DataLoader` objects
    
    :param dataset: The dataset object
    :param val_ratio: Validaiton ratio
    :param batch_size: Batch size
    :return:
    """
    n_val = int(len(dataset) * val_ratio)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return n_train, train_loader, n_val, val_loader


def get_model_dir(patch_size, step, depth, batch_size, lr, decay, padding, batch_norm,
                  dropout, rotation, init_filters, leaky, balance, augmentation, dice, create=True):
    """
    Returns the directory name of the model
    """
    name = f"depth{depth}_BS{batch_size}_lr{lr}_PS{patch_size}_ST{step}_WF{init_filters}"
    if padding:
        name += "_padding"
    if batch_norm:
        name += "_batchnorm"
    if rotation:
        name += "_rot"
    if decay > 0:
        name += f"_decay"
    if dropout > 0:
        name += f"_dropout{dropout}"
    if leaky:
        name += "_leaky"
    if balance:
        name += "_balance"
    if dice:
        name += '_dice'
    if augmentation:
        name += "_augmentation"
    
    model_dir = os.path.join(MODELS_DIR, name)
    dir_checkpoint = os.path.join(model_dir, "checkpoints/")
    if create and not os.path.exists(dir_checkpoint):
        os.makedirs(dir_checkpoint, exist_ok=True)
    return dir_checkpoint, name


def train(epochs, lr, decay, val_ratio, batch_size, patch_size, step, depth, num_filters, padding, batch_norm, dropout,
          leaky, rotations, balance, dice, augmentation):
    """
    Trains a model given the parameters
    """
    
    # Get model directory and create it if necessary
    dir_checkpoint, name = get_model_dir(patch_size, step, depth, batch_size, lr, decay, padding, batch_norm, dropout,
                                         rotations, num_filters, leaky, balance, augmentation, dice)
    
    # Load the training set
    dataset = TrainingSet(IMAGE_DIR, MASK_DIR, mask_threshold=MASK_THRESHOLD,
                          rotation_angles=ROTATION_ANGLES if rotations else None, patch_size=patch_size,
                          step=step if step is None else int(step), augmentation=augmentation)
    
    # Split train/val
    n_train, train_loader, n_val, val_loader = split_train_val(dataset, val_ratio, batch_size)
    
    # Writer for tensorboard
    writer = SummaryWriter(comment=name)
    
    # Create the model
    net = UNet(n_channels=NUM_CHANNELS, n_classes=N_CLASSES, depth=depth, init_filters=num_filters, padding=padding,
               batch_norm=batch_norm, dropout=dropout, leaky=leaky)
    
    # Optimizer ADAM
    optimizer = optim.Adam(net.parameters(), lr=lr)
    
    # Loss function
    criterion = nn.BCEWithLogitsLoss()
    if dice:
        criterion = dice_loss_withlogits
    
    # Learning rate decay
    lr_scheduler = None
    if decay:
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    
    # Start training
    train_model(epochs, criterion, optimizer, lr_scheduler, net, train_loader, val_loader, dir_checkpoint, logger, n_train,
                n_val, batch_size, writer, val_ratio, balance)


@click.command()
@click.option("--epochs", default=35, help="Number of epochs to train (default 35)")
@click.option("--lr", default=0.001, help="Learning rate (default 0.001)")
@click.option("--decay", is_flag=True, help="Decay the learning rate on plateau")
@click.option("--val-ratio", default=0.2, help="Validation ratio (default 0.2)")
@click.option("--batch-size", default=128, help="Batch size (default 128)")
@click.option("--patch-size", default=80, help="Patch size (default 80)")
@click.option("--step", default=20, help="Patch step (default 20)")
@click.option("--depth", default=3, help="U-Net depth (default 3)")
@click.option("--num-filters", default=64, help="Number of filters at first layer (default 64)")
@click.option("--padding", is_flag=True, help="Use padding")
@click.option("--batch-norm", is_flag=True, help="Use Batch Normalization")
@click.option("--dropout", default=0., help="Use Dropout (default 0)")
@click.option("--leaky", is_flag=True, help="Use leaky ReLU activation")
@click.option("--rotations", is_flag=True, help="Rotate original images")
@click.option("--balance", is_flag=True, help="Use BCE with class balance (not to be used with --dice)")
@click.option("--dice", is_flag=True, help="Use Dice loss")
@click.option("--augmentation", is_flag=True, help="Use stochastic data augmentation")
def main(epochs, lr, decay, val_ratio, batch_size, patch_size, step, depth, num_filters, padding, batch_norm, dropout,
         leaky, rotations, balance, dice, augmentation):
    """
    Trains a U-Net model given the parameters

    """
    
    train(epochs, lr, decay, val_ratio, batch_size, patch_size, step, depth, num_filters, padding, batch_norm, dropout,
          leaky, rotations, balance, dice, augmentation)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    main()
