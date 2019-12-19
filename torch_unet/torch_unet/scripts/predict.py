import gc
import logging
import os

import click
import matplotlib.image as mpimg
import torch
from torch_unet.globals import *
from torch_unet.tools.dataset import TestSet
from torch_unet.unet import UNet, predict_full_image
from tqdm import tqdm


def predict(model_path, model_depth, padding, num_filters, batch_norm, dropout, leaky, model_path_2, rotate):
    # Create prediction destination dir
    if not os.path.exists(PREDICTION_DEST_DIR):
        os.makedirs(PREDICTION_DEST_DIR)
    
    # Load testset
    test_set = TestSet(TEST_SET)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    
    # Load first model
    net_1 = UNet(n_channels=NUM_CHANNELS, n_classes=N_CLASSES, depth=int(model_depth), padding=padding,
                 batch_norm=batch_norm,
                 init_filters=num_filters, dropout=dropout, leaky=leaky)
    
    net_1.to(device=device)
    net_1.load_state_dict(state_dict=torch.load(model_path, map_location=device))
    
    # If specified, load second model
    if model_path_2 is not None:
        net_2 = UNet(n_channels=NUM_CHANNELS, n_classes=N_CLASSES, depth=int(model_depth), padding=padding,
                     batch_norm=batch_norm,
                     init_filters=num_filters, dropout=dropout, leaky=leaky)
        
        net_2.to(device=device)
        
        net_2.load_state_dict(state_dict=torch.load(model_path_2, map_location=device))
        net_2.eval()
    
    net_1.eval()
    for i in tqdm(range(len(test_set)), desc="Predicting"):
        img = test_set.get_raw_image(i)
        idx = test_set[i]['id']
        prediction_1 = predict_full_image(net_1, img, device, rotate)[0]
        
        if model_path_2 is not None:
            prediction_2 = predict_full_image(net_2, img, device, rotate)[0]
            img = prediction_1 * 0.5 + prediction_2 * 0.5
        else:
            img = prediction_1
        mpimg.imsave(PREDICTION_DEST_DIR + idx + ".png", img)
        gc.collect()


@click.command()
@click.option("--model-path", help="Model checkpoint path")
@click.option("--model-depth", default=3, help="Model depth")
@click.option("--padding", is_flag=True, help="Use padding")
@click.option("--num-filters", default=64, help="Number of filters at first layer (default 64)")
@click.option("--batch-norm", is_flag=True, help="Use batch normalization")
@click.option("--dropout", default=0., help="Dropout probability (default 0)")
@click.option("--leaky", is_flag=True, help="Use leaky activation")
@click.option("--model-path-2", default=None, help="Path of second model (optional)")
@click.option("--rotate", is_flag=True, help="Predict also on rotated images")
def main(model_path, model_depth, padding, num_filters, batch_norm, dropout, leaky, model_path_2, rotate):
    """Predicts the test images given the model path. Has the possibility of combining two models' outputs to get better accuracy"""
    predict(model_path, model_depth, padding, num_filters, batch_norm, dropout, leaky, model_path_2, rotate)


if __name__ == "__main__":
    main()
