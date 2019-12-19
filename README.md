# Road segmentation with U-Net architecture
_Edoardo Tarek Hölzl, Hugo Vincent Eliot Bertran Roussel_

This project presents our solution to the problem of segmenting Satellite images into two classes: Road and Background.

We use the [U-Net](https://arxiv.org/pdf/1505.04597.pdf) architecture that allows us to output per pixel probability of Road.
The project was done in the context of the Machine Learning course (CS-433) at EPFL.

After training for a few hours, our best model obtains an accuracy of 90.9 % on the provided test set.


## ADD EXAMPLE IMAGE

## Dependencies
The code base requires the following dependencies (also in requirements.txt): 

* numpy 1.17.4
* pytorch 1.3.1
* opencv-python 4.1.2.30
* scikit-image 0.16.2
* click 7.0
* albumentations 0.4.3
* imgaug 0.2.5

All dependencies, as well as the project library can be installed by using `pip install .` from within `road_segmentation/torch_unet`


## Training and Predicting
There are two main scripts : training and predicting.
### 1. Training
The train script must be run from `road_segmentation/torch_unet/` :
```bash
python torch_unet/scripts/train.py --help

Usage: train.py [OPTIONS]

  Trains a U-Net model given the parameters

Options:
  --epochs INTEGER       Number of epochs to train (default 35)
  --lr FLOAT             Learning rate (default 0.001)
  --decay                Decay the learning rate on plateau
  --val-ratio FLOAT      Validation ratio (default 0.2)
  --batch-size INTEGER   Batch size (default 128)
  --patch-size INTEGER   Patch size (default 80)
  --step INTEGER         Patch step (default 20)
  --depth INTEGER        U-Net depth (default 3)
  --num-filters INTEGER  Number of filters at first layer (default 64)
  --padding              Use padding
  --batch-norm           Use Batch Normalization
  --dropout FLOAT        Use Dropout (default 0)
  --leaky                Use leaky ReLU activation
  --rotations            Rotate original images
  --balance              Use BCE with class balance (not to be used with
                         --dice)
  --dice                 Use Dice loss
  --augmentation         Use stochastic data augmentation
  --help                 Show this message and exit.
```

The default parameters have been set to be the ones that give the best model. If you would like to train it again, run the following command:
```bash
python torch_unet/train.py --val-ratio=0 --padding --dropout=0.2 --batch-norm --rotations --augmentation --dice
```


All trained models will be saved in `road_segmentation/models/`

Also using tensorboard to monitor training is possible. Simply run tensorboard and point the logdir to `runs/`
### 2. Predicting
To predict the test images on a trained model 
```bash
python torch_unet/predict.py --help

Usage: predict.py [OPTIONS]

  Predicts the test images given the model path. Has the possibility of
  combining two models' outputs to get better accuracy

Options:
  --model-path TEXT      Model checkpoint path
  --model-depth INTEGER  Model depth
  --padding              Use padding
  --num-filters INTEGER  Number of filters at first layer (default 64)
  --batch-norm           Use batch normalization
  --dropout FLOAT        Dropout probability (default 0)
  --leaky                Use leaky activation
  --model-path-2 TEXT    Path of second model (optional)
  --rotate               Predict also on rotated images
  --help                 Show this message and exit.
```


In order to simply re-produce our last submission, just run `python run.py` from within `road_segmentation/torch_unet/`.
Image predictions will be saved in `road_segmentation/predictions` and the submission file in `road_segmentation/submissions`.