import os

from torch_unet.globals import PREDICTION_DEST_DIR, SUBMISSION_DIR
from torch_unet.mask_to_submission import masks_to_submission
from torch_unet.scripts import predict, train

TRAIN = False  # Change this to train model

MODEL_DEPTH = 3
LR = 0.001

PATCH_SIZE, STEP = 80, 20
PADDING = True
NUM_FILTERS = 64
BATCH_SIZE = 128
AUGMENTATION, ROTATION = True, True
DECAY, BATCHNORM, DROPOUT = False, True, 0.2
NUM_EPOCHS = 27
DICE, BALANCE, LEAKY = True, False, False
ROTATE = True

MODEL_PATH = ""


def run():
    if TRAIN:
        print("Starting to train model")
        train.train(NUM_EPOCHS, LR, DECAY, 0, BATCH_SIZE, PATCH_SIZE, STEP, MODEL_DEPTH, NUM_FILTERS,
                    PADDING, BATCHNORM, DROPOUT, LEAKY, ROTATION, BALANCE, DICE, AUGMENTATION)
    model_dir, _ = train.get_model_dir(PATCH_SIZE, STEP, MODEL_DEPTH, BATCH_SIZE, LR, DECAY, PADDING, BATCHNORM,
                                       DROPOUT, ROTATION, NUM_FILTERS, LEAKY, BALANCE, AUGMENTATION, DICE, create=False)
    model_name = f"CP_epoch{NUM_EPOCHS}.pth"
    model_path = os.path.join(model_dir, model_name)
    
    print(f"Predicting using model {model_path}")
    
    predict.predict(model_path=model_path, model_depth=MODEL_DEPTH, padding=PADDING, num_filters=NUM_FILTERS,
                    batch_norm=BATCHNORM, dropout=DROPOUT, leaky=LEAKY, model_path_2=None, rotate=ROTATE)
    
    submission_filename = os.path.join(SUBMISSION_DIR, "submission.csv")
    if not os.path.exists(SUBMISSION_DIR):
        os.makedirs(SUBMISSION_DIR)
    image_filenames = []
    for i in range(1, 51):
        image_filename = os.path.join(PREDICTION_DEST_DIR, "test_" + "%.d" % i + ".png")
        image_filenames.append(image_filename)
        masks_to_submission(submission_filename, *image_filenames)


if __name__ == "__main__":
    run()
