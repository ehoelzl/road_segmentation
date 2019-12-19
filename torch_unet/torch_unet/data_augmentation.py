from albumentations import HorizontalFlip, RandomRotate90, Transpose, VerticalFlip


class StochasticAugmentation:
    
    def __init__(self, prob):
        self.horizontal = HorizontalFlip(p=prob)
        self.vertical = VerticalFlip(p=prob)
        self.rotate = RandomRotate90(p=prob)
        self.transpose = Transpose(p=prob)
    
    def augment_image(self, img, mask):
        # Input must be HWC
        augmented = self.horizontal(image=img, mask=mask)
        augmented = self.vertical(**augmented)
        augmented = self.rotate(**augmented)
        augmented = self.transpose(**augmented)
        
        return augmented['image'], augmented['mask']
