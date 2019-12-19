from skimage.util import view_as_windows


def flatten_patches(patches):
    p_h, p_w = patches.shape[:2]
    new_shape = (p_h * p_w,) + patches.shape[2:]
    return patches.reshape(new_shape)


def extract_patches_with_step(img, patch_size, step):
    if len(img.shape) == 2:  # single channel
        shape = (patch_size, patch_size)
    elif len(img.shape) == 3:  # RGB Channel
        shape = (patch_size, patch_size, 3)
    else:
        raise ValueError("Image must be of dimension 2 or 3")
    
    overlapped_patches = view_as_windows(img, shape, step)
    if len(img.shape) == 3:
        overlapped_patches = overlapped_patches.squeeze(2)  # Remove additional dimension
    
    return flatten_patches(overlapped_patches)


def extract_non_overlapping_patches(img, patch_size):
    h = img.shape[0]
    assert h % patch_size == 0, f"Cannot extract patch for img shape {h}x{h} patch_size={patch_size}"
    
    return extract_patches_with_step(img, patch_size, patch_size)


def get_image_patches(img, patch_size, step=None):
    if step is None:
        return extract_non_overlapping_patches(img, patch_size)
    else:
        return extract_patches_with_step(img, patch_size, step)
