import numpy as np
from scipy.ndimage import distance_transform_edt


def unmask_nearestneighbor(image, mask):
    """
    Fills in masked pixels by taking the nearest unmasked value.

    image: 2d numpy array image
    mask: 2d numpy boolean array with 1 or True indicating a pixel is masked
    """
    # identify the nearest unmasked pixel for every masked pixel
    nearest_neighbor = distance_transform_edt(
        mask == 1, return_distances=False, return_indices=True
    )

    # Replace the masked values with the nearest neighbor unmasked pixels
    image[mask == 1] = image[
        nearest_neighbor[0][mask == 1], nearest_neighbor[1][mask == 1]
    ]

    return image
