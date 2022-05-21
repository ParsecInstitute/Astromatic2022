import numpy as np
from scipy.ndimage import distance_transform_edt


def unmask_nearestneighbor(image, mask):

    nearest_neighbor = distance_transform_edt(
        mask == 1, return_distances=False, return_indices=True
    )

    image[mask == 1] = image[
        nearest_neighbor[0][mask == 1], nearest_neighbor[1][mask == 1]
    ]
