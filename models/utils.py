import matplotlib.pyplot as plt
import torch
import numpy as np
from argparse import ArgumentTypeError


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


def init_weights(m) -> None:
    """
    Apply the weight initialization to a single layer.
    Use this with your_module.apply(init_weights).
    The single layer is a module that has the weights parameter. This does not yield for all modules.
    :param m: the layer to apply the init to
    :return: None
    """
    if type(m) in [torch.nn.Linear, torch.nn.Conv2d]:
        # Note: There is also xavier_normal_ but the paper does not state which one they used.
        torch.nn.init.xavier_uniform_(m.weight)


def remove_out_of_bounds_points(pc, y, x_min, x_max, y_min, y_max, z_min, z_max):
    # Max needs to be exclusive because the last grid cell on each axis contains
    # [((grid_size - 1) * cell_size) + *_min, *_max).
    #   E.g grid_size=512, cell_size = 170/512 with min=-85 and max=85
    # For z-axis this is not necessary, but we do it for consistency
    mask = (pc[:, 0] >= x_min) & (pc[:, 0] < x_max) \
           & (pc[:, 1] >= y_min) & (pc[:, 1] < y_max) \
           & (pc[:, 2] >= z_min) & (pc[:, 2] < z_max)
    pc_valid = pc[mask]
    y_valid = None
    if y is not None:
        y_valid = y[mask]
    return pc_valid, y_valid


def create_pillars_matrix(pc_valid, grid_cell_size, x_min, y_min, z_min, z_max, n_pillars_x):
    """
    Compute the pillars using matrix operations.
    :param pc: point cloud data. (N_points, features) with the first 3 features being the x,y,z coordinates.
    :return: augmented_pointcloud, grid_cell_indices, y_valid
    """
    num_laser_features = pc_valid.shape[1] - 3  # Calculate the number of laser features that are not the coordinates.

    # Calculate the cell id that this entry falls into
    # Store the X, Y indices of the grid cells for each point cloud point
    grid_cell_indices = np.zeros((pc_valid.shape[0], 2), dtype=int)
    grid_cell_indices[:, 0] = ((pc_valid[:, 0] - x_min) / grid_cell_size).astype(int)
    grid_cell_indices[:, 1] = ((pc_valid[:, 1] - y_min) / grid_cell_size).astype(int)

    # Initialize the new pointcloud with 8 features for each point
    augmented_pc = np.zeros((pc_valid.shape[0], 6 + num_laser_features))
    # Set every cell z-center to the same z-center
    augmented_pc[:, 2] = z_min + ((z_max - z_min) * 1 / 2)
    # Set the x cell center depending on the x cell id of each point
    augmented_pc[:, 0] = x_min + 1 / 2 * grid_cell_size + grid_cell_size * grid_cell_indices[:, 0]
    # Set the y cell center depending on the y cell id of each point
    augmented_pc[:, 1] = y_min + 1 / 2 * grid_cell_size + grid_cell_size * grid_cell_indices[:, 1]

    # Calculate the distance of the point to the center.
    # x
    augmented_pc[:, 3] = pc_valid[:, 0] - augmented_pc[:, 0]
    # y
    augmented_pc[:, 4] = pc_valid[:, 1] - augmented_pc[:, 1]
    # z
    augmented_pc[:, 5] = pc_valid[:, 2] - augmented_pc[:, 2]

    # Take the two laser features
    augmented_pc[:, 6:] = pc_valid[:, 3:]
    # augmented_pc = [cx, cy, cz,  Δx, Δy, Δz, l0, l1]

    # Convert the 2D grid indices into a 1D encoding
    # This 1D encoding is used by the models instead of the more complex 2D x,y encoding
    # To make things easier we transform the 2D indices into 1D indices
    # The cells are encoded as j = x * grid_width + y and thus give an unique encoding for each cell
    # E.g. if we have 512 cells in both directions and x=1, y=2 is encoded as 512 + 2 = 514.
    # Each new row of the grid (x-axis) starts at j % 512 = 0.
    grid_cell_indices = grid_cell_indices[:, 0] * n_pillars_x + grid_cell_indices[:, 1]

    return augmented_pc, grid_cell_indices


if __name__ == "__main__":

    a = torch.rand((7,1,640,640))
    visualise_tensor(a)