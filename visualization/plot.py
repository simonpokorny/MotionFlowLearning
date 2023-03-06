import math

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def visualise_tensor(tensor_batch, legend="Plot of tensor"):
    """
    Visualizes a PyTorch tensor in the shape [batch size, channels, height, width] as a grid of images using Matplotlib.

    Parameters:
        tensor_batch (torch.Tensor): The tensor to visualize, with shape [batch size, channels, height, width].
        legend (str): The title of the plot (default: "Plot of tensor").

    Example:
        # Create a random tensor with 2 images, 3 channels, and 64x64 resolution
        tensor = torch.randn(2, 3, 64, 64)

        # Visualize the tensor as a grid of images
        visualise_tensor(tensor)
    """
    assert tensor_batch.ndim == 4
    # Get number of channels
    batch_size, channels, _, _ = tensor_batch.shape
    nrows = ncols = int(np.ceil(np.sqrt(batch_size)))

    # Convert the tensor to a numpy array and transpose the dimensions to [batch size, height, width, channels]
    tensor_np = tensor_batch.numpy().transpose(0, 2, 3, 1)

    # Create a figure with subplots for each image in the batch
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)

    # Loop over the images in the batch and plot them in the corresponding subplot
    for i in range(nrows * ncols):
        if i < batch_size:
            fig.axes[i].imshow(tensor_np[i])
        else:
            fig.axes[i].set_axis_off()

    # Show the plot
    plt.legend(legend)
    plt.show()


def plot_pillars_heatmap(indices, x_max, x_min, y_max, y_min, grid_cell_size):
    fig = plt.figure(figsize=(15, 15))
    x, y = indices[:, 0], indices[:, 1]
    n_pillars_x = math.floor((x_max - x_min) / grid_cell_size)

    plt.hist2d(x, y, bins=n_pillars_x, cmap='YlOrBr', density=True)

    cb = plt.colorbar()
    cb.set_label('Number of points in pillar')

    plt.title('Heatmap of pillars (bev projection)')
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.show()


def plot_pillars(indices, x_max, x_min, y_max, y_min, grid_cell_size):
    fig = plt.figure(figsize=(15, 15))
    ax = plt.axes(projection="3d")

    n_pillars_x = math.floor((x_max - x_min) / grid_cell_size)
    n_pillars_y = math.floor((y_max - y_min) / grid_cell_size)
    pillar_matrix = np.zeros(shape=(n_pillars_x, n_pillars_y, 1))

    for x, y in indices:
        pillar_matrix[x, y] += 1

    x_pos, y_pos, z_pos = [], [], []
    x_size, y_size, z_size = [], [], []

    for i in range(pillar_matrix.shape[0]):
        for j in range(pillar_matrix.shape[1]):
            x_pos.append(i * grid_cell_size)
            y_pos.append(j * grid_cell_size)
            z_pos.append(0)

            x_size.append(grid_cell_size)
            y_size.append(grid_cell_size)
            z_size.append(int(pillar_matrix[i, j]))

    ax.bar3d(x_pos, y_pos, z_pos, x_size, y_size, z_size)
    plt.title("3D projection of pillar map")
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.show()


def plot_tensor(tensor, tittle="None"):
    plt.imshow(tensor.numpy())
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.title(tittle)
    plt.show()


def plot_2d_point_cloud(pc):
    fig, ax = plt.subplots(figsize=(15, 15))

    x, y = [], []
    for p in pc:
        x.append(p[0])
        y.append(p[1])
    ax.scatter(x, y, color="green")
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.title("2D projection of pointcloud")
    plt.show()


def visualize_point_cloud(points):
    """ Input must be a point cloud of shape (n_points, 3) """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([point_cloud])


def visualize_flows(vis, points, flows):
    """
    Visualize a 3D point cloud where is point is flow-color-coded
    :param vis: visualizer created with open3D, for example:

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1280, height=720)

    :param points: (n_points, 3)
    :param flows: (n_points, 3)
    """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(flows)
    # vis.destroy_window()
