import matplotlib.pyplot as plt
import torch
import numpy as np


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


if __name__ == "__main__":

    a = torch.rand((7,1,640,640))
    visualise_tensor(a)