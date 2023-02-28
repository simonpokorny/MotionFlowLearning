import numpy as np
import pytorch_lightning as pl
import torch

from models.models.BaseModelSlim import BaseModelSlim
from models.models.utils import init_weights, visualise_tensor
from models.networks import PillarFeatureNetScatter, PointFeatureNet, MovingAverageThreshold, RAFT
from models.networks.slimdecoder import OutputDecoder


class SLIM(BaseModelSlim):
    def __init__(self, n_pillars_x=640,
                 n_pillars_y=640,
                 point_features=8,
                 corr_levels=4,
                 corr_radius=3,
                 alternate_corr=False,
                 feature_downsampling_factor=8,
                 predict_weight_for_static_aggregation=True,
                 hdim=96,
                 cdim=64,
                 iters=6,
                 flow_maps_archi="single",
                 corr_module="all",
                 predict_logits=True,
                 learn_upsampling_mask=False
                 ):
        """
        Args:
            n_pillars_x (int): The number of pillars in the x dimension of the voxel grid.
            n_pillars_y (int): The number of pillars in the y dimension of the voxel grid.
            point_features (int): The number of features in each point of the input point clouds.
            corr_levels (int): The number of correlation levels for the correlation module in the network.
            corr_radius (int): The radius for the correlation module in the network.
            alternate_corr (bool): A boolean indicating whether to use alternate correlation in the network.
            feature_downsampling_factor (int): The downsampling factor for the features in the network.
            predict_weight_for_static_aggregation (bool): A boolean indicating whether to predict weight for static aggregation in the network.
            hdim (int): The number of hidden dimensions in the network.
            cdim (int): The number of correlation dimensions in the network.
            iters (int): The number of iterations for the optical flow algorithm in the network.
            flow_maps_archi (str): The architecture to use for the flow maps in the network (either "vanilla" or "single").
            corr_module (str): The correlation module to use in the network (either "all" or "full").
            predict_logits (bool): A boolean indicating whether to predict logits or probabilities in the network.
            learn_upsampling_mask (bool): A boolean indicating whether to learn the upsampling mask in the network.
        """

        super(SLIM, self).__init__()
        self.save_hyperparameters()  # Store the constructor parameters into self.hparams
        self.n_pillars_x = n_pillars_x
        self.n_pillars_y = n_pillars_y

        self._point_feature_net = PointFeatureNet(in_features=point_features, out_features=64)
        self._point_feature_net.apply(init_weights)

        self._pillar_feature_net = PillarFeatureNetScatter(n_pillars_x=n_pillars_x, n_pillars_y=n_pillars_y)
        self._pillar_feature_net.apply(init_weights)

        self._raft = RAFT(iters=iters,
                          corr_levels=corr_levels,
                          corr_radius=corr_radius,
                          alternate_corr=alternate_corr,
                          feature_downsampling_factor=feature_downsampling_factor,
                          predict_weight_for_static_aggregation=predict_weight_for_static_aggregation,
                          hdim=hdim,
                          cdim=cdim,
                          feat_for_corr_dim=feature_downsampling_factor,
                          flow_maps_archi=flow_maps_archi,
                          corr_module=corr_module,
                          predict_logits=predict_logits,
                          learn_upsampling_mask=learn_upsampling_mask
                          )

        # TODO change parameters of moving threshold for dataset
        self._moving_dynamicness_threshold = MovingAverageThreshold(unsupervised=True,
                                                                    num_train_samples=10,
                                                                    num_moving=1234786)

        # TODO propagate params
        self._decoder_fw = OutputDecoder()
        self._decoder_bw = OutputDecoder()

    def _transform_point_cloud_to_embeddings(self, pc, mask):
        """
         A method that takes a point cloud and a mask and returns the corresponding embeddings.
         The method flattens the point cloud and mask, applies the point feature network,
         and then reshapes the embeddings to their original dimensions.
        """
        pc_flattened = pc.flatten(0, 1)
        mask_flattened = mask.flatten(0, 1)
        # Init the result tensor for our data. This is necessary because the point net
        # has a batch norm and this needs to ignore the masked points
        batch_pc_embedding = torch.zeros((pc_flattened.size(0), 64), device=pc.device, dtype=pc.dtype)
        # Flatten the first two dimensions to get the points as batch dimension
        batch_pc_embedding[mask_flattened] = self._point_feature_net(pc_flattened[mask_flattened])
        # This allows backprop towards the MLP: Checked with backward hooks. Gradient is present.
        # Output is (batch_size * points, embedding_features)
        # Retransform into batch dimension (batch_size, max_points, embedding_features)
        batch_pc_embedding = batch_pc_embedding.unflatten(0, (pc.size(0), pc.size(1)))
        # 241.307 MiB    234
        return batch_pc_embedding

    def _batch_grid_2D(self, batch_grid):
        """
        A method that takes a batch of grid indices and returns the corresponding 2D grid coordinates.
        The method calculates the x and y indices of the grid points using the number of pillars in
        the x and y dimensions, respectively, and then concatenates them along the second dimension.
        """
        grid = np.hstack(
            ((batch_grid // self.n_pillars_x)[:, np.newaxis], (batch_grid % self.n_pillars_y)[:, np.newaxis]))
        return np.moveaxis(grid, -1, 1)

    def _filled_pillar_mask(self, batch_grid, batch_mask):
        """
        A method that takes a batch of grid indices and masks and returns a tensor with a 1 in the location
        of each grid point and a 0 elsewhere. The method creates a tensor of zeros with the same shape as
        the voxel grid, and then sets the locations corresponding to the grid points in the batch to 1.
        """
        bs = batch_grid.shape[0]
        # pillar mask
        pillar_mask = torch.zeros((bs, 1, self.n_pillars_x, self.n_pillars_y))
        #
        x = batch_grid[batch_mask][..., 0]
        y = batch_grid[batch_mask][..., 1]
        pillar_mask[:, :, x, y] = 1
        return pillar_mask

    def forward(self, x, transforms_matrices):
        """
        The usual forward pass function of a torch module
        :param x:
        :param transforms_matrices:
        :return:
        """
        # 1. Do scene encoding of each point cloud to get the grid with pillar embeddings
        # Input is a point cloud each with shape (N_points, point_features)

        # The input here is more complex as we deal with a batch of point clouds
        # that do not have a fixed amount of points
        # x is a tuple of two lists representing the batches
        previous_batch, current_batch = x
        # trans is a tuple of two tensors representing transforms to global coordinate system
        G_T_P, G_T_C = transforms_matrices
        previous_batch_pc, previous_batch_grid, previous_batch_mask = previous_batch
        current_batch_pc, current_batch_grid, current_batch_mask = current_batch
        # For some reason the datatype of the input is not changed to correct precision
        previous_batch_pc = previous_batch_pc.type(self.dtype)
        current_batch_pc = current_batch_pc.type(self.dtype)

        # pointwise_voxel_coordinates_fs is in shape [BS, num points, 2],
        # where last dimension belong to location of point in voxel grid
        current_voxel_coordinates = self._batch_grid_2D(current_batch_grid)
        previous_voxel_coordinates = self._batch_grid_2D(previous_batch_grid)

        # Create bool map of filled/non-filled pillars
        current_batch_pillar_mask = self._filled_pillar_mask(current_voxel_coordinates, current_batch_mask)
        previous_batch_pillar_mask = self._filled_pillar_mask(previous_voxel_coordinates, previous_batch_mask)

        # batch_pc = (batch_size, N, 8) | batch_grid = (n_batch, N, 2) | batch_mask = (n_batch, N)
        # The grid indices are (batch_size, max_points) long. But we need them as
        # (batch_size, max_points, feature_dims) to work. Features are in all necessary cases 64.
        # Expand does only create multiple views on the same datapoint and not allocate extra memory
        current_batch_grid = current_batch_grid.unsqueeze(-1).expand(-1, -1, 64)
        previous_batch_grid = previous_batch_grid.unsqueeze(-1).expand(-1, -1, 64)

        # Pass the whole batch of point clouds to get the embedding for each point in the cloud
        # Input pc is (batch_size, max_n_points, features_in)
        # per each point, there are 8 features: [cx, cy, cz,  Δx, Δy, Δz, l0, l1], as stated in the paper
        previous_batch_pc_embedding = self._transform_point_cloud_to_embeddings(previous_batch_pc,
                                                                                previous_batch_mask)
        # previous_batch_pc_embedding = [n_batch, N, 64]
        # Output pc is (batch_size, max_n_points, embedding_features)
        current_batch_pc_embedding = self._transform_point_cloud_to_embeddings(current_batch_pc,
                                                                               current_batch_mask)

        # Now we need to scatter the points into their 2D matrix
        # batch_pc_embeddings -> (batch_size, N, 64)
        # batch_grid -> (batch_size, N, 64)
        # No learnable params in this part
        previous_pillar_embeddings = self._pillar_feature_net(previous_batch_pc_embedding, previous_batch_grid)
        current_pillar_embeddings = self._pillar_feature_net(current_batch_pc_embedding, current_batch_grid)
        # pillar_embeddings = (batch_size, 64, 640, 640)

        # Concatenate the previous and current batches along a new dimension.
        # This allows to have twice the amount of entries in the forward pass
        # of the encoder which is good for batch norm.
        pillar_embeddings = torch.stack((previous_pillar_embeddings, current_pillar_embeddings), dim=1)
        # This is now (batch_size, 2, 64, 640, 640) large
        pillar_embeddings = pillar_embeddings.flatten(0, 1)
        # Flatten into (batch_size * 2, 64, 512, 512) for encoder forward pass.

        # The grid map is ready in shape (BS, 64, 640, 640)

        # 2. RAFT Encoder with motion flow backbone
        # Output for forward pass and backward pass
        # Each of the output is a list of num_iters x [1, 9, n_pillars_x, n_pillars_x]
        # logits, static_flow, dynamic_flow, weights are concatinate in channels in shapes [4, 2, 2, 1]
        outputs_fw, outputs_bw = self._raft(pillar_embeddings)

        # Transformation matrix Current (t1) to Previous (t0)
        C_T_P = np.linalg.inv(G_T_C) @ G_T_P
        # Transformation matrix Previous (t0) to Current (t1)
        P_T_C = np.linalg.inv(G_T_P) @ G_T_C

        predictions_fw = []
        predictions_bw = []

        # TODO only visualization of filled pillar mask
        visualise_tensor(current_batch_pillar_mask)

        for it, (raft_output_0_1, raft_output_1_0) in enumerate(zip(outputs_fw, outputs_bw)):
            prediction_fw = self._decoder_fw(
                network_output=raft_output_0_1,
                dynamicness_threshold=self._moving_dynamicness_threshold.value(),  # TODO
                pc=previous_batch_pc,
                pointwise_voxel_coordinates_fs=previous_voxel_coordinates,  # torch.randint(0, 640, (1, 95440, 2)),
                pointwise_valid_mask=previous_batch_mask,  # torch.randint(0, 2, (1, 95440)).type(torch.bool),
                filled_pillar_mask=previous_batch_pillar_mask.type(torch.bool),
                # torch.randint(0, 2, (1, 1, 640, 640)).type(torch.bool),
                odom=P_T_C,  # TODO má tam být P_T_C ?
                inv_odom=C_T_P)

            prediction_bw = self._decoder_bw(
                network_output=raft_output_1_0,
                dynamicness_threshold=self.moving_dynamicness_threshold.value(),
                pc=current_batch_pc,
                pointwise_voxel_coordinates_fs=current_voxel_coordinates,
                pointwise_valid_mask=current_batch_mask,
                filled_pillar_mask=current_batch_pillar_mask.type(torch.bool),
                odom=C_T_P,
                inv_odom=P_T_C,
            )

            predictions_fw.append(prediction_fw)
            predictions_bw.append(prediction_bw)

            # TODO optimization of _moving_dynamicness_threshold

        return predictions_fw, predictions_bw


if __name__ == "__main__":
    model = SLIM()

    grid_cell_size = 0.109375
    from datasets.waymoflow.WaymoDataModule import WaymoDataModule

    dataset_path = "../../data/waymoflow_subset"
    # dataset_path = "/Users/simonpokorny/mnt/data/waymo/raw/processed/training"

    data_module = WaymoDataModule(dataset_directory=dataset_path,
                                  grid_cell_size=grid_cell_size,
                                  x_min=-35,
                                  x_max=35,
                                  y_min=-35,
                                  y_max=35,
                                  z_min=-10,
                                  z_max=10,
                                  batch_size=1,
                                  has_test=False,
                                  num_workers=0,
                                  n_pillars_x=640,
                                  n_points=None, apply_pillarization=True)

    trainer = pl.Trainer(fast_dev_run=True, num_sanity_val_steps=0)  # Add Trainer hparams if desired
    trainer.fit(model, data_module)
    # trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    print("done")
