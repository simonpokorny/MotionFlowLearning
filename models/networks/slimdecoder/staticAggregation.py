import numpy as np
import torch

from .weightedKabsch import WeightedKabschAlgorithm


class StaticAggregatedFlow(torch.nn.Module):
    def __init__(self, use_eps_for_weighted_pc_alignment=False):
        """
        Initializes a StaticAggregatedFlow module.

        Args:
            use_eps_for_weighted_pc_alignment (bool): If True, adds a small epsilon to the denominator
                of the weighted point cloud alignment computation to avoid division by zero errors.
                Defaults to False.
        """
        super().__init__()
        self.use_eps_for_weighted_pc_alignment = use_eps_for_weighted_pc_alignment
        self.kabsch = WeightedKabschAlgorithm()

    def forward(
            self,
            static_flow,
            staticness,
            pc,
            pointwise_voxel_coordinates_fs,
            pointwise_valid_mask,
            voxel_center_metric_coordinates,
    ):
        """
        Computes the forward pass of the StaticAggregatedFlow module.

        Args:
            static_flow (torch.Tensor): A tensor of shape (batch_size, 2, H, W) representing the
                static flow field.
            staticness (torch.Tensor): A tensor of shape (batch_size, H, W) representing the
                staticness field.
            pc (torch.Tensor): A tensor of shape (batch_size, N, 3) representing the input point
                clouds.
            pointwise_voxel_coordinates_fs (torch.Tensor): A tensor of shape (batch_size, N, 2)
                representing the 2D voxel coordinates of each point in the static flow field.
            pointwise_valid_mask (torch.Tensor): A tensor of shape (batch_size, N) containing boolean
                values indicating whether each point in the input point clouds is valid.
            voxel_center_metric_coordinates (torch.Tensor): A tensor of shape (H, W, 3) representing
                the 3D metric coordinates of the centers of the voxels in the static flow field.

        Returns:
            A tuple containing:
            - static_aggr_flow (torch.Tensor): A tensor of shape (batch_size, N, 2) representing the
              static aggregated flow field.
            - trafo (torch.Tensor): A tensor of shape (batch_size, 4, 4) representing the transformation
              matrix used to align the input point clouds to the static flow field.
            - not_enough_points (torch.Tensor): A tensor of shape (batch_size,) containing boolean values
              indicating whether there were not enough valid points in the input point clouds to perform
              the weighted point cloud alignment.
        """
        assert len(static_flow.shape) == 4
        assert static_flow.shape[1] == 2
        assert (pointwise_voxel_coordinates_fs >= 0).all()
        # To static flow is also add a third coord z with value 0
        static_3d_flow_grid = torch.cat([static_flow, torch.zeros_like(static_flow[:, :1])], dim=1)
        bs = static_flow.shape[0]
        # grid_flow_3d [1,3,640,640]
        # pointwise_voxel_coordinates_fs [1, num points in pcl, 2]
        # representing the coordinates of each point in the BEV grid
        bs, ch, h, w = static_3d_flow_grid.shape
        x, y = pointwise_voxel_coordinates_fs[:, :, 0], pointwise_voxel_coordinates_fs[:, :, 1]
        pointwise_flow = static_3d_flow_grid[torch.arange(bs)[:, None], :, x, y]
        # point-wise flow is in shape [bs, num_points, 3]
        # We transform the flow from bev image to pcl

        voxel_center_metric_coordinates_f32 = voxel_center_metric_coordinates.astype(np.float32)
        # also constructing centers of voxels
        pc0_grid = torch.tensor(
            np.concatenate(
                [voxel_center_metric_coordinates_f32,
                 np.zeros_like(voxel_center_metric_coordinates_f32[..., :1])], axis=-1))

        static_3d_flow_grid = static_3d_flow_grid.permute(0, 2, 3, 1)
        # change order from [BS, CH, H, W] -> [BS, H, W, CH]
        assert pc0_grid.shape == static_3d_flow_grid.shape[1:]
        grid_shape = static_3d_flow_grid.shape[:1] + (pc0_grid.shape[0],) + pc0_grid.shape[1:]
        batched_pc0_grid = pc0_grid.expand(grid_shape)
        # batched_pc0_grid is in shape [BS, 640, 640, 3]

        # getting x,y coordinates where the individual point belongs in bev image
        x, y = pointwise_voxel_coordinates_fs[:, :, 0], pointwise_voxel_coordinates_fs[:, :, 1]
        # Probability of being static for each point in pcl
        pointwise_staticness = staticness.unsqueeze(0)[torch.arange(bs)[:, None], :, x, y]

        # Masking unfilled pillars with zeros
        pointwise_staticness = torch.where(
            pointwise_valid_mask.unsqueeze(-1),
            pointwise_staticness,
            torch.zeros_like(pointwise_staticness),
        )

        # pc_xyz is in shape [BS, Num Points, 3]
        pc_xyz = pc[:, :, :3]
        # Adding flow
        pointwise_flow_xyz = pc_xyz + pointwise_flow

        # Computing of weighted kabsch algorithm
        transformation, not_enough_points = self.kabsch(cloud_t0=pc_xyz,
                                                        cloud_t1=(pc_xyz + pointwise_flow_xyz),
                                                        weights=pointwise_staticness[:, :, 0])

        homogeneous_batched_pc0_grid = torch.cat([batched_pc0_grid,
                                                  torch.ones_like(batched_pc0_grid[..., 0][..., None])], dim=-1)
        # Constructing of static_aggr_flow
        static_aggr_flow = torch.einsum(
            "bij,bhwj->bhwi",
            transformation - torch.eye(4),
            homogeneous_batched_pc0_grid)[..., 0:2]

        # Change static aggr flow to default BS CH H W
        static_aggr_flow = static_aggr_flow.permute(0, 3, 1, 2)

        return static_aggr_flow, transformation, not_enough_points
