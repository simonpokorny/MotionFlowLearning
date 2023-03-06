import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor

from .artificialNetwork import ArtificialNetworkOutput


class OutputDecoder(pl.LightningModule):
    def __init__(self,
                 predict_weight_for_static_aggregation=True,
                 use_static_aggr_flow_for_aggr_flow=False,
                 use_dynamic_aggr_flow_for_aggr_flow=False,
                 dynamic_flow_is_non_rigid_flow=False,
                 artificial_network_config=None):
        """
        Initializes the OutputDecoder module.

        Args:
            - predict_weight_for_static_aggregation (bool): If True, predicts weights for static aggregation
            - use_static_aggr_flow_for_aggr_flow (bool): If True, uses static flow for aggregation flow
            - use_dynamic_aggr_flow_for_aggr_flow (bool): If True, uses dynamic flow for aggregation flow
              representing the one-hot encoded ground truth labels for static/dynamic/ground objects in the BEV grid
            - dynamic_flow_is_non_rigid_flow (bool): If True, indicates that the dynamic flow is non-rigid
            - overwrite_non_filled_pillars_with_default_flow (bool): If True, overwrites non-filled pillars with the
                default flow
            - overwrite_non_filled_pillars_with_default_logits (bool): If True, overwrites non-filled pillars with the
                default logits
            - artificial_network_config (dict): Config for ArtificialNetworkOutput()
        """

        super().__init__()

        self.flow_dim = 2

        self.predict_weight_for_static_aggregation = predict_weight_for_static_aggregation
        self.use_static_aggr_flow_for_aggr_flow = use_static_aggr_flow_for_aggr_flow
        self.use_dynamic_aggr_flow_for_aggr_flow = use_dynamic_aggr_flow_for_aggr_flow
        self.dynamic_flow_is_non_rigid_flow = dynamic_flow_is_non_rigid_flow

        assert type(artificial_network_config) == dict, "Wrong type for artificial_network_config"
        self.artificial_network_output = ArtificialNetworkOutput(**artificial_network_config)

    def forward(self, network_output: Tensor,
                dynamicness_threshold,
                pc: Tensor,
                pointwise_voxel_coordinates_fs: Tensor,
                pointwise_valid_mask,
                filled_pillar_mask,
                odom,
                inv_odom,
                gt_flow_bev=None,
                ohe_gt_stat_dyn_ground_label_bev_map=None):

        """
        Args:
            - network_output: a tensor of shape [batch_size, height, width, num_classes + 2 * num_flow_channels + 1]
              representing the output of a neural network. In our case num of channels should be 9.
                - Disapiring Logit 1
                - Static Logic 1
                - Dynamic Logic 1
                - Ground Logic 1
                - Static flow 2
                - Dynamic flow 2
                - Weights 1
            - dynamicness_threshold: a float value representing the dynamicness threshold used for separating static
              and dynamic objects
            - pc: a tensor of shape [batch_size, num_points, 5 or 4] representing the point cloud input.
              Channels should be in order [x, y, z, feature1, feature2]
            - pointwise_voxel_coordinates_fs: a tensor of shape [batch_size, num_points, 2] representing the coordinates
              of each point in the BEV grid
            - pointwise_valid_mask: a tensor of shape [batch_size, num_points] representing whether each point is valid or not
            - filled_pillar_mask: a tensor of shape [batch_size, height, width] and type bool representing whether each pillar
              in the BEV grid has been filled with points or not
            - inv_odom: a tensor of shape [batch_size, 4, 4] representing the inverse
              of the ground truth transformation matrix

             Below args are optional (default SLIM experiments with ground labels):

            - gt_flow_bev: a tensor of shape [batch_size, height, width, 2] representing the ground truth flow in the BEV grid
            - ohe_gt_stat_dyn_ground_label_bev_map: (optional) a tensor of shape [batch_size, height, width, 3] representing
              the one-hot encoded ground truth labels for static/dynamic/ground objects in the BEV grid

        Returns:
            - pointwise_outputs
            - static_aggr_trafo
            - dynamicness_threshold
            - modified_output_bev_img

        """

        # Check the shapes and dimensions
        assert filled_pillar_mask.ndim == 4
        assert network_output.ndim == 4
        assert filled_pillar_mask.shape[-2:] == network_output.shape[-2:], (
        filled_pillar_mask.shape, network_output.shape)
        assert pointwise_voxel_coordinates_fs.shape[-1] == 2
        # Check the correct dimensions of the output of the network
        assert network_output.shape[1] == 4 + 2 * self.flow_dim + 1
        assert pointwise_voxel_coordinates_fs.shape[-1] == 2

        network_output_dict = self._create_output_dict(network_output)
        homog_metric_voxel_center_coords, voxel_center_metric_coordinates = self._create_voxel_coords(network_output)
        gt_static_flow = self._create_gt_static_flow(inv_odom, homog_metric_voxel_center_coords)

        # Forward pass of artificial network output, following steps are done:
        # - Computed the source for logits and flow
        # - Masked the non-filled pillars
        # - Construct class probs from logits and decision of class is made based on dynamicness_threshold
        # - Static aggregation flow is computed (Kabsch)
        network_output_dict, static_aggr_trafo, not_enough_points = self.artificial_network_output(
            network_output_dict=network_output_dict,
            dynamicness_threshold=dynamicness_threshold,
            ohe_gt_stat_dyn_ground_label_bev_map=ohe_gt_stat_dyn_ground_label_bev_map,
            gt_flow_bev=gt_flow_bev,
            gt_static_flow=gt_static_flow,
            filled_pillar_mask=filled_pillar_mask,
            pc=pc,
            pointwise_voxel_coordinates_fs=pointwise_voxel_coordinates_fs,
            pointwise_valid_mask=pointwise_valid_mask,
            voxel_center_metric_coordinates=voxel_center_metric_coordinates,
        )

        # Slicing the output from artificial_network_output
        disappearing_logit = network_output_dict["disappearing_logit"][:, 0]
        disappearing = torch.sigmoid(disappearing_logit)
        class_logits = network_output_dict["class_logits"]
        class_probs = network_output_dict["class_probs"]
        staticness = network_output_dict["staticness"]
        dynamicness = network_output_dict["dynamicness"]
        groundness = network_output_dict["groundness"]
        is_static = network_output_dict["is_static"]
        is_dynamic = network_output_dict["is_dynamic"]
        is_ground = network_output_dict["is_ground"]
        static_flow = network_output_dict["static_flow"]
        static_aggr_flow = network_output_dict["static_aggr_flow"]
        dynamic_flow = network_output_dict["dynamic_flow"]
        dynamic_aggr_flow = network_output_dict.get("dynamic_aggr_flow", None)
        masked_dynamic_aggr_flow = network_output_dict.get("masked_dynamic_aggr_flow", None)
        masked_static_aggr_flow = network_output_dict["masked_static_aggr_flow"]

        if self.flow_dim == 2:
            dynamic_flow = torch.cat([dynamic_flow, torch.zeros_like(dynamic_flow[:, :1])], dim=1)
            static_flow = torch.cat([static_flow, torch.zeros_like(static_flow[:, :1])], dim=1)
            static_aggr_flow = torch.cat([static_aggr_flow, torch.zeros_like(static_aggr_flow[:, :1])], dim=1)
            masked_static_aggr_flow = torch.cat(
                [masked_static_aggr_flow, torch.zeros_like(masked_static_aggr_flow[:, :1])], dim=1)
            if dynamic_aggr_flow is not None:
                dynamic_aggr_flow = torch.cat([dynamic_aggr_flow, torch.zeros_like(dynamic_aggr_flow[:, :1])], dim=1)
            if masked_dynamic_aggr_flow is not None:
                masked_dynamic_aggr_flow = torch.cat(
                    [masked_dynamic_aggr_flow, torch.zeros_like(masked_dynamic_aggr_flow[:, :1])], dim=1)

        if self.use_static_aggr_flow_for_aggr_flow:
            static_flow_for_aggr = masked_static_aggr_flow
        else:
            static_flow_for_aggr = static_flow

        assert len(is_static.shape) == 3
        assert len(groundness.shape) == 3

        if self.dynamic_flow_is_non_rigid_flow:
            aggregated_flow = torch.where(is_static[:, None],
                                          static_flow_for_aggr,
                                          (static_flow_for_aggr + dynamic_flow) * (1.0 - groundness[..., None]))
        else:
            aggregated_flow = torch.where(is_static[:, None],
                                          static_flow_for_aggr,
                                          dynamic_flow * (1.0 - groundness[:, None]))

        if (self.use_dynamic_aggr_flow_for_aggr_flow and dynamic_aggr_flow is not None
                and "mask_has_dynamic_aggr_output" in network_output_dict.keys()):
            aggregated_flow = torch.where(
                network_output_dict["mask_has_dynamic_aggr_output"],
                dynamic_aggr_flow,
                aggregated_flow,
            )

        modified_output_bev_img = {
            'disappearing': disappearing,
            'disappearing_logit': disappearing_logit,
            'class_probs': class_probs,
            'class_logits': class_logits,
            'staticness': staticness,
            'dynamicness': dynamicness,
            'groundness': groundness,
            'is_static': is_static,
            'is_dynamic': is_dynamic,
            'is_ground': is_ground,
            'dynamic_flow': dynamic_flow,
            'static_flow': static_flow,
            'aggregated_flow': aggregated_flow,
            'static_aggr_flow': static_aggr_flow,
            'dynamic_aggr_flow': dynamic_aggr_flow,
        }

        # Transform pillars to pointcloud
        pointwise_output = self._apply_flow_to_points(
            modified_output_bev_img=modified_output_bev_img,
            pointwise_voxel_coordinates_fs=pointwise_voxel_coordinates_fs,
            pointwise_valid_mask=pointwise_valid_mask,
        )

        return pointwise_output, static_aggr_trafo, dynamicness_threshold, modified_output_bev_img

    def _create_output_dict(self, network_output):
        """
        Create the output dict, all decoder use it
        """
        # Partition the output by channels
        network_output_dict = {}
        if self.predict_weight_for_static_aggregation is not False:
            network_output_dict["weight_logits_for_static_aggregation"] = network_output[:, -1]
            network_output = network_output[:, :-1]

        network_output_dict.update({
            "disappearing_logit": network_output[:, 0:1],
            "static_logit": network_output[:, 1:2],
            "dynamic_logit": network_output[:, 2:3],
            "ground_logit": network_output[:, 3:4],
            "static_flow": network_output[:, 4: 4 + self.flow_dim],
            "dynamic_flow": network_output[:, 4 + self.flow_dim: 4 + 2 * self.flow_dim]})

        return network_output_dict

    def _create_gt_static_flow(self, inv_odom, homog_metric_voxel_center_coords):
        """
        Compute other self supervise component -> gt_static_flow
        """
        # gt_static_flow is (P_T_G - eye), which results in flow, shape
        # TODO should be here inverse?
        print("UnCheck")
        gt_static_flow = inv_odom - torch.eye(4, dtype=torch.float64)
        gt_static_flow = torch.einsum("bij,hwj->bhwi", gt_static_flow,
                                      homog_metric_voxel_center_coords)
        # normalization of the 4th coordinates
        gt_static_flow = gt_static_flow / gt_static_flow[..., -1].unsqueeze(-1)
        # we take only xy coords
        gt_static_flow = gt_static_flow[..., :2]
        # Transform it to default order [BS, CH, H, W]
        gt_static_flow = gt_static_flow.permute(0, 3, 1, 2)
        return gt_static_flow

    def _create_voxel_coords(self, network_output):
        """
        Creation of homog_metric_voxel_center_coords and voxel_center_metric_coordinates
        """
        final_grid_size = network_output.shape[-2:]

        # Creating of bev grid mesh
        bev_extent = np.array([-35.0, -35.0, 35.0, 35.0])
        net_output_shape = final_grid_size  # Net out shape is [640, 640]
        voxel_center_metric_coordinates = (
                np.stack(
                    np.meshgrid(np.arange(net_output_shape[0]), np.arange(net_output_shape[1]), indexing="ij"),
                    axis=-1,
                )
                + 0.5
        )  # now we have voxels in shape [640, 640, 2] from 0.5 to 639.5

        voxel_center_metric_coordinates /= net_output_shape
        voxel_center_metric_coordinates *= bev_extent[2:] - bev_extent[:2]
        voxel_center_metric_coordinates += bev_extent[:2]
        # now we have coordinates with centres of the voxel, for example in
        # voxel_center_metric_coordinates[0, 0] is [-34.9453125, -34.9453125]
        # The resolution (width of voxels) is 70m/640 = 0.1093..
        homog_metric_voxel_center_coords = torch.tensor(np.concatenate(
            [
                voxel_center_metric_coordinates,
                np.zeros_like(voxel_center_metric_coordinates[..., :1]),
                np.ones_like(voxel_center_metric_coordinates[..., :1]),
            ],
            axis=-1,
        ))
        # homog_metric_voxel_center_coords only add z coord to 0 and 4th dimension for homogeneous coordinates
        return homog_metric_voxel_center_coords, voxel_center_metric_coordinates

    def _apply_flow_to_points(self, *,
                              modified_output_bev_img,
                              pointwise_voxel_coordinates_fs,
                              pointwise_valid_mask):

        concat_bool_vals = torch.stack([modified_output_bev_img["is_static"],
                                        modified_output_bev_img["is_dynamic"],
                                        modified_output_bev_img["is_ground"]], dim=1, )

        concat_flt_vals = torch.stack([modified_output_bev_img["disappearing"],
                                       modified_output_bev_img["disappearing_logit"],
                                       modified_output_bev_img["staticness"],
                                       modified_output_bev_img["dynamicness"],
                                       modified_output_bev_img["groundness"]], dim=1, )
        concat_flt_vals = torch.cat([concat_flt_vals,
                                     modified_output_bev_img["class_probs"],
                                     modified_output_bev_img["class_logits"],
                                     modified_output_bev_img["dynamic_flow"],
                                     modified_output_bev_img["static_flow"],
                                     modified_output_bev_img["aggregated_flow"],
                                     modified_output_bev_img["static_aggr_flow"]], dim=1)

        if modified_output_bev_img["dynamic_aggr_flow"] is not None:
            concat_flt_vals = torch.cat([concat_flt_vals,
                                         modified_output_bev_img["dynamic_aggr_flow"]], dim=1)
            num_required_concat_vals = 26
        else:
            num_required_concat_vals = 23

        assert (pointwise_voxel_coordinates_fs >= 0).all(), "negative pixel coordinates found"
        assert (pointwise_voxel_coordinates_fs < concat_bool_vals.shape[2]).all(), "too large pixel coordinates found"

        bs = pointwise_voxel_coordinates_fs.shape[0]
        x, y = pointwise_voxel_coordinates_fs[:, :, 0], pointwise_voxel_coordinates_fs[:, :, 1]

        pointwise_concat_bool_vals = concat_bool_vals[torch.arange(bs)[:, None], :, x, y]
        pointwise_concat_flt_vals = concat_flt_vals[torch.arange(bs)[:, None], :, x, y]

        pointwise_concat_bool_vals = torch.where(pointwise_valid_mask.unsqueeze(-1),
                                                 pointwise_concat_bool_vals,
                                                 torch.zeros_like(pointwise_concat_bool_vals))
        pointwise_concat_flt_vals = torch.where(pointwise_valid_mask.unsqueeze(-1),
                                                pointwise_concat_flt_vals,
                                                torch.tensor(float("nan"), dtype=torch.float32).repeat(
                                                    *pointwise_concat_flt_vals.shape))

        assert pointwise_concat_bool_vals.shape[-1] == 3, pointwise_concat_bool_vals.shape
        pointwise_is_static = pointwise_concat_bool_vals[..., 0]
        pointwise_is_dynamic = pointwise_concat_bool_vals[..., 1]
        pointwise_is_ground = pointwise_concat_bool_vals[..., 2]

        assert pointwise_concat_flt_vals.shape[-1] == num_required_concat_vals, pointwise_concat_flt_vals.shape
        pointwise_disappearing = pointwise_concat_flt_vals[..., 0]
        pointwise_disappearing_logit = pointwise_concat_flt_vals[..., 1]
        pointwise_staticness = pointwise_concat_flt_vals[..., 2]
        pointwise_dynamicness = pointwise_concat_flt_vals[..., 3]
        pointwise_groundness = pointwise_concat_flt_vals[..., 4]
        pointwise_class_probs = pointwise_concat_flt_vals[..., 5:8]
        pointwise_class_logits = pointwise_concat_flt_vals[..., 8:11]
        pointwise_dynamic_flow = pointwise_concat_flt_vals[..., 11:14]
        pointwise_static_flow = pointwise_concat_flt_vals[..., 14:17]
        pointwise_aggregated_flow = pointwise_concat_flt_vals[..., 17:20]
        pointwise_static_aggregated_flow = pointwise_concat_flt_vals[..., 20:23]
        if modified_output_bev_img["dynamic_aggr_flow"] is not None:
            pointwise_dynamic_aggregated_flow = pointwise_concat_flt_vals[..., 23:26]
        else:
            pointwise_dynamic_aggregated_flow = None
        retval = {
            "disappearing_logit": pointwise_disappearing_logit,
            "disappearing": pointwise_disappearing,
            "class_logits": pointwise_class_logits,
            "class_probs": pointwise_class_probs,
            "staticness": pointwise_staticness,
            "dynamicness": pointwise_dynamicness,
            "groundness": pointwise_groundness,
            "is_static": pointwise_is_static,
            "is_dynamic": pointwise_is_dynamic,
            "is_ground": pointwise_is_ground,
            "dynamic_flow": pointwise_dynamic_flow,
            "static_flow": pointwise_static_flow,
            "aggregated_flow": pointwise_aggregated_flow,
            "static_aggr_flow": pointwise_static_aggregated_flow,
            "dynamic_aggr_flow": pointwise_dynamic_aggregated_flow,
        }
        return retval


if __name__ == "__main__":
    # for debug purposes
    test_input = [torch.rand((1, 9, 640, 640)) for x in range(6)]
    model = OutputDecoder()

    prediction_fw = model(network_output=test_input[0],
                          dynamicness_threshold=0.5,
                          pc=torch.rand((1, 95440, 5)),
                          pointwise_voxel_coordinates_fs=torch.randint(0, 640, (1, 95440, 2)),
                          pointwise_valid_mask=torch.randint(0, 2, (1, 95440)).type(torch.bool),
                          filled_pillar_mask=torch.randint(0, 2, (1, 1, 640, 640)).type(torch.bool),
                          odom=torch.rand((1, 4, 4)),
                          inv_odom=torch.rand((1, 4, 4)))
