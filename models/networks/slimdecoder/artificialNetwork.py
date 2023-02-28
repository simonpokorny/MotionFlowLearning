from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn

from .staticAggregation import StaticAggregatedFlow
from .utils import normalized_sigmoid_sum, scale_gradient


class ArtificialNetworkOutput(nn.Module):
    def __init__(self, overwrite_non_filled_pillars_with_default_flow: bool = True,
                 overwrite_non_filled_pillars_with_default_logits: bool = True,
                 static_logit="net",
                 dynamic_logit="net",
                 ground_logit=False,
                 use_epsilon_for_weighted_pc_alignment=False,
                 disappearing_logit=False,
                 static_flow="net",
                 dynamic_flow="net",
                 dynamic_flow_is_non_rigid_flow=False,
                 predict_weight_for_static_aggregation=False
                 ):
        """
        Initializes the `ArtificialNetworkOutput` class.

        Args:
        - overwrite_non_filled_pillars_with_default_flow (bool): Whether to overwrite non-filled pillars with default flow.
        - overwrite_non_filled_pillars_with_default_logits (bool): Whether to overwrite non-filled pillars with default logits.
        - static_logit (bool): Whether to use static logits.
        - dynamic_logit (bool): Whether to use dynamic logits.
        - ground_logit (bool): Whether to use ground logits.
        - use_epsilon_for_weighted_pc_alignment (bool): Whether to use epsilon for weighted point cloud alignment (Kabsch algorithm).
        - disappearing_logit (bool): Whether to use disappearing logits.
        - static_flow (str): Type of the static flow.
        - dynamic_flow (str): Type of the dynamic flow.
        - dynamic_flow_is_non_rigid_flow (bool): Whether the dynamic flow is non-rigid.
        """
        super().__init__()

        self.overwrite_non_filled_pillars_with_default_flow = overwrite_non_filled_pillars_with_default_flow
        self.overwrite_non_filled_pillars_with_default_logits = overwrite_non_filled_pillars_with_default_logits
        self.static_logit = static_logit
        self.dynamic_logit = dynamic_logit
        self.ground_logit = ground_logit
        self.use_epsilon_for_weighted_pc_alignment = use_epsilon_for_weighted_pc_alignment
        self.predict_weight_for_static_aggregation = predict_weight_for_static_aggregation

        # TODO params in
        self.compute_static_aggregated_flow = StaticAggregatedFlow(
            use_eps_for_weighted_pc_alignment=use_epsilon_for_weighted_pc_alignment)
        self.artificial_flow_network_output = ArtificialFlowNetworkOutput(static_flow=static_flow,
                                                                          dynamic_flow=dynamic_flow,
                                                                          dynamic_flow_is_non_rigid_flow=dynamic_flow_is_non_rigid_flow)
        self.artificial_logit_network_output = ArtificialLogitNetwork(static_logit=static_logit,
                                                                      dynamic_logit=dynamic_logit,
                                                                      ground_logit=ground_logit,
                                                                      disappearing_logit=disappearing_logit)

    def forward(
            self,
            network_output_dict: Dict[str, torch.Tensor],
            dynamicness_threshold: torch.Tensor,
            ohe_gt_stat_dyn_ground_label_bev_map: torch.Tensor,
            gt_flow_bev: torch.Tensor,
            gt_static_flow: torch.Tensor,
            filled_pillar_mask: torch.Tensor,
            pc: torch.Tensor,
            pointwise_voxel_coordinates_fs: torch.Tensor,
            pointwise_valid_mask: torch.Tensor,
            voxel_center_metric_coordinates: np.array
    ) -> tuple[Any, Any, Any]:
        """
        Computes the output of the artificial network.

        Args:
        - network_output_dict: a dictionary of network output tensors.
        - dynamicness_threshold: tensor
        - ohe_gt_stat_dyn_ground_label_bev_map: a tensor of one-hot encoded gt labels.
        - gt_flow_bev: a tensor of ground truth flow in the bird's eye view.
        - gt_static_flow: a tensor of ground truth static flow in the bird's eye view.
        - filled_pillar_mask: a tensor of filled pillar masks.
        - pc: a tensor of point cloud data.
        - pointwise_voxel_coordinates_fs: a tensor of pointwise voxel coordinates.
        - pointwise_valid_mask: a tensor of pointwise valid masks.
        - voxel_center_metric_coordinates: an array of voxel center metric coordinates.

        Returns:
        - network_output_dict: a dictionary of network output tensors.
        """
        assert network_output_dict["static_flow"].dim() == filled_pillar_mask.dim(), (
            network_output_dict["static_flow"].shape,
            filled_pillar_mask.shape,
        )

        # Flow
        # TODO useless for self supervised training - delete
        network_output_dict = self.artificial_flow_network_output(
            network_output_dict=network_output_dict,
            gt_flow_bev=gt_flow_bev,
            gt_static_flow=gt_static_flow,
        )

        # Logits TODO some modification are done
        network_output_dict = self.artificial_logit_network_output(
            network_output_dict=network_output_dict,
            ohe_gt_stat_dyn_ground_label_bev_map=ohe_gt_stat_dyn_ground_label_bev_map,
            gt_flow_bev=gt_flow_bev,
            gt_static_flow=gt_static_flow,
        )

        # Choosing the mask for unfilled pillars.
        # Mask non-filled pillars
        default_values_for_nonfilled_pillars = {
            "disappearing_logit": -100.0,
            "static_logit": -100.0 if self.static_logit is False else 0.0,
            "dynamic_logit": 0.0 if self.dynamic_logit is True else -100.0,
            "ground_logit": 0.0 if self.ground_logit is True else -100.0,
            "static_flow": 0.0,
            "dynamic_flow": 0.0,
            "static_aggr_flow": 0.0,
        }

        # Dict for choosing if the non filled pillar should be filled with default value
        modification_taboo_keys = []
        if not self.overwrite_non_filled_pillars_with_default_flow:
            modification_taboo_keys += [
                "static_flow",
                "dynamic_flow",
                "static_aggr_flow",
            ]

        if not self.overwrite_non_filled_pillars_with_default_logits:
            modification_taboo_keys += [
                "disappearing_logit",
                "static_logit",
                "dynamic_logit",
                "ground_logit",
            ]

        # We are filling empty pillars with default values
        for k in network_output_dict:
            if k == "weight_logits_for_static_aggregation":
                continue
            assert network_output_dict[k].ndim == filled_pillar_mask.ndim, (
                k, network_output_dict[k].shape, filled_pillar_mask.shape)

            if k in modification_taboo_keys:
                continue

            network_output_dict[k] = torch.where(
                filled_pillar_mask,
                network_output_dict[k],
                default_values_for_nonfilled_pillars[k]
                * torch.ones_like(network_output_dict[k]),
            )

        # construct_class_probs from concationation of logits
        network_output_dict["class_logits"] = torch.cat(
            [network_output_dict["static_logit"],
             network_output_dict["dynamic_logit"],
             network_output_dict["ground_logit"]], dim=1)

        # Softmax on class probs
        network_output_dict["class_probs"] = torch.nn.functional.softmax(network_output_dict["class_logits"], dim=1)

        # Probs of invidual classes are separated into individual keys in dict
        network_output_dict["staticness"] = network_output_dict["class_probs"][:, 0]
        network_output_dict["dynamicness"] = network_output_dict["class_probs"][:, 1]
        network_output_dict["groundness"] = network_output_dict["class_probs"][:, 2]

        # Creating the output based on probs of individual classes and dynamicness threshold
        # DYNAMIC
        network_output_dict["is_dynamic"] = network_output_dict["dynamicness"] >= dynamicness_threshold
        # STATIC
        is_static = torch.logical_and((network_output_dict["staticness"] >= network_output_dict["groundness"]),
                                      torch.logical_not(network_output_dict["is_dynamic"]))
        network_output_dict["is_static"] = is_static
        # GROUND
        network_output_dict["is_ground"] = torch.logical_not(torch.logical_or(network_output_dict["is_static"],
                                                                              network_output_dict["is_dynamic"]))

        # Construction of the static aggregation
        # static_aggr_weight_map = probs of static * bool mask of filled pillars
        static_aggr_weight_map = network_output_dict["staticness"] * filled_pillar_mask[:, 0]

        # TODO predict_weight_for_static_aggregation ???
        if self.predict_weight_for_static_aggregation is not False:
            mode = self.predict_weight_for_static_aggregation
            assert mode in {"sigmoid", "softmax"}
            if mode == "softmax":
                network_output_dict["masked_weights_for_static_aggregation"] = torch.where(
                    filled_pillar_mask[..., 0],
                    network_output_dict["weight_logits_for_static_aggregation"],
                    torch.ones_like(network_output_dict["weight_logits_for_static_aggregation"]
                                    ) * (
                            torch.min(network_output_dict["weight_logits_for_static_aggregation"]) - 1000.0))

                curshape = network_output_dict["masked_weights_for_static_aggregation"].shape
                assert len(curshape) == 3, curshape
                prodshape = curshape[-1] * curshape[-2]
                network_output_dict["masked_weights_for_static_aggregation"] = torch.reshape(
                    torch.nn.functional.softmax(
                        torch.reshape(
                            network_output_dict[
                                "masked_weights_for_static_aggregation"
                            ],
                            (-1, prodshape),
                        ),
                        dim=-1
                    ),
                    (-1, *curshape[-2:]),
                )
            else:
                assert mode == "sigmoid"
                grid_size = filled_pillar_mask.shape[-3:-1]
                prod_size = grid_size[0] * grid_size[1]
                network_output_dict["masked_weights_for_static_aggregation"] = torch.reshape(
                    normalized_sigmoid_sum(
                        logits=torch.reshape(
                            network_output_dict["weight_logits_for_static_aggregation"],
                            [-1, prod_size],
                        ),
                        mask=torch.reshape(filled_pillar_mask[..., 0], [-1, prod_size]),
                    ),
                    [-1, *grid_size],
                )
            static_aggr_weight_map = (
                    static_aggr_weight_map
                    * network_output_dict["masked_weights_for_static_aggregation"]
            )

        # Computing of static aggregation
        static_aggr_flow, static_aggr_trafo, not_enough_points = self.compute_static_aggregated_flow(
            static_flow=network_output_dict["static_flow"],
            staticness=static_aggr_weight_map,
            pc=pc,
            pointwise_voxel_coordinates_fs=pointwise_voxel_coordinates_fs,
            pointwise_valid_mask=pointwise_valid_mask,
            voxel_center_metric_coordinates=voxel_center_metric_coordinates)

        # Adding new values to out dict
        network_output_dict["static_aggr_flow"] = static_aggr_flow
        network_output_dict["static_aggr_trafo"] = static_aggr_trafo
        network_output_dict["not_enough_points"] = not_enough_points

        # masking Static agge flow
        network_output_dict["masked_static_aggr_flow"] = torch.where(
            filled_pillar_mask,
            network_output_dict["static_aggr_flow"],
            torch.zeros_like(network_output_dict["static_aggr_flow"]),
        )

        # Masking gt static flow
        network_output_dict["masked_gt_static_flow"] = torch.where(
            filled_pillar_mask,
            gt_static_flow,
            torch.zeros_like(network_output_dict["masked_static_aggr_flow"]),
        )

        return network_output_dict, static_aggr_trafo, not_enough_points


class ArtificialFlowNetworkOutput(nn.Module):
    def __init__(self, static_flow, dynamic_flow, dynamic_flow_is_non_rigid_flow, dynamic_flow_grad_scale=1):
        """
        Initializes an instance of ArtificialFlowNetworkOutput.

        Args:
            static_flow: Specifies how to compute the static flow output. Can be one of "net", "gt", or "zero".
            dynamic_flow: Specifies how to compute the dynamic flow output. Can be one of "net", "gt", or "zero".
            dynamic_flow_is_non_rigid_flow: Boolean indicating whether the dynamic flow is non-rigid or not.
            dynamic_flow_grad_scale: Scale factor for the gradient of the dynamic flow.

        """
        super().__init__()
        self.static_flow = static_flow
        self.dynamic_flow = dynamic_flow
        self.dynamic_flow_grad_scale = dynamic_flow_grad_scale
        self.dynamic_flow_is_non_rigid_flow = dynamic_flow_is_non_rigid_flow

    def forward(self, network_output_dict, gt_flow_bev, gt_static_flow):
        """
        Computes the output of the ArtificialFlowNetwork. We can modify from which source the flow will be taken.

        Args:
            network_output_dict: A dictionary containing intermediate outputs from the network.
            gt_flow_bev: Ground truth flow in bird's-eye view format.
            gt_static_flow: Ground truth static flow.

        Returns:
            A dictionary containing the final outputs of the network.
        """

        # We can choose from which source the static flow will be taken.
        # #region static_flow
        if self.static_flow == "net":
            pass
        elif self.static_flow == "gt":
            network_output_dict["static_flow"] = gt_static_flow
        elif self.static_flow == "zero":
            network_output_dict["static_flow"] = torch.zeros_like(
                network_output_dict["static_flow"]
            )
        else:
            raise ValueError("unknown output mode: %s" % str(self.static_flow))
        # #endregion static_flow

        # We can choose from which source the dynamic flow will be taken.
        # #region dynamic_flow
        if self.dynamic_flow == "net":
            pass
        elif self.dynamic_flow == "gt":
            network_output_dict["dynamic_flow"] = gt_flow_bev
            if self.dynamic_flow_is_non_rigid_flow:
                network_output_dict["dynamic_flow"] = (
                        network_output_dict["dynamic_flow"] - network_output_dict["static_flow"]
                )
        elif self.dynamic_flow == "zero":
            network_output_dict["dynamic_flow"] = torch.zeros_like(
                network_output_dict["dynamic_flow"]
            )
        else:
            raise ValueError("unknown output mode: %s" % str(self.dynamic_flow))
        network_output_dict["dynamic_flow"] = scale_gradient(
            network_output_dict["dynamic_flow"],
            self.dynamic_flow_grad_scale,
        )
        # #endregion dynamic_flow
        return network_output_dict


class ArtificialLogitNetwork(nn.Module):
    def __init__(self, static_logit=True, dynamic_logit=True, ground_logit=False, disappearing_logit="net"):
        """
        ArtificialLogitNetwork class constructor.

        Parameters:
            static_logit (bool or str): If True, compute the static logits. If False, assume they're not needed.
                If 'net', compute them as the network output. If 'gt_label_based', compute them as the ground-truth labels
                in `ohe_gt_stat_dyn_ground_label_bev_map`. If 'gt_flow_based', compute them from the ground-truth flow in
                `gt_flow_bev` and the ground-truth static flow in `gt_static_flow`. Default is True.
            disappearing_logit (bool or str): If True, compute the disappearing logits. If False, assume they're not needed.
                If 'net', compute them as the network output. If 'gt', raise a NotImplementedError. Default is False.
            ground_logit (bool or str): If True, compute the ground logits. If False, assume they're not needed.
                If 'net', compute them as the network output. If 'gt', raise a NotImplementedError.
                If 'gt_label_based', compute them as the ground-truth labels in `ohe_gt_stat_dyn_ground_label_bev_map`.
                Default is False.
        """
        super(ArtificialLogitNetwork, self).__init__()
        self.static_logit = static_logit
        self.dynamic_logit = dynamic_logit
        self.ground_logit = ground_logit
        self.disappearing_logit = disappearing_logit

    def forward(self, network_output_dict, ohe_gt_stat_dyn_ground_label_bev_map, gt_flow_bev, gt_static_flow):
        """
        Forward pass of ArtificialLogitNetwork.

        Parameters:
            network_output_dict (dict): Dictionary containing the output tensors of the network.
            ohe_gt_stat_dyn_ground_label_bev_map (torch.Tensor): Tensor of shape `(B, 3, H, W)` containing the ground-truth
                labels, static flow, and ground truth maps.
            gt_flow_bev (torch.Tensor): Tensor of shape `(B, 2, H, W)` containing the ground-truth flow vectors in the
                BEV space.
            gt_static_flow (torch.Tensor): Tensor of shape `(B, 2, H, W)` containing the ground-truth static flow vectors
                in the BEV space.

        Returns:
            output_dict (dict): Dictionary containing the output tensors of the network with the logits added.
        """

        ones = torch.ones_like(network_output_dict["static_logit"])

        # #region disappearing_logit
        if self.disappearing_logit == "net":
            pass
        elif self.disappearing_logit == "gt":
            raise NotImplementedError()
        elif self.disappearing_logit is True:
            network_output_dict["disappearing_logit"] = 0 * ones
        elif self.disappearing_logit is False:
            network_output_dict["disappearing_logit"] = -100 * ones
        else:
            raise ValueError(
                "unknown output mode: %s" % str(self.disappearing_logit)
            )
        # #endregion disappearing_logit

        # #region static_logit
        if self.static_logit == "net":
            pass
        elif self.static_logit == "gt_label_based":
            assert self.dynamic_logit == "gt_label_based"
            if self.ground_logit is False:
                # add gt labels to static if ground == off
                gt_stat = (
                        ohe_gt_stat_dyn_ground_label_bev_map[..., 0:1]
                        | ohe_gt_stat_dyn_ground_label_bev_map[..., 2:3]
                )
                gt_stat_flt = gt_stat.to(torch.float32)
                network_output_dict["static_logit"] = 100.0 * (gt_stat_flt - 1.0)
            elif self.ground_logit == "gt_label_based":
                network_output_dict["static_logit"] = 100.0 * (
                        ohe_gt_stat_dyn_ground_label_bev_map[..., 0:1].to(torch.float32) - 1.0
                )
            else:
                raise AssertionError(
                    "when using gt_label for cls then ground_logit must be `gt_label_based` or `off`, not %s"
                    % self.ground_logit
                )
        elif self.static_logit == "gt_flow_based":
            assert self.dynamic_logit == "gt_flow_based"
            assert self.ground_logit is False

            norig_flow = gt_flow_bev - gt_static_flow
            bev_is_static_map = torch.linalg.norm(norig_flow, dim=-1, keepdim=True) <= 0.05
            bev_is_static_map = bev_is_static_map.to(torch.float32)
            network_output_dict["static_logit"] = 100.0 * (bev_is_static_map - 1.0)
        elif self.static_logit is True:
            assert self.dynamic_logit is False
            assert self.ground_logit is False

            # TODO  !!!
            network_output_dict["static_logit"] = (
                    torch.max(torch.stack([network_output_dict["dynamic_logit"], network_output_dict["ground_logit"]]),
                              dim=0)[0] + 100.0 * torch.ones_like(network_output_dict["dynamic_logit"])).detach()
        elif self.static_logit is False:
            assert (self.dynamic_logit is not False or self.ground_logit is not False)
            network_output_dict["static_logit"] = (
                    torch.min(
                        torch.stack([
                            network_output_dict["dynamic_logit"],
                            network_output_dict["ground_logit"]
                        ]),
                        dim=0
                    )[0]
                    - 100.0 * ones
            )
        else:
            raise ValueError("unknown output mode: %s" % str(self.static_logit))
        # #endregion static_logit

        # #region dynamic_logit
        if self.dynamic_logit == "net":
            pass
        elif self.dynamic_logit == "gt_label_based":
            assert self.static_logit == "gt_label_based"
            network_output_dict["dynamic_logit"] = 100.0 * (
                    ohe_gt_stat_dyn_ground_label_bev_map[..., 1:2].type(torch.float32) - 1.0
            )
        elif self.dynamic_logit == "gt_flow_based":
            network_output_dict["dynamic_logit"] = (
                    100.0 - network_output_dict["static_logit"]
            )
        elif self.dynamic_logit is True:
            # assert out_mod_cfg.static_logit is False
            # assert out_mod_cfg.ground_logit is False
            network_output_dict["dynamic_logit"] = (
                    torch.max(
                        torch.stack([
                            network_output_dict["static_logit"],
                            network_output_dict["ground_logit"]
                        ]),
                        dim=0
                    )[0]
                    + 100.0 * ones
            )
        elif self.dynamic_logit is False:
            network_output_dict["dynamic_logit"] = (
                    torch.min(
                        torch.stack([
                            network_output_dict["static_logit"],
                            network_output_dict["ground_logit"]
                        ]),
                        dim=0
                    )[0]
                    - 100.0 * ones
            )
        else:
            raise ValueError("unknown output mode: %s" % str(self.dynamic_logit))
        # #endregion dynamic_logit

        # #region ground_logit
        if self.ground_logit == "net":
            pass
        elif self.ground_logit == "gt_label_based":
            assert self.static_logit == "gt_label_based"
            assert self.dynamic_logit == "gt_label_based"
            network_output_dict["ground_logit"] = 100.0 * (
                    ohe_gt_stat_dyn_ground_label_bev_map[..., 2:3].type(torch.float32) - 1.0
            )
        elif self.ground_logit is True:
            assert self.static_logit is False
            assert self.dynamic_logit is False
            network_output_dict["ground_logit"] = (
                    torch.max(
                        torch.stack([
                            network_output_dict["static_logit"],
                            network_output_dict["dynamic_logit"]
                        ]),
                        dim=0
                    )[0]
                    + 100.0 * ones
            )
        elif self.ground_logit is False:
            network_output_dict["ground_logit"] = torch.min(
                torch.stack(
                    [network_output_dict["static_logit"], network_output_dict["dynamic_logit"]],
                    dim=0,
                ),
                dim=0,
                keepdim=False,
            )[0] - 100.0 * ones
        else:
            raise ValueError("unknown output mode: %s" % str(self.ground_logit))
        # #endregion ground_logit
        return network_output_dict
