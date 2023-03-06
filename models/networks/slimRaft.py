import pytorch_lightning as pl
import torch
import torch.nn as nn

from .corrBlock import CorrBlock, coords_grid
from .resnetEncoder import ResnetEncoder
from .updateBlock import SlimUpdateBlock


class RAFT(pl.LightningModule):
    def __init__(self,
                 iters=6,
                 corr_levels=4,
                 corr_radius=3,
                 alternate_corr=False,
                 feature_downsampling_factor=8,
                 predict_weight_for_static_aggregation=True,
                 hdim=96,
                 cdim=64,
                 flow_maps_archi="vanilla",
                 corr_module="all",
                 learn_upsampling_mask=True):
        """
         Initialize the RAFT model.

         Args:
             iters (int): number of refinement iterations.
             corr_levels (int): number of levels in correlation.
             corr_radius (int): radius of correlation.
             alternate_corr (bool): whether to use alternate correlation TODO (not implemented).
             feature_downsampling_factor (int): feature downsampling factor.
             predict_weight_for_static_aggregation (bool): whether to predict weight for static aggregation.
             hdim (int): hidden dimension in context encoder.
             cdim (int): context dimension in context encoder.
             flow_maps_archi (str): architecture of flow maps to use. "vanilla" / "single"
             corr_module (str): correlation module to use.
             learn_upsampling_mask (bool): whether to learn upsampling mask.
         """
        super().__init__()
        self.alternate_corr = alternate_corr
        self.feature_downsampling_factor = feature_downsampling_factor
        self.predict_weight_for_static_aggregation = predict_weight_for_static_aggregation
        self.hdim = hdim
        self.cdim = cdim
        self.corr_search_radius = corr_radius
        self.corr_num_levels = corr_levels
        self.iters = iters
        self.flow_maps_archi = flow_maps_archi
        self.corr_module = corr_module

        assert flow_maps_archi in ["single", "vanilla"]
        predict_logits = True if flow_maps_archi == "single" else False

        # Encoder for correlation and input to RAFT-S
        self._features_encoder_net = ResnetEncoder(input_dim=64, output_dim=128, norm_fn='instance', dropout=0.0)
        # Encoder for context in shape hidden_dim=96 + context_dim=64
        self._context_encoder_net = ResnetEncoder(input_dim=64, output_dim=hdim + cdim, norm_fn='none', dropout=0.0)
        # RAFT update block, SLIM is using smaller version RAFT-S
        self._update_block = SlimUpdateBlock(corr_levels=corr_levels, corr_radius=corr_radius, hidden_dim=96,
                                             predict_weight_for_static_aggregation=predict_weight_for_static_aggregation,
                                             predict_logits=predict_logits,
                                             learn_upsampling_mask=learn_upsampling_mask,
                                             feature_downsampling_factor=feature_downsampling_factor)

    def forward(self, x):
        """
        Forward pass of the RAFT model.

        Args:
            x (torch.Tensor): pillar embeddings in the shape of (bs * 2, 640, 640, 64)

        Returns:
            A tuple of two dicts containing forward and backward flow maps and classes.
        """

        assert x.shape == (2, 64, 640, 640)
        # RAFT Encoder step
        # Note that we are using two encoder, one is for features extraction
        # and second one is for context extraction.
        # Feature encoder:
        batch_size = int(x.shape[0] / 2)
        pillar_features = self._features_encoder_net(x)
        t0_features, t1_features = torch.split(pillar_features, [batch_size, batch_size], dim=0)
        # frames features are in shape [BS, 128, 80, 80]
        previous_pillar_embeddings, current_pillar_embeddings = torch.split(x, [batch_size, batch_size], dim=0)
        # pillar embeddings are in shape [BS, 64, 640, 640]

        # RAFT motion flow backbone
        # Note that we predict motion from t0 to t1 and t1 to t0
        retvals_forward = self._predict_single_flow_and_classes(previous_pillar_embeddings=previous_pillar_embeddings,
                                                                t0_features=t0_features,
                                                                t1_features=t1_features)

        retvals_backward = self._predict_single_flow_and_classes(previous_pillar_embeddings=current_pillar_embeddings,
                                                                 t0_features=t1_features,
                                                                 t1_features=t0_features)
        return retvals_forward, retvals_backward

    def _predict_single_flow_and_classes(self, previous_pillar_embeddings, t0_features, t1_features):
        """
        Predicts a single flow and classes.

        Args:
            previous_pillar_embeddings (torch.Tensor): pillar embeddings at time step t-1 in shape (bs, 64, 640, 640)
            t0_features (torch.Tensor): features at time step t-1 in shape (1, 128, 80, 80).
            t1_features (torch.Tensor): features at time step t in shape (1, 128, 80, 80).

        Returns:
            A dictionary containing flow maps and classes.
        """
        # 3. RAFT Flow Backbone

        # Initialization of the flow
        # Coordinates should be in shape [:, w, h, :] = [h, w] where img_t0 shape is [B, H, W, C] ???
        # toto poradi piuziva keras ale pytorch ma [BS, CH, H, W]
        coords_t0, coords_t1 = self._initialize_flow(previous_pillar_embeddings, indexing="ij")
        bs, c, h, w = coords_t0.shape

        # Decide if to use default RAFT (vanilla) or modified to SLIM (single)
        # vanilla version do not process logits for static dynamic.
        if self.flow_maps_archi == "vanilla":
            logits = None
            upsampled_dummy_logits = torch.zeros([bs, 4,
                                                  h * self.feature_downsampling_factor,
                                                  w * self.feature_downsampling_factor])
        else:
            logits = torch.zeros([bs, 4, h, w])

        # Initialization for weights for Kabsch algorithm
        if self.predict_weight_for_static_aggregation is not False:
            assert self.flow_maps_archi != "vanilla"
            weight_logits_for_static_aggregation = torch.zeros([bs, 1, h, w])
        else:
            weight_logits_for_static_aggregation = None

        # Setup correlation values "all" or "kernel" (kernel is not implemented)
        assert self.corr_module == "all", self.corr_module
        corr_fn = CorrBlock(fmap1=t0_features,
                            fmap2=t1_features,
                            num_levels=self.corr_num_levels,
                            radius=self.corr_search_radius)

        # Context encoder
        cnet = self._context_encoder_net(previous_pillar_embeddings)  # context features shape [BS, 160, 80, 80]
        net, inp = torch.split(cnet, [self.hdim, self.cdim], dim=1)
        net = nn.functional.tanh(net)
        inp = nn.functional.relu(inp)

        intermediate_flow_predictions = []
        for _i in range(self.iters):
            coords_t1 = coords_t1.detach()

            if self.flow_maps_archi != "vanilla":
                logits = logits.detach()
            if self.predict_weight_for_static_aggregation is not False:
                weight_logits_for_static_aggregation = weight_logits_for_static_aggregation.detach()

            corr = corr_fn(coords_t1)  # index correlation volume
            flow = coords_t1 - coords_t0

            net, delta_flow, delta_logits, mask, delta_weight_logits_for_static_aggr = \
                self._update_block(net, inp, corr, flow, logits, weight_logits_for_static_aggregation)

            coords_t1 = coords_t1 + delta_flow
            flow = flow + delta_flow

            assert mask is None
            if self.flow_maps_archi != "vanilla":
                logits = logits + delta_logits
            if self.predict_weight_for_static_aggregation is not False:
                weight_logits_for_static_aggregation = weight_logits_for_static_aggregation + \
                                                       delta_weight_logits_for_static_aggr

            upsampled_flow = self._upflow8(flow, n=self.feature_downsampling_factor)
            upsampled_flow_usfl_convention = self._change_flow_convention_from_raft2usfl(upsampled_flow)

            # Upsample the weight logits for static aggregation
            if weight_logits_for_static_aggregation is not None:
                upsampled_weight_logits_for_static_aggregation = self._uplogits8(weight_logits_for_static_aggregation,
                                                                                 n=self.feature_downsampling_factor)

            # Upsample the logits for flow maps
            if self.flow_maps_archi == "vanilla":
                upsampled_logits = upsampled_dummy_logits
            else:
                upsampled_logits = self._uplogits8(logits, n=self.feature_downsampling_factor)

            # Concatenate the upsampled logits, flow convention, and weight logits to the network output
            conc_out = self._concat2outputs(upsampled_logits,
                                            upsampled_flow_usfl_convention,
                                            upsampled_flow_usfl_convention,
                                            upsampled_weight_logits_for_static_aggregation)
            intermediate_flow_predictions.append(conc_out)

        return intermediate_flow_predictions

    def _concat2outputs(self, logits, static_flow, dynamic_flow, weight_logits_for_static_aggregation=None):
        """
        Concatenates the logits, static_flow and dynamic_flow tensors along the second dimension.

        Args:
        logits (torch.Tensor): A tensor of shape (batch_size, num_classes).
        static_flow (torch.Tensor): A tensor of shape (batch_size, 2).
        dynamic_flow (torch.Tensor): A tensor of shape (batch_size, 2).
        weight_logits_for_static_aggregation (Optional[torch.Tensor]): A tensor of shape (batch_size, 1) representing the
            weight to be used for static flow aggregation. This argument is optional and can be set to None.

        Returns:
        torch.Tensor: A tensor of shape (batch_size, num_classes + 4) if weight_logits_for_static_aggregation is None,
        else a tensor of shape (batch_size, num_classes + 5).
        """
        assert logits.shape[1] == 4
        assert static_flow.shape[1] == 2
        assert dynamic_flow.shape[1] == 2
        assert (weight_logits_for_static_aggregation is None) == (
            not self.predict_weight_for_static_aggregation)

        if weight_logits_for_static_aggregation is None:
            return torch.cat([logits, static_flow, dynamic_flow], dim=1)

        assert weight_logits_for_static_aggregation.shape[1] == 1
        return torch.cat([logits, static_flow, dynamic_flow, weight_logits_for_static_aggregation], dim=1)

    def _change_flow_convention_from_raft2usfl(self, flow):
        """
        Converts the optical flow representation from RAFT convention to USFL convention.

        "RAFT" is a popular optical flow method that outputs optical flow in a convention where
        positive X component indicates movement to the right, and positive Y component indicates
        movement downwards. This convention is used in popular computer vision benchmarks such as KITTI.

        "USFL" is short for "up-sampled flow", and it refers to a convention used in the Lyft Level 5 AV dataset,
        where positive X component indicates movement to the right, and positive Y component indicates movement upwards.

        Args:
        flow (torch.Tensor): A tensor of shape (batch_size, 2, height, width) representing the optical flow in RAFT convention.

        Returns:
        torch.Tensor: A tensor of shape (batch_size, 2, height, width) representing the optical flow in USFL convention.
        """
        # x,y - resolution of bev map
        resolution_adapter = torch.tensor([70 / 640, 70 / 640], dtype=torch.float32, ).reshape((1, -1, 1, 1))
        flow_meters = torch.flip(flow, dims=[-1]) * resolution_adapter
        return flow_meters

    def _upflow8(self, flow, n, mode='bilinear'):
        """
        Upsamples the input optical flow tensor by a factor of 8 or different.

        Args:
        flow (torch.Tensor): A tensor of shape (batch_size, 2, height, width) representing the optical flow.
        n (int): Upsampling factor.
        mode (str): Interpolation mode to be used. Defaults to 'bilinear'.

        Returns:
        torch.Tensor: A tensor of shape (batch_size, 2, n*height, n*width) representing the upsampled optical flow.
        """
        new_size = (n * flow.shape[2], n * flow.shape[3])
        return n * torch.nn.functional.interpolate(flow, size=new_size, mode=mode, align_corners=True)

    def _uplogits8(self, flow, n, mode='bilinear'):
        """
        Upsamples the input logits tensor by a factor of 8.

        Args:
        flow (torch.Tensor): A tensor of shape (batch_size, num_classes, height, width) representing the logits.
        n (int): Upsampling factor.
        mode (str): Interpolation mode to be used. Defaults to 'bilinear'.

        Returns:
        torch.Tensor: A tensor of shape (batch_size, num_classes, n*height, n*width) representing the upsampled logits.
        """
        new_size = (n * flow.shape[2], n * flow.shape[3])
        return torch.nn.functional.interpolate(flow, size=new_size, mode=mode, align_corners=True)

    def _initialize_flow(self, img, indexing="xy"):
        """
        Initializes the optical flow tensor.

        Args:
        img (torch.Tensor): A tensor of shape (batch_size, num_channels, height, width) representing the input image.
            indexing (str): The type of coordinate indexing to be used. Can be set to 'xy' or 'ij'. Defaults to 'xy'.

        Returns:
        Tuple[torch.Tensor]: A tuple of two tensors of shape (batch_size, 2, height / feature_downsampling_factor,
            width / feature_downsampling_factor) representing the initial and
            final coordinate grids for computing optical flow.
        """
        coords0 = coords_grid(batch=img, downscale_factor=self.feature_downsampling_factor, indexing=indexing)
        coords1 = coords_grid(batch=img, downscale_factor=self.feature_downsampling_factor, indexing=indexing)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1
