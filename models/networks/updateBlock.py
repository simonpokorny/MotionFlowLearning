"""
This is update block for SLIM with GRU, motion encoder ...
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowHead(nn.Module):
    def __init__(self, input_dim=96, out_dim=2, hidden_dim=256):
        super(FlowHead, self).__init__()
        assert out_dim in [2, 3, 4], \
            "choose out_dims=2 for flow or out_dims=4 for classification or 3 if the paper DL is dangerously close"
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, out_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=96, input_dim=192+96):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (3, 3), padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (3, 3), padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (3, 3), padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q
        return h

class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))


    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        return h

class SmallMotionEncoder(nn.Module):
    def __init__(self, corr_levels, corr_radius):
        super(SmallMotionEncoder, self).__init__()
        cor_planes = corr_levels * (2*corr_radius + 1)**2
        self.convc1 = nn.Conv2d(cor_planes, 96, 1, padding=0)
        self.convf1 = nn.Conv2d(2, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv = nn.Conv2d(128, 80, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)

class BasicMotionEncoder(nn.Module):
    def __init__(self, corr_levels, corr_radius):
        super(BasicMotionEncoder, self).__init__()
        cor_planes = corr_levels * (2*corr_radius + 1)**2
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+192, 128-2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)

class SlimMotionEncoder(nn.Module):
    def __init__(self, predict_logits=True):
        """
        Args:
            predict_logits (bool): Whether to predict logits.
        """
        super(SlimMotionEncoder, self).__init__()
        self.conv_stat_corr1 = nn.Conv2d(196, 96, kernel_size=(1, 1), stride=(1, 1), padding=0)
        self.conv_flow1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(1, 1), padding=3)
        self.conv_flow2 = nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.predict_logits = predict_logits
        if self.predict_logits:
            self.conv_class1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(1, 1), padding=3)
            self.conv_class2 = nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv = nn.Conv2d(160, 80, kernel_size=(3, 3), stride=(1, 1), padding=1)

    def forward(self, flow, corr, L_cls):
        corr = F.relu(self.conv_stat_corr1(corr))

        flow = F.relu(self.conv_flow1(flow))
        flow = F.relu(self.conv_flow2(flow))

        concat_vals = [corr, flow]
        if self.predict_logits:
            L_cls = F.relu(self.conv_class1(L_cls))
            L_cls = F.relu(self.conv_class2(L_cls))
            concat_vals.append(L_cls)
        else:
            assert L_cls is None

        cor_flo_logits = torch.cat(concat_vals, dim=1)
        out = F.relu(self.conv(cor_flo_logits))

        if self.predict_logits:
            return torch.cat([out, L_cls, flow], dim=1)
        else:
            return torch.cat([out, flow], dim=1)

class SlimUpdateBlock(nn.Module):
    def __init__(self, corr_levels, corr_radius,
                 hidden_dim=96,
                 predict_weight_for_static_aggregation=False,
                 predict_logits=True,
                 learn_upsampling_mask=True,
                 feature_downsampling_factor=8):
        """
        Initializes the SmallUpdateBlock module.

        Args:
            corr_levels (int): The number of correlation levels.
            corr_radius (int): The correlation radius.
            hidden_dim (int, optional): The number of channels in the hidden state of the ConvGRU.
        """
        super(SlimUpdateBlock, self).__init__()

        self.predict_weight_for_static_aggregation = predict_weight_for_static_aggregation
        self.predict_logits = predict_logits

        # AAA
        num_stat_flow_head_channels = 3 if predict_weight_for_static_aggregation else 2
        self.static_flow_head = FlowHead(input_dim=hidden_dim, out_dim=num_stat_flow_head_channels, hidden_dim=256)

        #
        self.motion_encoder = SlimMotionEncoder(predict_logits=predict_logits)
        self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=208)
        #self.flow_head = FlowHead(hidden_dim, hidden_dim=128)

        if self.predict_logits:
            self.classification_head = FlowHead(input_dim=hidden_dim, out_dim=4, hidden_dim=256)

        self.learn_upsampling_mask = learn_upsampling_mask
        if self.learn_upsampling_mask:
            self.upsampling_mask_head = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, feature_downsampling_factor ** 2 * 9, kernel_size=1, stride=1),
            )




    def forward(self, net, inp, corr, flow, L_cls, L_wgt):
        """
        Args:
            net (torch.Tensor): The hidden state tensor.
            inp (torch.Tensor): The input tensor.
            corr (torch.Tensor): The correlation tensor.
            flow (torch.Tensor): The optical flow tensor.
            logits (torch.Tensor): The segmentation logits tensor.
            weight_logits_for_static_aggregation (float): The weight for the segmentation logits tensor.

        Returns:
            net (torch.Tensor): The hidden state tensor of the ConvGRU.
            delta_static_flow (torch.Tensor): The static flow tensor.
            delta_logits (torch.Tensor): The logit tensor.
            mask (torch.Tensor): The upsampling mask tensor.
            delta_weights (torch.Tensor): The weight delta tensor.

        """
        # motion_features = self.encoder(flow, corr)
        # inp = torch.cat([inp, motion_features], dim=1)
        # net = self.gru(net, inp)
        # delta_flow = self.flow_head(net)

        assert L_cls.shape == (1, 4, 80, 80)
        assert L_wgt.shape == (1, 1, 80, 80)
        assert corr.shape == (1, 196, 80, 80)
        assert flow.shape == (1, 2, 80, 80)
        assert inp.shape == (1, 64, 80, 80)
        assert net.shape == (1, 96, 80, 80)

        if self.predict_weight_for_static_aggregation:
            motion_features = self.motion_encoder(torch.cat([flow, L_wgt], dim=1), corr, L_cls)
        else:
            assert L_wgt is None
            motion_features = self.motion_encoder(flow, corr, L_cls)

        inp = torch.cat([inp, motion_features], dim=1)

        # Iteration in GRU block
        net = self.gru(h=net, x=inp)

        if self.predict_weight_for_static_aggregation:
            delta = self.static_flow_head(net)
            delta_static_flow = delta[:, :2]
            delta_weights = delta[:, -1].unsqueeze(1)
        else:
            delta_static_flow = self.static_flow_head(net)
            delta_weights = None

        if self.predict_logits:
            delta_logits = self.classification_head(net)
        else:
            delta_logits = None

        if self.learn_upsampling_mask:
            raise NotImplementedError()
        else:
            mask = None

        return net, delta_static_flow, delta_logits, mask, delta_weights,
        #return net, None, delta_flow

class BasicUpdateBlock(nn.Module):
    def __init__(self, corr_levels, corr_radius, hidden_dim=128, input_dim=128):
        super(BasicUpdateBlock, self).__init__()
        self.encoder = BasicMotionEncoder(corr_levels, corr_radius)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))

    def forward(self, net, inp, corr, flow, upsample=True):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)

        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        # scale mask to balence gradients
        mask = .25 * self.mask(net)
        return net, mask, delta_flow


