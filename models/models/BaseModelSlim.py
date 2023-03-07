"""
MIT License

Copyright (c) 2021 Felix (Jabb0), Aron (arndz), Carlos (cmaranes)
Source: https://github.com/Jabb0/FastFlow3D

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from argparse import ArgumentParser
from collections import defaultdict

import pytorch_lightning as pl
from pytorch3d.ops.knn import knn_points
import torch

from models.utils import str2bool


# def NN_loss(x, y, x_lengths=None, y_lengths=None, reduction='mean'):
#
#     # max_pts_p = np.argmax((x.shape[1], y.shape[1]))
#     # N_pts_x = x.shape[1]
#
#     # if max_pts_p == 0:
#     #     N_pad = x.shape[1] - y.shape[1]
#     #     y = torch.nn.functional.pad(y, (0,0,0,N_pad,0,0))
#
#     # else:
#     #     N_pad = y.shape[1] - x.shape[1]
#     #     x = torch.nn.functional.pad(y, (0,0,0,N_pad,0,0))
#
#     x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, K=1, norm=1)
#     y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, K=1, norm=1)
#
#     # hack, maybe can be done better
#     # if max_pts_p == 0:
#
#     cham_x = x_nn.dists[..., 0]  # (N, P1)
#     cham_y = y_nn.dists[..., 0]  # (N, P2)
#
#     nearest_to_y = x_nn[1]
#
#     # else:
#     #     cham_x = x_nn.dists[:N_pts_x, 0]  # (N, P1)
#     # cham_y = y_nn.dists[:N_pts_x, 0]  # (N, P2)
#     #
#     # nearest_to_y = x_nn[1][:,N_pts_x]
#
#     nn_loss = (cham_x + cham_y) / 2
#
#     if reduction == 'mean':
#         nn_loss = nn_loss.mean()
#     elif reduction == 'sum':
#         nn_loss = nn_loss.sum()
#     elif reduction == 'none':
#         nn_loss = nn_loss
#     else:
#         raise NotImplementedError
#
#     # breakpoint()
#     return nn_loss, nearest_to_y

class BaseModelSlim(pl.LightningModule):
    def __init__(self,
                 learning_rate=1e-6,
                 adam_beta_1=0.9,
                 adam_beta_2=0.999,
                 background_weight=0.1):
        super(BaseModelSlim, self).__init__()
        self._background_weight = background_weight
        # ----- Metrics information -----
        # TODO delete no flow class
        self._classes = [(0, 'background'), (1, 'vehicle'), (2, 'pedestrian'), (3, 'sign'), (4, 'cyclist')]
        self._thresholds = [(1, '1_1'), (0.1, '1_10')]  # 1/1 = 1, 1/10 = 0.1
        self._min_velocity = 0.5  # If velocity higher than 0.5 then it is considered as the object is moving

    def compute_metrics(self, y, y_hat, labels):
        """

        :param y: predicted data (points, 3)
        :param y_hat: ground truth (points, 3)
        :param labels: class labels for each point (points, 3)
        :return:
        """
        squared_root_difference = torch.sqrt(torch.sum((y - y_hat) ** 2, dim=1))
        # We compute the weighting vector for background_points
        # weights is a mask which background_weight value for backgrounds and 1 for no backgrounds, in order to
        # downweight the background points
        weights = torch.ones((squared_root_difference.shape[0]),
                             device=squared_root_difference.device,
                             dtype=squared_root_difference.dtype)  # weights -> (batch_size * N)
        weights[labels == 0] = self._background_weight

        loss = torch.sum(weights * squared_root_difference) / torch.sum(weights)
        # ---------------- Computing rest of metrics (Paper Table 3)-----------------

        # We compute a dictionary with 3 different metrics:
        # mean: L2 mean. This computes the L2 mean per each class. We also distinguish the state of a class element,
        # which can be moving or stationary. A class element is considered as it is moving when the flow vector magnitude
        # is >= self._min_velocity and it is stationary when is less than _min_velocity.
        # Then, metrics = {mean: L2_mean,
        #                   ...: .....}
        # {mean: {all: {label1: xxx, label2: yyy, ...}, moving: {label1: xxx, label2: yyy}, stationary: {...}}, otherMetric: {...}, ...}
        # "all" does not distinghish the label state

        # We also compute the accuracy, which stands for the percentage of points with error below 0.1 m/s and 1.0 m/s (self.self._thresholds)
        # Depending on the threshold, we will have an item in the dictionary which is:
        # {mean: {...}, 1_1: {all: {label1: xxx, label2: yyy, ...}, moving: {label1: xxx, label2: yyy}, stationary: {...}}}
        # 1_1 stands for 1 / 1, which is 1 m/s, 1_10 stands for 1/10, which is 0.1 m/s.

        # Use detach as those metrics do not need a gradient
        L2_without_weighting = squared_root_difference.detach()
        flow_vector_magnitude_gt = torch.sqrt(torch.sum(y ** 2, dim=1))

        L2_mean = {}
        nested_dict = lambda: defaultdict(nested_dict)
        L2_thresholds = nested_dict()  # To create nested dict
        all_labels = {}  # L2 mean for labels without distinguish state
        moving_labels = {}  # L2 mean computed for each of the labels but only taking into account moving points
        stationary_labels = {}  # L2 mean computed for each of the labels but only taking into account stationary points
        for label, class_name in self._classes:
            # ----------- Computing L2 mean -------------

            # --- stationary, moving and all (sum of both) elements of the class ---
            # To generate boolean mask that will help us filter elements of the label we are iterating
            label_mask = labels == label

            # with label_mask we only take items of label we are iterating
            L2_label = L2_without_weighting[label_mask]
            flow_vector_magnitude_label = flow_vector_magnitude_gt[label_mask]

            stationary_mask = flow_vector_magnitude_label < self._min_velocity
            stationary = L2_label[stationary_mask]  # Extract stationary flows
            moving = L2_label[~stationary_mask]  # Extract flows in movement

            if L2_label.numel() != 0:
                all_labels[class_name] = L2_label.mean()
            if moving.numel() != 0:
                moving_labels[class_name] = moving.mean()
            if stationary.numel() != 0:
                stationary_labels[class_name] = stationary.mean()

            # ----------- Computing L2 accuracy with threshold -------------
            for threshold, name in self._thresholds:
                if L2_label.numel() != 0:
                    all_accuracy = (L2_label <= threshold).float().mean()
                    L2_thresholds[name]['all'][class_name] = all_accuracy
                if stationary.numel() != 0:
                    stationary_accuracy = (stationary <= threshold).float().mean()
                    L2_thresholds[name]['stationary'][class_name] = stationary_accuracy
                if moving.numel() != 0:
                    moving_accuracy = (moving <= threshold).float().mean()
                    L2_thresholds[name]['moving'][class_name] = moving_accuracy

        L2_mean['all'] = all_labels
        L2_mean['moving'] = moving_labels
        L2_mean['stationary'] = stationary_labels

        metrics = {'mean': L2_mean}
        metrics.update(L2_thresholds)
        return loss, metrics



    def general_step(self, batch, batch_idx, mode):
        """
        A function to share code between all different steps.
        :param batch: the batch to perform on
        :param batch_idx: the id of the batch
        :param mode: str of "train", "val", "test". Useful if specific things are required.
        :return:
        """
        # (batch_previous, batch_current), batch_targets
        x, y, trans = batch
        y_hat = self(x, trans)
        # x is a list of input batches with the necessary data
        # For loss calculation we need to know which elements are actually present and not padding
        # Therefore we need the mask of the current frame as batch tensor
        # It is True for all points that just are NOT padded and of size (batch_size, max_points)
        current_frame_masks = x[1][2]
        # Remove all points that are padded


        # dim 0 - forward and backward flow
        # dim 1 - outputs based on raft iteration (len == nbr of iter)
        # dim 2 - pointwise outputs, static aggr, dynamic threshold, bev outputs

        fw_pointwise = y_hat[0][-1][0]  # -1 for last raft output?
        fw_bev = y_hat[0][-1][3]
        fw_trans = y_hat[0][-1][1].cuda()
        bw_trans = y_hat[1][-1][1].cuda()

        # todo backward pass as well
        # todo if more samples in batch than 1, it needs to be verified
        # NN
        # forward
        # .cuda() should be optimized
        p_i = x[0][0][..., :3].float().cuda() # pillared - should be probably returned to previous form
        #p_i = p_i[..., :3] + p_i[..., 3:6]# previous form
        p_j = x[1][0][..., :3].float().cuda()
        # p_j = p_j[..., :3] + p_j[..., 3:6]# previous form

        # this is ambiguous, not sure if there is difference between static_flow and dynamic_flow
        # static ---> aggregated_static?
        raw_flow_i = fw_pointwise['dynamic_flow'].cuda()
        rigid_flow = fw_pointwise['static_aggr_flow'].cuda()

        fw_raw_flow_nn = knn_points(p_i + raw_flow_i, p_j, lengths1=None, lengths2=None, K=1, norm=1)
        fw_rigid_flow_nn = knn_points(p_i + rigid_flow, p_j, lengths1=None, lengths2=None, K=1, norm=1)

        # todo remove outlier percentage
        cham_x = fw_raw_flow_nn.dists[..., 0] + fw_rigid_flow_nn.dists[..., 0]  # (N, P1)

        # nearest_to_p_j = fw_raw_flow_nn[1]
        nn_loss = cham_x.max()

        # Rigid Cycle
        trans_p_i = torch.cat((p_i, torch.ones((len(p_i), p_i.shape[1], 1), device=p_i.device)), dim=2)
        bw_fw_trans = bw_trans @ fw_trans - torch.eye(4, device=fw_trans.device)
        # todo check this in visualization, if the points are transformed as in numpy
        res_trans = torch.matmul(bw_fw_trans, trans_p_i.permute(0, 2, 1)).norm(dim=1)

        rigic_cycle_loss = res_trans.mean()

        # Artificial label loss - for previous time not second
        def construct_batched_cuda_grid(pts, feature, x_min=-35, y_min=-35, grid_size=640):
            '''
            Assumes BS x N x CH (all frames same number of fake pts with zeros in the center)
            :param pts:
            :param feature:
            :param cfg:
            :return:
            '''
            BS = len(pts)
            bs_ind = torch.cat([bs_idx * torch.ones(pts.shape[1], dtype=torch.long, device=pts.device) for bs_idx in range(BS)])

            feature_grid = - torch.ones(BS, grid_size, grid_size, device=pts.device).long()

            cell_size = torch.abs(2 * torch.tensor(x_min / grid_size))

            coor_shift = torch.tile(torch.tensor((x_min, y_min), dtype=torch.float, device=pts.device), dims=(BS, 1, 1))

            feature_ind = ((pts[:, :, :2] - coor_shift) / cell_size).long()

            # breakpoint()
            feature_grid[bs_ind, feature_ind.flatten(0, 1)[:, 0], feature_ind.flatten(0, 1)[:, 1]] = feature.flatten().long()

            return feature_grid


        dynamic_states = (fw_raw_flow_nn.dists[..., 0] > fw_rigid_flow_nn.dists[..., 0]) + 1
        art_label_grid = construct_batched_cuda_grid(p_i, dynamic_states, x_min=-35, y_min=-35, grid_size=640)

        p_i_grid_class = fw_bev['class_logits'].cuda()

        # In paper, they have Sum instead of Mean, we should check original codes
        art_CE = torch.nn.CrossEntropyLoss(ignore_index=-1)
        artificial_class_loss = art_CE(p_i_grid_class, art_label_grid)

        # y = y[current_frame_masks]
        loss = 2. * nn_loss + 1. * rigic_cycle_loss + 0.1 * artificial_class_loss
        # loss.backward()   # backward probehne

        print(f"NN loss: {nn_loss.item():.4f}, Rigid cycle loss: {rigic_cycle_loss.item():.4f}, Artificial label loss: {artificial_class_loss.item():.4f}")

        metrics = 0
        # y_hat = y_hat[current_frame_masks]
        # This will yield a (n_real_points, 3) tensor with the batch size being included already


        # The first 3 dimensions are the actual flow. The last dimension is the class id.
        # y_flow = y[:, :3]
        # Loss computation
        # labels = y[:, -1].int()  # Labels are actually integers so lets convert them
        # Remove datapoints with no flow assigned (class -1)
        # mask = labels != -1
        # y_hat = y_hat[mask]
        # y_flow = y_flow[mask]
        # labels = labels[mask]
        # loss, metrics = self.compute_metrics(y_flow, y_hat, labels)

        return loss, metrics

    def log_metrics(self, loss, metrics, phase):
        # phase should be training, validation or test
        metrics_dict = {}
        for metric in metrics:
            for state in metrics[metric]:
                for label in metrics[metric][state]:
                    metrics_dict[f'{phase}/{metric}/{state}/{label}'] = metrics[metric][state][label]

        # Do not log the in depth metrics in the progress bar
        self.log(f'{phase}/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict(metrics_dict, on_step=True, on_epoch=True, prog_bar=False, logger=True)

    def training_step(self, batch, batch_idx):
        """
        This method is specific to pytorch lightning.
        It is called for each minibatch that the model should be trained for.
        Basically a part of the normal training loop is just moved here.

        model.train() is already set!
        :param batch: (data, target) of batch size
        :param batch_idx: the id of this batch e.g. for discounting?
        :return:
        """
        phase = "train"
        loss, metrics = self.general_step(batch, batch_idx, phase)
        # Automatically reduces this metric after each epoch
        self.log_metrics(loss, metrics, phase)
        # Return loss for backpropagation
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Similar to the train step.
        Already has model.eval() and torch.nograd() set!
        :param batch:
        :param batch_idx:
        :return:
        """
        phase = "val"
        loss, metrics = self.general_step(batch, batch_idx, phase)
        # Automatically reduces this metric after each epoch
        self.log_metrics(loss, metrics, phase)

    def test_step(self, batch, batch_idx):
        """
        Similar to the train step.
        Already has model.eval() and torch.nograd() set!
        :param batch:
        :param batch_idx:
        :return:
        """
        phase = "test"
        loss, metrics = self.general_step(batch, batch_idx, phase)
        # Automatically reduces this metric after each epoch
        self.log_metrics(loss, metrics, phase)

    def configure_optimizers(self):
        """
        Also pytorch lightning specific.
        Define the optimizers in here this will return the optimizer that is used to train this module.
        Also define learning rate scheduler in here. Not sure how this works...
        :return: The optimizer to use
        """
        # Defaults are the same as for pytorch
        #betas = (
        #    self.hparams.adam_beta_1 if self.hparams.adam_beta_1 is not None else 0.9,
        #    self.hparams.adam_beta_1 if self.hparams.adam_beta_2 is not None else 0.999)
        # lr = self.hparams.learning_rate

        return torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=0)

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Method to add all command line arguments specific to this module.
        Used to dynamically add the correct arguments required.
        :param parent_parser: The current argparser to add the options to
        :return: the new argparser with the new options
        """
        parent_parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = parent_parser.add_argument_group("General Model Params")
        parser.add_argument('--learning_rate', type=float, default=1e-6)
        parser.add_argument('--use_group_norm', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--background_weight', default=0.1, type=float)
        return parent_parser
