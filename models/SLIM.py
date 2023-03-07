import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from pytorch3d.ops.knn import knn_points

from models.networks import PillarFeatureNetScatter, PointFeatureNet, MovingAverageThreshold, RAFT
from models.networks.slimdecoder import OutputDecoder
from models.utils import init_weights
from visualization.plot import plot_2d_point_cloud, plot_tensor, visualise_tensor

VISUALIZATION = False


class SLIM(pl.LightningModule):
    def __init__(self, config):
        """
        Args:
            n_pillars_x (int): The number of pillars in the x dimension of the voxel grid.
            config (dict): Config is based on configs from configs/slim.yaml
        """
        super(SLIM, self).__init__()
        assert type(config) == dict

        self.save_hyperparameters()  # Store the constructor parameters into self.hparams
        self.n_pillars_x = config["default"]["n_pillars_x"]
        self.n_pillars_y = config["default"]["n_pillars_y"]

        self._point_feature_net = PointFeatureNet(in_features=config["default"]["point_features"], out_features=64)
        self._point_feature_net.apply(init_weights)

        self._pillar_feature_net = PillarFeatureNetScatter(n_pillars_x=self.n_pillars_x, n_pillars_y=self.n_pillars_y)
        self._pillar_feature_net.apply(init_weights)

        self._raft = RAFT(**config["raft"])

        self._moving_dynamicness_threshold = MovingAverageThreshold(**config["moving_threshold"])

        self._decoder_fw = OutputDecoder(**config["decoder"])
        self._decoder_bw = OutputDecoder(**config["decoder"])

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
        pillar_mask[:, :, y, x] = 1
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

        # Check VISUALIZATION
        if VISUALIZATION:
            plot_2d_point_cloud(current_batch_pc[0])
            # plot_pillars(current_voxel_coordinates[0], 35, -35, 35, -35., 0.109375)
            plot_tensor(current_batch_pillar_mask[0][0])
            visualise_tensor(current_batch_pillar_mask)
            # import matplotlib.pyplot as plt

        # 2. RAFT Encoder with motion flow backbone
        # Output for forward pass and backward pass
        # Each of the output is a list of num_iters x [1, 9, n_pillars_x, n_pillars_x]
        # logits, static_flow, dynamic_flow, weights are concatinate in channels in shapes [4, 2, 2, 1]
        outputs_fw, outputs_bw = self._raft(pillar_embeddings)

        # Transformation matrix Current (t1) to Previous (t0)
        C_T_P = torch.linalg.inv(G_T_C) @ G_T_P
        # Transformation matrix Previous (t0) to Current (t1)
        P_T_C = torch.linalg.inv(G_T_P) @ G_T_C

        predictions_fw = []
        predictions_bw = []

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
                dynamicness_threshold=self._moving_dynamicness_threshold.value(),
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

    def configure_optimizers(self):
        """
        Also pytorch lightning specific.
        Define the optimizers in here this will return the optimizer and schedulars that is used to train this module.
        :return: The optimizer and schedular to use

        SLIM official setup:

        initial: 0.0001
        step_decay:
        decay_ratio: 0.5
        step_length: 60000

        warm_up:
        initial: 0.01
        step_length: 2000

        """
        self.lr = 0.0001
        optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr)

        decay_ratio = 0.5
        decay = lambda step: decay_ratio ** int(step / 6000)
        scheduler_decay = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[decay])
        scheduler_decay = {'scheduler': scheduler_decay,
                           'interval': 'step',  # or 'epoch'
                           'frequency': 1}

        warm_up = lambda step: 0.01 ** (step / 2000) if (step < 2000) else 1
        scheduler_warm_up = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[warm_up])
        scheduler_warm_up = {'scheduler': scheduler_warm_up,
                             'interval': 'step',  # or 'epoch'
                             'frequency': 1}

        return [optimizer], [scheduler_decay, scheduler_warm_up]

    def log_metrics(self, loss, metrics, phase):
        # phase should be training, validation or test

        if phase != "train":
            raise NotImplementedError()
            metrics_dict = {}
            for metric in metrics:
                for state in metrics[metric]:
                    for label in metrics[metric][state]:
                        metrics_dict[f'{phase}/{metric}/{state}/{label}'] = metrics[metric][state][label]

        # Do not log the in depth metrics in the progress bar
        #self.log(f'{phase}/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
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
        fw_trans = y_hat[0][-1][1]
        bw_trans = y_hat[1][-1][1]

        # todo backward pass as well
        # todo if more samples in batch than 1, it needs to be verified
        # NN
        # forward
        # .cuda() should be optimized
        p_i = x[0][0][..., :3].float()  # pillared - should be probably returned to previous form
        # p_i = p_i[..., :3] + p_i[..., 3:6]# previous form
        p_j = x[1][0][..., :3].float()
        # p_j = p_j[..., :3] + p_j[..., 3:6]# previous form

        # this is ambiguous, not sure if there is difference between static_flow and dynamic_flow
        # static ---> aggregated_static?
        raw_flow_i = fw_pointwise['dynamic_flow']
        rigid_flow = fw_pointwise['static_aggr_flow']

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
            bs_ind = torch.cat(
                [bs_idx * torch.ones(pts.shape[1], dtype=torch.long, device=pts.device) for bs_idx in range(BS)])

            feature_grid = - torch.ones(BS, grid_size, grid_size, device=pts.device).long()

            cell_size = torch.abs(2 * torch.tensor(x_min / grid_size))

            coor_shift = torch.tile(torch.tensor((x_min, y_min), dtype=torch.float, device=pts.device), dims=(BS, 1, 1))

            feature_ind = ((pts[:, :, :2] - coor_shift) / cell_size).long()

            # breakpoint()
            feature_grid[
                bs_ind, feature_ind.flatten(0, 1)[:, 0], feature_ind.flatten(0, 1)[:, 1]] = feature.flatten().long()

            return feature_grid

        dynamic_states = (fw_raw_flow_nn.dists[..., 0] > fw_rigid_flow_nn.dists[..., 0]) + 1
        art_label_grid = construct_batched_cuda_grid(p_i, dynamic_states, x_min=-35, y_min=-35, grid_size=640)

        p_i_grid_class = fw_bev['class_logits']

        # In paper, they have Sum instead of Mean, we should check original codes
        art_CE = torch.nn.CrossEntropyLoss(ignore_index=-1)
        artificial_class_loss = art_CE(p_i_grid_class, art_label_grid)

        # y = y[current_frame_masks]
        loss = 2. * nn_loss + 1. * rigic_cycle_loss + 0.1 * artificial_class_loss
        # loss.backward()   # backward probehne

        print(
            f"NN loss: {nn_loss.item():.4f}, Rigid cycle loss: {rigic_cycle_loss.item():.4f}, Artificial label loss: {artificial_class_loss.item():.4f}")

        self.log(f'{mode}/loss', nn_loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'{mode}/loss', rigic_cycle_loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'{mode}/loss', artificial_class_loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)

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


if __name__ == "__main__":

    ### CONFIG ###
    with open("../configs/slim.yaml", "r") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    ### MODEL ####
    model = SLIM(config=cfg)

    ### DATAMODULE ###
    from datasets.waymoflow.WaymoDataModule import WaymoDataModule

    dataset_path = "../data/waymoflow_subset"
    # dataset_path = "/Users/simonpokorny/mnt/data/waymo/raw/processed/training"
    grid_cell_size = 0.109375
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

    ### TRAINER ###
    trainer = pl.Trainer(fast_dev_run=True, num_sanity_val_steps=0)  # Add Trainer hparams if desired
    trainer.fit(model, data_module)
    # trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    print("done")
