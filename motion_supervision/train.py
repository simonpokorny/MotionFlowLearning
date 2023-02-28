import sys

import numpy as np
import torch
import time
import os
from tqdm import tqdm
from models.models import FastFlow3DModelScatter
from models.data.util import custom_collate_batch

from datasets.argoverse.argoverse2 import Argoverse2_Sequence
from pytorch3d.ops.knn import knn_points

from motion_supervision.dataset import SceneFlowLoader


# cannot import it from rci, IDK why

def get_device_idx_for_port():  # TODO not for multiple gpus
    gpu_txt = open('/home/vacekpa2/gpu.txt', 'r').readlines()
    os.system('nvidia-smi -L > /home/vacekpa2/gpu_all.txt')

    time.sleep(0.1)
    gpu_all_txt = open('/home/vacekpa2/gpu_all.txt', 'r').readlines()

    gpu_all_txt = [text[7:] for text in gpu_all_txt]
    device_idx = 0
    for idx, gpu_id in enumerate(gpu_all_txt):
        if gpu_txt[0][7:] == gpu_id:
            device_idx = idx

    return device_idx

def get_device():
    if torch.cuda.is_available():
        device_idx = get_device_idx_for_port()
        device = torch.device(device_idx)
    else:
        device = torch.device('cpu')
    return device

def print_gpu_memory():
    if torch.cuda.is_available():
        free_memory = torch.cuda.mem_get_info()[0] / 1024 / 1024
        max_memory = torch.cuda.mem_get_info()[1] / 1024 / 1024
        memory_consumed = max_memory - free_memory
        print(f"Memory consumption: {memory_consumed:.0f} MB")

def store_batch_data(prev_pts, curr_pts, flow, mos, ego_label, loss, data_dir=f'{os.path.expanduser("~")}/data/toy/'):
    for i in range(len(prev_pts)):
        np.save(f'{data_dir}/prev_pts_{i:06d}.npy', prev_pts[i].cpu().detach().numpy())
        np.save(f'{data_dir}/curr_pts_{i:06d}.npy', curr_pts[i].cpu().detach().numpy())
        np.save(f'{data_dir}/flow_{i:06d}.npy', flow[i].cpu().detach().numpy())
        np.save(f'{data_dir}/mos_{i:06d}.npy', mos[i].cpu().detach().numpy())
        np.save(f'{data_dir}/ego_{i:06d}.npy', ego_label[i].cpu().detach().numpy())
        np.save(f'{data_dir}/loss_{i:06d}.npy', loss[i].cpu().detach().numpy())
        # print('storing only first sample from batch')
        break
    # store it inteligently


def NN_loss(x, y, x_lengths=None, y_lengths=None, reduction='mean'):
    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, K=1, norm=1)
    y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, K=1, norm=1)

    cham_x = x_nn.dists[..., 0]  # (N, P1)
    cham_y = y_nn.dists[..., 0]  # (N, P2)

    if reduction == 'mean':
        nn_loss = (cham_x.mean() + cham_y.mean()) / 2
    elif reduction == 'sum':
        nn_loss = (cham_x.sum() + cham_y.sum()) / 2
    else:
        raise NotImplementedError

    return nn_loss


def ego_loss(mos, ego_label):
    # Ego label
    mos = (mos - mos.min()) / (mos.max() - mos.min())  # normalize , probably keep the sigmoid function
    # mos = mos * 10 - 5  # for sigmoid range, is it correct?
    # I can normalize to -6 and 6 and then use sigmoid for 0-1 range
    mos = torch.sigmoid(mos)  # sigmoid  # this will make the 0.5 values on "zeros in visualization"
    # todo important - add weights for that
    nbr_dyn = (ego_label == 1).sum()
    nbr_stat = (ego_label == 0).sum()
    nbr_ego = nbr_dyn + nbr_stat

    ego_dynamic_loss = - torch.log(mos[ego_label == 1]).mean() if (ego_label == 1).sum() > 0 else 0
    ego_static_loss = - torch.log(1 - mos[ego_label == 0]).mean() if (ego_label == 0).sum() > 0 else 0

    MOS_loss = (nbr_stat / nbr_ego) * ego_dynamic_loss + (nbr_dyn / nbr_ego) * ego_static_loss  # dynamic from ego and static from ego road 1/0, mean reduction

    return MOS_loss

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, ce_weights=(1, 1), reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ce_weights = ce_weights
        self.reduction = reduction

        self.CE = nn.CrossEntropyLoss(weight=torch.tensor(ce_weights), ignore_index=-1, reduction='none')

    def forward(self, logits, target):
        # Logits: B, N, C, but to CrossEntropy it needs to be B, C, N
        # Target: B, N
        # for speed, there can be only one softmax I guess
        logits = logits.permute(0, 2, 1)

        CE_loss = self.CE(logits, target)

        logits = F.log_softmax(logits, dim=1)

        pt = logits.permute(0, 2, 1)

        pt = pt.flatten(start_dim=0, end_dim=1)
        target_gather = target.flatten()

        ignore_index = -1
        valid_mask = target_gather != ignore_index
        valid_target = target_gather[valid_mask]
        valid_pt = pt[valid_mask]
        CE_loss = CE_loss.flatten()[valid_mask]

        valid_target = valid_target.tile(2,1).permute(1,0)    # get the same shape as pt
        only_probs_as_target = torch.gather(valid_pt, 1, valid_target)[:,0]

        loss = (1 - only_probs_as_target) ** self.gamma * CE_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == "none":
            return loss

def get_real_lengths_in_batch(prev_pts, curr_pts):
    # This can be done by storing the lengths after padding to speed up
    # flow_non_zero = [flow[i][curr_pts[i].abs().sum(dim=1).bool()] for i in range(len(curr_pts))]
    prev_list = [pts[pts.abs().sum(dim=1).bool()] for pts in prev_pts]
    curr_list = [pts[pts.abs().sum(dim=1).bool()] for pts in curr_pts]
    # nbr_pts = np.min([pts.shape[0] for pts in prev_list] + [pts.shape[0] for pts in curr_list])# can be done beforehand

    x_lengths = torch.tensor([pts.shape[0] for pts in prev_list], dtype=torch.long)
    y_lengths = torch.tensor([pts.shape[0] for pts in curr_list], dtype=torch.long)

    if torch.cuda.is_available():
        x_lengths = x_lengths.cuda()
        y_lengths = y_lengths.cuda()

    return x_lengths, y_lengths


if __name__ == "__main__":
    cfg = {'x_max' : 35.0,  # orig waymo 85m
           'x_min' : -35.0,
           'y_max' : 35.0,
           'y_min' : -35.0,
           'z_max' : 3.0,
           'z_min' : -3.0,
           'grid_size' : 640,   # slim 640
           'background_weight' : 0.1,
           'learning_rate' : 0.01,
           'weight_decay' : 0.0001,
           'use_group_norm' : False,
           'BS' : 10,
           }

    device = get_device()

    # number of workers matters for speed, pillarization etc.

    model = FastFlow3DModelScatter(n_pillars_x=cfg['grid_size'], n_pillars_y=cfg['grid_size'],
                                   background_weight=cfg['background_weight'], point_features=8,
                                   learning_rate=cfg['learning_rate'],
                                   use_group_norm=cfg['use_group_norm']).cuda()


    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    # optimizer = torch.optim.Adam([weights], lr=cfg['learning_rate'])

    # Conclussion, the fast flow model cannot express the segmentation as well.
    # I need to do it in the level of unet output grid

    # todo final flow is based on grid embedding - maybe calculate losses and metrics on that?
    # flow is predicted on current frame!



    # tried weights, tried model, check outputs input, tried CE loss without weights
    # trying group norm if batch norm is the problem - nope
    # It was the logits, I need to use pytorch crossentropy with normalized inputs, otherwise it is not stable

    # flow_dataset.collect_data()
    # sys.exit("1")
    exp_folder = os.path.expanduser("~") + '/data/fastflow/'
    os.makedirs(exp_folder, exist_ok=True)

    max_epoch = 700
    # max_iter = int(len(dataloader) / cfg["BS"])

    for epoch in range(max_epoch):
        model = model.train()
        loss_list = []

        # just to iterate over it
        dataset = Argoverse2_Sequence(sequence_nbr=epoch)
        flow_dataset = SceneFlowLoader(sequence=dataset, cfg=cfg)
        dataloader = torch.utils.data.DataLoader(flow_dataset, batch_size=cfg['BS'], shuffle=True,
                                                 num_workers=np.min((0, cfg['BS'])), collate_fn=custom_collate_batch)

        for idx, batch in enumerate(dataloader):

            prev, curr = batch

            flow, mos = model(batch)
            max_of_logits = mos.max().item()

            # todo assign normalization to model, when you are 100% sure
            # todo label -1 should be assigned even to the points that are not in the orig pts
            mos = (mos - mos.min()) / (mos.max() - mos.min())  # normalize , probably keep the sigmoid function
            # mos = mos * 10 - 5  # for sigmoid range, is it correct?

            ego_label = curr[3].to(device) # this needs to be check during refactor!

            ce_weights = torch.tensor((45. / 1000, 1118. / 1000), device=device)
            Loss_Function = FocalLoss(gamma=2, ce_weights=ce_weights, reduction='mean')
            loss = Loss_Function(mos, ego_label.long())


            dynamic_prediction = torch.argmax(mos.flatten(0,1), dim=1)

            loss.backward()

            loss_list.append(loss.item())

            optimizer.step()
            optimizer.zero_grad()
            print(f"Epoch: {epoch:03d}/{max_epoch:03d} Iter: {idx:04d}, Overall Loss: {np.mean(loss_list):.4f}, MOS per-batch loss: {loss.item():.4f}, stored mos max: {mos.max()}, dynamic points {(dynamic_prediction == 1).sum().item()}, 'static points' {(dynamic_prediction == 0).sum().item()}")

        torch.save(model.state_dict(), exp_folder + f'/{epoch:03d}_fastflow_weights.pth')


            # This is for getting back the original points
            # prev_pts = (prev[0][...,:3] + prev[0][...,3:6]).to(device)
            # curr_pts = (curr[0][...,:3] + curr[0][...,3:6]).to(device) # in future solve inconsistent sending to cuda
            # ego_label = ego_label.long().to(device)
            # x_lengths, y_lengths = get_real_lengths_in_batch(prev_pts, curr_pts)

            # x = prev_pts + flow
            # y = curr_pts

            # todo cycle consistency and anchor NN point (it is part of the cycle consistency)
            # todo frame back and forth consistency
            # todo artificial labels
            # todo truncate flow?
            # https://just-go-with-the-flow.github.io/

            # loss = loss_per_point.mean()

            # loss = MOS_loss




            # store_batch_data(prev_pts, curr_pts, flow, mos, ego_label, loss_per_point)

            # breakpoint()
            # break
        # # one iteration for storing results
        # todo do this intelligently after refactor
        # with torch.no_grad():
        #     model = model.eval()
        #     batch = custom_collate_batch([flow_dataset.__getitem__(34)]) # tryouts
        #     flow, mos = model(batch)
        #     mos = (mos - mos.min()) / (mos.max() - mos.min())
        #     prev, curr = batch
        #     prev_pts = (prev[0][..., :3] + prev[0][..., 3:6]).to(device)
        #     curr_pts = (curr[0][..., :3] + curr[0][..., 3:6]).to(device)
        #     ego_label = curr[3]
        #     loss_per_point = CE(mos.permute(0, 2, 1), ego_label.long().cuda())
        #     store_batch_data(prev_pts, curr_pts, flow, mos, ego_label, loss_per_point)
            # MOS_loss = ego_loss(mos, ego_label)
        # sceneflow_weights = model.state_dict()
        # torch.save(sceneflow_weights, 'sceneflow_weights.pth')

        # print(f"Overall Loss: {np.mean(loss_list):.4f}, MOS validation loss: {MOS_loss.item():.4f}, stored mos max: {mos.max():.2f}, loss length: {len(loss_list)}")
        # print(f"Overall Loss: {np.mean(loss_list):.4f}, max logit: {max_of_logits:.2f}, loss length: {len(loss_list)}")



    print_gpu_memory()
