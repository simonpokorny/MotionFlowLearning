import importlib
import os

import numpy as np
import torch
import multiprocessing

from models.data.util import ApplyPillarization, custom_collate_batch
from models.utils.pillars import remove_out_of_bounds_points





class SceneFlowLoader(torch.utils.data.Dataset):
    # todo add flow and extend to all sequences - that might be done somewhere else
    # todo connect it with metrics in batch



    def __init__(self, sequence, cfg):
        self.sequence = sequence
        self.cfg = cfg
        self.apply_transformation = True
        grid_cell_size = (cfg['x_max'] + abs(cfg['x_min'])) / cfg['grid_size']

        n_pillars_x = cfg['grid_size']
        n_pillars_y = cfg['grid_size']

        # This can be later split in datasetclass
        self.pilarization = ApplyPillarization(grid_cell_size=grid_cell_size, x_min=cfg['x_min'], y_min=cfg['y_min'],
                                          z_min=cfg['z_min'], z_max=cfg['z_max'], n_pillars_x=n_pillars_x,
                                          )

    def collect_data(self):
        info = self.sequence.info()

        nbr_of_seqs = info['nbr_of_seqs']

        from datasets.kitti.semantic_kitti import SemanticKitti_Sequence
        # zahodit posledni
        data_files = []

        all_seqs = [SemanticKitti_Sequence(seq) for seq in range(nbr_of_seqs)]

        frames_of_seqs = [[curr_seq.sequence_nbr, list(range(len(curr_seq)))] for curr_seq in all_seqs]

        # do it better with respect to training splits!
    


        pass

    def __len__(self):
        return len(self.sequence) - 1 # to use the last current label as well

    def __getitem__(self, idx):

        pts1, pts2 = self.sequence.get_two_synchronized_frames(idx, idx+1, pts_source='lidar')
        ego_label1 = self.sequence.get_feature(idx, name='dynamic_label')   # for prev frame    # tmp dynamic label
        ego_label2 = self.sequence.get_feature(idx+1, name='dynamic_label')   # for current frame

        # todo add if ego_label none, then create or -1s
        # first eliminate out of boundaries
        pts1, mask1 = remove_out_of_bounds_points(pts1, self.cfg['x_min'], self.cfg['x_max'], self.cfg['y_min'],
                                                  self.cfg['y_max'],
                                                  self.cfg['z_min'], self.cfg['z_max'])
        pts2, mask2 = remove_out_of_bounds_points(pts2, self.cfg['x_min'], self.cfg['x_max'], self.cfg['y_min'],
                                                  self.cfg['y_max'],
                                                  self.cfg['z_min'], self.cfg['z_max']) # add here the AV points

        ego_label1 = ego_label1[mask1]
        ego_label2 = ego_label2[mask2]

        # Subsample to get previous and current the same number of points
        min_nbr_pts = np.min([pts1.shape[0], pts2.shape[0]])
        # subsample_idx1 = np.random.choice(pts1.shape[0], min_nbr_pts)
        # subsample_idx2 = np.random.choice(pts2.shape[0], min_nbr_pts)
        # TODO: check if this is correct
        subsample_idx1 = np.arange(min_nbr_pts)
        subsample_idx2 = np.arange(min_nbr_pts)

        pts1 = pts1[subsample_idx1]
        pts2 = pts2[subsample_idx2]
        ego_label1 = ego_label1[subsample_idx1]
        ego_label2 = ego_label2[subsample_idx2]


        # this can be changed in config
        pts1 = np.insert(pts1, 4, 1, axis=1)
        pts2 = np.insert(pts2, 4, 1, axis=1)



        pts1, grid1 = self.pilarization(pts1)
        pts2, grid2 = self.pilarization(pts2)
        mask1 = np.ones(pts1[:, 0].shape, dtype=bool)
        mask2 = np.ones(pts2[:, 0].shape, dtype=bool)   # is right?

        flow = np.random.rand(pts1.shape[0], 3)

        prev_batch = (pts1, grid1, mask1, ego_label1)
        current_batch = (pts2, grid2, mask2, ego_label2)    # normal pts as well?

        x = (prev_batch, current_batch) # is this really the pts1 and pts2?

        return x

    def return_dataloader(self, batch_size=1, num_workers=0, shuffle=True):
        cpu_count = multiprocessing.cpu_count()

        num_workers = np.min((cpu_count, self.cfg['BS'])) if self.cfg["BS"] > 1 else 0

        dataloader = torch.utils.data.DataLoader(self, batch_size=self.cfg['BS'], shuffle=shuffle,
                                                 num_workers=num_workers, collate_fn=custom_collate_batch)

        return dataloader

if __name__ == '__main__':
    from datasets.kitti.semantic_kitti import SemanticKitti_Sequence
    from datasets.waymo.waymo import Waymo_Sequence
    from datasets.argoverse.argoverse2 import Argoverse2_Sequence

    import glob
    from tqdm import tqdm


    DATASETS = [SemanticKitti_Sequence]#, Waymo_Sequence, Argoverse2_Sequence]


    # gen_labels = ['ego_prior_label', 'corrected_ego_prior_label', 'visibility_prior', 'corrected_visibility_prior', 'prior_static_mask', 'road_proposal']
    gen_labels = ['final_prior_label']


    for dataset_idx in range(len(DATASETS)):

        sequence = DATASETS[dataset_idx](0)

        os.makedirs('meta', exist_ok=True)
        data_desc = open(f'meta/{DATASETS[dataset_idx].__name__}_metadata.txt', 'w')


        # data_desc.write('sequence frame nbr_of_dyn_pts ' + " ".join(gen_labels) +  '\n')

        for seq_id, log_id in enumerate(tqdm(sequence.log_ids)):

            seq = DATASETS[dataset_idx](seq_id)

            label_source = 'final_prior_label'


            for frame in range(len(seq)):

                dyn_label = seq.get_feature(frame, name=label_source)
                nbr_dyn_pts = np.sum(dyn_label == 1)
                str_line = f"{seq_id} {frame} {nbr_dyn_pts}"

                # for gen_label in gen_labels:
                #     gen_label_values = seq.get_feature(frame, name=gen_label)
                #
                #     if gen_label == 'prior_static_mask':
                #         gen_label_values = gen_label_values == 30 # for static values to be mapped as static pts
                #
                #     nbr_gen_pts = np.sum(gen_label_values == 1)
                #
                #     str_line += f" {nbr_gen_pts}"

                # print(str_line)
                data_desc.write(str_line + '\n')


        data_desc.close()
