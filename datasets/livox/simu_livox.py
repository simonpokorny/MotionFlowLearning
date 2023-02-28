import os

import numpy as np
import glob
from tqdm import tqdm
from datasets import paths
from datasets.argoverse import argoverse2
from timespace.box_utils import get_point_mask
import torch
from motion_supervision.motionflow import pytorch3d_ICP


seg_dict = {'unknown' : 0,
            'car' : 1,
            'truck' : 2,
            'bus' : 3,
            'bicycle' : 4,
            'motor' : 4,    # motorcyclist probably
            'bimo' : 4,
            'pedestrian' : 6,
            'dog' : 7,
            'road' : 8,
            'ground' : 9,
            'building' : 10,
            'fence' : 11,
            'tree' : 12,
            'pole' : 13,
            'greenbelt' : 14}

class Simu_Livox(basics.Basic_Dataprocessor):
    def __init__(self, data_dir=machine_paths.simu_livox, sequence=0, max_len=np.inf):
        super().__init__(data_dir=data_dir, max_len=max_len)
        self.data_dir = data_dir
        self.sequence = os.path.basename(data_dir)

        from_frame = int(self.sequence) * 500
        to_frame = (int(self.sequence) + 1) * 500

        print(self.data_dir, from_frame, to_frame)
        self.anno_files = sorted(glob.glob(self.data_dir + '/../../raw/anno/*.txt'))[from_frame : to_frame]
        self.pts_files = sorted(glob.glob(self.data_dir + '/../../raw/points/*.txt'))[from_frame : to_frame]
        # how to ?
        self.ego_box = argoverse2.Argoverse2_Sequence().ego_box
        self.ego_box[2] = - 0.8  # priblizne

    def parse_anno(self, frame):
        ''' orig box: [id, class, x, y, z, l, w, h, yaw]
            new box: [x, y, z, l, w, h, yaw, class, id] '''
        anno = open(self.anno_files[frame])
        anno_lines = anno.readlines()
        boxes_li = []
        for line in anno_lines:
            line = line.strip('\n')
            line = line.split(',')
            cls_mapping = seg_dict[line[1]]
            box = line[2:] + [cls_mapping] + line[0:1]
            box = np.array(box, dtype=float)
            box[8] += 1 # increase id by one for motion flow and to have id_mask uint with 0 reserved for nothing

            boxes_li.append(box)

        boxes = np.stack(boxes_li)

        return boxes


    def parse_pts(self, frame):
        ''' x,y,z,motion state, segmentation, lidar number'''
        pts = open(self.pts_files[frame])
        lines = pts.readlines()
        pcl = []

        for line in lines:
            line = line.strip('\n')
            line = line.split(',')
            point = np.array(line, dtype=float)
            pcl.append(point)

        pcl = np.stack(pcl)

        return pcl

    def generate_poses_from_ICP(self, frame):
        if frame == 0:
            return np.eye(4)

        pts1 = np.load(f"{self.data_dir}/lidar/{frame-1:06d}.npy")
        pts2 = np.load(f"{self.data_dir}/lidar/{frame:06d}.npy")

        # get rid of the moving points
        pts1 = pts1[pts1[:, 3] == 0]
        pts2 = pts2[pts2[:, 3] == 0]

        # to torch
        pts1 = torch.tensor(pts1).cuda()
        pts2 = torch.tensor(pts2).cuda()

        T_mat, transformed_pts = pytorch3d_ICP(pts1, pts2, verbose=False)

        # T_mat is for frame + 1
        return np.linalg.inv(T_mat.detach().cpu().numpy())

    def store_own_format(self, seq_name='.', from_frame=0, to_frame=np.inf):
        for folder in ['lidar', 'boxes','relative_pose','pose','seg_labels','id_mask','motion_state','gt_flow', 'ego_dynamic', 'ego_box_time']:
            os.makedirs(self.data_dir + f"{seq_name}/{folder}", exist_ok=True)

        curr_pose = np.eye(4)

        for frame in tqdm(range(from_frame, to_frame)):
            if frame < from_frame or frame > to_frame: continue

            pts = self.parse_pts(frame)
            boxes = self.parse_anno(frame)
            id_mask = np.zeros(pts.shape[0], dtype=int)

            for box in boxes:
                mask = get_point_mask(pts[:, :3], box[:7])
                id_mask[mask] = box[8]

            np.save(f"{self.data_dir}/{seq_name}/lidar/{frame:06d}.npy", pts[:,[0,1,2,5]])
            np.save(f"{self.data_dir}/{seq_name}/boxes/{frame:06d}.npy", boxes)
            np.save(f"{self.data_dir}/{seq_name}/motion_state/{frame:06d}.npy", pts[:,3])
            np.save(f"{self.data_dir}/{seq_name}/seg_labels/{frame:06d}.npy", pts[:,4])
            np.save(f"{self.data_dir}/{seq_name}/id_mask/{frame:06d}.npy", id_mask)

            T_mat = self.generate_poses_from_ICP(frame)
            np.save(f"{self.data_dir}/{seq_name}/relative_pose/{frame:06d}.npy", T_mat)

            T_mat = T_mat @ curr_pose
            np.save(f"{self.data_dir}/{seq_name}/pose/{frame:06d}.npy", T_mat)

            print(curr_pose, T_mat)

            curr_pose = T_mat.copy() # assign newest pose

            # flow
            frame_flow = self.get_flow_from_rigid_boxes(frame)
            self.store_feature(frame_flow, frame, name=f'{seq_name}/gt_flow')


    def rename_frames_to_zero(self):
        sequences = glob.glob(self.data_dir + '/sequences/*')

        for seq in sequences:
            for folder in os.listdir(seq):

                for idx, file in enumerate(sorted(os.listdir(seq + '/' + folder))):
                    src = seq + '/' + folder + '/' + file
                    dst = seq + '/' + folder + '/' + f"{idx:06d}.npy"
                    print(src, dst)
                    os.rename(src, dst)

if __name__ == "__main__":
    # command to run on rci
    # for i in {0..10}; do change_line "python -u data_utils" "python -u data_utils/livox/simu_livox.py $i" simu_livox.sh && sleep 0.2 && sbatch simu_livox.sh && sleep 0.2; done
    # frame flow metrics
    import sys
    seq = int(sys.argv[1])

    if len(sys.argv) < 2:
        from_frame = 0
        to_frame = np.inf
    else:
        from_frame = seq * 500
        to_frame = (seq+1) * 500

    seq_nbr = 0
    dataset = Simu_Livox(data_dir=os.path.expanduser("~") + f'/data/simu_livox/sequences/{seq_nbr:02d}')

    dataset.store_own_format(seq_name=f".", from_frame=from_frame, to_frame=to_frame)

    # flow = dataset.get_feature(10, 'gt_flow')
    # pts1 = dataset.get_global_pts(10, 'lidar')
    # pts2 = dataset.get_global_pts(11, 'lidar')
    # visualizer.visualizer_flow3d(pts1, pts2, flow)

    # visualizer.visualize_points3D(pts, pts[:,3])
    # visualizer.visualize_points3D(pts, pts[:,4])
    # visualizer.visualize_points3D(pts, pts[:,5])
