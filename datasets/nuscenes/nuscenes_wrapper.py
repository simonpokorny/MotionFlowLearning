import os.path

from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
import os
import json
from typing import List, Dict, Any
from tqdm import tqdm

from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation
import numpy as np
import fire
from PIL import Image
import matplotlib.pyplot as plt

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.utils.geometry_utils import BoxVisibility
from nuscenes.utils.data_classes import RadarPointCloud, LidarPointCloud, Box, load_bin_file
from nuscenes.utils.splits import create_splits_logs
from nuscenes.utils.kitti import KittiDB


class NuscenesConverter:
    def __init__(self,
                 output_dir: str = f'{os.path.expanduser("~")}/data/nuscenes/output/',
                 nusc_dir: str = f'{os.path.expanduser("~")}/data/nuscenes/',
                 nusc_version: str = 'v1.0-mini',
                 split: str = 'mini_train',
                 cam_name: str = 'CAM_FRONT',
                 lidar_name: str = 'LIDAR_TOP',
                 lidar_sweeps: int=10,
                 radar_sweeps: int=1,
                 image_count: int = None,
                 use_symlinks: bool = False):
        """
        :param output_dir: Where to write the KITTI-style annotations.
        :param cam_name: Name of the camera to export.
        :param lidar_name: Name of the lidar sensor.
        :param lidar_sweeps: Number of lidar sweeps
        :param radar_sweeps: Number of radar sweeps
        :param image_count: Number of images to convert.
        :param nusc_version: nuScenes version to use.
        :param split: Dataset split to use.
        """

        self.output_dir = os.path.expanduser(output_dir)
        self.cam_name = cam_name
        self.lidar_name = lidar_name
        self.lidar_sweeps = lidar_sweeps
        self.radar_sweeps = radar_sweeps
        self.image_count = image_count
        self.nusc_version = nusc_version
        self.split = split
        self.nusc_dir = nusc_dir
        self.use_symlinks = use_symlinks

        # Create output_dir.
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

        # Select subset of the data to look at.
        self.nusc = NuScenes(version=nusc_version,
                             dataroot=nusc_dir)
        self.nbr_of_sequences = len(self.nusc.scene)

    def get_data_from_sample(self):
        pass

    def get_ego_pose(self, sample):
        sd_rec = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        ego_pose = self.nusc.get('ego_pose', sd_rec['token'])
        ego_quats = ego_pose['rotation']
        ego_quats = np.array((ego_quats))[[3, 0, 1, 2]]  # shift from w,x,y,z to x,y,z,w for scipy rotation
        ego_trans = ego_pose['translation']

        ego_rot_mat = Rotation.from_quat(ego_quats).as_matrix()
        T_mat = np.eye(4)
        T_mat[:3, :3] = ego_rot_mat
        T_mat[:3, -1] = ego_trans

        return T_mat



    def get_boxes(self, sample):
        boxes_list = []

        for box_token in sample['anns']:

            box_dict = self.nusc.get('sample_annotation', box_token)
            x, y, z = box_dict['translation']
            w, l, h = box_dict['size']

            rot_quats = np.array(box_dict['rotation'])[[3, 0, 1, 2]]
            rot_vec = Rotation.from_quat(rot_quats).as_rotvec()
            yaw = rot_vec[2]
            cls = box_dict['category_name']
            idx = box_dict['instance_token']
            box = [x,y,z,l,w,h,yaw,cls,idx]

            boxes_list.append(box)

        boxes = np.stack(boxes_list)

        return boxes

    def get_seg_label(self, sample):

        lidar_top_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        lidar_token = lidar_top_data['token']

        try:
            lidarseg_filename = os.path.join(self.nusc_dir, 'lidarseg', self.nusc_version, lidar_token + '_lidarseg.bin')
            seg_label = load_bin_file(lidarseg_filename)

            panoptic_filename = os.path.join(self.nusc_dir, 'panoptic', self.nusc_version, lidar_token + '_panoptic.npz')
            panoptic_label = np.load(panoptic_filename, allow_pickle=True)['data']
        except:
            print("Error: Segmentation and panoptic labels aren't available")
            return None, None

        return seg_label, panoptic_label




    def one_scene(self, idx : int):
        # Create output folders.
        scene_token = self.nusc.scene[idx]['token']

        label_folder = os.path.join(self.output_dir, scene_token, self.split, 'seg_label')
        panoptic_folder = os.path.join(self.output_dir, scene_token, self.split, 'panoptic')
        lidar_folder = os.path.join(self.output_dir, scene_token, self.split, 'lidar')
        pose_folder = os.path.join(self.output_dir, scene_token, self.split, 'pose')
        boxes_folder = os.path.join(self.output_dir, scene_token, self.split, 'boxes')

        for folder in [label_folder, panoptic_folder, lidar_folder, pose_folder, boxes_folder]:
            if not os.path.isdir(folder):
                os.makedirs(folder)


        first_sample_token = self.nusc.scene[idx]['first_sample_token']
        last_sample_token = self.nusc.scene[idx]['last_sample_token']
        description = self.nusc.scene[idx]['description']

        sample = self.nusc.get('sample', first_sample_token)
        sample_token = sample['token']

        sample = {'next' : first_sample_token} # tmp for while loop
        frame = 0
        while sample['next'] != '':
            print(frame)
            sample_token = sample['next']
            sample = self.nusc.get('sample', sample_token)

            lidar_sample = self.nusc.get("sample_data", sample['data']["LIDAR_TOP"])
            for _ in range(300):

                lidar_token = lidar_sample['next']
                lidar_sample = self.nusc.get("sample_data", lidar_token)
                print(_, lidar_token) # sample like this, then interpolate boxes and flows
                ego_token = lidar_sample['ego_pose_token']
                ego_pose = self.nusc.get('ego_pose', ego_token)

                breakpoint()
            T_mat = self.get_ego_pose(sample)


            pcl, _ = LidarPointCloud.from_file_multisweep(
                    nusc=self.nusc,
                    sample_rec=sample,
                    chan=self.lidar_name,
                    ref_chan=self.lidar_name,
                    nsweeps=1,
                    min_distance=1)

            lidar = pcl.points.T

            # Annotation
            boxes = self.get_boxes(sample)

            lidarseg, panoptic = self.get_seg_label(sample)

            data_dict = {'lidar' : lidar,
                         'pose' : T_mat,
                         'frame' : frame,
                         'boxes' : boxes,
                         'seg_label': lidarseg,
                         'panoptic' : panoptic}

            for folder in [label_folder, panoptic_folder, lidar_folder, pose_folder, boxes_folder]:
                key = os.path.basename(folder)
                np.save(os.path.dirname(folder) + f'{frame:06d}.npy' , data_dict[key])

            frame += 1

        breakpoint()
    def convert_to_my_format(self):
        for seq_nbr in range(self.nbr_of_sequences):
            self.one_scene(seq_nbr)

if __name__ == "__main__":
    # this is in 2 HZ still, I will need to interpolate the labels etc.
    dataset = NuscenesConverter()
    dataset.convert_to_my_format()

