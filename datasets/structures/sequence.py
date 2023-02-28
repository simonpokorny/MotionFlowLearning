import numpy as np
import matplotlib.pyplot as plt
import os
import glob

import yaml

from scipy.spatial.transform.rotation import Rotation
from PIL import Image

from timespace.box_utils import get_boxes_from_ego_poses
from timespace.trajectory import construct_transform_matrix
from timespace import box_utils

class Basic_Dataprocessor(object):

    def __init__(self, data_dir, max_len=np.inf):
        self.data_dir = data_dir
        self.max_len = max_len  # todo hack

    def unified_preload(self, path : str):
        if path.endswith('.bin'):
            return np.fromfile(path, dtype=np.float32).reshape(-1,4)

        elif path.endswith('.npy'):
            return np.load(path, allow_pickle=True)

        elif path.endswith('.png') or path.endswith('.jpg'):
            return np.asarray(Image.open(path))

        elif path.endswith('.label'):
            return NotImplementedError("SemanticKitti format")
        else:
            raise NotImplementedError("Different formats")

    def unified_store(self, feature, path:str):
        if path.endswith('.bin'):
            return np.fromfile(path, dtype=np.float32).reshape(-1,4)

        elif path.endswith('.npy'):
            return np.load(path, allow_pickle=True)

        elif path.endswith('.png') or path.endswith('.jpg'):
            return np.asarray(Image.open(path))

    @classmethod
    def pts_to_frame(self, pts, pose):
        '''

        :param pts: point cloud
        :param pose: 4x4 transformation matrix
        :return:
        '''
        transformed_pts = pts.copy()
        transformed_pts[:, 3] = 1
        transformed_pts[:, :3] = (transformed_pts[:,:4] @ pose.T)[:, :3]

        transformed_pts[:,3:] = pts[:,3:]

        return transformed_pts

    @classmethod
    def frame_to_pts(self, pts, pose):
        raise NotImplementedError("Not yet needed")

    def get_global_pts(self, idx, name):
        pts = self.get_feature(idx, name)
        pose = self.get_feature(idx, 'pose')
        global_pts = self.pts_to_frame(pts, pose)

        return global_pts

    def get_two_synchronized_frames(self, frame1, frame2, pts_source='lidar'):
        '''

        :param frame1: Frame one is the one in origin
        :param frame2: Different frame, that would be synchronized to the first one (shifter from origin)
        :return:
        '''
        pose1 = self.get_feature(frame1, 'pose')

        pts1 = self.get_global_pts(frame1, pts_source)
        pts2 = self.get_global_pts(frame2, pts_source)

        tmp_pts1 = pts1.copy()
        tmp_pts1[:, 3] = 1

        tmp_pts2 = pts2.copy()
        tmp_pts2[:, 3] = 1

        back_pts1 = (np.linalg.inv(pose1) @ tmp_pts1.T).T
        back_pts2 = (np.linalg.inv(pose1) @ tmp_pts2.T).T

        return back_pts1, back_pts2


    def get_feature(self, idx : int, name : str, ext='npy'):
        path_to_feature = self.data_dir + name + f'/{idx:06d}.{ext}'

        return self.unified_preload(path_to_feature)

    def has_feature(self, idx, name, ext='npy'):
        file_name = self.data_dir + name + f'/{idx:06d}.{ext}'

        return os.path.exists(file_name)

    def store_feature(self, feature, idx, name):

        if 'rgb' in name:
            path_to_feature = self.data_dir + name + f'/{idx:06d}.png'
            os.makedirs(os.path.dirname(path_to_feature), exist_ok=True)
            Image.fromarray(feature).save(path_to_feature)

        else:
            path_to_feature = self.data_dir + name + f'/{idx:06d}.npy'
            os.makedirs(os.path.dirname(path_to_feature), exist_ok=True)
            np.save(path_to_feature, feature)



    def get_range_of_feature(self, start, end, name):
        features = [self.get_feature(i, name) for i in range(start, end)]
        return features

    def get_ego_poses(self, format='array'):
        # poses_dict = {}
        # for t in range(self.__len__()):
        #     if t == 100: break # TODO TMP!
        #     poses_dict[t] = self.get_feature(t, 'pose')
            # {t: self.get_feature(t, name='pose') for t in range(self.__len__())}
        # return poses_dict
        if format == 'dict':
            poses = {t: self.get_feature(t, name='pose') for t in range(self.__len__())}
        else:
            # poses = [self.get_feature(t, name='pose') for t in range(self.__len__())]
            files = glob.glob(self.data_dir + '/pose/*.npy')
            poses = np.stack([np.load(f) for f in files])

        return poses

    def get_ego_boxes(self):
        pose_list = list(self.get_ego_poses().values())
        ego_boxes = get_boxes_from_ego_poses(pose_list, self.ego_box)

        return ego_boxes

    def __len__(self):
        # todo hack for now
        nbr_lidar_frames = len(glob.glob(self.data_dir + '/lidar/*'))
        if self.max_len >= nbr_lidar_frames:
            return len(glob.glob(self.data_dir + '/lidar/*'))
        else:
            return self.max_len



    def calculate_flow(self, pts1, box1, box2, ego_pose1, ego_pose2, undefined_box_z=0.2, move_threshold=0.1):  # pts and boxes in local frame
        flow = np.zeros((pts1.shape[0], 4), dtype=float)
        dynamic = np.zeros(pts1.shape[0])

        tmp_pts1 = pts1.copy()
        tmp_pts1[:, 3] = 1
        T2_to_T1 = np.linalg.inv(ego_pose1) @ ego_pose2

        pts1_in_pts2 = tmp_pts1 @ T2_to_T1.T

        flow[:, :3] = - pts1_in_pts2[:, :3] + pts1[:, :3]  # rigid flow from ego-motion

        # per object flow
        id_box_dict = {box['uuid']: box for box in box2}
        for one_box in box1:

            box1_uuid = one_box['uuid']

            label_mask = box_utils.get_point_mask(pts1, one_box)    # to prevent oversegmentation
            flow[label_mask, 3] = -1
            dynamic[label_mask] = -1

            pts_in_box = box_utils.get_point_mask(pts1, one_box, z_add=( - undefined_box_z, 0))

            if box1_uuid not in id_box_dict.keys():
                continue  # ended here

            second_box = id_box_dict[box1_uuid]  # solve

            # find the same
            box1_T_mat = construct_transform_matrix(one_box['rotation'], one_box['translation'])
            box2_T_mat = construct_transform_matrix(second_box['rotation'], second_box['translation'])

            obj_shift = box2_T_mat @ np.linalg.inv(box1_T_mat)
            # separate points of object and background (rigid_flow is already included in annotation)
            tmp_obj_pts = pts1[pts_in_box].copy()
            tmp_obj_pts[:, 3] = 1

            transformed_obj_pts = tmp_obj_pts[:, :4] @ obj_shift.T
            shift_flow = transformed_obj_pts[:, :3] - pts1[pts_in_box, :3]

            # Dynamic
            box1_dyn = box1_T_mat[:, -1] @ ego_pose1.T
            box2_dyn = box2_T_mat[:, -1] @ ego_pose2.T

            # breakpoint()
            velocity = box2_dyn[:3] - box1_dyn[:3]
            vector_velocity = np.sqrt((velocity ** 2)).sum(0)

            if vector_velocity > move_threshold and one_box['potentially_dynamic']:
                dynamic[pts_in_box] = 1
                flow[pts_in_box, :3] = velocity         # shift flow here would be unsynchronized flow
                flow[pts_in_box, 3] = 1

            else:
                dynamic[pts_in_box] = 0
                flow[pts_in_box, :3] = 0
                flow[pts_in_box, 3] = 1 # is static and has flow ~ 0

        return pts1_in_pts2, flow, dynamic

    def get_frame(self, idx, remove_sensor_outlier=True):
        pts = self.get_feature(idx, 'lidar')
        pose = self.get_feature(idx, 'pose')
        global_pts = self.get_global_pts(idx=idx, name='lidar')

        # eliminate points in point cloud by radius small
        # transform to ego frame and eliminate points in ego frame by radius small
        if remove_sensor_outlier:
            sensor_valid_mask = np.linalg.norm(pts[:, :3] - pose[:3, -1], axis=1) > 1.8
        else:
            sensor_valid_mask = np.ones(pts.shape[0], dtype=bool)

        id_mask = self.get_feature(idx, 'id_mask')

        data = {'pts' : pts[sensor_valid_mask],
                'global_pts' : global_pts[sensor_valid_mask],
                'pose' : pose,
                'id_mask' : id_mask[sensor_valid_mask],
                }

        return data


    def info(self):
        seq_path = os.path.dirname(self.data_dir[:-1])
        nbr_of_seqs = len(os.listdir(seq_path))


        info_dict = {"nbr_of_seqs": nbr_of_seqs}

        return info_dict



if __name__ == '__main__':
    pass
