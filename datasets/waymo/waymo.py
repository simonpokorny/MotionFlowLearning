import numpy as np
import glob
import os
from tqdm import tqdm

from scipy.spatial.transform.rotation import Rotation
from datasets.structures.sequence import Basic_Dataprocessor
from datasets.paths import WAYMO_PATH
from timespace import box_utils

import motion_supervision.constants as C

LIDAR_LOCAL_POSE = np.array([[-8.54212716e-01, -5.19923095e-01, -7.81823797e-04, 1.43000000e+00],
                             [ 5.19918423e-01, -8.54209872e-01,  3.21373964e-03,  0.00000000e+00],
                             [-2.33873907e-03,  2.33873267e-03,  9.99994530e-01,  2.18400000e+00],  # this height is also upper plane of ego box
                             [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

# Can be extracted from one-time-waymo.py, while loading the range images and projecting them to the points


def get_waymo_ego_boxx():
    # 0-z is on ground
    ### KITTI EGO Parameters
    l = 3.5
    w = 1.8
    h = LIDAR_LOCAL_POSE[2, 3]  # 2.184
    x, y, z = 0, 0, h / 2
    angle = 0
    ego_box = {}

    ego_box['translation'] = np.array((x, y, z))
    ego_box['size'] = np.array((l, w, h))
    ego_box['rotation'] = Rotation.from_rotvec([0, 0, angle]).as_matrix()

    return ego_box


class Waymo_Sequence(Basic_Dataprocessor):

    def __init__(self, sequence_nbr, init_preprocessing=False):

        self.name = 'waymo'
        self.sequence_nbr = sequence_nbr
        self.ego_box = get_waymo_ego_boxx()
        self.framerate = 10 # 10Hz

        self.min_sensor_radius = 2.5
        self.sensor_position = np.array([0, 0, self.ego_box['size'][2]])  # in kitti velodyne is origin of coordinate system?
        self.ground_ego_height = 0 # later to cfg

        self.log_ids = sorted(glob.glob(WAYMO_PATH + '/*'))
        self.log_nbr = {i: seq for i, seq in zip(range(len(self.log_ids)), self.log_ids)}


        self.init_preprocessing = init_preprocessing
        self.sequence_path = self.log_nbr[self.sequence_nbr] + '/'
        super().__init__(self.sequence_path)

        self._init_preprocessing()

    def _init_preprocessing(self):
        processed_lidar_paths = sorted(glob.glob(f'/{self.sequence_path}/lidar/*.npy'))
        dynamic_label_paths = sorted(glob.glob(f'/{self.sequence_path}/dynamic_label/*.npy'))

        if len(dynamic_label_paths) != len(processed_lidar_paths) or self.init_preprocessing:
            init_preprocessing = True

        else:
            init_preprocessing = False

        if init_preprocessing:

            box_ids = []
            for frame in range(len(processed_lidar_paths)):
                boxes = np.load(f'/{self.sequence_path}/boxes/{frame:06d}.npy', allow_pickle=True)
                for box in boxes:
                    box_ids.append(box['uuid'])

            track_mapping = {i: j + 1 for i, j in zip(np.unique(box_ids), range(0, len(np.unique(box_ids))))}  # start from 1 for ids

            for folder in ['lidar', 'poses', 'id_mask', 'lidarseg', 'dynamic_label', 'flow_label']:
                os.makedirs(f'/{self.sequence_path}/{folder}', exist_ok=True)

            print('Preprocessing data...')
            for frame in tqdm(range(len(processed_lidar_paths))):


                pts = np.load(f'/{self.sequence_path}/lidar/{frame:06d}.npy')


                boxes = np.load(f'/{self.sequence_path}/boxes/{frame:06d}.npy', allow_pickle=True)
                lidarseg = np.zeros(pts.shape[0], dtype=np.int32)
                id_mask = np.zeros(pts.shape[0], dtype=np.int32)
                dynamic_label = np.zeros(pts.shape[0], dtype=np.int32)
                flow_label = np.zeros((pts.shape[0], 4), dtype=np.float32)  # x,y,z, flow asigned

                non_label_mask = pts[:, 5] == 1 # not labeled

                lidarseg[non_label_mask] = -1
                id_mask[non_label_mask] = -1
                dynamic_label[non_label_mask] = -1
                flow_label[non_label_mask] = -1

                for instance in boxes:
                    # transform to rotation matrix
                    instance['rotation'] = Rotation.from_rotvec(instance['rotation']).as_matrix()

                    label_mask = box_utils.get_point_mask(pts, instance, z_add=(-0,0))

                    lidarseg[label_mask] = -1
                    id_mask[label_mask] = -1
                    dynamic_label[label_mask] = -1
                    flow_label[label_mask] = -1

                    object_mask = box_utils.get_point_mask(pts, instance, z_add=(-C.data['undefined_box_z'],0))   # to prevent oversegmentation

                    motion_state = (np.sqrt((instance['velocity'][:2] ** 2).sum()) / self.framerate) > C.data['moving_threshold'] # motion thershold
                    # motion state is now w.r.t frame, not m/s from annotation
                    flow_label[object_mask, :3] = instance['velocity'] / self.framerate    # per frame
                    flow_label[object_mask, 3] = 1

                    lidarseg[object_mask] = instance['class']
                    id_mask[object_mask] = track_mapping[instance['uuid']]
                    dynamic_label[object_mask] = motion_state


                np.save(f"/{self.sequence_path}/lidarseg/{frame:06d}.npy", lidarseg)
                np.save(f"/{self.sequence_path}/id_mask/{frame:06d}.npy", id_mask)
                np.save(f"/{self.sequence_path}/dynamic_label/{frame:06d}.npy", dynamic_label)
                np.save(f"/{self.sequence_path}/flow_label/{frame:06d}.npy", flow_label)

if __name__ == '__main__':
    import sys
    # todo match the segmentation labels to get rid of road below boxes?
    # todo add additional dynamic label in form of "object has moved at least once"

    seq = int(sys.argv[1])
    sequence = Waymo_Sequence(seq, init_preprocessing=True)
