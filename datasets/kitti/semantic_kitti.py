import os.path
import numpy as np
import yaml
import glob

from scipy.spatial.transform import Rotation

from datasets.structures.sequence import Basic_Dataprocessor
from datasets.paths import SEMANTICKITTI_PATH



label_name_mapping = {
    0: 'unlabeled',
    1: 'outlier', #  previously "outlier"
    10: 'car',
    11: 'bicycle',
    13: 'bus',
    15: 'motorcycle',
    16: 'on-rails',
    18: 'truck',
    20: 'other-vehicle',
    30: 'person',
    31: 'bicyclist',
    32: 'motorcyclist',
    40: 'road',
    44: 'parking',
    48: 'sidewalk',
    49: 'other-ground',
    50: 'building',
    51: 'fence',
    52: 'other-structure',
    60: 'lane-marking',
    70: 'vegetation',
    71: 'trunk',
    72: 'terrain',
    80: 'pole',
    81: 'traffic-sign',
    99: 'other-object',
    252: 'moving-car',
    253: 'moving-bicyclist',
    254: 'moving-person',
    255: 'moving-motorcyclist',
    256: 'moving-on-rails',
    257: 'moving-bus',
    258: 'moving-truck',
    259: 'moving-other-vehicle'
}



kept_labels = ['unlabeled', 'road', 'sidewalk', 'parking', 'other-ground', 'building', 'car', 'truck',
    'bicycle', 'motorcycle', 'other-vehicle', 'vegetation', 'trunk', 'terrain',
    'person', 'bicyclist', 'motorcyclist', 'fence', 'pole', 'traffic-sign'
]

def get_label_mapping():
    reverse_label_name_mapping = {}
    label_map = np.zeros(260, dtype=int)
    cnt = 0

    for label_id in label_name_mapping:
        if label_id > 250:
            if label_name_mapping[label_id].replace('moving-',
                                                    '') in kept_labels:
                label_map[label_id] = reverse_label_name_mapping[
                    label_name_mapping[label_id].replace('moving-', '')]
            else:
                label_map[label_id] = 255
        elif label_id == 0:
            label_map[label_id] = 255
        else:
            if label_name_mapping[label_id] in kept_labels:
                label_map[label_id] = cnt
                reverse_label_name_mapping[
                    label_name_mapping[label_id]] = cnt
                cnt += 1
            else:
                label_map[label_id] = 255

    reverse_label_name_mapping = reverse_label_name_mapping
    num_classes = cnt

    return reverse_label_name_mapping, num_classes

def reannotate_moving_by_config(label):
    learning_map = label_name_mapping
    for k, v in learning_map.items():
        if 'moving' in v:
            label[label == k] = 1
        elif 'unlabeled' in v or 'outlier' in v:
            label[label == k] = -1
        else:
            label[label == k] = 0

    return label

def reannotate_by_config(label):
    # as long as the learning map is from lowest to highest index, it is fine
    learning_map = label_name_mapping
    for k, v in learning_map.items():

        if 'moving' in v:
            v = v.split('-')[1]
            # print(k, v)

        if v in kept_labels:
            label[label == k] = kept_labels.index(v)
        else:
            label[label == k] = -1

    return label


def get_ego_bbox():
    ### KITTI EGO Parameters
    # https://towardsdatascience.com/kitti-coordinate-transformations-125094cd42fb
    l = 3.5
    w = 1.8
    h = 1.73
    x, y, z = 0, 0, -h / 2  # cause vehicle system is in Lidar frame

    ego_box = {}

    ego_box['translation'] = np.array((x, y, z))
    ego_box['size'] = np.array((l, w, h))
    ego_box['rotation'] = Rotation.from_rotvec([0, 0, 0]).as_matrix()

    return ego_box


def read_bin_pts(filename):
    return np.fromfile(filename, dtype=np.float32).reshape(-1, 4)

def read_annotation_file(filename):
    label = np.fromfile(filename, dtype=np.uint32)
    label = label.reshape((-1))
    label = label.astype(np.int32)  # for -1 in remapping

    sem_label = label & 0xFFFF  # semantic label in lower half
    inst_label = label >> 16  # instance id in upper half

    return sem_label, inst_label

def parse_calibration(filename):
    """ read calibration file with given filename
        Returns
        -------
        dict
            Calibration matrices as 4x4 numpy arrays.
    """
    calib = {}

    calib_file = open(filename)
    for line in calib_file:
        key, content = line.strip().split(":")
        values = [float(v) for v in content.strip().split()]

        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        calib[key] = pose

    calib_file.close()

    return calib


def parse_poses(filename, calibration):
    """ read poses file with per-scan poses from given filename
        Returns
        -------
        list
            list of poses as 4x4 numpy arrays.
    """
    # with open(filename, 'r') as f:
    #   file = f.readlines()

    file = open(filename, mode='r')

    poses = []

    Tr = calibration["Tr"]
    Tr_inv = np.linalg.inv(Tr)

    for line in file:
        values = [float(v) for v in line.strip().split()]

        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))

    file.close()
    return np.stack(poses)


def load_yaml(path):
    import yaml
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.Loader)


class SemanticKitti_Sequence(Basic_Dataprocessor):

    def __init__(self, sequence_nbr):
        # TODO Semantic Segmentation is shifted, the dynamic labels are correct
        self.name = 'semantic_kitti'
        self.sequence_nbr = sequence_nbr

        self.min_sensor_radius = 2.5
        self.sensor_position = np.array([0, 0, 0]) # in kitti velodyne is origin of coordinate system
        self.ego_box = get_ego_bbox()
        self.ground_ego_height = self.ego_box['translation'][2] - self.ego_box['size'][2] / 2

        self.log_ids = sorted(glob.glob(SEMANTICKITTI_PATH + '/*'))
        self.log_nbr = {i: seq for i, seq in zip(range(len(self.log_ids)), self.log_ids)}
        self.label_mapping, self.num_classes = get_label_mapping()


        self.sequence_path = self.log_nbr[self.sequence_nbr] + '/'
        super().__init__(self.sequence_path)

        self._init_preprocessing()

    def _init_preprocessing(self):
        processed_lidar_paths = sorted(glob.glob(f'/{self.sequence_path}/lidar/*.npy'))
        raw_lidar_paths = sorted(glob.glob(f'/{self.sequence_path}/velodyne/*.bin'))

        if len(raw_lidar_paths) != len(processed_lidar_paths):
            init_preprocessing = True

        else:
            init_preprocessing = False

        if init_preprocessing:

            for folder in ['lidar', 'poses', 'id_mask', 'lidarseg', 'dynamic_label']:
                os.makedirs(f'/{self.sequence_path}/{folder}', exist_ok=True)

            poses = parse_poses(self.sequence_path + '/poses.txt', parse_calibration(self.sequence_path + '/calib.txt'))

            for frame in range(len(raw_lidar_paths)):
                print('Preprocessing data...', frame, '/', len(raw_lidar_paths))
                # poses
                self.store_feature(poses[frame], frame, 'pose')

                # lidars
                pts = read_bin_pts(self.sequence_path + '/velodyne/' + str(frame).zfill(6) + '.bin')
                self.store_feature(pts, frame, 'lidar')


                if os.path.exists(self.sequence_path + '/labels/' + str(frame).zfill(6) + '.label'):
                    sem_label, inst_mask = read_annotation_file(self.sequence_path + '/labels/' + str(frame).zfill(6) + '.label')
                else:
                    sem_label, inst_mask = np.zeros((pts.shape[0])), np.zeros((pts.shape[0]))

                # if changing label format, do it in reannotate_by_config function or label lists/dicts
                mapped_label = reannotate_by_config(sem_label.copy())

                self.store_feature(mapped_label, frame, 'lidarseg')
                self.store_feature(inst_mask, frame, 'id_mask')

                dynamic_label = reannotate_moving_by_config(sem_label.copy())

                # Remap the labels from KittiRoad
                dynamic_label[dynamic_label == 9] = 0
                dynamic_label[dynamic_label == 251] = 1

                self.store_feature(dynamic_label, frame, 'dynamic_label')

            # id_mask



if __name__ == '__main__':
    seq = 4
    dataset = SemanticKitti_Sequence(seq)

