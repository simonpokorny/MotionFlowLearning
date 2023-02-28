import os

# import open3d as o3d
from datasets.visualizer import *
import datasets.paths as machine_paths

from datasets.structures.sequence import Basic_Dataprocessor

LIVOX_DIR = machine_paths.livox

def get_ego_livox():

    # From Argoverse2 by now
    l = 191.8 * 0.0254  # in inches
    w = 83.5 * 0.0254
    h = 58.0 * 0.0254
    x, y, z = 0, 0, 0.6  # calibration observing points
    angle = 0
    EGO_BBOX = np.array((x, y, z, l, w, h, angle))

    return EGO_BBOX

class Livox_Sequence(Basic_Dataprocessor):

    def __init__(self, data_dir=machine_paths.livox, dataset_type='.', sequence_nbr=0):

        self.ego_box = get_ego_livox()
        self.framerate = 0.1

        # remap sequences
        log_ids = sorted(glob.glob(data_dir + '/' + dataset_type + '/*'))
        log_nbr = {i: seq for i, seq in zip(range(len(log_ids)), log_ids)}
        self.sequence_nbr = sequence_nbr
        self.sequence = os.path.basename(log_nbr[sequence_nbr])
        self.sequence_path = os.path.join(data_dir, dataset_type, self.sequence)

        super().__init__(self.sequence_path + '/')


        self.__init_preprocessing()
    #
    def __init_preprocessing(self):

        self.raw_npz_path = sorted(glob.glob(self.sequence_path + '/unpack/*.npz'))

        os.makedirs(self.sequence_path + '/lidar', exist_ok=True)
        for i in range(len(self.raw_npz_path)):
            lidar = np.load(self.raw_npz_path[i], allow_pickle=True)['point_cloud']
            np.save(self.sequence_path + f'/lidar/{i:06d}.npy', lidar)

        self.pose_file = glob.glob(self.sequence_path + '/' +self.sequence + '.xml')[0]

        import xml.etree.ElementTree as ET
        tree = ET.parse(self.pose_file)
        root = tree.getroot()
        odometry_tag = root.findall('odometry')[0]
        poses_txt = [i.text[1:-1] for i in odometry_tag]
        poses = [np.array(i.split(" "), dtype=float).reshape(4,4) for i in poses_txt]

        os.makedirs(self.sequence_path + '/pose', exist_ok=True)
        _ = [np.save(self.sequence_path + f'/pose/{idx:06d}.npy', poses[idx]) for idx in range(len(poses))]



if __name__ == "__main__":
    # first convert "read_lvx_format.py", then run this
    sequence = Livox_Sequence()
    # Todo I dont know which data can be sused ... I know now ...
    pcls = [sequence.get_global_pts(idx, 'lidar') for idx in range(40,70)]
    pts = sequence.get_feature(10, 'lidar')

    without_mid_pcl = [i[i[:,4] != 1] for i in pcls]

    visualize_points3D(pts, pts[:,4])
    visualize_multiple_pcls(*without_mid_pcl)
