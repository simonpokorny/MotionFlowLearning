import os.path
import glob
import socket
import numpy as np

from datasets.structures.sequence import Basic_Dataprocessor
from timespace.timestamps import find_nearest_timestamps

from datasets.paths import DELFT_PATH



class Delft_Sequence(Basic_Dataprocessor):

    def __init__(self, sequence=0):



        self.sequence = sequence


        self.sequence_dict = {seq_nbr : seq_name for seq_nbr, seq_name in enumerate(sorted(glob.glob(DELFT_PATH + '/*/')))}
        self.sequence_path = self.sequence_dict[sequence] + '/' # this can be unified, repeating bugs
        # self.raw_features = ['raw_camera_pts', 'raw_depth', 'raw_rgb', 'raw_velodyne', 'raw_poses']
        self.useable_features = os.listdir(self.sequence_path)


        Basic_Dataprocessor.__init__(self, data_dir=self.sequence_path)

        # print(self.sequence_dict)
        self.get_available_data()
        # self.synchronize_by_feature(feature_name='pose')    # synchronization to lowest frame rate
        # self.index_timestamp()
        # self.store_by_data_frames()
        self._init_preprocessing()

    # this is messy, but it works
    def index_timestamp(self):
        from shutil import copy2
        for folder in self.useable_features:
            files = sorted(glob.glob(self.sequence_path + folder + '/*'))

            if os.path.exists(self.sequence_path + 'sync_' + folder):
                continue

            os.makedirs(self.sequence_path + 'sync_' + folder, exist_ok=True)

            # if files[0][:-4].endswith('0'.zfill(6)):
            #     break

            for idx in range(len(self.data_frames)):
                copy2(self.data_frames[idx][folder], self.sequence_path + 'sync_' + folder + '/' + str(idx).zfill(6) + files[idx][-4:])

                # os.rename(files[idx], self.sequence_path + 'sync_' + folder + '/' + str(idx).zfill(6) + files[idx][-4:])


    def _init_preprocessing(self):
        pass

    def get_available_data(self):
        '''
        Scans through main folder. It assumes that all data are one level bellow the root
        :return:
        '''
        self.data_paths = {}

        for key in self.useable_features:

            self.data_paths[key] = sorted(glob.glob(f'{self.sequence_dict[self.sequence]}/' + key + '/*'))


    def synchronize_by_feature(self, feature_name):
        '''
        #todo in future to the basic dataprocessor
        :param feature_name: Choose one raw feature and find the closest nearby frames from other features
        :return: list of the frames with all file_paths without RAW PREFFIX!
        '''
        self.data_frames = []

        frames = self.data_paths[feature_name]

        for idx, frame in enumerate(frames):
            data_dict = {}
            timestamp = int(os.path.basename(frame).split('.')[0])

            for corres_feature in self.useable_features:
                all_timestamps = [int(os.path.basename(feat_f).split('.')[0]) for feat_f in self.data_paths[corres_feature]]

                index = find_nearest_timestamps(all_timestamps, timestamp)
                data_dict[corres_feature] = self.data_paths[corres_feature][index]

            self.data_frames.append(data_dict)

        return self.data_frames

    def store_by_data_frames(self):
        '''

        :return: Store the data into same folder
        '''
        for idx, frame_dict in enumerate(self.data_frames):

            for key, value in frame_dict.items():
                feature = self.unified_preload(value)

                self.store_feature(feature, idx, name="sync_" + key)

            # print(idx)

    def __getitem__(self, idx):
        data = {}
        for key, value in self.data_frames[idx].items():
            data[key] = self.unified_preload(value)

        return data

    def __len__(self):
        return len(glob.glob(self.sequence_path + 'sync_pose' + '/*'))



if __name__ == '__main__':
    sequence = Delft_Sequence()
    data = sequence.__getitem__(0)
    # you are missing scripts for running, regeneration etc. the final products

    from datasets.visualizer import *
    visualize_points3D(data['camera_pts'], data['camera_pts'][:,3:])
