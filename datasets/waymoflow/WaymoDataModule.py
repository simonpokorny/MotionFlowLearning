"""
MIT License

Copyright (c) 2021 Felix (Jabb0), Aron (arndz), Carlos (cmaranes)
Source: https://github.com/Jabb0/FastFlow3D

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from pathlib import Path
from typing import Optional, Union, List, Dict

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from datasets.waymoflow.WaymoDataset import WaymoDataset
from datasets.waymoflow.util import ApplyPillarization, drop_points_function, custom_collate_batch


class WaymoDataModule(pl.LightningDataModule):
    """
    Data module to prepare and load the waymo dataset.
    Using a data module streamlines the data loading and preprocessing process.
    """
    def __init__(self, dataset_directory,
                 # These parameters are specific to the dataset
                 grid_cell_size, x_min, x_max, y_min, y_max, z_min, z_max, n_pillars_x,
                 batch_size: int = 32,
                 has_test=False,
                 num_workers=1,
                 n_points=None,
                 apply_pillarization=True,
                 shuffle_train=True):
        super(WaymoDataModule, self).__init__()
        self._dataset_directory = Path(dataset_directory)
        self._batch_size = batch_size
        self._train_ = None
        self._val_ = None
        self._test_ = None
        self._shuffle_train = shuffle_train
        self.apply_pillarization = apply_pillarization

        self._pillarization_transform = ApplyPillarization(grid_cell_size=grid_cell_size, x_min=x_min,
                                                           y_min=y_min, z_min=z_min, z_max=z_max,
                                                           n_pillars_x=n_pillars_x)

        # This returns a function that removes points that should not be included in the pillarization.
        # It also removes the labels if given.
        self._drop_points_function = drop_points_function(x_min=x_min,
                                                          x_max=x_max, y_min=y_min, y_max=y_max,
                                                          z_min=z_min, z_max=z_max)
        self._has_test = has_test
        self._num_workers = num_workers

        self._collate_fn = custom_collate_batch
        self._n_points = n_points

    def prepare_data(self) -> None:
        """
        Preprocessing of the data only called on 1 GPU.
        Download and process the datasets here. E.g., tokenization.
        Everything that is not random and only necessary once.
        This is used to download the dataset to a local storage for example.
            Later the dataset is then loaded by every worker in the setup() method.
        :return: None
        """
        # No need to download stuff
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Setup of the datasets. Called on every GPU in distributed training.
        Do splits and build model internals here.
        :param stage: either 'fit', 'validate', 'test' or 'predict'
        :return: None
        """
        self._train_ = WaymoDataset(self._dataset_directory.joinpath("train"),
                                    point_cloud_transform=self._pillarization_transform,
                                    drop_invalid_point_function=self._drop_points_function,
                                    n_points=self._n_points, apply_pillarization=self.apply_pillarization)
        self._val_ = WaymoDataset(self._dataset_directory.joinpath("valid"),
                                  point_cloud_transform=self._pillarization_transform,
                                  drop_invalid_point_function=self._drop_points_function,
                                  apply_pillarization=self.apply_pillarization,
                                  n_points=self._n_points)
        if self._has_test:
            self._test_ = WaymoDataset(self._dataset_directory.joinpath("test"),
                                       point_cloud_transform=self._pillarization_transform,
                                       drop_invalid_point_function=self._drop_points_function,
                                       apply_pillarization=self.apply_pillarization
                                       )

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        """
        Return a data loader for training
        :return: the dataloader to use
        """
        return DataLoader(self._train_, self._batch_size, num_workers=self._num_workers,
                          shuffle=self._shuffle_train,
                          collate_fn=self._collate_fn)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        """
        Return a data loader for validation
        :return: the dataloader to use
        """
        return DataLoader(self._val_, self._batch_size, shuffle=False, num_workers=self._num_workers,
                          collate_fn=self._collate_fn)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        """
        Return a data loader for testing
        :return: the dataloader to use
        """
        if not self._has_test:
            raise RuntimeError("No test dataset specified. Maybe set has_test=True in DataModule init.")
        return DataLoader(self._test_, self._batch_size, shuffle=False, num_workers=self._num_workers,
                          collate_fn=self._collate_fn)