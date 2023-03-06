# motion-learning

## Changes

#### 6.3.
- Changed the structure of the project (seems more useful and organized)
- all hyperparameters of slim are stored in config with description 
- pytorch3d folder was removed from repo, but added to requirements 
- merged BaseModelSlim and Slim to one script slim.py
- slim decoder is rewritten to more readable format 
- added optimization and lr changes to slim based on official repo 
- losses are also added


## Todo list

#### without assignment 
- [ ] Nothing yet
#### Patrik
- [ ] Lossy
#### Simon
- [x] Udělat todo list na readme :)
- [ ] Kitti raw daataset
- [ ] BaseDataset & Base Dataloader
- [ ] Train on raw kitti


### Repo folder structure

    .
    ├── configs                 # storage of yaml files with default hyperparameters
    ├── data                    # in this directory all dataset or debug subsets can be stored
    ├── dataset                 # all implemented datasets with datamodules for pl.pytorch_lightning
    ├── losses                  # implemented losses 
    ├── models                  # all implemented models
    │   └── networks            # shared modules across models 
    ├── motion_supervision      #
    ├── scripts                 # all runable scripts for train, preprocessing ... 
    ├── time_space              # 
    ├── visualization           # scripts for visualization
    ├── requirements.txt
    ├── LICENSE
    └── README.md




