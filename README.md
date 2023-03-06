# motion-learning



### Repo folder structure

    .
    ├── configs                 # storage of yaml files with default hyperparameters
    ├── data                    # in this directory all dataset or debug subsets can be stored
    ├── dataset                 # all implemented datasets with datamodules for pl.pytorch_lightning
    ├── losses                  # implemented losses 
    ├── models                  # all implemented models
    │   └── networks            # shared modules across models 
    ├── motion_supervision      # 
    ├── pytorch3d               # external library
    ├── scripts                 # all runable scripts for train, preprocessing ... 
    ├── time_space              # 
    ├── visualization           # scripts for visualization
    ├── requirements.txt
    ├── LICENSE
    └── README.md



### Source files