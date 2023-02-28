import socket
import os

if socket.gethostname() == 'Patrik' and False:
    ARGOVERSE2_PATH = f'{os.path.expanduser("~")}/patrik_data/argoverse2/'
    livox = f'{os.path.expanduser("~")}/patrik_data/livox/'
    simu_livox = f'{os.path.expanduser("~")}/data/simu_livox/'

# RCI
else:
    ARGOVERSE2_PATH = f'{os.path.expanduser("~")}/data/argoverse2/sensor/train/'   # split other paths
    livox = f'{os.path.expanduser("~")}/data/livox/'
    simu_livox = f'{os.path.expanduser("~")}/data/simu_livox/'
    DELFT_PATH = f"{os.path.expanduser('~')}/data/drone/"

DELFT_PATH = f"{os.path.expanduser('~')}/patrik_data/drone/sequences/"
WAYMO_PATH = f"{os.path.expanduser('~')}/data/waymo/training/"
# DELFT_PATH = f"{os.path.expanduser('~')}/data/drone/"

SEMANTICKITTI_PATH = f"{os.path.expanduser('~')}/data/semantic_kitti/dataset/sequences/"

