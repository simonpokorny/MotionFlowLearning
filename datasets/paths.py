import socket
import os

# from datasets.kitti.semantic_kitti import SemanticKitti_Sequence
# from datasets.waymo.waymo import Waymo_Sequence
# from datasets.argoverse.argoverse2 import Argoverse2_Sequence

HOME_DIR = f'{os.path.expanduser("~")}/'
TMP_VIS = f'{os.path.expanduser("~")}/data/tmp_vis/'

# if socket.gethostname() == 'Patrik' and False:
#     ARGOVERSE2_PATH = f'{os.path.expanduser("~")}/patrik_data/argoverse2/'
#     livox = f'{os.path.expanduser("~")}/patrik_data/livox/'
#     simu_livox = f'{os.path.expanduser("~")}/data/simu_livox/'

# RCI
# else:
ARGOVERSE2_PATH = f'{os.path.expanduser("~")}/data/argoverse2/sensor/train/'   # split other paths
livox = f'{os.path.expanduser("~")}/data/livox/'
simu_livox = f'{os.path.expanduser("~")}/data/simu_livox/'
DELFT_PATH = f"{os.path.expanduser('~')}/patrik_data/drone/sequences/"
WAYMO_PATH = f"{os.path.expanduser('~')}/data/waymo/training/"
SEMANTICKITTI_PATH = f"{os.path.expanduser('~')}/data/semantic_kitti/dataset/sequences/"

# DELFT_PATH = f"{os.path.expanduser('~')}/data/drone/"

SK_TOY = {#'dataset_class' : SemanticKitti_Sequence,
          'data_dir' : HOME_DIR + 'data/semantic_kitti/dataset/sequences',
          'used_seqs' : [4],
          'label_source' : 'final_prior_label'}

WAYMO_TOY = {#'dataset_class' : Waymo_Sequence,
          'data_dir' : HOME_DIR + "/data/waymo/training/",
          'used_seqs' : list(range(2)),
          'label_source' : 'dynamic_label'} # tmp changed

ARGO2_TOY = {#'dataset_class' : Argoverse2_Sequence,
          'data_dir' : HOME_DIR + '/data/argoverse2/sensor/train/',
          'used_seqs' : [0],
          'label_source' : 'final_prior_label'}

SK_TRN = {#'dataset_class' : SemanticKitti_Sequence,
          'data_dir' : HOME_DIR + 'data/semantic_kitti/dataset/sequences',
          'used_seqs' : list(range(34)),
          'label_source' : 'final_prior_label'}

ARGO2_TRN = {#'dataset_class' : Argoverse2_Sequence,
             'data_dir' : HOME_DIR + '/data/argoverse2/sensor/train/',
             'used_seqs' : list(range(700)),
             'label_source' : 'final_prior_label'}

WAYMO_TRN = {#'dataset_class' : Waymo_Sequence,
             'data_dir' : HOME_DIR + "/data/waymo/training/",
             'used_seqs': list(range(50)),
             'label_source': 'final_prior_label'
             }

WAYMO_TST = {#'dataset_class' : Waymo_Sequence,
             'data_dir' : HOME_DIR + "/data/waymo/training/",
             'used_seqs': list(range(50,55)),
             'label_source': 'dynamic_label'
             }

# SK_TRN_AUTOGEN = {'dataset_class' : SemanticKitti_Sequence,
#                   'data_dir' : HOME_DIR + 'data/semantic_kitti/dataset/sequences',
#                   'used_seqs' : list(range(11)),
#                   'label_source' : 'dynamic_label'}



