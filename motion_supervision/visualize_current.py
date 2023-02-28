from datasets import visualizer

import numpy as np
# sequence = Argoverse2_Sequence(sequence_nbr=7)

import sys

dataset = int(sys.argv[1])
seq = int(sys.argv[2])
frame = int(sys.argv[3])

if dataset == 0:
    from datasets.kitti.semantic_kitti import SemanticKitti_Sequence
    sequence = SemanticKitti_Sequence(sequence_nbr=seq)

elif dataset == 1:
    from datasets.waymo.waymo import Waymo_Sequence
    sequence = Waymo_Sequence(sequence_nbr=seq)

elif dataset == 2:
    from datasets.argoverse.argoverse2 import Argoverse2_Sequence
    sequence = Argoverse2_Sequence(sequence_nbr=seq)

else:
    raise ValueError("Dataset not supported")

# local_pts = sequence.get_feature(0, name='lidar')
# v = visualizer.visualize_points3D(local_pts, local_pts[:,3])


pts1 = sequence.get_global_pts(idx=frame, name='lidar')

time_pts = [sequence.get_global_pts(idx=idx, name='lidar') for idx in range(frame, frame+10)]
visualizer.visualize_multiple_pcls(*time_pts)

freespace = []
for i in range(frame, frame + 20, 5):
    one_freespace = sequence.get_feature(idx=i, name='accum_freespace')
    pose = sequence.get_feature(idx=i, name='pose')
    one_freespace = sequence.pts_to_frame(one_freespace, pose)

    freespace.append(one_freespace)


freespace = np.concatenate(freespace, axis=0)
# freespace = freespace[freespace[:, 2] > 0.5]

visualizer.visualize_multiple_pcls(*[freespace, time_pts[0], time_pts[1]])
# pts2 = np.concatenate([np.insert(sequence.get_global_pts(idx=i, name='lidar'), 4, i, 1) for i in range(0, 10)])
# visualizer.visualize_points3D(pts2, pts2[:,4])

# sys.exit('0')
static_value = sequence.get_feature(idx=frame, name='prior_static_mask')
dynamic_label = sequence.get_feature(idx=frame, name='dynamic_label')
ego_prior = sequence.get_feature(idx=frame, name='ego_prior_label')
visibility_prior = sequence.get_feature(idx=frame, name='visibility_prior')
corrected_visibility_prior = sequence.get_feature(idx=frame, name='corrected_visibility_prior')
road_proposal = sequence.get_feature(idx=frame, name='road_proposal')
# flow_label = sequence.get_feature(idx=frame, name='flow_label')
# lidarseg = sequence.get_feature(idx=frame, name='lidarseg')
# accum_freesace = sequence.get_feature(idx=frame, name='accum_freespace')

# visualizer.visualize_points3D(pts1, static_mask)
# valid_mask = ego_prior == 1
# diff_mask = dynamic_label == ego_prior
# visualizer.visualize_points3D(pts1[valid_mask], diff_mask[valid_mask])




visualizer.visualize_points3D(pts1, visibility_prior)
visualizer.visualize_points3D(pts1, corrected_visibility_prior)
# visualizer.visualize_points3D(pts1, ego_prior)
# visualizer.visualize_points3D(pts1, static_value)

final_prior = sequence.get_feature(frame, 'final_prior_label')

# to function, road maybe as well? - finish 3pm? then 5 pm Training of mos, 8 pm codes upload github and Fastflow mos

# Final dynamic is the complete final prior
# final_dynamic = sequence.get_feature(frame, 'final_prior_label')




# prop_prior[prop_prior == 1] = keep_dynamic_mask[prop_prior == 1]

# variance
# velocity = sequence.get_feature(frame, name='flow_label')

visualizer.visualize_points3D(pts1, dynamic_label)
visualizer.visualize_points3D(pts1, final_prior)
# visualizer.visualize_points3D(pts1, final_dynamic)
visualizer.visualize_points3D(pts1, static_value)
# visualizer.visualize_points3D(pts1, velocity[:,:3])  # labels?
# global_pts = np.concatenate([np.insert(sequence.get_global_pts(i, 'lidar'), 4, i, axis=1) for i in range(frame, frame + 4)])
# global_ego = np.concatenate([sequence.get_feature(i, 'corrected_ego_prior_label') for i in range(frame, frame + 4)])

# visualizer.visualize_points3D(global_pts, global_ego)
# visualizer.visualize_points3D(global_pts, global_pts[:,4])


# error_static = (static_value == 30) & (dynamic_label == 1)
# visualizer.visualize_points3D(pts1, error_static)

# prior_flow = sequence.get_feature(idx=frame, name='prior_visibility_flow')
# visualizer.visualize_flow3d(time_pts[0], time_pts[1], prior_flow)

# var_label = ego_prior.copy()

# prop_pts = pts1[var_label == 1]
# visualizer.visualize_points3D(prop_pts)
