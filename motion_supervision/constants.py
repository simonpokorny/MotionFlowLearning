# Indecided / unlabelled : 0
# Static : 1
# Dynamic : 2
# Flow vector : x,y,z, valid_flow


data = {'moving_threshold' : 0.1,
        'undefined_box_z' : 0.2, # to prevent wrong semantic label from bounding boxes when not precise. Value of -1 is assigned in band between road and object
        }

rays = {'x_max' : 50.0,  # orig waymo fast flow 85m
       'x_min' : -50.0,
       'y_max' : 50.0,
       'y_min' : -50.0,
       'z_max' : 3.0,
       'z_min' : -3.0,
       'grid_size' : 512,
       'background_weight' : 0.1,
       'freespace_min_height' : 0.5,
       'freespace_max_height' : 2.0,
       'size_of_block' : 5,
       'cell_size' : (0.2, 0.2, 0.2),
               }

ego_prior_params = {"cell_size" : (0.2, 0.2),
            'min_road_pts' : 10,    # throw away road with less than this few points
            'z_var_road' : 0.1,        # This is ego driven box specific ground
            'z_var_proposal' : 0.8, # consider dynamic only place, where it is this high, more or less middle of the ego vehicle
            'z_var_outlier' : 4,
            }    # if there is something above, or noisy point below ground, fillter it out

# todo add here paths as well?

cfg = {'x_max' : 70.0,  # orig waymo fast flow 85m
       'x_min' : -70.0,
       'y_max' : 70.0,
       'y_min' : -70.0,
       'z_max' : 3.0,
       'z_min' : -3.0,
       'grid_size' : 512,
       'background_weight' : 0.1,
       'correction_time' : 4,   # 4 / 10 HZ = 4 adjacent frames
       'cell_size' : (0.2, 0.2, 0.2),

        'z_var_road' : 0.1, # this is DUPLICATED, TOdo, unify with ego prior cfg
       'static_cell_size' : (0.2, 0.2, 0.2),
       'past_static_frames' : 5,
       'required_static_time' : 30,
       'future_static_frames' : 5,  # for cases
               }
# split it by prior to avoid losing some detections
