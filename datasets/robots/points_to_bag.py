import os

import numpy as np
import open3d as o3d
import glob
from tqdm import tqdm


import rosbag
from pyrosenv.sensor_msgs import point_cloud2, PointField
from pyrosenv import sensor_msgs, std_msgs

from sensor_msgs.msg import Imu

files = sorted(glob.glob('/home/patrik/patrik_data/drone/hydra_data/pts/*.npy'))

pcd = o3d.geometry.PointCloud()

os.makedirs(os.path.dirname(files[0]).replace('sync_pts', 'sync_pcd'), exist_ok=True)
# bag = rosbag.Bag('/home/patrik/patrik_data/drone/bags/2023-01-30.bag', "r")
bag = rosbag.Bag('/home/patrik/patrik_data/drone/bags/try.bag', "w")


def point_cloud(points, parent_frame):
    """ Creates a point cloud message.
    Args:
        points: Nx7 array of xyz positions (m) and rgba colors (0..1)
        parent_frame: frame in which the point cloud is defined
    Returns:
        sensor_msgs/PointCloud2 message
    """
    ros_dtype = PointField.FLOAT32
    dtype = np.float32
    itemsize = np.dtype(dtype).itemsize

    data = points.astype(dtype).tobytes()

    fields = [sensor_msgs.PointField(
        name=n, offset=i*itemsize, datatype=ros_dtype, count=1)
        for i, n in enumerate('xyzrgba')]

    header = std_msgs.Header(frame_id=parent_frame, stamp=rospy.Time.now())

    return sensor_msgs.PointCloud2(
        header=header,
        height=1,
        width=points.shape[0],
        is_dense=False,
        is_bigendian=False,
        fields=fields,
        point_step=(itemsize * 7),
        row_step=(itemsize * 7 * points.shape[0]),
        data=data
    )


for i in tqdm(range(len(files))):
    pts_path = files[0]
    timestamp = int(os.path.basename(pts_path).split('.')[0])
    pts = np.load(pts_path)[:,:3]
    msg = point_cloud2.PointCloud2()

    msg.data = pts.tostring()
    msg.header.seq = i
    msg.header.stamp.secs = int(str(timestamp)[:-9])
    msg.header.stamp.nsecs = int(str(timestamp)[-9:])
    msg.header.frame_id = '/world'

    bag.write('/camera/depth/points', msg)

bag.close()

for topic, msg, t in bag.read_messages():

    if 'depth/image_rect_raw' in topic:
        print(msg)
        pts_path = '/home/patrik/patrik_data/drone/hydra_data/pts' + f'/{t}.npy'
        os.listdir(os.path.dirname(pts_path))
        pts = np.load(pts_path)
        break

# for file in tqdm(files):
#     data = np.load(file)
#
#     pcd.points = o3d.utility.Vector3dVector(data[:, :3])
#     pcd.colors = o3d.utility.Vector3dVector(data[:, 3:6])
#     o3d.io.write_point_cloud(file.replace('npy', 'pcd').replace('sync_pts', 'sync_pcd'), pcd)

