# import sensor_msgs.point_cloud2 as pc2
import ctypes
import glob
import shutil

import numpy as np
import rosbag
import os
import yaml
import open3d as o3d
from pyrosenv.sensor_msgs import point_cloud2
import subprocess

import sys
import rospy
from PIL import Image
from scipy.spatial.transform.rotation import Rotation
# from cv_bridge import CvBridge

from datasets.structures.rgbd import project_depth_to_pcl

def run_in_shell(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    process.wait()
    print(process.returncode)


def load_pcd_to_npy(file):

    pcd = o3d.io.read_point_cloud(file)
    out_arr = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    pcl_rgb = np.concatenate((out_arr, colors), axis=1)

    return pcl_rgb



def imgmsg_to_cv2(img_msg):
    if img_msg.encoding != "bgr8":
        rospy.logerr("This Coral detect node has been hardcoded to the 'bgr8' encoding.  Come change the code if you're actually trying to implement a new camera")
    dtype = np.dtype("uint8") # Hardcode to 8 bits...
    dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
    image_opencv = np.ndarray(shape=(img_msg.height, img_msg.width, 3), # and three channels of data. Since OpenCV works with bgr natively, we don't need to reorder the channels.
                    dtype=dtype, buffer=img_msg.data)
    # If the byt order is different between the message and the system.
    if img_msg.is_bigendian == (sys.byteorder == 'little'):
        image_opencv = image_opencv.byteswap().newbyteorder()
    return image_opencv




def unpack_bag(bag_file, data_dir, **kwargs):

    bag = rosbag.Bag(bag_file, "r")

    cx = kwargs['cx']
    cy = kwargs['cy']
    fx = kwargs['fx']
    fy = kwargs['fy']

    info_dict = yaml.load(rosbag.Bag(bag_file, 'r')._get_yaml_info(), Loader=yaml.Loader)
    # Print the information contained in the info_dict
    info_dict.keys()
    for topic in info_dict["topics"]:
        print("-" * 50)
        for k, v in topic.items():
            print(k.ljust(20), v)

    # THIS CAN THROW ERROR, SHOULD BE RUN MANUALLY IN THAT CASE
    # run_in_shell([f'rosrun pcl_ros bag_to_pcd {bag_file} /camera/depth/color/points {data_dir}/realsense_pcl/'])




    for topic, msg, t in bag.read_messages():

        # / camera / aligned_depth_to_color / camera_info
        # / camera / color / camera_info
        # / camera / color / image_rect_color
        # / camera / depth / camera_info
        # / camera / depth / color / points
        # / camera / extrinsics / depth_to_color
        # / clock
        # / mavros / vision_pose2 / pose
        # / rosout
        # / rosout_agg
        # / run_subscribe_msckf / pathimu
        # / tf
        # / tf_static

        if 'camera/depth/color/points' in topic:
            # https://answers.ros.org/question/344096/subscribe-pointcloud-and-convert-it-to-numpy-in-python/

            # pc2.read_points(cloud, skip_nans=True, field_names=("x", "y", "z"))
            # lidar = point_cloud2.read_points(msg, field_names=("x", "y", "z", "rgb"))

            xyz = []
            rgb = []
            gen = point_cloud2.read_points(msg, skip_nans=True)
            int_data = list(gen)
            import struct
            for x in int_data:
                test = x[3]
                # cast float32 to int so that bitwise operations are possible
                s = struct.pack('>f', test)
                i = struct.unpack('>l', s)[0]
                # you can get back the float value by the inverse operations
                pack = ctypes.c_uint32(i).value
                r = (pack & 0x00FF0000) >> 16
                g = (pack & 0x0000FF00) >> 8
                b = (pack & 0x000000FF)
                # prints r,g,b values in the 0-255 range
                # x,y,z can be retrieved from the x[0],x[1],x[2]
                xyz.append([x[0], x[1], x[2]])
                rgb.append([r, g, b])

            xyz = np.stack(xyz)
            rgb = np.stack(rgb)

            xyzrgb = np.concatenate((xyz, rgb), axis=1).astype('float32')

            # lidar = np.array(list(lidar), dtype=float)

            os.makedirs(data_dir + '/pts', exist_ok=True)
            np.save(f'{data_dir}/pts/{t}.npy', xyzrgb)

        if 'tf' in topic:
            pass

        if 'camera/depth/color/points' in topic:
            print(topic, 'implemented outside the python')
            pass

        if 'camera/depth/image_rect_raw' in topic:
            os.makedirs(data_dir + '/rgb', exist_ok=True)
            depth = np.ndarray(shape=(msg.height, msg.width), dtype=np.dtype('uint16'), buffer=msg.data)
            img = Image.fromarray(depth)
            img.save(f'{data_dir}/depth/{t}.png')

        if 'camera/color/image' in topic:
            # color_img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            os.makedirs(data_dir + '/rgb', exist_ok=True)

            color_img = imgmsg_to_cv2(msg)
            img = Image.fromarray(color_img)
            img.save(f'{data_dir}/rgb/{t}.png')
            # resize it here?
        if 'camera/infra1/image_rect_raw' in topic:
            pass

        if 'camera/infra2/image_rect_raw' in topic:
            pass

        if 'camera/depth/camera_info' in topic:
            # print(msg)
            pass
        # if 'camera/color/image_raw' in topic:
        #     color_img = imgmsg_to_cv2(msg)
        #     img = Image.fromarray(color_img)
        #     img.save(f'{data_dir}/rgb/{t}.png')
        #
        # if 'camera/aligned_depth_to_color/image_raw' in topic:
        #     depth = np.ndarray(shape=(msg.height, msg.width), dtype=np.dtype('uint16'), buffer=msg.data)
        #     img = Image.fromarray(depth)
        #     img.save(f'{data_dir}/depth/{t}.png')

        # if 'run_subscribe_msckf/poseimu' in topic:
        if 'mavros/vision_pose2/pose' in topic:
            os.makedirs(data_dir + '/pose', exist_ok=True)
            # breakpoint()
            translation = msg.pose.position

            orientation = msg.pose.orientation
            w = orientation.w
            x = orientation.x
            y = orientation.y
            z = orientation.z

            rot_mat = Rotation.from_quat([x, y, z, w]).as_matrix()

            T = np.eye(4)
            T[:3, :3] = rot_mat
            T[:3, 3] = [translation.x, translation.y, translation.z]

            np.save(f'{data_dir}/pose/{t}.npy', T)



    # rgb_files = sorted(glob.glob(f"{data_dir}/rgb/*.png"))
    # depth_files = sorted(glob.glob(f"{data_dir}/depth/*.png"))
    #
    # for rgb_file, depth_file in zip(rgb_files, depth_files):
    #
    #     depth = Image.open(depth_file)
    #     depth = np.asarray(depth)
    #     depth = depth / 1024.
    #
    #     rgb = Image.open(rgb_file)
    #     rgb = rgb.resize(depth.shape[::-1]) # why it should be swapped? Working, but i dont get PIL
    #     rgb = np.asarray(rgb)
    #     rgb = rgb / 255.
    #
    #     # print(rgb.shape, depth.shape)
    #     pcl = project_depth_to_pcl(depth, rgb=rgb, cx=cx, cy=cy, fx=fx, fy=fy)
    #
    #     # import pptk
    #     # pptk.viewer(pcl[:,:3], pcl[:,3:] / 255)
    #     np.save(rgb_file.replace('rgb', 'pts')[:-4], pcl)
    #     # break

def get_realsense_params():
    import pyrealsense2 as rs

    print("Getting realsense intrinsics parameters, make sure you have camera connected. If this fails, reconnect camera")
    pipeline = rs.pipeline()
    config = rs.config()

    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()

    config.enable_stream(rs.stream.depth, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    profile = pipeline.get_active_profile()
    depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
    depth_intrinsics = depth_profile.get_intrinsics()
    w, h = depth_intrinsics.width, depth_intrinsics.height

    cx = depth_intrinsics.ppx
    cy = depth_intrinsics.ppy
    fx = depth_intrinsics.fx
    fy = depth_intrinsics.fy

    instrinsic = open('intrinsic.txt', 'w')
    instrinsic.write(f'cx: {cx}\n'
                     f'cy: {cy}\n'
                     f'fx: {fx}\n'
                     f'fy: {fy}\n'
                     f'fps: 30\n'
                     f'width: {w}\n'
                     f'height: {h}\n')
    instrinsic.close()

def read_realsense_params(file):
    with open(file, 'r') as f:
        lines = f.readlines()
        params = {}
        for line in lines:
            key = line.split(':')[0]
            value = line.split(':')[1].strip()
            params[key] = float(value)
    # print(params)
    return params

if __name__ == "__main__":
    import sys

    param_file = '/home/patrik/patrik_data/drone/intrinsic.txt'
    if os.path.exists(param_file):
        params = read_realsense_params(param_file)
    else:
        get_realsense_params()
        params = read_realsense_params(param_file)

    cx = params['cx']
    cy = params['cy']
    fx = params['fx']
    fy = params['fy']

    width = 424
    height = 240
    fx, cx = 211.6538848876953, 211.9246063232422
    fy, cy = 211.6538848876953, 118.16419219970703
    print("WIDTH AND HEIGHT ARE HARDCODED, CHANGE THEM IF YOU WANT TO USE DIFFERENT RESOLUTION")
    unpack_bag(sys.argv[1], sys.argv[2], cx=cx, cy=cy, fx=fx, fy=fy, width=width, height=height)
