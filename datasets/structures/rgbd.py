import os.path
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from PIL import Image


def project_depth_to_pcl(depth, rgb=None, seg=None, confidence=None, seg_label=None, cx=None, cy=0, fx=0, fy=0):

    depth_coors = np.argwhere(depth > -50).astype(np.float32) # to get indices of all elements, filtration is done later
    depth_values = depth.flatten()
    depth_pts = np.insert(depth_coors, 2, depth_values, axis=1)

    depth_pts[:, 0] = (depth_pts[:, 0] - cx) * depth_pts[:, 2] / fx
    depth_pts[:, 1] = (depth_pts[:, 1] - cy) * depth_pts[:, 2] / fy

    if rgb is not None:
        depth_pts = np.insert(depth_pts, depth_pts.shape[1], rgb[:,:,0].flatten(), axis=1)
        depth_pts = np.insert(depth_pts, depth_pts.shape[1], rgb[:,:,1].flatten(), axis=1)
        depth_pts = np.insert(depth_pts, depth_pts.shape[1], rgb[:,:,2].flatten(), axis=1)

    if seg is not None:
        depth_pts = np.insert(depth_pts, depth_pts.shape[1], seg[:,:,0].flatten(), axis=1)
        depth_pts = np.insert(depth_pts, depth_pts.shape[1], seg[:,:,1].flatten(), axis=1)
        depth_pts = np.insert(depth_pts, depth_pts.shape[1], seg[:,:,2].flatten(), axis=1)

    if confidence is not None:
        depth_pts = np.insert(depth_pts, depth_pts.shape[1], confidence.flatten(), axis=1)

    if seg_label is not None:
        depth_pts = np.insert(depth_pts, depth_pts.shape[1], seg_label.flatten(), axis=1)

    pcl = depth_pts[(depth_pts[:, 2] > 0.1) & (depth_pts[:, 2] < 20)]   # filter invalid disparity


    return pcl

def plot_bev(pcl, save=None, lim=None):
    fig, ax = plt.subplots()
    ax.scatter(pcl[:, 0], pcl[:, 1], color=pcl[:, 6:9], marker='.', s=0.2)
    ax.set_xlim([-.5, 6])
    ax.set_ylim([-4, 4])

    if save is not None:
        fig.savefig(top_down_inference)
    else:
        plt.show()

    plt.close()

def plot_realsense(pcl, features, save=None, vmin=0, vmax=1, cb=True):
    fig = plt.figure(figsize=(4, 4), dpi=300)
    ax = fig.add_subplot(projection='3d')

    xs = pcl[:, 0]
    ys = pcl[:, 1]
    zs = pcl[:, 2]


    colorbar_data = ax.scatter(xs, ys, zs, marker='.', s=0.3, c=features, alpha=0.6, vmin=vmin, vmax=vmax, cmap='jet')
    ax.set_xlim([-.5, 6])
    ax.set_ylim([-4, 4])
    ax.set_zlim([-4, 4])

    ax.view_init(elev=25, azim=210)
    ax.dist = 4

    if cb:
        fig.colorbar(colorbar_data)

    plt.axis('off')

    if save is not None:
        os.makedirs(os.path.dirname(save), exist_ok=True)
        plt.savefig(save)
    else:
        plt.show()

    plt.close()



