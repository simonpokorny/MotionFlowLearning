import pyrealsense2 as rs
import numpy as np
import cv2
import time

points = rs.points()
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)
profile = pipeline.start(config)

count = 0
from PIL import Image
try:
    while True:
        frames = pipeline.wait_for_frames()
        nir_lf_frame = frames.get_infrared_frame(1)
        nir_rg_frame = frames.get_infrared_frame(2)
        if not nir_lf_frame or not nir_rg_frame:
            continue
        nir_lf_image = np.asanyarray(nir_lf_frame.get_data())
        nir_rg_image = np.asanyarray(nir_rg_frame.get_data())

        time.sleep(0.1)

        left = Image.fromarray(nir_lf_image)
        right = Image.fromarray(nir_rg_image)

        left.save(f'/home/patrik/patrik_data/drone/stereo/{count:06d}_left.png')
        right.save(f'/home/patrik/patrik_data/drone/stereo/{count:06d}_right.png')

        count += 1

        if count == 100:
            break
        # horizontal stack
        # image=np.hstack((nir_lf_image,nir_rg_image))
        # cv2.namedWindow('NIR images (left, right)', cv2.WINDOW_AUTOSIZE)
        # cv2.imshow('IR Example', image)
        # key = cv2.waitKey(1)
        #
        # Press esc or 'q' to close the image window
        # if key & 0xFF == ord('q') or key == 27:
        #     cv2.destroyAllWindows()
        #     break
finally:
    pipeline.stop()
