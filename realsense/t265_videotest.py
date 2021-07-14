import pyrealsense2 as rs

from pprint import pprint
import numpy as np
import cv2
from math import tan, pi

"""
Returns R, T transform from src to dst
"""
def get_extrinsics(src, dst):
    extrinsics = src.get_extrinsics_to(dst)
    R = np.reshape(extrinsics.rotation, [3,3]).T
    T = np.array(extrinsics.translation)
    return (R, T)

"""
Returns a camera matrix K from librealsense intrinsics
"""
def camera_matrix(intrinsics):
    return np.array([[intrinsics.fx,             0, intrinsics.ppx],
                     [            0, intrinsics.fy, intrinsics.ppy],
                     [            0,             0,              1]])

"""
Returns the fisheye distortion from librealsense intrinsics
"""
def fisheye_distortion(intrinsics):
    return np.array(intrinsics.coeffs[:4])
# Get realsense pipeline
pipe = rs.pipeline()

# Configure the pipeline
cfg = rs.config()
cfg.enable_stream(rs.stream.pose) # Positional data (translation, rotation, velocity etc)
cfg.enable_stream(rs.stream.fisheye, 1) # Left camera
cfg.enable_stream(rs.stream.fisheye, 2) # Right camera

# Prints a list of available streams, not all are supported by each device
# print('Available streams:')
# pprint(dir(rs.stream))

# Start the configured pipeline
pipe.start(cfg)

try:
    min_disp = 0
    # must be divisible by 16
    num_disp = 160 - min_disp
    max_disp = min_disp + num_disp
    profiles = pipe.get_active_profile()
    streams = {"left"  : profiles.get_stream(rs.stream.fisheye, 1).as_video_stream_profile(),
               "right" : profiles.get_stream(rs.stream.fisheye, 2).as_video_stream_profile()}
    intrinsics = {"left"  : streams["left"].get_intrinsics(),
                  "right" : streams["right"].get_intrinsics()}
    # Translate the intrinsics from librealsense into OpenCV
    K_left  = camera_matrix(intrinsics["left"])
    D_left  = fisheye_distortion(intrinsics["left"])
    K_right = camera_matrix(intrinsics["right"])
    D_right = fisheye_distortion(intrinsics["right"])
    (width, height) = (intrinsics["left"].width, intrinsics["left"].height)
    # Get the relative extrinsics between the left and right camera
    (R, T) = get_extrinsics(streams["left"], streams["right"])
    stereo_fov_rad = 90 * (pi/180)  # 90 degree desired fov
    stereo_height_px = 700          # 300x300 pixel stereo output
    stereo_focal_px = stereo_height_px/2 / tan(stereo_fov_rad/2)
    # We set the left rotation to identity and the right rotation
    # the rotation between the cameras
    R_left = np.eye(3)
    R_right = R
    # The stereo algorithm needs max_disp extra pixels in order to produce valid
    # disparity on the desired output region. This changes the width, but the
    # center of projection should be on the center of the cropped image
    stereo_width_px = stereo_height_px + max_disp
    stereo_size = (stereo_width_px, stereo_height_px)
    stereo_cx = (stereo_height_px - 1)/2 + max_disp
    stereo_cy = (stereo_height_px - 1)/2
    P_left = np.array([[stereo_focal_px, 0, stereo_cx, 0],
                       [0, stereo_focal_px, stereo_cy, 0],
                       [0,               0,         1, 0]])
    P_right = P_left.copy()
    P_right[0][3] = T[0]*stereo_focal_px
    # Construct Q for use with cv2.reprojectImageTo3D. Subtract max_disp from x
    # since we will crop the disparity later
    Q = np.array([[1, 0,       0, -(stereo_cx - max_disp)],
                  [0, 1,       0, -stereo_cy],
                  [0, 0,       0, stereo_focal_px],
                  [0, 0, -1/T[0], 0]])
    (w, h) = (intrinsics["left"].width, intrinsics["left"].height)
    mapx_l,mapy_l = cv2.fisheye.initUndistortRectifyMap(K_left,D_left,R_left,P_left,(w,h),cv2.CV_32FC1)
    mapx_r,mapy_r = cv2.fisheye.initUndistortRectifyMap(K_right, D_right, R_right, P_right,(w,h),cv2.CV_32FC1)
    while True:
        frames = pipe.wait_for_frames()
        left = frames.get_fisheye_frame(1)
        
        left_data = np.asanyarray(left.get_data())

        right = frames.get_fisheye_frame(2)
        right_data = np.asanyarray(right.get_data())

        dst_left = cv2.remap(left_data,mapx_l,mapy_l,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)

        dst_right = cv2.remap(right_data,mapx_r,mapy_r,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)


        # crop the image

        #cv2.imwrite('calibresult.png',dst)
        # cv2.imshow('left', dst_left)
        cv2.imshow('left_ori', left_data)
        # cv2.imshow('right', dst_right)
        # cv2.imshow('right_ori', right_data)
        pose = frames.get_pose_frame()
        if pose:
            data = pose.get_pose_data()
        #     print('\nFrame number: ', pose.frame_number)
            print('Position: ', data.translation)
        #     print('Velocity: ', data.velocity)
        #     print('Acceleration: ', data.acceleration)
        #     print('Rotation: ', data.rotation)


        cv2.waitKey(1)

finally:
    pipe.stop()
