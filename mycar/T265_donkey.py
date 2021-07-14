# -*-coding: utf-8-*-
"""
Intel RealSense T265 Tracking Camera Parts Class for Donkeycar.
A rewrite of Donkeycar v3.1.1 realsense2.py.
This code is MIT license compliant.
"""
import time
import logging
import math as m
import numpy as np
import pickle
import cv2
# from T265_Stereo import center_undistorted

try:
    import pyrealsense2 as rs
except:
    print('[RealSenseT265] This module requires pyrealsense2 package!')
    raise


class FullDataReader:
    """
    Donkey Part class that gets all the data that can be obtained from T265.
    Created based on the code above Donkeycar v3.1.1.
    External wheel odometry correction is not supported.
    """

    def __init__(self, image_output=False, debug=False):
        """
        RealSense T265 A parts class that retrieves data from a tracking camera.
        argument:
        image_output From one of the two fisheye cameras on the T265
        Get an image stream(USB3.0 recommended).
        The default is False, None is always returned when run is executed.
        debug:
        Debug flag. When set to the true value, the log is output to the standard output.
        Return value: None
        """
        self.debug = debug
        self.image_output = image_output

        # Declare a RealSense pipeline and encapsulate real devices and sensors
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.pose)

        if self.image_output:
            # Both cameras need to be enabled at this time
            cfg.enable_stream(rs.stream.fisheye, 1)  # left camera
            cfg.enable_stream(rs.stream.fisheye, 2)  # right camera

        # Start streaming with the requested calibration
        self.pipe.start(cfg)

        # Start thread
        self.running = True

        zero_vec = (0.0, 0.0, 0.0)
        self.pos = zero_vec
        self.vel = zero_vec
        self.acc = zero_vec
        self.e_vel = zero_vec
        self.e_acc = zero_vec
        self.rot = (0.0, 0.0, 0.0, 0.0)
        self.ang = zero_vec
        self.posemap_conf = 0X0  # fail
        self.pose_conf = 0X0  # fail
        self.left_img = None
        self.right_img = None

        self.image_correction()

    # def StereoPair(self):
    #     """
    #     This will take the two images and combine them into a single image
    #     One in red, the other in green, and diff in blue channel.
    #     """
    #     if self.left_img is not None and self.right_img is not None:
    #         width, height = self.left_img.shape
    #         # grey_a = dk.utils.rgb2gray(self.left_img)
    #         # grey_b = dk.utils.rgb2gray(self.right_img)
    #         # grey_c = grey_a - grey_b
    #         grey_a = cv2.cvtColor(self.left_img, cv2.COLOR_BGR2GRAY)
    #         grey_b = cv2.cvtColor(self.right_img, cv2.COLOR_BGR2GRAY)
    #         grey_c = grey_a - grey_b
    #
    #         stereo_image = np.zeros([width, height, 3], dtype=np.dtype('B'))
    #         stereo_image[..., 0] = np.reshape(grey_a, (width, height))
    #         stereo_image[..., 1] = np.reshape(grey_b, (width, height))
    #         stereo_image[..., 2] = np.reshape(grey_c, (width, height))
    #     else:
    #         stereo_image = None
    #
    #     return stereo_image

    def image_correction(self):
        """Load camera matrix and distortion coeffs from file"""
        camera_data = pickle.load(open("RS_CalibrationData.p", "rb"))
        self.RCamIn = np.array(camera_data.get('R_camera_In'))
        self.RCamEx = np.array(camera_data.get('R_camera_Ex'))
        self.LCamIn = np.array(camera_data.get('L_camera_In'))
        self.LCamEx = np.array(camera_data.get('L_camera_Ex'))

    def undistortImage(self, imageL, imageR):
        # Returns the undistorted image
        imgL = cv2.undistort(imageL, self.LCamIn, self.LCamEx, None, self.LCamIn)
        imgR = cv2.undistort(imageR, self.RCamIn, self.RCamEx, None, self.RCamIn)
        # convert the image to grayscale and remove noise
        return imgL, imgR

    # # Calculates Rotation Matrix given euler angles.
    # @staticmethod
    # def eulerAnglesToRotationMatrix(theta):

    #     R_x = np.array([[1, 0, 0],
    #                     [0, m.cos(theta[0]), -m.sin(theta[0])],
    #                     [0, m.sin(theta[0]), m.cos(theta[0])]
    #                     ])

    #     R_y = np.array([[m.cos(theta[1]), 0, m.sin(theta[1])],
    #                     [0, 1, 0],
    #                     [-m.sin(theta[1]), 0, m.cos(theta[1])]
    #                     ])

    #     R_z = np.array([[m.cos(theta[2]), -m.sin(theta[2]), 0],
    #                     [m.sin(theta[2]), m.cos(theta[2]), 0],
    #                     [0, 0, 1]
    #                     ])

    #     R = np.dot(R_z, np.dot(R_y, R_x))

    #     return R

    # @staticmethod
    # def truncate(n, decimals=0):
    #     multiplier = 10 ** decimals
    #     return int(n * multiplier) / multiplier

    def poll(self):
        try:
            frames = self.pipe.wait_for_frames()
        except Exception as e:
            if self.debug:
                print(e)
            logging.error(e)
            return

        if self.image_output:
            # Get image from left fisheye camera
            left = frames.get_fisheye_frame(1)
            self.left_img = np.asanyarray(left.get_data())
            # Get image from right fisheye camera
            right = frames.get_fisheye_frame(2)
            self.right_img = np.asanyarray(right.get_data())

        # Fetch location information frame
        pose = frames.get_pose_frame()

        if pose:
            data = pose.get_pose_data()
            # Position coordinates
            self.pos = (data.translation.x, data.translation.y, data.translation.z)
            # Speed
            self.vel = (data.velocity.x, data.velocity.y, data.velocity.z)
            # Acceleration
            self.acc = (data.acceleration.x, data.acceleration.y, data.acceleration.z)
            # Angular velocity
            self.e_vel = (data.angular_velocity.x, data.angular_velocity.y, data.angular_velocity.z)
            # Angular acceleration
            self.e_acc = (data.angular_acceleration.x, data.angular_acceleration.y, data.angular_acceleration.z)
            # Quaternion
            self.rot = (data.rotation.w, - data.rotation.z, data.rotation.x, - data.rotation.y)
            # Euler angles
            self.ang = self.get_eular_angle()
            # Map reliability: 0x0-failure, 0x1-low, 0x2-medium, 0x3-high
            self.posemap_conf = data.mapper_confidence
            # Position Coordinate Reliability: 0x0-Failure, 0x1-Low, 0x2-Medium, 0x3-High
            self.pose_conf = data.tracker_confidence
            logging.debug('[RealSenseT265] poll () pos (% f,% f,% f)' % (self.pos[0], self.pos[1], self.pos[2]))
            # logging.debug ('[RealSenseT265] poll () ang (% f,% f,% f)'% (self.ang [0], self.ang [1], self.ang [2]))
            if self.debug:
                print('[RealSenseT265] poll() pos(%f, %f, %f)' % (self.pos[0], self.pos[1], self.pos[2]))
                print('[RealSenseT265] poll() vel(%f, %f, %f)' % (self.vel[0], self.vel[1], self.vel[2]))
                print('[RealSenseT265] poll() ang(%f, %f, %f)' % (self.ang[0], self.ang[1], self.ang[2]))
                print('[RealSenseT265] poll() rot(%f, %f, %f, %f)' % (self.rot[0], self.rot[1], self.rot[2],
                                                                      self.rot[3]))
                print('[RealSenseT265] poll() eular vel(%f, %f, %f)' % (self.e_vel[0], self.e_vel[1], self.e_vel[2]))
                print('[RealSenseT265] poll() eular acc(%f, %f, %f)' % (self.e_acc[0], self.e_acc[1], self.e_acc[2]))
                print('[RealSenseT265] poll() conf map:%d pose:%d' % (self.posemap_conf, self.pose_conf))
                print('[RealSenseT265] poll() left is None:{} right is None:{}'.format(str(self.left_img is None),
                                                                                       str(self.right_img is None)))

    def get_eular_angle(self):
        """
        Calculate the attitude angular velocity from the instance variable `self.rot`(quaternion).
        argument:
        None
        Return value:
        (roll, pitch, yaw) Euler angles based on initial position(radians)
        """
        w, x, y, z = self.rot[0], self.rot[1], self.rot[2], self.rot[3]
        roll = m.atan2(2.0 * (w * x + y * z), w * w - x * x - y * y + z * z) * 180.0 / m.pi
        pitch = -m.asin(2.0 * (x * z - w * y)) * 180.0 / m.pi
        yaw = m.atan2(2.0 * (w * z + x * y), w * w + x * x - y * y - z * z) * 180.0 / m.pi
        return roll, pitch, yaw

    def update(self):
        """
        This method is called when another thread is created.
        Get sensor data from T265.
        Execute poll () while the instance variable running is true.
        argument:
        None
        Return value:
        None
        """
        while self.running:
            self.poll()

    def run_threaded(self):
        """
        One of the parts class Template Method. If threaded is true
        Called instead of run ().
        Return all the latest sensor data available on the T265.
        The latest sensor data is not acquired when this method is executed (separate thread).
        Updated by).
        argument:
        None
        Return value:
        pos_x Location information X-axis (unit: meters)
        pos_y Location information Y-axis (unit: meters)
        pos_z Position information Z axis (unit: meter)
        vel_x Velocity X-axis (unit: meters / second)
        vel_y Velocity Y axis (unit: meters / second)
        vel_z Velocity Z axis (unit: meters / second)
        e_vel_x Angular velocity X axis, equivalent to gyr_x (unit: radians / second)
        e_vel_y Angular velocity Y axis, equivalent to gyr_y (unit: radians / second)
        e_vel_z Angular velocity Z axis, equivalent to gyr_z (Unit: radians / second)
        acc_x Acceleration X-axis (Unit: meters / second ^ 2)
        acc_y Acceleration Y-axis (Unit: meters / second ^ 2)
        acc_z Acceleration Z axis (Unit: meters / second ^ 2)
        e_acc_x Angular acceleration X-axis (Unit: radians / sec ^ 2)
        e_acc_y Angular acceleration Y-axis (Unit: radians / sec ^ 2)
        e_acc_z Angular acceleration Z axis (Unit: radians / sec ^ 2)
        rot_i Quaternion (Qi)
        rot_j Quaternion (Qj)
        rot_k Quaternion (Qk)
        rot_l Quaternion (Ql)
        ang_x Euler angles X axis (roll) (unit: radians)
        ang_y Euler angles Y-axis (pitch) (unit: radians)
        ang_z Euler angles Z axis (yaw) (unit: radians)
        posemap_conf posemap reliability: 0x0-failure, 0x1-low, 0x2-medium, 0x3-high
        pose_conf pose Confidence: 0x0-Failure, 0x1-Low, 0x2-Medium, 0x3-High
        left_image_array Left camera image (nd.array type, (800,848) format)
        right_image_array Right camera image (nd.array type, (800,848) format)
        """
        """
        return self.pos[0], self.pos[1], self.pos[2],\
            self.vel[0], self.vel[1], self.vel[2],\
            self.e_vel[0], self.e_vel[1], self.e_vel[2],\
            self.acc[0], self.acc[1], self.acc[2],\
            self.e_acc[0], self.e_acc[1], self.e_acc[2],\
            self.rot[0], self.rot[1], self.rot[2],\
            self.rot[3], self.posemap_conf, self.pose_conf,\
            self.ang[0], self.ang[1], self.ang[2],\
            self.left_img, self.right_img
            """
        x = -self.pos[2]
        y = -self.pos[0]
        z = self.pos[1]
        # vtx = np.asanyarray([x, y, z])
        # with open("track1.csv", "a+") as myfile:
        #     myfile.write(str(vtx).lstrip('[').rstrip(']'))
        #     myfile.write('\n')

        # x = self.truncate(x, 3)
        # y = self.truncate(y, 3)
        # z = self.truncate(z, 3)

        # vel = np.linalg.norm(self.e_vel)
        # R = self.eulerAnglesToRotationMatrix(self.ang)
        # T = np.array([[0.0, 0.0, 0.17]])
        # R_full = np.concatenate((R, T.T), axis=1)

        # w = self.rot[0]
        # pitch = -m.asin(2.0 * (x * z - w * y)) * 180.0 / m.pi
        # roll = m.atan2(2.0 * (w * x + y * z), w * w - x * x - y * y + z * z) * 180.0 / m.pi
        # yaw = m.atan2(2.0 * (w * z + x * y), w * w + x * x - y * y - z * z) * 180.0 / m.pi
        # if self.left_img is not None:
        #     self.left_img, self.right_img = self.undistortImage(self.left_img, self.right_img)
        #     cv2.imwrite('/home/sagi/Desktop/Right3.jpg', self.right_img)
        #     cv2.imwrite('/home/sagi/Desktop/Left3.jpg', self.left_img)
        # return {'pos': (x, z, y), 'cte': 'none', 'speed': -self.vel[2], 'hit': 'none',
        #         'transformation': [R_full]}, None, self.left_img, self.right_img 

        return {'pos': (x, z, y), 'cte': 'none', 'speed': -self.vel[2], 'hit': 'none'}, (self.ang[2]*m.pi/180)

    def run(self):
        """
        One of the parts class Template Method. If threaded is false
        Called instead of run_threaded ().
        All the latest sensor data that can be acquired by T265 (when this method is called)
        (Acquired to) will be returned.
        argument:
        None
        Return value:
        pos_x Location information X-axis (unit: meters)
        pos_y Location information Y-axis (unit: meters)
        pos_z Position information Z axis (unit: meter)
        vel_x Velocity X-axis (unit: meters / second)
        vel_y Velocity Y axis (unit: meters / second)
        vel_z Velocity Z axis (unit: meters / second)
        e_vel_x Angular velocity X axis, equivalent to gyr_x (unit: radians / second)
        e_vel_y Angular velocity Y axis, equivalent to gyr_y (unit: radians / second)
        e_vel_z Angular velocity Z axis, equivalent to gyr_z (Unit: radians / second)
        acc_x Acceleration X-axis (Unit: meters / second ^ 2)
        acc_y Acceleration Y-axis (Unit: meters / second ^ 2)
        acc_z Acceleration Z axis (Unit: meters / second ^ 2)
        e_acc_x Angular acceleration X-axis (Unit: radians / sec ^ 2)
        e_acc_y Angular acceleration Y-axis (Unit: radians / sec ^ 2)
        e_acc_z Angular acceleration Z axis (Unit: radians / sec ^ 2)
        rot_i Quaternion (Qi)
        rot_j Quaternion (Qj)
        rot_k Quaternion (Qk)
        rot_l Quaternion (Ql)
        ang_x Euler angles X axis (roll) (unit: radians)
        ang_y Euler angles Y-axis (pitch) (unit: radians)
        ang_z Euler angles Z axis (yaw) (unit: radians)
        posemap_conf posemap reliability: 0x0-failure, 0x1-low, 0x2-medium, 0x3-high
        pose_conf pose Confidence: 0x0-Failure, 0x1-Low, 0x2-Medium, 0x3-High
        left_image_array Left camera image (nd.array type, (800,848) format)
        right_image_array Right camera image (nd.array type, (800,848) format)
        """
        self.poll()
        return self.run_threaded()

    def shutdown(self):
        """
        One of the parts class Template Method. Processing at the end.
        Close the multithreaded loop and stop the T265 pipe.
        argument:
        None
        Return value:
        None
        """
        self.running = False
        time.sleep(0.1)
        self.pipe.Stop()
