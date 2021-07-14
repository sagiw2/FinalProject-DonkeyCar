import cv2
import numpy as np
from numpy.linalg import inv
import time
import pickle


# f = open("pid.txt", 'w')


class PointAndGo(object):

    def __init__(self):
        self.running = True
        # image vars #
        self.img = None
        self.out_img = None
        # self.L_or_R = 0
        self.left_click_location = [0, 0]
        # car state vars #
        self.is_simulated = False
        self.info = {'pos': (49.69002, 0.5552713, 47.83797), 'cte': 1.907349e-05, 'speed': 1.547668e-06, 'hit': 'none'}
        # dic_struct={'pos': (49.69002, 0.5552713, 47.83797), 'cte': 1.907349e-05, 'speed': 1.547668e-06, 'hit': 'none'}
        self.previous_info = {'pos': (47.74339, 0.5552713, 33.71123), 'cte': 1.907349e-05, 'speed': 1.547668e-06,
                              'hit': 'none'}
        self.pos = [0.0, 0.0]
        self.depth_locations = None
        self.vel = 0.0
        self.driving_direction = 1
        self.throttle = 0.0
        self.mode = 'user'
        self.recording = False
        self.angle = 0.0
        self.heading_angle = np.pi / 2
        self.donkey_angle = None
        # camera calibration vars #
        self.need_to_be_calibrated = True
        # point and go vars #
        self.has_location = False
        self.nav_data = None
        # calsses vars #
        self.lateral = LateralDynamic
        self.calibrate_camera = ImageCorrection
        self.ImageCorrection_ref = ImageCorrection()
        self.CoordinationTransformer = CoordinationTransformer
        self.CoordinationTransformer_ref = CoordinationTransformer()
        self.Velocity_PID = PID
        self.Velocity_PID_ref = PID()

    def update(self):
        """calibrate the camera and return an undistorted image for the user interface """
        while self.running:
            if self.img is not None:
                if self.is_simulated:
                    if self.need_to_be_calibrated:
                        self.calibrate_camera.load_camera_parameters(self.ImageCorrection_ref)
                        self.need_to_be_calibrated = False
                    self.out_img = self.calibrate_camera.undistortImage(self.ImageCorrection_ref, self.img)
                else:
                    # if self.need_to_be_calibrated:
                    #     self.CoordinationTransformer.load_camera_matrix(self.CoordinationTransformer_ref)
                    #     self.need_to_be_calibrated = False
                    # self.CoordinationTransformer.transform_matrix_calculator(self.CoordinationTransformer_ref, self.info.get('transformation'))
                    self.out_img = self.img
            else:
                self.out_img = None

    def calculate_heading(self, prev_heading_angle):
        """calculate the car heading based on current and previous location"""

        if self.is_simulated is False:
            # the angle of the diff vector is roughly equal to the car heading angle
            delta_x = self.info.get('pos')[0] - self.previous_info.get('pos')[0]
            delta_y = self.info.get('pos')[2] - self.previous_info.get('pos')[2]

            # if the vehicle hasn't moved the heading angle remains the same
            if delta_x == 0 and delta_y == 0:
                return prev_heading_angle
            # if delta_x is zero then the arctan2 is not defined and his value at that point is +/-90deg
            elif delta_x == 0:
                return np.sign(delta_y) * (np.pi / 2)
            heading_angle = np.arctan2(delta_y, delta_x)
            if abs(heading_angle - prev_heading_angle) < 0.17:
                heading_angle = prev_heading_angle
            return heading_angle
        else:
            return self.heading_angle

    def calculate_point_to_nav(self):
        """calculate the x and y in simulation coordinates of the click location"""
        '''X is the forward direction, Y is the left direction'''

        # calculate the target location in car coordinates
        delta_x, delta_y = CoordinationTransformer.transform(self.CoordinationTransformer_ref,
                                                             self.left_click_location, self.is_simulated,
                                                             self.depth_locations)

        # rotate the target location to a coordinate system parallel to the world system
        # x_factor = delta_x * np.cos(self.heading_angle) - delta_y * np.sin(self.heading_angle)
        # y_factor = delta_x * np.sin(self.heading_angle) + delta_y * np.cos(self.heading_angle)

        x_factor = delta_x
        y_factor = delta_y

        # add the relative car location
        x = self.pos[0] + x_factor
        y = self.pos[1] + y_factor
        # print(x, y, "tambal", self.pos)
        return x, y

    def run_threaded(self, sim_img, rs_cam, info, tracking_info,donkey_angle, depth_locations):
        """main part loop"""

        if self.is_simulated:
            self.img = sim_img
        else:
            info = tracking_info
            # info = self.info
            self.donkey_angle = donkey_angle
            self.img = rs_cam
            self.depth_locations = depth_locations

        if info:

            self.vel = info.get('speed')
            des_speed = 0

            # calculate car heading angle
            delta_location = np.subtract([self.info.get('pos')[0], self.info.get('pos')[2]],
                                         [self.previous_info.get('pos')[0], self.previous_info.get('pos')[2]])

            dist = np.linalg.norm(delta_location)

            # if the vehicle hasn't moved enough than the calculation should be done with the previous iteration info
            if dist > 0.01:
                self.previous_info = self.info
            self.info = info
            self.heading_angle = self.calculate_heading(self.heading_angle)

            # parse info for easier calculation and readability
            self.pos = [self.info.get('pos')[0], self.info.get('pos')[2]]

            # show image to user and set up mouse input from image
            if self.out_img is not None:
                cv2.namedWindow("DonkeyCar VideoFeed", cv2.WINDOW_NORMAL)
                cv2.imshow('DonkeyCar VideoFeed', self.out_img)
                cv2.resizeWindow('DonkeyCar VideoFeed', 848, 480)
                cv2.setMouseCallback('DonkeyCar VideoFeed', self.mouse_callback)
                cv2.waitKey(1)
                # if cv2.getTrackbarPos('Left/Right Camera', 'DonkeyCar VideoFeed'):
                #     self.L_or_R = 1
                # else:
                #     self.L_or_R = 0

            # if the car has a target location to nav to
            if self.has_location:

                # calculate distance between current location and target location
                delta_location = np.subtract(self.nav_data, self.pos)
                dist = np.linalg.norm(delta_location)

                # check to see if current location is inside the tolerance for target location
                if dist <= 0.1:
                    self.has_location = False
                    des_speed = 0.0
                    # print("Reached Goal")
                else:
                    # if vehicle is not close enough then calculate the steering angle to that location and
                    # drive with constant speed
                    self.angle, theta = self.lateral.calculate_turn_radius(LateralDynamic(), delta_location,
                                                                           self.heading_angle)
                    des_speed = 1.0 * np.sign(delta_location[0])

                    # if 0.8 < abs(theta) < 4.71 and dist < 3:
                    #     print("can't reach")
                    #     des_speed = 0.0     # need to be 0, changed for testing
            else:
                # if the vehicle has no target location then a mock target location is set
                des_speed = 0.0
                self.nav_data = self.pos
        else:
            return self.angle, self.throttle, self.mode, self.recording

        self.throttle = self.Velocity_PID.pid(self.Velocity_PID_ref, self.vel, des_speed)
        self.angle = self.angle*np.sign(self.throttle)
        # self.angle = 0.0
        # self.throttle = 0.0
        # print(self.throttle)
        # print(self.angle)
        # print(self.vel)
        print(dist)
        # self.throttle = -0.75
        # print(self.heading_angle*180/np.pi)
        # print(self.donkey_angle - self.heading_angle)
        return self.angle, self.throttle, self.mode, self.recording

    def mouse_callback(self, event, u, v, flags, param):
        """register a mouse left click on camera feed window"""
        if event == 1:
            self.has_location = True
            self.left_click_location = [u, v]
            self.nav_data[0], self.nav_data[1] = self.calculate_point_to_nav()
            self.Velocity_PID.reset_pid(self.Velocity_PID_ref)
            # print("new target acquired")
            # print(self.nav_data)


    def shutdown(self):
        """terminate part"""
        self.running = False
        cv2.destroyAllWindows()
        time.sleep(0.2)


class ImageCorrection:

    def __init__(self):
        self.cameraIntrinsicValues = None
        # Distortion coefficients
        self.cameraExtrinsicValues = None
        self.parameters_are_loaded = False

    def load_camera_parameters(self):
        """Load camera matrix and distortion coeffs from file"""
        camera_data = pickle.load(open("CalibrationData_horizontal.p", "rb"))
        self.cameraIntrinsicValues = np.array(camera_data.get('camera_In'))
        self.cameraExtrinsicValues = np.array(camera_data.get('camera_Ex'))
        self.parameters_are_loaded = True

    def undistortImage(self, image):
        # Returns the undistorted image
        if self.parameters_are_loaded:
            image = cv2.undistort(image, self.cameraIntrinsicValues, self.cameraExtrinsicValues, None,
                                  self.cameraIntrinsicValues)
            # convert the image to grayscale and remove noise
            image = self.grayscale(image)
            image = self.remove_noise(image, 3)
        return image

    @staticmethod
    def grayscale(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def remove_noise(image, kernel_size=5):
        """Applies a Gaussian Noise kernel"""
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


class LateralDynamic:

    def __init__(self):
        self.max_angle = 16 * np.pi / 180
        # vehicle length
        self.L = 1.7

    def angle_to_steering(self, angle):
        """calculate the steering angle for the desired angle"""
        """steering angle range [left, right]=[-1,1], steering constraints [16,-16], 
        angle range [left, right]=[90,-90]"""

        # if the angle is bigger then the steering angle constrains then set the steering angle to its max/min
        if abs(angle) >= self.max_angle:
            return np.sign(angle)

        # map the desired angle to an acceptable steering angle
        steering_angle = (1 / self.max_angle) * angle
        return steering_angle

    def calculate_turn_radius(self, delta_Location, heading_angle):
        """calculate the turn radius and extract the desired delta_angle (front wheels angle)"""

        # if delta_x is zero then the arctan2 is not defined and his value is 90deg
        if delta_Location[0] == 0.0:
            theta = np.pi / 2
        else:
            theta = np.arctan2(delta_Location[1], delta_Location[0])

        # the desired angle is the angle to the target - the car heading angle
        theta = theta - heading_angle

        # if the angle is 0 or 180 then turn radius is -/+ inf -> the vehicle needs to drive straight
        if np.cos(theta) == 1:
            return -self.angle_to_steering(np.pi / 2), theta

        # calculate the turning circle string length
        a = np.linalg.norm(delta_Location)

        # extract turn radius from cosin theorem
        turn_r = np.sqrt((a ** 2) / (2 * (1 - np.cos(theta))))

        # calculate front wheels angle
        delta = np.sign(np.sin(theta)) * self.L / turn_r

        # return the minus angle
        return -self.angle_to_steering(delta), theta


class PID:

    def __init__(self):
        self.tol = 0.1
        self.Kp = 0.6
        self.Kd = 0.01
        self.Ki = 0.001
        self.error_sum = 0.0
        self.prev_error = 0.0
        self.vel = 0.0
        self.prev_velocity = 0.0
        self.delta_time = 0.05
        self.throttle_cap = 0.5  # values bigger than that have caused undesired behavior
        self.car_stuck = [10, 10]

    def pid(self, velocity, des_velocity):
        error = des_velocity - velocity
        acc = self.calculate_pid(error)

        # if the acceleration needed is bigger then the user defined cap then acc = acc_cap
        if abs(acc) > self.throttle_cap:
            acc = self.throttle_cap * np.sign(acc)
        else:
            acc = 0

        # # if the car tries to move but dosent, give her a boost
        # if abs(acc) > 0.5 and abs(velocity) < 0.1:
        #     self.car_stuck[0] -= 1
        # if not self.car_stuck:
        #     acc = des_velocity
        #     if not self.car_stuck[1]:
        #         self.car_stuck[0] = 10
        #     else:
        #        self.car_stuck -= 1

        # pid calibration file
        # global f
        # f.write(f'{acc},{error}\n')
        return acc

    def calculate_pid(self, error):
        if abs(error) < self.tol:
            self.error_sum = 0
            self.prev_error = 0
            return 0
        control_signal = self.Kp * error + self.Kd * (
                    error - self.prev_error) / self.delta_time + self.Ki * self.error_sum
        self.error_sum = self.error_sum + error
        self.prev_error = error
        return control_signal

    def reset_pid(self):
        self.error_sum = 0.0
        self.prev_error = 0.0


class CoordinationTransformer:

    def __init__(self):
        self.x_pix_to_sim_coefs = [5.3816739719997114E-13, -3.7138216600172993E-10, 9.86240869386419E-8,
                                   -1.2062173017819998E-5, 0.00067447213758620541, -0.0069585826204507191,
                                   0.0012329507038856518]
        self.delta_y_from_x_coefs = [-0.67608541125349886, 202.74510355130579]
        # self.camera_matrix_L = None
        # self.camera_matrix_R = None
        # self.world_transform = [[]]
        self.resolution_sim = [640, 480]
        self.resolution_realsense = [848, 480]
        self.cameraview_to_car_cg = 1.7 / 2 + 0.7

    # def transform_matrix_calculator(self, R):
    #     self.world_transform = R
    #     return

    # def load_camera_matrix(self):
    #     camera_data = pickle.load(open("RS_CalibrationData.p", "rb"))
    #     self.camera_matrix_L = np.array(camera_data.get('L_camera_In'))
    #     self.camera_matrix_R = np.array(camera_data.get('R_camera_In'))
    #     return

    def transform(self, pixel, is_simulated, depth_locations):
        """calculate target point location in the car coordinates system based on handmade calibration"""
        resolution = self.resolution_sim if is_simulated else self.resolution_realsense
        # if is_simulated:
        #     resolution = self.resolution_sim

        # else:
        #     resolution = self.resolution_realsense
        #     if camera:  # transform change based in camera location in relation to IMU
        #         self.world_transform[0][1][1] = 0.107
        #         camera_mat = self.camera_matrix_L
        #     else:
        #         self.world_transform[0][1][1] = -0.533
        #         camera_mat = self.camera_matrix_R

        if is_simulated:
            x_pix = resolution[1] - pixel[1] - 1
            y_pix = resolution[0] / 2 - pixel[0]
            x = np.polyval(self.x_pix_to_sim_coefs, x_pix) + self.cameraview_to_car_cg
            y = y_pix / np.polyval(self.delta_y_from_x_coefs, x_pix)

        else:
            # need to add the transform from the D435 to the T265
            x = depth_locations[resolution[0] * pixel[1] + pixel[0]][2]
            y = -depth_locations[resolution[0] * pixel[1] + pixel[0]][0]
        
        return x, y
        # else:
        #     mat = np.matmul(camera_mat, self.world_transform[0])
        #     inv_mat = np.linalg.pinv(mat)
        #     [x, y, _, _] = np.matmul(inv_mat, np.array([[x_pix], [y_pix], [0.0]]))
        #     return x[0], y[0]

