import pyrealsense2 as rs
import numpy as np
import cv2
import time

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.animation import FuncAnimation



class RealSense435(object):

    def __init__(self):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()
        # pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        # pipeline_profile = config.resolve(pipeline_wrapper)

        config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)

        # Start streaming
        profile = self.pipeline.start(config)

        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print("Depth Scale is: ", depth_scale)

        clipping_distance_in_meters = 3 #1 meter
        clipping_distance = clipping_distance_in_meters / depth_scale


        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        self.align = rs.align(align_to)
        self.running  = True
        self.color_image = None
        self.pixle_world_location = None

        self.O_grid = OccupencyGrid
        self.O_grid_ref = OccupencyGrid()

    def update(self):

        while self.running:
            time.sleep(0.2)
            # Get frameset of color and depth
            frames = self.pipeline.wait_for_frames()

            # Align the depth frame to color frame
            aligned_frames = self.align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 848x480 depth image
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            self.color_image = np.asanyarray(color_frame.get_data())

            pc = rs.pointcloud()
            points = pc.calculate(aligned_depth_frame)
            pc.map_to(color_frame)
            self.pixle_world_location = np.asanyarray(points.get_vertices())

    def run_threaded(self):
        return self.color_image, self.pixle_world_location

    def run(self):
        
        self.update()
        return self.run_threaded()

    def shutdown(self):
        self.running = False
        time.sleep(2) # give thread enough time to shutdown

        # done running
        self.pipeline.stop()
        self.pipeline = None
        self.align = None
        self.color_image = None
        self.pixle_world_location = None




class OccupencyGrid(object):


    def __init__(self):
        # self.x_lim = [-2, 2]
        # self.z_lim = [-1, 3]
        # # self.fig = plt.figure()
        # # self.ax = self.fig.add_subplot(111, aspect='equal')
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim((-1, 3))
        xticks = np.arange(-2, 2.1, 0.1)
        yticks = np.arange(-1, 3.1, 0.1)
        self.ax.set_xticks(xticks)
        self.ax.set_yticks(yticks)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ln, = plt.plot([], [], 's')

    def update_fig(self, point_cloud):
        if point_cloud is not None:
            pc = point_cloud
            pc = pc.tolist()
            pc = np.array(pc)
            # remove points outside of area of interest and empty points
            pc = pc[np.linalg.norm(pc, axis=1) < 3]
            pc = pc[pc[:, 0] != 0]
            # resize area of interest in y axis
            pc = pc[pc[:, 1] < 0.23]
            pc = pc[pc[:, 1] > 0.15]
            # remove points with the same x and different z
            pc = pc.round(3)
            # pc = np.array([pc[:, 0], pc[:, 2]])
            self.ln.set_data(pc[:,0], pc[:,2])
            return self.ln,

    def grid(self, pc):
        ani = FuncAnimation(self.fig, self.update_fig, point_cloud=pc, blit=True)
        # pc = self.update_fig(point_cloud)
        # plt.clf()
        # plt.plot(pc[0], pc[1], 'o')
        # plt.draw()
        plt.show()

    # def convert_pc2grid(self, point_cloud):
    #     if point_cloud is not None:
    #         print("tambal")
    #         pc = point_cloud
    #         pc = pc.tolist()
    #         pc = np.array(pc)
    #         # remove points outside of area of interest and empty points
    #         pc = pc[np.linalg.norm(pc, axis=1) < 3]
    #         pc = pc[pc[:, 0] != 0]
    #         # resize area of interest in y axis
    #         pc = pc[pc[:, 1] < 0.23]
    #         pc = pc[pc[:, 1] > 0.15]
    #         # remove points with the same x and different z
    #         pc = pc.round(3)
    #         pc = pc[np.unique(pc[:, 2], axis=0, return_index=True)[1]]
    #         forplot = np.array([pc[:, 0], pc[:, 2]])
    #         forplot_1 = np.floor(forplot*10)/10
    #         forplot_2 = abs(forplot_1 - np.ceil(forplot*10)/10)
    #         rect = []
    #         for i in range(0, forplot.shape[1]):
    #             rect.append(patches.Rectangle((forplot_1[0, i], forplot_1[1, i]), forplot_2[0, i], forplot_2[1, i],linewidth=1, edgecolor='none', facecolor='blue'))
    #         fig = plt.figure()
    #         ax = fig.add_subplot(111, aspect='equal')
    #         ax.set_xlim(-2, 2)
    #         ax.set_ylim((-1, 3))
    #         xticks = np.arange(-2, 2.1, 0.1)
    #         yticks = np.arange(-1, 3.1, 0.1)
    #         ax.set_xticks(xticks)
    #         ax.set_yticks(yticks)
    #         ax.grid()
    #         ax.add_collection(PatchCollection(rect))
    #         # ax = fig.add_subplot(projection='3d')
    #         # ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2])
    #         ax.set_xlabel('x')
    #         ax.set_ylabel('y')
    #         # ax.set_zlabel('z')
    #         plt.show()
