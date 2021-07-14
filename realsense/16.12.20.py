## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2
import csv

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)

config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

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
align = rs.align(align_to)

i = 0

def mouse_callback(event, u, v, flags, param):
        """register a mouse left click on camera feed window"""
        if event == 1:
            global vtx, i
            np.savetxt("depth{}.csv".format(i), vtx, delimiter=",")
            i+=1
            # with open('employee_file.csv', mode='w') as employee_file:
            #     employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            #     employee_writer.writerow(vtx)
            print("tambal")
            # left_click_location = [u, v]
            # pixel_3d_coords = 848*left_click_location[1]+left_click_location[0]
            # print(vtx[pixel_3d_coords])
            # print(vtx[pixel_3d_coords][0])
            # print(vtx[pixel_3d_coords][1])
            # print(vtx[pixel_3d_coords][2])

            

try:
    while True:


        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        # Remove background - Set pixels further than clipping_distance to grey
        grey_color = 153
        depth_image_3d = np.dstack(
            (depth_image, depth_image, depth_image))  # depth image is 1 channel, color is 3 channels
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        images = np.hstack((bg_removed, depth_colormap))
        # Stack both images horizontally
###############################################

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)

        # Stack both images horizontally

        #images = np.hstack((color_image, depth_colormap))
        pc = rs.pointcloud()
        points = pc.calculate(aligned_depth_frame)
        pc.map_to(color_frame)
        vtx = np.asanyarray(points.get_vertices())

        # x_matrix = np.array([x[0] for x in vtx]).reshape((480, 640))
        # y_matrix = np.array([x[1] for x in vtx]).reshape((480, 640))
        # z_matrix = np.array([x[2] for x in vtx]).reshape((480, 640))

        cv2.imshow('Image', depth_colormap)
        cv2.setMouseCallback('Image', mouse_callback)


        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

finally:

    # Stop streaming
    pipeline.stop()
