import cv2
import csv
import pyrealsense2 as rs
from pynput import keyboard
import numpy as np


pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.pose)
cfg.enable_stream(rs.stream.fisheye, 1)  # left camera
cfg.enable_stream(rs.stream.fisheye, 2)  # right camera
pipe.start(cfg)


te

def csv_writer(x, y, pos):
    with open('coords_file_rs_left.csv', mode='a') as coords_file:
        coords_writer = csv.writer(coords_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        u = 800-y-1
        v = 848-x-1
        if u < 0:
            u = u+1
        elif v < 0:
            v = v+1
        x_location = input("Enter x location where x = 1 is one square forward\n")
        y_location = input("Enter y location where y = 1 is one square right\n")
        if not x_location == 'x':
            coords_writer.writerow([u, v, x_location, y_location, -pos.z, pos.x])
            coords_file.close()
        else:
            cv2.destroyAllWindows()
            quit()


def mouse_callback(event, x, y, flags, params):
    frames = pipe.wait_for_frames()
    # left-click event value is 1
    if event == 1:
        pose = frames.get_pose_frame()
        data = pose.get_pose_data()
        csv_writer(x, y, data.translation)


def main():
    frames = pipe.wait_for_frames()
    rs.align(frames)
    left = None
    while left is None:
        print("press the spacebar to take a snapshot")
        with keyboard.Events() as events:
            for event in events:
                if event.key == keyboard.Key.space:
                    left = frames.get_fisheye_frame(1)
                    break
    left = np.asanyarray(left.get_data())
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.imshow('frame', left)
    cv2.resizeWindow('frame', 848, 800)
    cv2.setMouseCallback('frame', mouse_callback)
    cv2.waitKey(0)


if __name__ == "__main__":
    # execute only if run as a script
    main()

