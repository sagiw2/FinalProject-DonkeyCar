# FinalProject-DonkeyCar

Current Maintainer: Sagi Weiss <sagi953@gmail.com>


# Introduction

My final project for my B.Sc in Mechanical Engineering at BGU.

This project goal is "Remote operation of vehicles with an emphasis on dealing with communication delays."

My solution is a local navigation algorithm called - Point & Go.
The navigation commands in this method are given by clicking on a video feed provided by a camera mounted on the robot; the click position is then converted to a real-world location - "Target Location". After a target location is obtained, the robot can navigate without user assistance using a control loop based on the lateral dynamics of vehicles and a PID controller.

# Installation

## Donkey-Car
This repo runs on a [Donkey-Car](https://www.donkeycar.com/) vehicle. The setup is as follows:
* clone this repo
* Setting up your car provided by [Ori's Code](https://ori.codes/)
  * [building your car](https://ori.codes/hardware/)
  * [setting up your RC car](https://ori.codes/software/donkeycar-rc/)
*there is no need to clone the donkey repo as all files are present in this one (pandemic version)*

## Realsense2
In this project, the donkey car is equipped with 2 Intel realsense cameras, T265 and D435.
To set up the cameras, please follow [Realsense](https://github.com/IntelRealSense/librealsense) installation guide.

*note that some of the files need to be transferred to the car sub-directory to use the cameras with the car*

# Parts

There are 3 parts designed for this algorithm:
* pointandgo.py - runs the Point & Go algorithm *now named donkey_car_test after bug fixed will be merged back to pointandgo.py*
* T265_donkey.py - pulls the relevant data from the SLAM camera
* D435_donkey.py - pulls the relevant data from the Depth camera

# Usage

After completing the installation, the system is plug n play. Boot up the car and connect through an ssh server to it (need an -XC prefix to receive video data). After a video feed is received simply click on the screen and the car will travel to the desired location.

# TODO
* remove non-relevant code snippets left from testing
* separate the open-cv video feed from the pointandgo.py and move it to another part to speed up the algorithm code
* finish occupancy grid from depth image and integrate it to the algorithm
