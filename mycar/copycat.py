import time
import csv
import numpy as np

class copy(object):

    def __init__(self):
        self.running = True
        self.data = np.array([0.0,0.0])


    def update(self):
        """calibrate the camera and return an undistorted image for the user interface """
        while self.running:
            pass

 
    def run_threaded(self, angle, throttle):
        """main part loop"""
            
        # self.data = np.append(self.data,[[angle, throttle]], axis=1)
        x = np.array([angle, throttle])
        if x[0] is not None:
            self.data = np.vstack((self.data, x))

        # with open("drive_commands.csv", "a+") as myfile:
        #     myfile.write(str(angle))
        #     myfile.write(',')
        #     myfile.write(str(throttle))
        #     myfile.write('\n')


    def run(self):
        return self.run_threaded()

    def shutdown(self):
        """terminate part"""
        self.running = False
        np.savetxt("drive_commands.csv", self.data, delimiter=",")
        time.sleep(5)


class paste(object):

    def __init__(self):
        self.running = True
        self.index = 0
        self.my_data = np.genfromtxt('drive_commands.csv', delimiter=',')
        self.idx = 1


    def update(self):
        """calibrate the camera and return an undistorted image for the user interface """
        while self.running:
            pass

 
    def run_threaded(self):
        """main part loop"""
            
        # with open("drive_commands.csv") as myfile:
        #     spamreader = csv.reader(myfile, delimiter=',')
        #     interestingrows=[row for idx, row in enumerate(spamreader) if idx == self.index]

        #     if np.mod(self.idx,2):
        #         angle = float(interestingrows[0][0])
        #         throttle = float(interestingrows[0][1])
        #         self.index +=1
        #     else:
        #         angle = 0.0
        #         throttle = 0.0
        #     self.idx +=1

        angle = 0.0
        throttle = 0.0
        if self.index < np.size(self.my_data, 0):
            angle = self.my_data[self.index][0]
            throttle = self.my_data[self.index][1]
            if np.mod(self.idx,2):
                self.index += 1
        self.idx += 1
        return angle, throttle

    def run(self):
        return self.run_threaded()
            


    def shutdown(self):
        """terminate part"""
        self.running = False

        time.sleep(0.2)



