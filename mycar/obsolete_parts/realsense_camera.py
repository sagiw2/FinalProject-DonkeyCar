import pyrealsense2 as rs


class RealsenseCam(object):

    def __init__(self):
        self.running = True
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.pose)
        self.pipe.start(cfg)
        self.frames = None

    def update(self):
        while self.running:
            self.frames = self.pipe.wait_for_frames()

    def run_threaded(self):
        if self.frames is not None:
            pose = self.frames.get_pose_frame()
            data = pose.get_pose_data()
            return data.translation

    def shutdown(self):
        self.running = False
        self.pipe.stop()
