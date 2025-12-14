from ml.classifier import PlastiTraceClassifier
from ui.camera_loop import CameraLoop

if __name__ == "__main__":
    clf = PlastiTraceClassifier("models/plastitrace.pth")
    CameraLoop(classifier=clf).run()
