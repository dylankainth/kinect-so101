"""Live Kinect v1 viewer: depth + RGB via libfreenect callbacks. Press q/Esc to quit."""
import cv2
import numpy as np
import freenect

latest = {}
done = False


def on_depth(dev, depth, timestamp):
    latest["depth"] = depth.copy()


def on_video(dev, video, timestamp):
    latest["rgb"] = video.copy()


def body(dev, ctx):
    global done
    if "depth" in latest:
        d = np.clip(latest["depth"], 0, 1023) // 4
        cv2.imshow("depth", cv2.applyColorMap(d.astype(np.uint8), cv2.COLORMAP_JET))
    if "rgb" in latest:
        cv2.imshow("rgb", cv2.cvtColor(latest["rgb"], cv2.COLOR_RGB2BGR))
    if cv2.waitKey(1) in (ord("q"), 27):
        raise freenect.Kill


freenect.runloop(depth=on_depth, video=on_video, body=body)
cv2.destroyAllWindows()
