"""Kinect v1 capture in a background thread.

Default depth format is DEPTH_REGISTERED: uint16 millimetres, 640x480, already
warped into the RGB camera frame, so pixel (u, v) in RGB == pixel (u, v) in depth.
0 means "no reading". Other modes: "mm" (depth-camera frame, mm) and "raw11"
(11-bit disparity, 2047 = invalid; NOT aligned to RGB).

Intrinsics (Nicolas Burrus calibration of a typical Kinect v1; per-unit error is
a few pixels, fine for 3 cm blocks). Registered depth lives in the RGB camera frame,
so the RGB set is used for it:
    RGB  : fx=529.2 fy=525.6 cx=328.9 cy=267.5   (default for registered/rgb)
    DEPTH: fx=594.2 fy=591.0 cx=339.3 cy=242.7   (used for "mm" / "raw11")
Camera frame for deproject(): x right, y down, z forward (optical, metres).
"""
import threading
import time

import numpy as np

HERE = __import__('os').path.dirname(__import__('os').path.abspath(__file__))

RGB_INTR = dict(fx=529.2, fy=525.6, cx=328.9, cy=267.5)
DEPTH_INTR = dict(fx=594.2, fy=591.0, cx=339.3, cy=242.7)


def depth_to_meters(depth, mode="registered"):
    """Raw frame -> float32 metres; invalid -> nan."""
    d = np.asarray(depth)
    if mode == "raw11":
        out = 1.0 / (d.astype(np.float32) * -0.0030711016 + 3.3309495161)
        out[(d >= 2047) | (out <= 0)] = np.nan
        return out
    out = d.astype(np.float32) / 1000.0
    out[d == 0] = np.nan
    return out


def deproject(u, v, depth_m, intr=RGB_INTR):
    """Pixel (u, v) + depth in metres -> camera-frame xyz (m). Works on arrays."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    z = np.asarray(depth_m, dtype=np.float64)
    return np.stack([(u - intr["cx"]) * z / intr["fx"],
                     (v - intr["cy"]) * z / intr["fy"], z], axis=-1)


class Kinect:
    def __init__(self, depth_mode="registered", open_timeout=5.0):
        self.depth_mode = depth_mode
        self.intr = DEPTH_INTR if depth_mode in ("mm", "raw11") else RGB_INTR
        self._open_timeout = open_timeout
        self._lock = threading.Lock()
        self._rgb = self._depth = None
        self._rgb_t = self._depth_t = 0.0
        self._n_rgb = self._n_depth = 0
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._error = None
        self._thread = None

    # -- lifecycle ---------------------------------------------------------
    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        threading.Thread(target=self._preview_loop, daemon=True).start()
        if not self._ready.wait(self._open_timeout) or self._error:
            self.stop()
            raise RuntimeError(f"Kinect failed to open: {self._error or 'timeout'}")
        return self

    def _preview_loop(self, period=0.2):
        """Share what the Kinect sees: RGB | depth written to control/live.jpg for kinect_live.py
        (only one process can open the device)."""
        import os
        import cv2
        out = os.path.join(os.path.dirname(HERE), "control", "live.jpg")
        tmp = out + ".tmp.jpg"
        while not self._stop.is_set():
            time.sleep(period)
            try:
                rgb, _, d, _ = self.get_frames()
                if rgb is None or d is None:
                    continue
                dm = self.depth_to_meters(d)
                dv = np.clip((np.nan_to_num(dm, nan=0) - 0.5) / 0.4 * 255, 0, 255).astype(np.uint8)
                dc = cv2.applyColorMap(255 - dv, cv2.COLORMAP_TURBO)
                dc[~np.isfinite(dm) | (dm <= 0)] = 0
                cv2.imwrite(tmp, np.hstack([cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), dc]))
                os.replace(tmp, out)
            except Exception:  # noqa: BLE001  the preview must never disturb the robot
                pass

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *a):
        self.stop()

    def _run(self):
        import freenect
        ctx = dev = None
        try:
            ctx = freenect.init()
            if freenect.num_devices(ctx) < 1:
                raise RuntimeError("no Kinect device found")
            dev = freenect.open_device(ctx, 0)
            if dev is None:
                raise RuntimeError("open_device failed (device held by another process?)")
            fmt = {"registered": freenect.DEPTH_REGISTERED, "mm": freenect.DEPTH_MM,
                   "raw11": freenect.DEPTH_11BIT}[self.depth_mode]
            freenect.set_depth_mode(dev, freenect.RESOLUTION_MEDIUM, fmt)
            freenect.set_video_mode(dev, freenect.RESOLUTION_MEDIUM, freenect.VIDEO_RGB)

            def on_depth(d, a, t):
                with self._lock:
                    self._depth, self._depth_t = a.copy(), time.time()
                    self._n_depth += 1

            def on_video(d, a, t):
                with self._lock:
                    self._rgb, self._rgb_t = a.copy(), time.time()
                    self._n_rgb += 1

            freenect.set_depth_callback(dev, on_depth)
            freenect.set_video_callback(dev, on_video)
            freenect.start_depth(dev)
            freenect.start_video(dev)
            self._ready.set()

            def body(*_):
                if self._stop.is_set():
                    raise freenect.Kill
            try:
                freenect.base_runloop(ctx, body)
            except freenect.Kill:
                pass
        except Exception as e:  # noqa: BLE001
            self._error = e
            self._ready.set()
        finally:
            try:
                if dev is not None:
                    freenect.stop_depth(dev)
                    freenect.stop_video(dev)
                    freenect.close_device(dev)
                if ctx is not None:
                    freenect.shutdown(ctx)
            except Exception:  # noqa: BLE001
                pass

    # -- access ------------------------------------------------------------
    def get_rgb(self):
        """-> (rgb HxWx3 uint8 RGB order, timestamp) or (None, 0)."""
        with self._lock:
            return (None if self._rgb is None else self._rgb.copy()), self._rgb_t

    def get_depth(self):
        """-> (raw depth array, timestamp) or (None, 0)."""
        with self._lock:
            return (None if self._depth is None else self._depth.copy()), self._depth_t

    def get_frames(self):
        with self._lock:
            return (None if self._rgb is None else self._rgb.copy(), self._rgb_t,
                    None if self._depth is None else self._depth.copy(), self._depth_t)

    def wait_for_frames(self, timeout=5.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            f = self.get_frames()
            if f[0] is not None and f[2] is not None:
                return f
            time.sleep(0.02)
        raise TimeoutError("no frames from Kinect")

    def counts(self):
        with self._lock:
            return self._n_rgb, self._n_depth

    def depth_to_meters(self, depth):
        return depth_to_meters(depth, self.depth_mode)

    def deproject(self, u, v, depth_m):
        return deproject(u, v, depth_m, self.intr)
