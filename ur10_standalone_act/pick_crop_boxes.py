"""Live crop-box (ROI) picker for the standalone UR10 follower cameras.

No JSON. Cameras are hardcoded here (same 3 RealSense serials as
record_ur10_follower.py / eval_ur10_follower.py). For each camera you draw one
rectangle on the live NATIVE-resolution feed; a side window shows the crop
resized to 224x224 (exactly what the policy sees). On exit it prints a
`CROP_BOXES = {...}` Python snippet in (top, left, height, width) order — paste it
over the `CROP_BOXES` dict in record_ur10_follower.py and eval_ur10_follower.py.

Controls per camera window:
    drag left mouse  -- draw rectangle
    c                -- confirm and move to the next camera
    r                -- reset the current selection
    s                -- skip (no crop for this camera; full frame is used)
    q / ESC          -- abort

Run (lerobot conda env):
    python ur10_standalone_act/pick_crop_boxes.py
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from lerobot.cameras import make_cameras_from_configs
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

# Same cameras as record_ur10_follower.py / eval_ur10_follower.py (keep in sync).
CAMERAS = {
    "front": RealSenseCameraConfig(serial_number_or_name="409122272284", fps=30, width=640, height=480),
    "side":  RealSenseCameraConfig(serial_number_or_name="409122273078", fps=30, width=640, height=480),
    "wrist": RealSenseCameraConfig(serial_number_or_name="323622272232", fps=30, width=640, height=480),
}
RESIZE = (224, 224)  # what the policy sees after cropping


def _reset_realsense() -> None:
    """Hardware-reset the configured RealSense devices (D405s can get stuck)."""
    serials = [c.serial_number_or_name for c in CAMERAS.values()]
    try:
        import pyrealsense2 as rs
    except Exception:
        return
    ctx = rs.context()
    reset_any = False
    for dev in ctx.query_devices():
        if dev.get_info(rs.camera_info.serial_number) in serials:
            print(f"hardware_reset {dev.get_info(rs.camera_info.serial_number)}")
            dev.hardware_reset()
            reset_any = True
    if reset_any:
        time.sleep(5.0)


class _RoiSelector:
    def __init__(self) -> None:
        self.drawing = False
        self.start = None
        self.current = None
        self.roi = None  # (top, left, height, width)

    def callback(self, event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start = (x, y)
            self.current = (x, y)
            self.roi = None
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.current = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.current = (x, y)
            x0, y0 = self.start
            x1, y1 = x, y
            top, bottom = sorted((y0, y1))
            left, right = sorted((x0, x1))
            self.roi = (top, left, bottom - top, right - left)

    def reset(self) -> None:
        self.__init__()


def _pick_roi(name: str, cam):
    win = f"ROI [{name}] (drag, c=confirm, r=reset, s=skip, q=abort)"
    prev = f"224 preview [{name}]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    sel = _RoiSelector()
    cv2.setMouseCallback(win, sel.callback)
    print(f"\n[{name}] drag to draw, c=confirm, r=reset, s=skip, q/ESC=abort")

    while True:
        frame = cam.async_read()
        if frame is None:
            continue
        disp = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if frame.ndim == 3 else frame.copy()

        if sel.drawing and sel.start and sel.current:
            cv2.rectangle(disp, sel.start, sel.current, (0, 255, 0), 2)
        elif sel.roi is not None:
            top, left, h, w = sel.roi
            cv2.rectangle(disp, (left, top), (left + w, top + h), (0, 255, 0), 2)
            cv2.putText(disp, f"top={top} left={left} h={h} w={w}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # Live 224x224 preview of exactly what the policy will see.
            if h > 1 and w > 1:
                crop = frame[top:top + h, left:left + w]
                if crop.size:
                    p = cv2.resize(crop, RESIZE)
                    cv2.imshow(prev, cv2.cvtColor(p, cv2.COLOR_RGB2BGR) if p.ndim == 3 else p)

        cv2.imshow(win, disp)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            cv2.destroyWindow(win); cv2.destroyWindow(prev)
            return "ABORT"
        if key == ord("r"):
            sel.reset(); cv2.destroyWindow(prev)
        if key == ord("s"):
            cv2.destroyWindow(win); cv2.destroyWindow(prev)
            return None
        if key == ord("c") and sel.roi is not None and sel.roi[2] > 1 and sel.roi[3] > 1:
            cv2.destroyWindow(win); cv2.destroyWindow(prev)
            return sel.roi


def main() -> None:
    _reset_realsense()
    cams = make_cameras_from_configs(CAMERAS)
    boxes: dict[str, tuple[int, int, int, int]] = {}
    try:
        for cam in cams.values():
            cam.connect()
        for name, cam in cams.items():
            roi = _pick_roi(name, cam)
            if roi == "ABORT":
                print("aborted; nothing written")
                return
            if roi is None:
                print(f"[{name}] skipped (no crop)")
                continue
            boxes[name] = roi
    finally:
        for cam in cams.values():
            try:
                cam.disconnect()
            except Exception:
                pass
        cv2.destroyAllWindows()

    print("\n--- paste this over CROP_BOXES in record_ur10_follower.py / eval_ur10_follower.py ---\n")
    print("CROP_BOXES = {")
    for name in CAMERAS:
        if name in boxes:
            t, l, h, w = boxes[name]
            print(f'    "{name}": ({t}, {l}, {h}, {w}),')
        else:
            print(f'    # "{name}": skipped (no crop)')
    print("}")


if __name__ == "__main__":
    main()
