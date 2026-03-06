"""
Reads frames from a USB camera.
Saves periodic screenshots and a per-frame average RGB txt for the center crop box.
"""

import os
import cv2

# import h5py
import numpy as np
import time

from datetime import datetime, timezone, timedelta

EPOCH_ORIGIN = datetime(2019, 1, 1, 0, 0, 0)


def to_polarspec_timestamp(dt=None, epoch=EPOCH_ORIGIN, prefix=""):
    if dt is None:
        dt = datetime.now(timezone.utc).replace(tzinfo=None)
    delta = dt - epoch
    return f"{prefix}{int(delta.total_seconds())}"


# def list_available_cameras(max_index=10):
#     available = []
#     for i in range(max_index):
#         cap = cv2.VideoCapture(i)
#         if cap.isOpened():
#             available.append({
#                 "index": i,
#                 "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
#                 "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
#                 "fps": cap.get(cv2.CAP_PROP_FPS),
#             })
#             cap.release()
#     return available


def add_frame_axes(
    frame_bgr,
    tick_interval=250,
    tick_length=6,
    margin=60,
    color=(0, 0, 0),
    bg_color=(255, 255, 255),
    font_scale=0.5,
    thickness=1,
):
    h, w = frame_bgr.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    canvas = np.full((h + margin, w + margin, 3), bg_color, dtype=np.uint8)
    canvas[margin : margin + h, margin : margin + w] = frame_bgr
    img_left, img_right = margin, margin + w
    img_top, img_bottom = margin, margin + h
    cv2.line(canvas, (img_left, img_top), (img_left, img_bottom), color, thickness)
    cv2.line(canvas, (img_left, img_top), (img_right, img_top), color, thickness)
    for x in range(0, w + 1, tick_interval):
        cx = x + margin
        cv2.line(canvas, (cx, img_top - tick_length), (cx, img_top), color, thickness)
        label = str(x)
        (lw, lh), _ = cv2.getTextSize(label, font, font_scale, thickness)
        cv2.putText(
            canvas,
            label,
            (cx - lw // 2, img_top - tick_length - 4),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
    for y in range(0, h + 1, tick_interval):
        cy = y + margin
        cv2.line(canvas, (img_left - tick_length, cy), (img_left, cy), color, thickness)
        label = str(y)
        (lw, lh), _ = cv2.getTextSize(label, font, font_scale, thickness)
        cv2.putText(
            canvas,
            label,
            (img_left - tick_length - lw - 4, cy + lh // 2),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
    return canvas


def init_avg_rgb_csv(output_dir, name):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}-avg-rgb.csv")
    f = open(path, "w")
    f.write("frame,timestamp,avg r,avg g,avg b\n")
    f.flush()
    print(f"  {name}-avg_rgb.csv opened: {path}")
    return f


def append_avg_rgb(csv_file, frame_rgb, frame_idx, ts, box_w, box_h, cx, cy):
    crop = frame_rgb[
        cy - box_h // 2 : cy + box_h // 2, cx - box_w // 2 : cx + box_w // 2
    ]
    avg_r, avg_g, avg_b = crop.mean(axis=(0, 1)).round(2)
    csv_file.write(f"{frame_idx},{ts},{avg_r},{avg_g},{avg_b}\n")
    csv_file.flush()


def save_screenshot(
    frame_bgr, frame_count, output_dir, box_w, box_h, center_x, center_y, axes_margin=60
):
    os.makedirs(output_dir, exist_ok=True)
    frame = add_frame_axes(frame_bgr, margin=axes_margin)
    cx = center_x if center_x is not None else frame_bgr.shape[1] // 2
    cy = center_y if center_y is not None else frame_bgr.shape[0] // 2
    top_left = (cx - box_w // 2 + axes_margin, cy - box_h // 2 + axes_margin)
    bottom_right = (cx + box_w // 2 + axes_margin, cy + box_h // 2 + axes_margin)
    cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 1)
    path = os.path.join(
        output_dir, f"frame_{frame_count:04d}_{to_polarspec_timestamp()}.png"
    )
    cv2.imwrite(path, frame)
    cv2.imwrite(os.path.join(output_dir, "latest.png"), frame)
    print(f"  Screenshot saved: {path}")


# def init_hdf5(path, box_w, box_h, camera_index, frame_width, frame_height):
#     ...

# def append_frame_hdf5(hdf5_file, crop, timestamp):
#     ...


def capture_frames(
    camera_index=0,
    test_length=10.0,
    show_preview=False,
    capture_interval=1,
    screenshot_interval=None,
    # screenshot_dir="./Results/rgb",
    output_dir="./Results",
    name="sample",
    box_w=24,
    box_h=24,
    center_x=None,
    center_y=None,
):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera at index {camera_index}.")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(
        f"Camera opened: {w}x{h} | capturing {int(test_length)} frame(s) over {test_length:.0f}s"
    )

    # hdf5_file = None
    # if hdf5_path and box_w and box_h:
    #     os.makedirs(os.path.dirname(hdf5_path), exist_ok=True)
    #     hdf5_file = init_hdf5(hdf5_path, box_w, box_h, camera_index, w, h)
    #     print(f"HDF5 file opened: {hdf5_path}")

    screenshot_dir = os.path.join(output_dir, f"{name}-screencaptures")
    csv_file = init_avg_rgb_csv(output_dir, name)

    # temporal_frames = []
    start_time = time.monotonic()
    next_capture = start_time

    cx = center_x if center_x is not None else w // 2
    cy = center_y if center_y is not None else h // 2

    frame_count = 0

    try:
        while True:
            now = time.monotonic()
            elapsed = now - start_time
            if elapsed >= test_length:
                break

            ret, frame_bgr = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            if now >= next_capture:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                # temporal_frames.append(frame_rgb.reshape(-1, 3).tolist())
                next_capture += capture_interval
                frame_count += 1
                ts = int(to_polarspec_timestamp())

                # if hdf5_file and box_w and box_h:
                #     crop = frame_rgb[cy - box_h//2 : cy + box_h//2,
                #                      cx - box_w//2 : cx + box_w//2]
                #     append_frame_hdf5(hdf5_file, crop, ts)

                append_avg_rgb(
                    csv_file, frame_rgb, frame_count, ts, box_w, box_h, cx, cy
                )

                if frame_count == 1 or (
                    screenshot_interval and frame_count % screenshot_interval == 0
                ):
                    save_screenshot(
                        frame_bgr, frame_count, screenshot_dir, box_w, box_h, cx, cy
                    )

                print(
                    f"  Frame {frame_count:4d} | t={elapsed:5.1f}s | {test_length - elapsed:.1f}s remaining"
                )

                if show_preview:
                    cv2.imshow("USB Camera Preview (Q to quit)", frame_bgr)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("User quit.")
                        break
    finally:
        cap.release()
        csv_file.close()
        print("avg_rgb.txt closed.")
        if show_preview:
            cv2.destroyAllWindows()
        # if hdf5_file:
        #     hdf5_file["frames"].attrs["count"] = frame_count
        #     hdf5_file.close()
        #     print(f"HDF5 file closed: {hdf5_path}")

    print(f"\nCaptured {frame_count} frames over {test_length:.0f}s.")


# def print_frames(frames, f):
#     ...


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":

    NAME = "R-C-5-6"  # the sample name -- CHANGE THIS EVERY EXPERIMENT

    CAMERA_INDEX = 1
    SHOW_PREVIEW = False

    BOX_W = 24
    BOX_H = 24
    CENTER_X = 300
    CENTER_Y = 275

    SCHEDULE = {"hours": 5, "minutes": 0, "seconds": 0}
    TEST_LENGTH = timedelta(**SCHEDULE).total_seconds()

    CAPTURE_INTERVAL = 4  # capture avg RGB every 4 seconds
    SCREENSHOT_EVERY = 75  # every 75 frames × 4s = every 5 minutes

    # print(list_available_cameras())

    capture_frames(
        camera_index=CAMERA_INDEX,
        test_length=TEST_LENGTH,
        show_preview=SHOW_PREVIEW,
        capture_interval=CAPTURE_INTERVAL,
        screenshot_interval=SCREENSHOT_EVERY,
        output_dir="./Results",
        name=NAME,
        box_w=BOX_W,
        box_h=BOX_H,
        center_x=CENTER_X,
        center_y=CENTER_Y,
    )

    # if frames:
    #     with open("./Results/rgb/out.txt", "w+") as f:
    #         twod_frames = []
    #         for frame in frames:
    #             twod_frame = [[]]
    #             row_index = 0
    #             for idx, pixel in enumerate(frame):
    #                 twod_frame[row_index].append(pixel)
    #                 if idx % FRAME_WIDTH == 0 and idx != 0:
    #                     row_index += 1
    #                     twod_frame.append([])
    #             twod_frames.append(twod_frame)
    #         print_frames(twod_frames, f)

    #     with open("./Results/rgb/centered.txt", "w+") as f:
    #         centered_frames = []
    #         for frame in twod_frames:
    #             start_row = CENTER_Y - BOX_H // 2
    #             end_row   = CENTER_Y + BOX_H // 2
    #             start_col = CENTER_X - BOX_W // 2
    #             end_col   = CENTER_X + BOX_W // 2
    #             box = [row[start_col:end_col] for row in frame[start_row:end_row]]
    #             centered_frames.append(box)
    #         print_frames(centered_frames, f)
