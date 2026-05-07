"""
Reads frames from a USB camera.
Saves periodic screenshots and a per-frame average RGB txt for the center crop box.
"""

import os
import cv2
import numpy as np
import time

from datetime import datetime, timezone, timedelta

EPOCH_ORIGIN = datetime(2019, 1, 1, 0, 0, 0)


def to_polarspec_timestamp(dt=None, epoch=EPOCH_ORIGIN, prefix=""):
    if dt is None:
        dt = datetime.now(timezone.utc).replace(tzinfo=None)
    delta = dt - epoch
    return f"{prefix}{int(delta.total_seconds())}"


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
    base_path = os.path.join(output_dir, f"{name}-avg-rgb.csv")

    # "test" runs always overwrite; real sample names get a counter to avoid clobbering data
    path = base_path
    if name != "test":
        counter = 1
        while os.path.exists(path):
            path = os.path.join(output_dir, f"{name}-avg-rgb_{counter}.csv")
            counter += 1

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


def capture_frames(
    camera_index=0,
    test_length=10.0,
    show_preview=False,
    capture_interval=1,
    screenshot_interval=None,
    output_dir="./Results",
    name="sample",
    box_w=24,
    box_h=24,
    center_x=None,
    center_y=None,
):
    cap = cv2.VideoCapture(
        "/dev/video0", cv2.CAP_V4L2
    )  # ← use device path + V4L2 backend
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # ← lock resolution
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera at index {camera_index}.")

    # ── Warm-up: let AWB and auto-exposure settle before recording starts ──────
    #  Without this, the first screenshot captures before the camera's auto white
    # balance and exposure have converged, producing a warmer/different tone than
    # all subsequent screenshots and skewing early RGB readings.
    WARMUP_SECONDS = 3
    print(f"  Warming up camera for {WARMUP_SECONDS}s...")
    warmup_deadline = time.monotonic() + WARMUP_SECONDS
    while time.monotonic() < warmup_deadline:
        cap.read()  # discard frames
    print("  Camera ready.")
    # ──────────────────────────────────────────────────────────────────────────

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(
        f"Camera opened: {w}x{h} | capturing {int(test_length)} frame(s) over {test_length:.0f}s"
    )

    screenshot_dir = os.path.join(output_dir, f"{name}-screencaptures")
    csv_file = init_avg_rgb_csv(output_dir, name)

    start_time = time.monotonic()
    next_capture = start_time

    cx = center_x if center_x is not None else w // 2
    cy = center_y if center_y is not None else h // 2

    frame_count = 0
    last_frame_bgr = None  # track the last captured frame

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
                next_capture += capture_interval
                frame_count += 1
                ts = int(to_polarspec_timestamp())

                last_frame_bgr = frame_bgr  # update on every capture

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
        # save the last frame with a timestamp after the loop ends
        if last_frame_bgr is not None:
            save_screenshot(
                last_frame_bgr, frame_count, screenshot_dir, box_w, box_h, cx, cy
            )
            print("  Final timestamped screenshot saved.")

        cap.release()
        csv_file.close()
        print("avg_rgb.txt closed.")
        if show_preview:
            cv2.destroyAllWindows()

    print(f"\nCaptured {frame_count} frames over {test_length:.0f}s.")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # NAME = "test"
    NAME = "SCS-C-1-3"  # the sample name -- CHANGE THIS EVERY EXPERIMENT

    CAMERA_INDEX = 0
    SHOW_PREVIEW = False

    BOX_W = 24
    BOX_H = 24
    CENTER_X = 675
    CENTER_Y = 300

    SCHEDULE = {"hours": 5, "minutes": 0, "seconds": 0}
    test_length = timedelta(**SCHEDULE).total_seconds()

    CAPTURE_INTERVAL = 6  # capture avg RGB every 6 seconds
    SCREENSHOT_INTERVAL_MINUTES = 1  # take a screenshot from the camera every N minutes
    screenshot_every = round((SCREENSHOT_INTERVAL_MINUTES * 60) / CAPTURE_INTERVAL)

    capture_frames(
        camera_index=CAMERA_INDEX,
        test_length=test_length,
        show_preview=SHOW_PREVIEW,
        capture_interval=CAPTURE_INTERVAL,
        screenshot_interval=screenshot_every,
        output_dir="./Results",
        name=NAME,
        box_w=BOX_W,
        box_h=BOX_H,
        center_x=CENTER_X,
        center_y=CENTER_Y,
    )
