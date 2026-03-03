"""
Reads frames from a USB camera and builds a temporal list of RGB values per pixel.
"""

import os
import cv2
import h5py
import numpy as np
import time

from datetime import datetime, timezone, timedelta

EPOCH_ORIGIN = datetime(2019, 1, 1, 0, 0, 0)  # January 1, 2019 00:00:00 UTC
# from polarspec logic: https://github.com/ayisakov/polarspec/blob/40684015acaf392959d8c877f10f04c0bda6095a/polarspec/StepRotateAndCapture.cpp#L149


def to_polarspec_timestamp(
    dt: datetime | None = None,
    epoch: datetime = EPOCH_ORIGIN,
    prefix: str = "",
) -> str:
    """
    Convert a datetime to a polarspec-style timestamp (seconds since a given epoch).

    Args:
        dt:     datetime to convert. Defaults to current UTC time if None.
        epoch:  The origin datetime to measure from. Defaults to 2019-01-01 UTC.
        prefix: String prefix for the timestamp. Defaults to empty string.

    Returns:
        A string in the format "<prefix><seconds>", e.g. "RR221688546".
    """
    if dt is None:
        dt = datetime.now(timezone.utc).replace(tzinfo=None)
    delta = dt - epoch
    return f"{prefix}{int(delta.total_seconds())}"


def list_available_cameras(max_index: int = 10) -> list[dict]:
    """
    Probe camera indices 0..max_index-1 and return info on every one that opens.

    Args:
        max_index: How many indices to probe (default 10).

    Returns:
        A list of dicts, one per working camera:
            {
                "index":  int,   # OpenCV camera index
                "width":  int,   # frame width  in pixels
                "height": int,   # frame height in pixels
                "fps":    float, # reported frame rate
            }
    """
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(
                {
                    "index": i,
                    "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    "fps": cap.get(cv2.CAP_PROP_FPS),
                }
            )
            cap.release()
    return available


def add_frame_axes(
    frame_bgr,
    tick_interval: int = 250,
    tick_length: int = 6,
    margin: int = 60,
    color: tuple[int, int, int] = (0, 0, 0),
    bg_color: tuple[int, int, int] = (255, 255, 255),
    font_scale: float = 0.5,
    thickness: int = 1,
):
    """
    Add pixel coordinate axes in a white border around the frame, styled like a plot.
    The image is untouched — axes, ticks, and labels live entirely in the border.
    (0, 0) is at the top-left corner where the two axes meet.

    Args:
        frame_bgr:     BGR frame from OpenCV (numpy array).
        tick_interval: Pixels between each tick mark and label.
        tick_length:   Length of each tick mark in pixels (protrudes into the border).
        margin:        Width of the border added to the left and top of the image.
        color:         BGR color for axis lines, ticks, and labels. Default black.
        bg_color:      BGR background color for the border area. Default white.
        font_scale:    Font size for tick labels.
        thickness:     Line thickness for axis lines and ticks.

    Returns:
        A new image with the original frame in the bottom-right and axes in the border.
        The original frame pixels are completely unmodified.
    """
    h, w = frame_bgr.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    canvas = np.full((h + margin, w + margin, 3), bg_color, dtype=np.uint8)
    canvas[margin : margin + h, margin : margin + w] = frame_bgr

    img_left = margin
    img_right = margin + w
    img_top = margin
    img_bottom = margin + h

    cv2.line(
        canvas, (img_left, img_top), (img_left, img_bottom), color, thickness
    )  # Y axis
    cv2.line(
        canvas, (img_left, img_top), (img_right, img_top), color, thickness
    )  # X axis

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


def save_screenshot(
    frame_bgr,
    frame_count: int,
    output_dir: str = "./Results/rgb",
    annotate_axes: bool = False,
    box_w: int | None = None,
    box_h: int | None = None,
    axes_margin: int = 60,
    center_x=None,
    center_y=None,
) -> str:
    """
    Save a BGR frame as a PNG with a polarspec-format timestamp in the filename.

    Args:
        frame_bgr:     BGR frame from OpenCV (numpy array).
        frame_count:   Current frame number, used in the filename.
        output_dir:    Directory to save the PNG.
        annotate_axes: If True, draw pixel coordinate axes around the image.
        box_w:         Width of the center crop box. If provided with box_h and
                       annotate_axes=True, draws a red rectangle showing the crop area.
        box_h:         Height of the center crop box.
        axes_margin:   Must match the margin used in add_frame_axes (default 60).

    Returns:
        The full path of the saved file.
    """
    os.makedirs(output_dir, exist_ok=True)

    if annotate_axes:
        frame = add_frame_axes(frame_bgr, margin=axes_margin)

        if box_w is not None and box_h is not None:
            h, w = frame_bgr.shape[:2]
            cx = center_x if center_x is not None else w // 2
            cy = center_y if center_y is not None else h // 2
            start_row = cy - box_h // 2
            end_row = cy + box_h // 2
            start_col = cx - box_w // 2
            end_col = cx + box_w // 2
            top_left = (start_col + axes_margin, start_row + axes_margin)
            bottom_right = (end_col + axes_margin, end_row + axes_margin)
            cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 1)
    else:
        frame = frame_bgr

    timestamp = to_polarspec_timestamp()
    # timestamp = datetime.now().isoformat()
    path = os.path.join(output_dir, f"frame_{frame_count:04d}_{timestamp}.png")
    cv2.imwrite(path, frame)
    cv2.imwrite(os.path.join(output_dir, "latest.png"), frame)
    print(f"  Screenshot saved: {path}")

    print(f"  Screenshot saved: {path}")
    return path


def init_hdf5(
    path: str,
    box_w: int,
    box_h: int,
    camera_index: int,
    frame_width: int,
    frame_height: int,
) -> h5py.File:
    """
    Create and initialise an HDF5 file for incremental frame storage.
    Creates two resizable datasets under /frames: crop and timestamps.

    Args:
        path:         Full path to the .h5 file to create.
        box_w:        Width of the centered crop box in pixels.
        box_h:        Height of the centered crop box in pixels.
        camera_index: Camera index used for this capture session.
        frame_width:  Full frame width in pixels.
        frame_height: Full frame height in pixels.

    Returns:
        An open h5py.File handle. Caller is responsible for closing it.
    """
    f = h5py.File(path, "w")

    # Top-level metadata
    f.attrs["camera_index"] = camera_index
    f.attrs["frame_width"] = frame_width
    f.attrs["frame_height"] = frame_height
    f.attrs["box_w"] = box_w
    f.attrs["box_h"] = box_h
    # f.attrs["start_timestamp"] = datetime.now().isoformat()
    f.attrs["start_timestamp"] = to_polarspec_timestamp()

    grp = f.create_group("frames")

    # Resizable dataset for crop data: shape (0, box_h, box_w, 3), grows along axis 0
    grp.create_dataset(
        "crop",
        shape=(0, box_h, box_w, 3),
        maxshape=(None, box_h, box_w, 3),
        dtype=np.uint8,
        chunks=(1, box_h, box_w, 3),  # one chunk per frame for efficient appending
    )

    # Resizable dataset for polarspec timestamps: shape (0,), grows along axis 0
    grp.create_dataset(
        "timestamps",
        shape=(0,),
        maxshape=(None,),
        dtype=np.int64,
    )

    return f


def append_frame_hdf5(
    hdf5_file: h5py.File,
    crop: np.ndarray,
    timestamp: int,
) -> None:
    """
    Append a single crop frame and its timestamp to an open HDF5 file.

    Args:
        hdf5_file: Open h5py.File handle (from init_hdf5).
        crop:      Numpy array of shape (box_h, box_w, 3), dtype uint8.
        timestamp: Polarspec timestamp (integer seconds since 2019-01-01 UTC).
    """
    ds_crop = hdf5_file["frames/crop"]
    ds_ts = hdf5_file["frames/timestamps"]

    n = ds_crop.shape[0]
    ds_crop.resize(n + 1, axis=0)
    ds_ts.resize(n + 1, axis=0)

    ds_crop[n] = crop
    ds_ts[n] = timestamp
    hdf5_file.flush()  # write to disk immediately — crash-safe


def capture_frames(
    camera_index: int = 0,
    test_length: float = 10.0,
    show_preview: bool = True,
    screenshot_interval: int | None = None,
    screenshot_dir: str = "./Results/rgb",
    box_w: int | None = None,
    box_h: int | None = None,
    hdf5_path: str | None = None,
    center_x=None,
    center_y=None,
) -> list[list[list[int]]]:
    """
    Capture one frame per second from a USB camera for a fixed duration.
    Every `screenshot_interval` frames, saves a .png with a polarspec timestamp.
    If hdf5_path is provided, the centered crop of every frame is written
    incrementally to an HDF5 file.

    Args:
        camera_index:        Index of the USB camera (0 = default/first camera).
        test_length:         Total recording duration in seconds.
        show_preview:        Whether to show a live preview window.
        screenshot_interval: Save a PNG every N captured frames. None disables screenshots.
        screenshot_dir:      Directory to save PNG screenshots.
        box_w:               Width of center crop box.
        box_h:               Height of center crop box.
        hdf5_path:           If provided, write crop data incrementally to this .h5 file.

    Returns:
        temporal_frames: A list of frames, where each frame is a flat list of
                         [R, G, B] values for every pixel in row-major order.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera at index {camera_index}.")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(test_length)
    print(
        f"Camera opened: {w}x{h} | capturing {total_frames} frame(s) over {test_length:.0f}s"
    )

    # Open HDF5 file if requested
    hdf5_file = None
    if hdf5_path and box_w and box_h:
        os.makedirs(os.path.dirname(hdf5_path), exist_ok=True)
        hdf5_file = init_hdf5(hdf5_path, box_w, box_h, camera_index, w, h)
        print(f"HDF5 file opened: {hdf5_path}")

    temporal_frames: list[list[list[int]]] = []
    start_time = time.monotonic()
    next_capture = start_time

    cx = center_x if center_x is not None else w // 2
    cy = center_y if center_y is not None else h // 2

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
                temporal_frames.append(frame_rgb.reshape(-1, 3).tolist())
                next_capture += 1.0
                frame_count = len(temporal_frames)
                ts = int(to_polarspec_timestamp())
                # ts = int(datetime.now().timestamp())

                # Append centered crop to HDF5
                if hdf5_file and box_w and box_h:
                    crop = frame_rgb[
                        cy - box_h // 2 : cy + box_h // 2,
                        cx - box_w // 2 : cx + box_w // 2,
                    ]
                append_frame_hdf5(hdf5_file, crop, ts)

                if screenshot_interval and frame_count % screenshot_interval == 0:
                    save_screenshot(
                        frame_bgr,
                        frame_count,
                        screenshot_dir,
                        annotate_axes=True,
                        box_w=box_w,
                        box_h=box_h,
                        center_x=center_x,
                        center_y=center_y,
                    )

                remaining = test_length - elapsed
                print(
                    f"  Frame {frame_count:4d}/{total_frames} | t={elapsed:5.1f}s | {remaining:.1f}s remaining"
                )

            if show_preview:
                cv2.imshow("USB Camera Preview (Q to quit)", frame_bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("User quit.")
                    break
    finally:
        cap.release()
        if show_preview:
            cv2.destroyAllWindows()
        if hdf5_file:
            hdf5_file["frames"].attrs["count"] = len(temporal_frames)
            hdf5_file.close()
            print(f"HDF5 file closed: {hdf5_path}")

    print(f"\nCaptured {len(temporal_frames)} frames over {test_length:.0f}s.")
    return temporal_frames


def print_frames(frames, f):
    for frame in frames:
        for row in frame:
            for pixel in row:
                r, g, b = pixel
                f.write(f"({r},{g},{b})")
            f.write("\n")
        f.write("\n\n")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    CAMERA_INDEX = 1
    MAX_FRAMES = 50
    SHOW_PREVIEW = False
    FRAME_WIDTH = 640

    # Center crop box — defined once, used everywhere
    BOX_W = 24
    BOX_H = 24
    CENTER_X = 690
    CENTER_Y = 460

    SCHEDULE = {"hours": 0, "minutes": 0, "seconds": 5}
    TEST_LENGTH = timedelta(**SCHEDULE).total_seconds()

    # print(list_available_cameras())  # look for 720p

    frames = capture_frames(
        camera_index=CAMERA_INDEX,
        test_length=TEST_LENGTH,
        show_preview=SHOW_PREVIEW,
        screenshot_interval=5,
        box_w=BOX_W,
        box_h=BOX_H,
        hdf5_path="./Results/rgb/capture.h5",
        center_x=CENTER_X,
        center_y=CENTER_Y,
    )

    if frames:

        with open("./Results/rgb/out.txt", "w+") as f:
            twod_frames = []
            for frame in frames:
                twod_frame = [[]]
                row_index = 0
                for idx, pixel in enumerate(frame):
                    twod_frame[row_index].append(pixel)
                    if idx % FRAME_WIDTH == 0 and idx != 0:
                        row_index += 1
                        twod_frame.append([])
                twod_frames.append(twod_frame)

            print_frames(twod_frames, f)

        with open("./Results/rgb/centered.txt", "w+") as f:
            centered_frames = []
            for frame in twod_frames:
                # ▼▼▼ CHANGE 8: use CENTER_X/CENTER_Y instead of frame center
                start_row = CENTER_Y - BOX_H // 2
                end_row = CENTER_Y + BOX_H // 2
                start_col = CENTER_X - BOX_W // 2
                end_col = CENTER_X + BOX_W // 2
                box = [row[start_col:end_col] for row in frame[start_row:end_row]]
                centered_frames.append(box)
            print_frames(centered_frames, f)
