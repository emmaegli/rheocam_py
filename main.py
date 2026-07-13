"""
Reads frames from a USB camera.
Saves periodic screenshots and a per-frame average RGB CSV for each defined AOI (area of interest).
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


def find_laser_point(frame_bgr, region, box_w, box_h, min_score=None):
    """
    Searches for the laser spot inside `region` (a dict with x, y, w, h in
    raw-frame pixel coordinates -- no axes margin, since this runs on the
    unannotated frame straight from the camera) and returns its (cx, cy) in
    the same raw-frame coordinates, or None if nothing clears `min_score`.

    Method matches the validated post-process approach: scores every
    box_w x box_h window with an excess-red index (2R - G - B) via
    cv2.boxFilter (a vectorized mean over every window in one pass), then
    takes the window with the highest score inside `region`. Excess-red
    beats raw R here because the oil background already reads high-R; this
    index instead highlights pixels that are red/pink *relative to green*,
    which isolates the laser spot even when it appears pink/magenta rather
    than pure red.

    Unlike the post-process script, no annotation-box masking is needed:
    this runs on the raw camera frame before any overlay box is drawn, so
    there's no red annotation rectangle in the image yet to accidentally
    score highly.

    min_score: if set, a detection is only accepted when the winning
    window's score is >= min_score; otherwise returns None (laser presumed
    off/occluded this frame). Leave as None to always take the best window
    in the region, matching the post-process script's default behavior.
    """
    h, w = frame_bgr.shape[:2]
    x_min, x_max = sorted((max(0, region["x"]), min(w, region["x"] + region["w"])))
    y_min, y_max = sorted((max(0, region["y"]), min(h, region["y"] + region["h"])))

    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    r_avg = cv2.boxFilter(img_rgb[:, :, 0], ddepth=-1, ksize=(box_w, box_h))
    g_avg = cv2.boxFilter(img_rgb[:, :, 1], ddepth=-1, ksize=(box_w, box_h))
    b_avg = cv2.boxFilter(img_rgb[:, :, 2], ddepth=-1, ksize=(box_w, box_h))
    score = 2 * r_avg - g_avg - b_avg  # excess-red index

    mask = np.full(score.shape, -np.inf, dtype=np.float32)
    mask[y_min:y_max, x_min:x_max] = score[y_min:y_max, x_min:x_max]

    cy, cx = np.unravel_index(np.argmax(mask), mask.shape)
    best_score = mask[cy, cx]

    if min_score is not None and best_score < min_score:
        return None

    return int(cx), int(cy)


def init_avg_rgb_csv(output_dir, name, aoi_labels):
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
    aoi_columns = ",".join(
        f"{label} avg r,{label} avg g,{label} avg b" for label in aoi_labels
    )
    f.write(f"frame,timestamp,{aoi_columns}\n")
    f.flush()
    print(f"  avg_rgb.csv opened: {path}")
    return f


def compute_avg_rgb(frame_rgb, box_w, box_h, cx, cy):
    crop = frame_rgb[
        cy - box_h // 2 : cy + box_h // 2, cx - box_w // 2 : cx + box_w // 2
    ]
    return crop.mean(axis=(0, 1)).round(2)


def append_avg_rgb(csv_file, frame_idx, ts, aoi_values):
    """
    aoi_values: list of (avg_r, avg_g, avg_b) tuples, one per AOI, in the same
    order the CSV header was built with.
    """
    values_str = ",".join(f"{r},{g},{b}" for (r, g, b) in aoi_values)
    csv_file.write(f"{frame_idx},{ts},{values_str}\n")
    csv_file.flush()


def draw_dashed_rect(img, top_left, bottom_right, color, thickness=1, dash_len=6):
    x1, y1 = top_left
    x2, y2 = bottom_right
    # top and bottom edges
    for x in range(x1, x2, dash_len * 2):
        cv2.line(img, (x, y1), (min(x + dash_len, x2), y1), color, thickness)
        cv2.line(img, (x, y2), (min(x + dash_len, x2), y2), color, thickness)
    # left and right edges
    for y in range(y1, y2, dash_len * 2):
        cv2.line(img, (x1, y), (x1, min(y + dash_len, y2)), color, thickness)
        cv2.line(img, (x2, y), (x2, min(y + dash_len, y2)), color, thickness)


def save_screenshot(frame_bgr, frame_count, output_dir, aois, axes_margin=60):
    """
    aois: list of dicts, each with keys:
        label (str), box_w, box_h, cx, cy, color (BGR tuple)
        optionally: search_region (dict with x, y, w, h) for auto-detect AOIs
    """
    os.makedirs(output_dir, exist_ok=True)
    frame = add_frame_axes(frame_bgr, margin=axes_margin)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for aoi in aois:
        color = aoi.get("color", (0, 0, 255))

        # For auto-detect AOIs, draw the larger search region as a dashed box
        region = aoi.get("search_region")
        if region is not None:
            rx1 = region["x"] + axes_margin
            ry1 = region["y"] + axes_margin
            rx2 = region["x"] + region["w"] + axes_margin
            ry2 = region["y"] + region["h"] + axes_margin
            draw_dashed_rect(frame, (rx1, ry1), (rx2, ry2), color, thickness=1)

        cx, cy = aoi["cx"], aoi["cy"]
        box_w, box_h = aoi["box_w"], aoi["box_h"]
        top_left = (cx - box_w // 2 + axes_margin, cy - box_h // 2 + axes_margin)
        bottom_right = (cx + box_w // 2 + axes_margin, cy + box_h // 2 + axes_margin)
        cv2.rectangle(frame, top_left, bottom_right, color, 1)
        # label just above the box
        label = aoi["label"]
        cv2.putText(
            frame,
            label,
            (top_left[0], top_left[1] - 4),
            font,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

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
    aois=None,
):
    """
    aois: list of dicts defining each area of interest. Two kinds:

    1) Fixed AOI — a static box at a known location:
        {"label": "aoi1", "type": "fixed", "box_w": 24, "box_h": 24,
         "center_x": 675, "center_y": 300, "color": (0, 0, 255)}
       ("type" defaults to "fixed" if omitted.) If center_x/center_y are
       omitted, it defaults to the frame center.

    2) Auto-detected laser AOI — searches a larger region each capture tick
       and centers a box_w x box_h box on wherever the laser scores highest:
        {"label": "laser", "type": "auto_laser", "box_w": 22, "box_h": 22,
         "search_region": {"x": 250, "y": 0, "w": 650, "h": 250},
         "color": (0, 255, 255), "min_score": None}
       `search_region` should be generous enough to contain the laser dot
       across experiments, but tight enough to avoid picking up other
       red/pink objects in the scene. box_w/box_h double as both the
       detection window size (scored via excess-red index) and the size of
       the box whose average RGB gets logged. `min_score` is optional --
       set a number to treat low-scoring frames as "laser not detected"
       (falls back to the last known position); leave as None/omit to
       always take the best-scoring window in the region.
    """
    if not aois:
        raise ValueError("capture_frames requires at least one AOI in `aois`.")

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

    # Resolve each AOI's center (defaulting to frame center for fixed AOIs;
    # auto_laser AOIs start at their search region's center until first detection).
    resolved_aois = []
    for aoi in aois:
        aoi_type = aoi.get("type", "fixed")
        entry = {
            "label": aoi["label"],
            "type": aoi_type,
            "box_w": aoi["box_w"],
            "box_h": aoi["box_h"],
            "color": aoi.get("color", (0, 0, 255)),
        }
        if aoi_type == "auto_laser":
            region = aoi["search_region"]
            entry["search_region"] = region
            entry["cx"] = region["x"] + region["w"] // 2
            entry["cy"] = region["y"] + region["h"] // 2
        else:
            cx = aoi.get("center_x")
            cy = aoi.get("center_y")
            entry["cx"] = cx if cx is not None else w // 2
            entry["cy"] = cy if cy is not None else h // 2
        resolved_aois.append(entry)

    # Single shared CSV; columns are grouped per AOI in the order given above.
    csv_file = init_avg_rgb_csv(
        output_dir, name, [aoi["label"] for aoi in resolved_aois]
    )

    start_time = time.monotonic()
    next_capture = start_time

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

                for aoi in resolved_aois:
                    if aoi["type"] == "auto_laser":
                        point = find_laser_point(
                            frame_bgr,
                            aoi["search_region"],
                            aoi["box_w"],
                            aoi["box_h"],
                            min_score=aoi.get("min_score"),
                        )
                        if point is not None:
                            aoi["cx"], aoi["cy"] = point
                        else:
                            print(
                                f"  WARNING: laser not detected for AOI "
                                f"'{aoi['label']}' at frame {frame_count} — "
                                f"reusing last known position ({aoi['cx']}, {aoi['cy']})."
                            )

                aoi_values = [
                    compute_avg_rgb(
                        frame_rgb, aoi["box_w"], aoi["box_h"], aoi["cx"], aoi["cy"]
                    )
                    for aoi in resolved_aois
                ]
                append_avg_rgb(csv_file, frame_count, ts, aoi_values)

                if frame_count == 1 or (
                    screenshot_interval and frame_count % screenshot_interval == 0
                ):
                    save_screenshot(
                        frame_bgr, frame_count, screenshot_dir, resolved_aois
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
            save_screenshot(last_frame_bgr, frame_count, screenshot_dir, resolved_aois)
            print("  Final timestamped screenshot saved.")

        cap.release()
        csv_file.close()
        print("avg_rgb.csv closed.")
        if show_preview:
            cv2.destroyAllWindows()

    print(f"\nCaptured {frame_count} frames over {test_length:.0f}s.")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":

    NAME = "test"
    # NAME = "SC-L-1-1"  # the sample name -- CHANGE THIS EVERY EXPERIMENT

    CAMERA_INDEX = 0
    SHOW_PREVIEW = False

    # ── Define your AOIs here. Add/remove entries as needed. ──────────────────
    # "color": the color of the box that defines each aoi in post process box draw, so (0,0,255)=red, (255,0,0)=blue, etc.
    AOIS = [
        {
            "label": "up_1",
            "type": "fixed",
            "box_w": 24,
            "box_h": 24,
            "center_x": 675,
            "center_y": 300,
            "color": (0, 0, 255),  # red
        },
        {
            "label": "up_2",
            "type": "fixed",
            "box_w": 24,
            "box_h": 24,
            "center_x": 520,  # ← set the second AOI's location
            "center_y": 310,
            "color": (160, 32, 255),  # purple
        },
        {
            "label": "up_3",
            "type": "fixed",
            "box_w": 24,
            "box_h": 24,
            "center_x": 430,  # ← set the second AOI's location
            "center_y": 200,
            "color": (255, 0, 0),  # blue
        },
        {
            "label": "pol_1",
            "type": "auto_laser",
            "box_w": 30,  # matches BOX_W from post-process analysis
            "box_h": 30,  # matches BOX_H from post-process analysis
            "search_region": {
                # Carried over from post-process SEARCH_X_MIN/MAX, SEARCH_Y_MIN/MAX.
                # These are raw-frame pixel coordinates -- no AXES_MARGIN offset
                # needed here, since this runs on the unannotated camera frame
                # (the post-process script only adds that margin because it's
                # indexing into *saved screenshots*, which have it baked in).
                "x": 740,
                "y": 220,
                "w": 110,  # SEARCH_X_MAX - SEARCH_X_MIN
                "h": 80,  # SEARCH_Y_MAX - SEARCH_Y_MIN
            },
            "color": (0, 255, 255),  # yellow
        },
    ]
    # ────────────────────────────────────────────────────────────────────────

    SCHEDULE = {"hours": 6, "minutes": 0, "seconds": 0}
    test_length = timedelta(**SCHEDULE).total_seconds()

    CAPTURE_INTERVAL = 6  # capture avg RGB in aoi's every 6 seconds
    SCREENSHOT_INTERVAL_MINUTES = (
        0.1  # take a screenshot from the camera every N minutes
    )
    screenshot_every = round((SCREENSHOT_INTERVAL_MINUTES * 60) / CAPTURE_INTERVAL)

    capture_frames(
        camera_index=CAMERA_INDEX,
        test_length=test_length,
        show_preview=SHOW_PREVIEW,
        capture_interval=CAPTURE_INTERVAL,
        screenshot_interval=screenshot_every,
        output_dir="./Results",
        name=NAME,
        aois=AOIS,
    )
