import csv
import time
import cv2
import numpy as np
from ultralytics import YOLO


("Baseline (warm-up, ignore)", {}),
("Baseline", {}),
# ---------- SETTINGS ----------
VIDEO_PATH = "input.mp4"

LINE_Y_RATIO = 0.5
ZONE_RATIO = [(0.05, 0.55), (0.35, 0.55), (0.35, 0.95), (0.05, 0.95)]

MANUAL_ENTRY = 0
MANUAL_EXIT = 0


MAX_FRAMES = None

# Baseline settings
BASELINE = {
    "model": "yolo11n.pt",
    "conf": 0.4,
    "imgsz": 640,
    "tracker": "botsort.yaml",
    "stride": 1,          # 1 = har frame process, 2 = har doosra frame
}

EXPERIMENTS = [
    ("Baseline", {}),
    ("Change 1: tracker ByteTrack", {"tracker": "bytetrack.yaml"}),
    ("Change 2: imgsz 960", {"imgsz": 960}),
    ("Change 3: conf 0.25", {"conf": 0.25}),
    ("Change 4: model yolo11s", {"model": "yolo11s.pt"}),
    ("Change 5: skip every 2nd frame", {"stride": 2}),
]
# ------------------------------


def run(cfg):
    model = YOLO(cfg["model"])  # har run ke liye fresh model (tracker reset)
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise SystemExit("Video open nahi hui, VIDEO_PATH check karo")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    line_y = int(h * LINE_Y_RATIO)
    zone = np.array([(int(x * w), int(y * h)) for x, y in ZONE_RATIO], dtype=np.int32)

    prev_y = {}
    entered = set()
    exited = set()
    alert_ids = set()
    frames_read = 0

    start = time.time()
    while True:
        if MAX_FRAMES and frames_read >= MAX_FRAMES:
            break
        ok, frame = cap.read()
        if not ok:
            break
        frames_read += 1

        if frames_read % cfg["stride"] != 0:
            continue

        results = model.track(frame, persist=True, classes=[0], conf=cfg["conf"],
                              imgsz=cfg["imgsz"], tracker=cfg["tracker"], verbose=False)
        boxes = results[0].boxes
        if boxes.id is None:
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.int().cpu().tolist()
        for (x1, y1, x2, y2), tid in zip(xyxy, ids):
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            if tid in prev_y:
                py = prev_y[tid]
                if py < line_y <= cy:
                    entered.add(tid)
                elif py >= line_y > cy:
                    exited.add(tid)
            prev_y[tid] = cy

            if cv2.pointPolygonTest(zone, (float(cx), float(cy)), False) >= 0:
                alert_ids.add(tid)

    elapsed = time.time() - start
    cap.release()

    return {
        "entry": len(entered),
        "exit": len(exited),
        "alerts": len(alert_ids),
        "fps": frames_read / elapsed,   # video ke kitne frames per second nipta diye
    }


rows = []
for name, change in EXPERIMENTS:
    cfg = {**BASELINE, **change}
    print(f"Running: {name} ...")
    r = run(cfg)
    error = abs(r["entry"] - MANUAL_ENTRY) + abs(r["exit"] - MANUAL_EXIT)
    rows.append([name, r["entry"], r["exit"], error, round(r["fps"], 2), r["alerts"]])
    print(f"   entry={r['entry']} exit={r['exit']} error={error} fps={r['fps']:.2f}")

header = ["Experiment", "Entry", "Exit", "Count error", "FPS", "Zone alerts"]

with open("results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

lines = ["| " + " | ".join(header) + " |",
         "|" + "---|" * len(header)]
for row in rows:
    lines.append("| " + " | ".join(str(v) for v in row) + " |")
lines.append("")
lines.append(f"Manual count: Entry = {MANUAL_ENTRY}, Exit = {MANUAL_EXIT}")
table = "\n".join(lines)

with open("results.md", "w") as f:
    f.write(table)

print("\n" + table)
print("\nSaved: results.csv, results.md")
