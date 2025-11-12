# =============================
# File: auto_scan.py
# =============================

"""
Auto-scan system for microscope stage with serpentine (column-first) scanning and YOLO11 detection.

Features
- User sets START and END by moving the stage manually; we capture those live positions.
- Column-first serpentine pattern: col1 bottom→top, col2 top→bottom, ... (per user preference)
- Step sizes in mm (user-editable via CLI args)
- At each tile (10×): capture image → run YOLO11 → collect candidates
- For each candidate: (dedup-aware) center candidate → switch to 50× → optional AF → capture high-mag image
- Avoid duplicates via a KD-tree in stage space (dedup radius in mm)
- Centering with pixel→stage conversion using calibration file (e.g., HQ_pixel_size.txt). Defaults are editable.
- LED brightness initialized to 15% (editable)
- Robust logging (CSV + JSON) and graceful abort on persistent device failures

Requires
- ultralytics (for YOLO11)  ->  pip install ultralytics
- numpy, pandas, scipy (KDTree)

This script uses MicroscopeController API methods such as set/get positions, magnification,
autofocus, LED, and take_picture (see your MicroscopeController.py).
"""

from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree as KDTree

# YOLO11 (Ultralytics)
try:
    from ultralytics import YOLO
except Exception as e:  # pragma: no cover
    YOLO = None

# Import the user's MicroscopeController
from MicroscopeController import MicroscopeController

# -------------------------
# Calibration utilities
# -------------------------

def load_um_per_px_from_txt(txt_path: str) -> Dict[str, float]:
    """Parse a simple TXT like:
    um/pixel
    5X: 0.69789
    10X: 0.34886
    50X: 0.063957
    Returns dict {"5X": value_um_per_px, ...}
    """
    d: Dict[str, float] = {}
    if not os.path.exists(txt_path):
        return d
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.lower().endswith('um/pixel'):
                continue
            if ':' in line:
                key, val = line.split(':', 1)
                key = key.strip().upper()  # e.g., "10X"
                try:
                    d[key] = float(val.strip())
                except ValueError:
                    pass
    return d

@dataclass
class CameraMapping:
    px_per_mm_10x: float
    px_per_mm_50x: float
    flip_x: bool = False   # if True, image +x corresponds to stage -x
    flip_y: bool = False   # if True, image +y corresponds to stage -y
    rotate_deg: float = 0.0  # rotation (deg) of image relative to stage axes

    def img_dxdy_px_to_stage_mm(self, dx_px: float, dy_px: float, use_50x: bool = False) -> Tuple[float, float]:
        """Convert image offset (pixels) to stage delta (mm), accounting for scale/flip/rotation.
        Image origin assumed at center; dx_px>0 means to the right, dy_px>0 means downward.
        """
        px_per_mm = self.px_per_mm_50x if use_50x else self.px_per_mm_10x
        # Convert to mm in image coords (before flips)
        dx_mm = dx_px / px_per_mm
        dy_mm = dy_px / px_per_mm
        # Apply flips: flip means invert the stage direction for that image axis
        if self.flip_x:
            dx_mm = -dx_mm
        if self.flip_y:
            dy_mm = -dy_mm
        # Apply rotation (image frame → stage frame)
        if abs(self.rotate_deg) > 1e-9:
            th = math.radians(self.rotate_deg)
            cos_t, sin_t = math.cos(th), math.sin(th)
            sx = dx_mm * cos_t - dy_mm * sin_t
            sy = dx_mm * sin_t + dy_mm * cos_t
            return sx, sy
        return dx_mm, dy_mm

# -------------------------
# Helpers
# -------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def timestamp() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def safe_call(fn, *args, retries: int = 2, sleep_s: float = 0.2, **kwargs):
    """Call MicroscopeController methods with retries; reraise on persistent failures."""
    last_exc = None
    for _ in range(retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # includes pipe reconnect handled internally
            last_exc = e
            time.sleep(sleep_s)
    raise last_exc


# -------------------------
# Core scanner
# -------------------------

@dataclass
class ScanConfig:
    # Stage & scan
    step_x_mm: float = 0.20
    step_y_mm: float = 0.20
    settle_ms: int = 50
    safe_z: Optional[float] = None  # if set, lift to this Z before XY moves

    # Imaging & optics
    led_percent: int = 15
    af_10x_per_tile: bool = True
    af_50x: bool = True
    af_distance_10x: float = 0.01
    af_distance_50x: float = 0.01

    # Detection (YOLO11)
    yolo_weights: str = r"E:\0_FlakesNew\runs\train\AI_BN_1002\weights\best.pt"
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    allow_classes: Optional[List[str]] = None  # e.g., ["thin", "mid", "thick"]. None = all

    # Candidate handling
    dedup_radius_mm: float = 0.10  # 100 µm default, user-editable
    center_tol_mm: float = 0.015   # between 0.01 and 0.02 per user
    max_center_iters: int = 3

    # Storage
    root_out: Optional[str] = None  # default: E:\Scans\{date}\{scan_name}
    scan_name: Optional[str] = None

    # Calibration
    calib_txt: Optional[str] = r"/mnt/data/HQ_pixel_size.txt"
    flip_x: bool = False
    flip_y: bool = False
    rotate_deg: float = 0.0


class AutoScanner:
    def __init__(self, mc: MicroscopeController, cfg: ScanConfig):
        self.mc = mc
        self.cfg = cfg
        self.mapping = self._build_mapping()
        self.model = self._load_model()
        self.dedup_tree = None  # KDTree over candidates (stage XY in mm)
        self.candidate_pts: List[Tuple[float, float]] = []
        self.results_csv_rows: List[Dict[str, Any]] = []
        self.results_json: Dict[str, Any] = {"tiles": [], "candidates": []}

    # --------- setup ---------
    def _build_mapping(self) -> CameraMapping:
        # Load um/px from txt
        um_px = load_um_per_px_from_txt(self.cfg.calib_txt) if self.cfg.calib_txt else {}
        # Compute px per mm from 10X and 50X
        def px_per_mm_for(mag_key: str, fallback_um_per_px: float) -> float:
            um_per_px = um_px.get(mag_key, fallback_um_per_px)
            if um_per_px <= 0:
                raise ValueError(f"Invalid um/px for {mag_key}")
            return 1000.0 / um_per_px
        # Fallbacks if not found in file
        px_per_mm_10x = px_per_mm_for("10X", 0.35)
        px_per_mm_50x = px_per_mm_for("50X", 0.064)
        return CameraMapping(
            px_per_mm_10x=px_per_mm_10x,
            px_per_mm_50x=px_per_mm_50x,
            flip_x=self.cfg.flip_x,
            flip_y=self.cfg.flip_y,
            rotate_deg=self.cfg.rotate_deg,
        )

    def _load_model(self):
        if YOLO is None:
            raise RuntimeError("Ultralytics is not available. Install with `pip install ultralytics`.\n")
        model = YOLO(self.cfg.yolo_weights)
        return model

    # --------- utilities ---------
    def _update_dedup_tree(self):
        if self.candidate_pts:
            self.dedup_tree = KDTree(np.array(self.candidate_pts))
        else:
            self.dedup_tree = None

    def _is_far_from_existing(self, x_mm: float, y_mm: float) -> bool:
        if self.dedup_tree is None:
            return True
        d, _ = self.dedup_tree.query([x_mm, y_mm], k=1)
        return d >= self.cfg.dedup_radius_mm

    def _maybe_lift_safe_z(self):
        if self.cfg.safe_z is not None:
            safe_call(self.mc.set_z_position, self.cfg.safe_z)

    def _settle(self):
        time.sleep(self.cfg.settle_ms / 1000.0)

    def _goto_xy(self, x_mm: float, y_mm: float):
        self._maybe_lift_safe_z()
        safe_call(self.mc.set_x_position, x_mm)
        safe_call(self.mc.set_y_position, y_mm)
        self._settle()

    def _autofocus(self, distance: float) -> bool:
        # returns True if OK (controller returns "OK"); failure triggers one retry here
        ok = bool(safe_call(self.mc.autofocus, distance))
        if not ok:
            time.sleep(0.1)
            ok = bool(safe_call(self.mc.autofocus, distance))
        return ok

    def _take_picture(self, path: str) -> bool:
        ensure_dir(os.path.dirname(path))
        ok = bool(safe_call(self.mc.take_picture, path))
        return ok

    # --------- detection pipeline ---------
    def run_detection_on_image(self, img_path: str) -> List[Dict[str, Any]]:
        """Run YOLO and return list of detections: [{bbox_xyxy, conf, cls_name, cx, cy}]."""
        res = self.model.predict(img_path, conf=self.cfg.conf_threshold, iou=self.cfg.iou_threshold, verbose=False)
        out: List[Dict[str, Any]] = []
        if not res:
            return out
        r0 = res[0]
        names = r0.names  # class id -> name
        if r0.boxes is None:
            return out
        boxes = r0.boxes.xyxy.cpu().numpy()
        confs = r0.boxes.conf.cpu().numpy()
        clss = r0.boxes.cls.cpu().numpy().astype(int)
        h, w = r0.orig_shape
        cx_img = w / 2.0
        cy_img = h / 2.0
        for (x1, y1, x2, y2), c, ci in zip(boxes, confs, clss):
            name = names.get(int(ci), str(ci))
            if self.cfg.allow_classes and name not in self.cfg.allow_classes:
                continue
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            out.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "conf": float(c),
                "cls": int(ci),
                "cls_name": name,
                "cx": float(cx),
                "cy": float(cy),
                "img_center": [cx_img, cy_img],
                "img_size": [w, h],
            })
        return out

    # --------- centering logic ---------
    def center_candidate(self, det: Dict[str, Any], current_xy_mm: Tuple[float, float]) -> Tuple[float, float, bool]:
        """Iteratively center candidate by moving stage based on image offset.
        Returns (x_mm, y_mm, centered_ok)
        """
        x_mm, y_mm = current_xy_mm
        for _ in range(self.cfg.max_center_iters):
            cx, cy = det["cx"], det["cy"]
            cx_img, cy_img = det["img_center"]
            dx_px = cx - cx_img
            dy_px = cy - cy_img
            dx_mm, dy_mm = self.mapping.img_dxdy_px_to_stage_mm(dx_px, dy_px, use_50x=False)
            # Move opposite to error to bring feature to center
            x_mm_new = x_mm + dx_mm
            y_mm_new = y_mm + dy_mm
            self._goto_xy(x_mm_new, y_mm_new)
            x_mm, y_mm = x_mm_new, y_mm_new
            # Check tolerance
            if math.hypot(dx_mm, dy_mm) <= self.cfg.center_tol_mm:
                return x_mm, y_mm, True
            # Take a fresh 10× image to re-evaluate center
            tmp_path = os.path.join(self.out_dir, "tmp", f"recenter_{timestamp()}.jpg")
            self._take_picture(tmp_path)
            dets = self.run_detection_on_image(tmp_path)
            if not dets:
                break
            # Find the detection closest (in pixel) to center
            det = sorted(dets, key=lambda d: (d["cx"]-det["img_center"][0])**2 + (d["cy"]-det["img_center"][1])**2)[0]
        return x_mm, y_mm, False

    # --------- main scan ---------
    def run_scan(self, start_xy: Tuple[float, float], end_xy: Tuple[float, float]):
        # Prepare output folder
        date_str = datetime.now().strftime('%Y%m%d')
        scan_name = self.cfg.scan_name or f"scan_{timestamp()}"
        root = self.cfg.root_out or os.path.join("E:\\Scans", date_str, scan_name)
        self.out_dir = root
        tiles_dir = os.path.join(root, "tiles_10x")
        cand_dir = os.path.join(root, "candidates_50x")
        ensure_dir(tiles_dir)
        ensure_dir(cand_dir)
        ensure_dir(os.path.join(root, "tmp"))

        # Initialize LED
        try:
            safe_call(self.mc.set_brightness, self.cfg.led_percent)
        except Exception:
            pass

        # Switch to 10× for scanning
        safe_call(self.mc.set_magnification, 10)

        # Normalize rectangle
        x0, y0 = start_xy
        x1, y1 = end_xy
        x_min, x_max = sorted([x0, x1])
        y_min, y_max = sorted([y0, y1])

        # Determine grid sizes (# of columns in X, # of rows in Y). We scan columns first (Y-sweeps)
        n_cols = max(1, int(round((x_max - x_min) / self.cfg.step_x_mm)) + 1)
        n_rows = max(1, int(round((y_max - y_min) / self.cfg.step_y_mm)) + 1)

        # Pre-log metadata
        meta = {
            "start_xy": [x0, y0],
            "end_xy": [x1, y1],
            "x_range": [x_min, x_max],
            "y_range": [y_min, y_max],
            "n_cols": n_cols,
            "n_rows": n_rows,
            "step_x_mm": self.cfg.step_x_mm,
            "step_y_mm": self.cfg.step_y_mm,
            "led_percent": self.cfg.led_percent,
            "af_10x_per_tile": self.cfg.af_10x_per_tile,
            "af_50x": self.cfg.af_50x,
            "dedup_radius_mm": self.cfg.dedup_radius_mm,
            "center_tol_mm": self.cfg.center_tol_mm,
        }
        self.results_json["meta"] = meta

        # Build column X positions and Y row positions
        xs = [x_min + c * self.cfg.step_x_mm for c in range(n_cols)]
        ys = [y_min + r * self.cfg.step_y_mm for r in range(n_rows)]

        # Serpentine by column: col0 bottom→top (in our screen convention +Y = bottom)
        for ci, x in enumerate(xs):
            col_ys = ys if (ci % 2 == 0) else list(reversed(ys))
            for ri, y in enumerate(col_ys):
                # Move to tile
                self._goto_xy(x, y)
                # Optional AF at 10× per tile
                if self.cfg.af_10x_per_tile:
                    self._autofocus(self.cfg.af_distance_10x)
                # Capture a 10× tile image
                tile_name = f"tile_x{ci:03d}_y{ri:03d}_X{x:.3f}_Y{y:.3f}.jpg"
                tile_path = os.path.join(tiles_dir, tile_name)
                ok = self._take_picture(tile_path)
                if not ok:
                    raise RuntimeError(f"Failed to capture 10× tile at X={x:.3f}, Y={y:.3f}")
                # Detect
                dets = self.run_detection_on_image(tile_path)
                # Record tile info
                self.results_json["tiles"].append({
                    "ci": ci, "ri": ri, "X": x, "Y": y,
                    "path": tile_path, "detections": dets,
                })
                # Visit all candidates (dedup-aware)
                for det in dets:
                    # Convert current stage
                    cur_x = x
                    cur_y = y
                    # Estimate candidate stage XY via centering loop (starts at current tile center)
                    if self._is_far_from_existing(cur_x, cur_y):
                        # Switch to 10× centering (already at 10×)
                        new_x, new_y, centered_ok = self.center_candidate(det, (cur_x, cur_y))
                        # If good, add to candidate list and KD-tree
                        if centered_ok and self._is_far_from_existing(new_x, new_y):
                            self.candidate_pts.append((new_x, new_y))
                            self._update_dedup_tree()
                            # Now switch to 50× and capture hi-mag
                            safe_call(self.mc.set_magnification, 50)
                            if self.cfg.af_50x:
                                self._autofocus(self.cfg.af_distance_50x)
                            cand_name = (
                                f"cand_X{new_x:.3f}_Y{new_y:.3f}_"
                                f"cls{det['cls_name']}_conf{det['conf']:.2f}.jpg"
                            )
                            cand_path = os.path.join(cand_dir, cand_name)
                            ok2 = self._take_picture(cand_path)
                            # Log candidate result
                            self.results_json["candidates"].append({
                                "X": new_x, "Y": new_y,
                                "path": cand_path if ok2 else None,
                                "det": det,
                                "centered": centered_ok,
                            })
                            self.results_csv_rows.append({
                                "type": "candidate_50x",
                                "X_mm": new_x, "Y_mm": new_y,
                                "tile_x_index": ci, "tile_y_index": ri,
                                "tile_path": tile_path,
                                "cand_path": cand_path if ok2 else "",
                                "cls": det["cls_name"], "conf": det["conf"],
                            })
                            # IMPORTANT: switch back to 10× to continue scanning
                            safe_call(self.mc.set_magnification, 10)
                        else:
                            # Not centered or too close to an existing candidate; skip
                            self.results_csv_rows.append({
                                "type": "candidate_skip",
                                "X_mm": new_x, "Y_mm": new_y,
                                "tile_x_index": ci, "tile_y_index": ri,
                                "tile_path": tile_path,
                                "reason": "not_centered_or_duplicate",
                            })
                    # else: too close to existing candidate; ignore

        # After scan, write CSV & JSON
        csv_path = os.path.join(root, "scan_log.csv")
        json_path = os.path.join(root, "scan_log.json")
        pd.DataFrame(self.results_csv_rows).to_csv(csv_path, index=False)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.results_json, f, indent=2)
        print(f"Saved logs: {csv_path}\n{json_path}")


# -------------------------
# CLI
# -------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Microscope auto-scan with YOLO11")

    # START/END acquisition options
    p.add_argument('--start_xy', type=float, nargs=2, metavar=('Xmm', 'Ymm'),
                   help='Start XY (mm). If omitted, query stage live and use current as START.')
    p.add_argument('--end_xy', type=float, nargs=2, metavar=('Xmm', 'Ymm'),
                   help='End XY (mm). If omitted, query stage live after user moves it, and press Enter to capture.')

    # Steps
    p.add_argument('--step_x_mm', type=float, default=0.20)
    p.add_argument('--step_y_mm', type=float, default=0.20)

    # Settling & safety
    p.add_argument('--settle_ms', type=int, default=50)
    p.add_argument('--safe_z', type=float, default=None)

    # AF
    p.add_argument('--af_10x_per_tile', action='store_true', default=True)
    p.add_argument('--no_af_10x_per_tile', dest='af_10x_per_tile', action='store_false')
    p.add_argument('--af_50x', action='store_true', default=True)
    p.add_argument('--no_af_50x', dest='af_50x', action='store_false')
    p.add_argument('--af_distance_10x', type=float, default=0.01)
    p.add_argument('--af_distance_50x', type=float, default=0.01)

    # Illumination
    p.add_argument('--led_percent', type=int, default=15)

    # Detection
    p.add_argument('--weights', type=str, default=r"E:\0_FlakesNew\runs\train\AI_BN_1002\weights\best.pt")
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--iou', type=float, default=0.45)
    p.add_argument('--allow_classes', type=str, nargs='*', default=None,
                   help='Allowed class names (e.g., thin mid thick). None = all classes.')

    # Candidate handling
    p.add_argument('--dedup_radius_mm', type=float, default=0.10)
    p.add_argument('--center_tol_mm', type=float, default=0.015)
    p.add_argument('--max_center_iters', type=int, default=3)

    # Storage
    p.add_argument('--out_root', type=str, default=None)
    p.add_argument('--scan_name', type=str, default=None)

    # Calibration
    p.add_argument('--calib_txt', type=str, default=r"/mnt/data/HQ_pixel_size.txt")
    p.add_argument('--flip_x', action='store_true', default=False)
    p.add_argument('--flip_y', action='store_true', default=False)
    p.add_argument('--rotate_deg', type=float, default=0.0)

    return p.parse_args()


def acquire_start_end(mc: MicroscopeController, args) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    # START
    if args.start_xy is not None:
        start_xy = (args.start_xy[0], args.start_xy[1])
    else:
        x = float(mc.get_x_position())
        y = float(mc.get_y_position())
        print(f"Captured START from live stage: X={x:.3f}, Y={y:.3f}")
        start_xy = (x, y)

    # END
    if args.end_xy is not None:
        end_xy = (args.end_xy[0], args.end_xy[1])
    else:
        input("Move stage to END point, then press <Enter>...")
        x = float(mc.get_x_position())
        y = float(mc.get_y_position())
        print(f"Captured END from live stage: X={x:.3f}, Y={y:.3f}")
        end_xy = (x, y)

    return start_xy, end_xy


def main():
    args = parse_args()

    # Build config
    cfg = ScanConfig(
        step_x_mm=args.step_x_mm,
        step_y_mm=args.step_y_mm,
        settle_ms=args.settle_ms,
        safe_z=args.safe_z,
        led_percent=args.led_percent,
        af_10x_per_tile=args.af_10x_per_tile,
        af_50x=args.af_50x,
        af_distance_10x=args.af_distance_10x,
        af_distance_50x=args.af_distance_50x,
        yolo_weights=args.weights,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        allow_classes=args.allow_classes,
        dedup_radius_mm=args.dedup_radius_mm,
        center_tol_mm=args.center_tol_mm,
        max_center_iters=args.max_center_iters,
        root_out=args.out_root,
        scan_name=args.scan_name,
        calib_txt=args.calib_txt,
        flip_x=args.flip_x,
        flip_y=args.flip_y,
        rotate_deg=args.rotate_deg,
    )

    # Initialize controller and connect (persistent connection is handled internally)
    mc = MicroscopeController()

    try:
        # Acquire START/END
        start_xy, end_xy = acquire_start_end(mc, args)
        # Run scan
        scanner = AutoScanner(mc, cfg)
        scanner.run_scan(start_xy, end_xy)
        print("Scan completed successfully.")
    except KeyboardInterrupt:
        print("\nScan interrupted by user.")
    except Exception as e:
        print(f"\nFATAL: {e}\nAborting scan and shutting down safely.")
    finally:
        try:
            mc.led_off()
        except Exception:
            pass
        mc.close()


if __name__ == '__main__':
    main()


# =============================
# File: scan_viewer.py
# =============================

"""
Simple local Flask viewer for the scan output. Click a tile to show its 10× image and any 50× candidate images.
Run:  python scan_viewer.py --root E:\Scans\20251009\scan_...\
Requires: Flask (pip install flask)
"""

from __future__ import annotations
import argparse
import json
import os
from flask import Flask, render_template_string, send_from_directory

TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Scan Viewer</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 1rem 2rem; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 12px; }
    .card { border: 1px solid #ccc; border-radius: 8px; padding: 10px; }
    img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 6px; }
    .path { font-size: 0.85em; color: #555; word-break: break-all; }
    .meta { margin-bottom: 1rem; }
    .badge { display: inline-block; padding: 2px 6px; border: 1px solid #666; border-radius: 6px; font-size: 0.8em; }
  </style>
</head>
<body>
<h2>Scan Viewer</h2>
<div class="meta">
  <div class="badge">{{meta.n_cols}} cols × {{meta.n_rows}} rows</div>
  <div class="badge">step: {{meta.step_x_mm}} × {{meta.step_y_mm}} mm</div>
  <div class="badge">LED: {{meta.led_percent}}%</div>
</div>

<h3>Tiles (10×)</h3>
<div class="grid">
{% for t in tiles %}
  <div class="card">
    <div><strong>Tile:</strong> x{{t.ci}} y{{t.ri}} @ ({{'%.3f'|format(t.X)}}, {{'%.3f'|format(t.Y)}})</div>
    <a href="/img?path={{t.path}}"><img src="/img?path={{t.path}}"/></a>
    <div class="path">{{t.path}}</div>
    {% if t.detections and t.detections|length > 0 %}
      <div><strong>Detections:</strong> {{t.detections|length}}</div>
      <ul>
      {% for d in t.detections %}
        <li>{{d.cls_name}} ({{'%.2f'|format(d.conf)}})</li>
      {% endfor %}
      </ul>
    {% else %}
      <div>No detections</div>
    {% endif %}
  </div>
{% endfor %}
</div>

<h3>Candidates (50×)</h3>
<div class="grid">
{% for c in cands %}
  <div class="card">
    <div><strong>Candidate @ ({{'%.3f'|format(c.X)}}, {{'%.3f'|format(c.Y)}})</strong></div>
    {% if c.path %}
      <a href="/img?path={{c.path}}"><img src="/img?path={{c.path}}"/></a>
      <div class="path">{{c.path}}</div>
    {% else %}
      <div>Image missing</div>
    {% endif %}
    {% if c.det %}
      <div>Det: {{c.det.cls_name}} ({{'%.2f'|format(c.det.conf)}})</div>
    {% endif %}
  </div>
{% endfor %}
</div>
</body>
</html>
"""

app = Flask(__name__)

@app.route('/img')
def img():
    from flask import request, abort
    path = request.args.get('path')
    if not path or not os.path.exists(path):
        abort(404)
    return send_from_directory(os.path.dirname(path), os.path.basename(path))

@app.route('/')
def index():
    log_path = app.config['LOG_PATH']
    with open(log_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    meta = data.get('meta', {})
    tiles = data.get('tiles', [])
    cands = data.get('candidates', [])
    return render_template_string(TEMPLATE, meta=meta, tiles=tiles, cands=cands)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='Root folder of one scan (contains scan_log.json)')
    ap.add_argument('--port', type=int, default=5000)
    args = ap.parse_args()
    log_path = os.path.join(args.root, 'scan_log.json')
    if not os.path.exists(log_path):
        raise SystemExit(f"scan_log.json not found under: {args.root}")
    app.config['LOG_PATH'] = log_path
    app.run(host='127.0.0.1', port=args.port, debug=False)

if __name__ == '__main__':
    main()
