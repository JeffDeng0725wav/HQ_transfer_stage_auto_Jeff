# autoscan_gui.py
# pip install PySide6 pywin32
import sys, json, time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple, List, Dict

from PySide6.QtWidgets import (
    QApplication, QWidget, QTabWidget, QFileDialog, QMessageBox, QGridLayout,
    QLabel, QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox, QCheckBox,
    QPlainTextEdit, QProgressBar, QHBoxLayout, QVBoxLayout, QComboBox
)
from PySide6.QtCore import Qt, QThread, Signal

# your controller
from MicroscopeController import MicroscopeController

# =======================
# settings model
# =======================
@dataclass
class Settings:
    # paths
    weights: str = ""
    calib_txt: str = ""       # calibration file (um/px)
    out_root: str = ""
    scan_name: str = ""
    # light/safety
    led_percent: int = 20
    safe_z: float = 0.0       # "lift to safety" = smaller Z (absolute)
    # region/steps
    start_x: Optional[float] = None
    start_y: Optional[float] = None
    end_x: Optional[float] = None
    end_y: Optional[float] = None
    step_x_mm: float = 0.20
    step_y_mm: float = 0.20
    settle_ms: int = 50
    # AF policy
    af_10x_every_n: int = 1
    af_after_mag_change: bool = True
    af_distance_10x: float = 0.05
    af_distance_50x: float = 0.02
    # detection (placeholder)
    conf: float = 0.25
    iou: float = 0.45
    allow_classes: str = ""
    # dedup/centering (placeholder)
    dedup_radius_mm: float = 0.10
    center_tol_mm: float = 0.015
    max_center_iters: int = 3
    # current magnification (UI)
    target_mag: float = 10.0

# =======================
# helpers: calibration & grid
# =======================
def parse_calib_um_per_px(path: str) -> Dict[float, float]:
    """Read {magnification: um_per_px} from txt."""
    mp = {}
    p = Path(path)
    if not p.exists(): return mp
    for line in p.read_text(encoding="utf-8").strip().splitlines():
        if ":" not in line: continue
        name, val = line.split(":", 1)
        name = name.strip().upper().replace("X", "")
        try:
            mag = float(name)
            umpx = float(val.strip())
            mp[mag] = umpx
        except:
            continue
    return mp

def build_snake_tiles(sx, sy, ex, ey, dx, dy) -> List[Tuple[float,float,int,int]]:
    """Return (x, y, row, col) with 1-based row/col, snake pattern."""
    nx = max(1, int(abs(ex - sx) / max(dx, 1e-9)) + 1)
    ny = max(1, int(abs(ey - sy) / max(dy, 1e-9)) + 1)
    xs = [sx + i * (dx if ex >= sx else -dx) for i in range(nx)]
    ys = [sy + j * (dy if ey >= sy else -dy) for j in range(ny)]
    tiles = []
    for r, y in enumerate(ys, start=1):
        xrow = xs if (r % 2 == 1) else list(reversed(xs))
        for c_idx, x in enumerate(xrow, start=1):
            col = c_idx if (r % 2 == 1) else (nx - c_idx + 1)
            tiles.append((x, y, r, col))
    return tiles

# =======================
# worker thread
# =======================
class ScanWorker(QThread):
    log = Signal(str)
    progress = Signal(int, int)    # done, total
    finished_ok = Signal()
    error = Signal(str)

    def __init__(self, s: Settings, ctrl: MicroscopeController, calib_map: Dict[float,float], parent=None):
        super().__init__(parent)
        self.s = s
        self.ctrl = ctrl
        self.calib_map = calib_map
        self._stop = False
        self._last_mag = None

    def stop(self):
        self._stop = True

    def _ensure_mag_and_af(self, target_mag: float):
        """If magnification changes, switch and autofocus immediately."""
        try:
            current_mag = float(self.ctrl.get_magnification())
        except Exception:
            current_mag = None

        if (current_mag is None) or (abs(current_mag - target_mag) > 1e-6):
            ok = self.ctrl.set_magnification(target_mag)
            self.log.emit(f"Switch magnification to {target_mag}x → {ok}")
            time.sleep(0.05)
            if self.s.af_after_mag_change:
                dist = self.s.af_distance_10x if target_mag <= 10 else self.s.af_distance_50x
                try:
                    ok_af = self.ctrl.autofocus(float(dist))
                    self.log.emit(f"AF after mag change (dist={dist}) → {ok_af}")
                except Exception as e:
                    self.log.emit(f"AF failed after mag change: {e}")

        self._last_mag = target_mag

    def run(self):
        try:
            if not self.ctrl or not self.ctrl.connected:
                self.error.emit("Not connected to microscope server")
                return

            sx, sy, ex, ey = self.s.start_x, self.s.start_y, self.s.end_x, self.s.end_y
            if None in (sx, sy, ex, ey):
                self.error.emit("Start/End points are not set")
                return

            tiles = build_snake_tiles(sx, sy, ex, ey, self.s.step_x_mm, self.s.step_y_mm)
            total = len(tiles)
            self.log.emit(f"Total tiles: {total}")
            self.progress.emit(0, total)

            # output folder
            root = Path(self.s.out_root) if self.s.out_root else Path.cwd() / "Scans"
            scan_name = self.s.scan_name.strip() or time.strftime("scan_%Y%m%d_%H%M%S")
            out_dir = root / scan_name / "images"
            out_dir.mkdir(parents=True, exist_ok=True)

            # light
            try:
                self.ctrl.set_brightness(int(self.s.led_percent))
                self.ctrl.led_on()
            except Exception as e:
                self.log.emit(f"Warning: failed to set LED: {e}")

            # ensure magnification + AF once
            self._ensure_mag_and_af(float(self.s.target_mag))

            done = 0
            for idx, (x, y, row, col) in enumerate(tiles):
                if self._stop:
                    self.log.emit("Stop signal received. Exiting scan loop…")
                    break

                # move
                self.ctrl.set_x_position(float(x))
                self.ctrl.set_y_position(float(y))
                time.sleep(self.s.settle_ms / 1000.0)

                # periodic AF
                if self.s.af_10x_every_n > 0 and (idx % self.s.af_10x_every_n == 0):
                    dist = self.s.af_distance_10x if self._last_mag and self._last_mag <= 10 else self.s.af_distance_50x
                    try:
                        ok = self.ctrl.autofocus(float(dist))
                        self.log.emit(f"Periodic AF @idx={idx}, dist={dist} → {ok}")
                    except Exception as e:
                        self.log.emit(f"Periodic AF failed: {e}")

                # capture as {row}_{col}.jpg
                img_path = out_dir / f"{row}_{col}.jpg"
                ok_pic = self.ctrl.take_picture(str(img_path))
                if not ok_pic:
                    self.log.emit(f"Capture failed: {img_path.name}")
                else:
                    self.log.emit(f"Saved: {img_path.name}")

                # TODO: detection + saving results
                # TODO: dedup + centering policy
                done += 1
                self.progress.emit(done, total)
                if done % 10 == 0:
                    self.log.emit(f"Progress {done}/{total}")

            # end: move Z to SAFE_Z (smaller), turn off LED
            try:
                self.ctrl.set_z_position(float(self.s.safe_z))
            except Exception:
                pass
            try:
                self.ctrl.led_off()
            except Exception:
                pass

            if not self._stop:
                self.log.emit("Scan finished.")
                self.finished_ok.emit()

        except Exception as e:
            self.error.emit(str(e))

# =======================
# main window
# =======================
class Main(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Auto-Scanner GUI · MicroscopeController")
        self.ctrl: Optional[MicroscopeController] = None
        self.s = Settings()
        self.worker: Optional[ScanWorker] = None
        self.calib_map: Dict[float,float] = {}
        self._build_ui()

    def _build_ui(self):
        tabs = QTabWidget()

        # --- Tab: Connection/Hardware ---
        w_conn = QWidget(); g0 = QGridLayout(w_conn)
        self.btn_connect = QPushButton("Connect Server")
        self.btn_disconnect = QPushButton("Disconnect")
        self.btn_homex = QPushButton("Home X")
        self.btn_homey = QPushButton("Home Y")
        self.btn_homez = QPushButton("Home Z")
        self.sp_led = QSpinBox(); self.sp_led.setRange(0,100); self.sp_led.setValue(self.s.led_percent)
        self.btn_ledset = QPushButton("Set Brightness")
        self.btn_led_on = QPushButton("LED ON")
        self.btn_led_off= QPushButton("LED OFF")
        self.sp_safez = QDoubleSpinBox(); self._mm(self.sp_safez, self.s.safe_z, -1000, 1000, 0.01)

        # magnification combo (switch → AF immediately)
        self.cbo_mag = QComboBox()
        self.cbo_mag.addItems(["5","10","20","40","50"])
        self.cbo_mag.setCurrentText(str(int(self.s.target_mag)))
        self.cbo_mag.currentTextChanged.connect(self._on_mag_change)

        g0.addWidget(self.btn_connect,0,0); g0.addWidget(self.btn_disconnect,0,1)
        g0.addWidget(self.btn_homex,1,0); g0.addWidget(self.btn_homey,1,1); g0.addWidget(self.btn_homez,1,2)
        g0.addWidget(QLabel("LED %"),2,0); g0.addWidget(self.sp_led,2,1); g0.addWidget(self.btn_ledset,2,2)
        g0.addWidget(self.btn_led_on,3,0); g0.addWidget(self.btn_led_off,3,1)
        g0.addWidget(QLabel("SAFE_Z (mm)"),4,0); g0.addWidget(self.sp_safez,4,1)
        g0.addWidget(QLabel("Magnification (x)"),5,0); g0.addWidget(self.cbo_mag,5,1)
        tabs.addTab(w_conn, "Connection / Hardware")

        # --- Tab: Paths ---
        w_paths = QWidget(); g1 = QGridLayout(w_paths)
        self.e_weights = QLineEdit(self.s.weights); b_w = QPushButton("Browse")
        self.e_calib = QLineEdit(self.s.calib_txt); b_c = QPushButton("Browse")
        self.e_out = QLineEdit(self.s.out_root); b_o = QPushButton("Choose Folder")
        self.e_name = QLineEdit(self.s.scan_name)
        g1.addWidget(QLabel("YOLO Weights"),0,0); g1.addWidget(self.e_weights,0,1); g1.addWidget(b_w,0,2)
        g1.addWidget(QLabel("Calibration (um/px)"),1,0); g1.addWidget(self.e_calib,1,1); g1.addWidget(b_c,1,2)
        g1.addWidget(QLabel("Output Root"),2,0); g1.addWidget(self.e_out,2,1); g1.addWidget(b_o,2,2)
        g1.addWidget(QLabel("Scan Name"),3,0); g1.addWidget(self.e_name,3,1)
        tabs.addTab(w_paths, "Paths")

        # --- Tab: Region / Step ---
        w_roi = QWidget(); g2 = QGridLayout(w_roi)
        self.sp_sx = QDoubleSpinBox(); self._mm(self.sp_sx, 0.0, -99999, 99999)
        self.sp_sy = QDoubleSpinBox(); self._mm(self.sp_sy, 0.0, -99999, 99999)
        self.sp_ex = QDoubleSpinBox(); self._mm(self.sp_ex, 1.0, -99999, 99999)
        self.sp_ey = QDoubleSpinBox(); self._mm(self.sp_ey, 1.0, -99999, 99999)
        self.btn_use_start = QPushButton("Use Current as Start")
        self.btn_use_end   = QPushButton("Use Current as End")
        self.sp_dx = QDoubleSpinBox(); self._mm(self.sp_dx, self.s.step_x_mm, 1e-6, 1e6, 0.001)
        self.sp_dy = QDoubleSpinBox(); self._mm(self.sp_dy, self.s.step_y_mm, 1e-6, 1e6, 0.001)
        self.sp_settle = QSpinBox(); self.sp_settle.setRange(0, 10000); self.sp_settle.setValue(self.s.settle_ms)
        self.lbl_est = QLabel("Estimate: — rows × — cols, — tiles total")
        g2.addWidget(QLabel("Start X"),0,0); g2.addWidget(self.sp_sx,0,1); g2.addWidget(QLabel("Start Y"),0,2); g2.addWidget(self.sp_sy,0,3); g2.addWidget(self.btn_use_start,0,4)
        g2.addWidget(QLabel("End   X"),1,0); g2.addWidget(self.sp_ex,1,1); g2.addWidget(QLabel("End   Y"),1,2); g2.addWidget(self.sp_ey,1,3); g2.addWidget(self.btn_use_end,1,4)
        g2.addWidget(QLabel("Step X (mm)"),2,0); g2.addWidget(self.sp_dx,2,1)
        g2.addWidget(QLabel("Step Y (mm)"),2,2); g2.addWidget(self.sp_dy,2,3)
        g2.addWidget(QLabel("Settle Wait (ms)"),3,0); g2.addWidget(self.sp_settle,3,1)
        g2.addWidget(self.lbl_est,4,0,1,4)
        tabs.addTab(w_roi, "Region / Step")

        # --- Tab: AF / Detection (placeholder) ---
        w_af = QWidget(); g3 = QGridLayout(w_af)
        self.sp_afn = QSpinBox(); self.sp_afn.setRange(1, 9999); self.sp_afn.setValue(self.s.af_10x_every_n)
        self.cb_af_after = QCheckBox("AF immediately after mag change"); self.cb_af_after.setChecked(self.s.af_after_mag_change)
        self.sp_af10 = QDoubleSpinBox(); self._mm(self.sp_af10, self.s.af_distance_10x, 0, 10, 0.001)
        self.sp_af50 = QDoubleSpinBox(); self._mm(self.sp_af50, self.s.af_distance_50x, 0, 10, 0.001)
        self.sp_conf = QDoubleSpinBox(); self.sp_conf.setRange(0,1); self.sp_conf.setSingleStep(0.01); self.sp_conf.setValue(self.s.conf)
        self.sp_iou  = QDoubleSpinBox(); self.sp_iou.setRange(0,1); self.sp_iou.setSingleStep(0.01); self.sp_iou.setValue(self.s.iou)
        self.e_allow = QLineEdit(self.s.allow_classes)
        g3.addWidget(QLabel("AF every N tiles @10x"),0,0); g3.addWidget(self.sp_afn,0,1); g3.addWidget(self.cb_af_after,0,2)
        g3.addWidget(QLabel("AF distance @10x (mm)"),1,0); g3.addWidget(self.sp_af10,1,1)
        g3.addWidget(QLabel("AF distance @≥20x (mm)"),1,2); g3.addWidget(self.sp_af50,1,3)
        g3.addWidget(QLabel("CONF"),2,0); g3.addWidget(self.sp_conf,2,1); g3.addWidget(QLabel("IOU"),2,2); g3.addWidget(self.sp_iou,2,3)
        g3.addWidget(QLabel("Allowed classes"),3,0); g3.addWidget(self.e_allow,3,1,1,3)
        tabs.addTab(w_af, "AF / Detection")

        # --- bottom control strip ---
        self.btn_confirm = QPushButton("Review Settings")
        self.btn_start   = QPushButton("Start Scan")
        self.btn_stop    = QPushButton("Emergency Stop"); self.btn_stop.setEnabled(False)
        self.btn_save    = QPushButton("Save Config"); self.btn_load = QPushButton("Load Config")
        self.pb = QProgressBar()
        self.log = QPlainTextEdit(); self.log.setReadOnly(True)

        layout = QVBoxLayout(self)
        layout.addWidget(tabs)
        row = QHBoxLayout()
        row.addWidget(self.btn_confirm); row.addStretch(1)
        row.addWidget(self.btn_save); row.addWidget(self.btn_load)
        row.addWidget(self.btn_start); row.addWidget(self.btn_stop)
        layout.addLayout(row)
        layout.addWidget(self.pb)
        layout.addWidget(self.log)

        # bindings
        self.btn_connect.clicked.connect(self._connect)
        self.btn_disconnect.clicked.connect(self._disconnect)
        self.btn_homex.clicked.connect(lambda: self._safe(lambda: self.ctrl.home_x()))
        self.btn_homey.clicked.connect(lambda: self._safe(lambda: self.ctrl.home_y()))
        self.btn_homez.clicked.connect(lambda: self._safe(lambda: self.ctrl.home_z()))
        self.btn_ledset.clicked.connect(lambda: self._safe(lambda: self.ctrl.set_brightness(int(self.sp_led.value()))))
        self.btn_led_on.clicked.connect(lambda: self._safe(lambda: self.ctrl.led_on()))
        self.btn_led_off.clicked.connect(lambda: self._safe(lambda: self.ctrl.led_off()))
        self.btn_use_start.clicked.connect(lambda: self._read_xy(to="start"))
        self.btn_use_end.clicked.connect(lambda: self._read_xy(to="end"))
        self.btn_confirm.clicked.connect(self._confirm)
        self.btn_start.clicked.connect(self._start)
        self.btn_stop.clicked.connect(self._stop)
        self.btn_save.clicked.connect(self._save_cfg)
        self.btn_load.clicked.connect(self._load_cfg)
        b_w.clicked.connect(lambda: self._pick_file(self.e_weights, "Choose YOLO Weights", "All (*.*)"))
        b_c.clicked.connect(self._pick_calib)
        b_o.clicked.connect(lambda: self._pick_dir(self.e_out))

        for w in (self.sp_sx, self.sp_sy, self.sp_ex, self.sp_ey, self.sp_dx, self.sp_dy):
            w.valueChanged.connect(self._update_est)
        self._update_est()

    # ==== utils ====
    def _mm(self, sp: QDoubleSpinBox, val, lo, hi, step=0.01):
        sp.setRange(lo, hi); sp.setDecimals(4); sp.setSingleStep(step); sp.setValue(val)

    def _pick_file(self, edit: QLineEdit, title: str, filt: str):
        p, _ = QFileDialog.getOpenFileName(self, title, "", filt)
        if p: edit.setText(p)

    def _pick_dir(self, edit: QLineEdit):
        d = QFileDialog.getExistingDirectory(self, "Choose Folder", "")
        if d: edit.setText(d)

    def _pick_calib(self):
        p, _ = QFileDialog.getOpenFileName(self, "Choose Calibration File (um/px)", "", "Text (*.txt);;All (*.*)")
        if not p: return
        self.e_calib.setText(p)
        self.calib_map = parse_calib_um_per_px(p)
        if self.calib_map:
            msg = "\n".join([f"{int(k)}x: {v} um/px" for k,v in sorted(self.calib_map.items())])
            QMessageBox.information(self, "Calibration Loaded", msg)
        else:
            QMessageBox.warning(self, "Calibration Failed", "No valid magnification → pixel size parsed.")

    def _update_est(self):
        sx, sy, ex, ey = self.sp_sx.value(), self.sp_sy.value(), self.sp_ex.value(), self.sp_ey.value()
        dx, dy = self.sp_dx.value(), self.sp_dy.value()
        nx = max(1, int(abs(ex - sx) / max(dx, 1e-9)) + 1)
        ny = max(1, int(abs(ey - sy) / max(dy, 1e-9)) + 1)
        self.lbl_est.setText(f"Estimate: {ny} rows × {nx} cols, {nx*ny} tiles total")

    def _safe(self, fn):
        try:
            if not self.ctrl:
                QMessageBox.warning(self, "Not Connected", "Please connect first.")
                return
            fn()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    # ==== connect/disconnect ====
    def _connect(self):
        if self.ctrl:
            QMessageBox.information(self, "Info", "Already connected.")
            return
        try:
            self.ctrl = MicroscopeController(debug=False)  # constructor connects
            QMessageBox.information(self, "Success", "Connected to \\\\.\\pipe\\HQ_server.")
        except Exception as e:
            self.ctrl = None
            QMessageBox.critical(self, "Connection Failed", str(e))

    def _disconnect(self):
        if not self.ctrl: return
        try:
            self.ctrl.close()
            self.ctrl = None
            QMessageBox.information(self, "Disconnected", "Connection closed.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    # ==== magnification change ====
    def _on_mag_change(self, txt: str):
        self.s.target_mag = float(txt or 10)
        if not self.ctrl: return
        try:
            ok = self.ctrl.set_magnification(self.s.target_mag)
            if not ok:
                QMessageBox.warning(self, "Magnification", f"Failed to switch to {self.s.target_mag}x")
            else:
                if self.cb_af_after.isChecked():
                    dist = self.sp_af10.value() if self.s.target_mag <= 10 else self.sp_af50.value()
                    self.ctrl.autofocus(float(dist))
        except Exception as e:
            QMessageBox.critical(self, "Magnification Failed", str(e))

    # ==== read current XY ====
    def _read_xy(self, to: str):
        if not self.ctrl:
            QMessageBox.warning(self, "Not Connected", "Please connect first.")
            return
        x = float(self.ctrl.get_x_position())
        y = float(self.ctrl.get_y_position())
        if to == "start":
            self.sp_sx.setValue(x); self.sp_sy.setValue(y)
        else:
            self.sp_ex.setValue(x); self.sp_ey.setValue(y)
        self._update_est()

    # ==== confirm/start/stop ====
    def _collect(self) -> Settings:
        s = Settings()
        s.weights = self.e_weights.text().strip()
        s.calib_txt = self.e_calib.text().strip()
        s.out_root = self.e_out.text().strip()
        s.scan_name = self.e_name.text().strip()
        s.led_percent = int(self.sp_led.value())
        s.safe_z = float(self.sp_safez.value())
        s.start_x, s.start_y = float(self.sp_sx.value()), float(self.sp_sy.value())
        s.end_x, s.end_y = float(self.sp_ex.value()), float(self.sp_ey.value())
        s.step_x_mm, s.step_y_mm = float(self.sp_dx.value()), float(self.sp_dy.value())
        s.settle_ms = int(self.sp_settle.value())
        s.af_10x_every_n = int(self.sp_afn.value())
        s.af_after_mag_change = self.cb_af_after.isChecked()
        s.af_distance_10x = float(self.sp_af10.value())
        s.af_distance_50x = float(self.sp_af50.value())
        s.conf = float(self.sp_conf.value())
        s.iou = float(self.sp_iou.value())
        s.allow_classes = self.e_allow.text().strip()
        s.target_mag = float(self.cbo_mag.currentText())
        return s

    def _confirm(self):
        s = self._collect()
        msg = json.dumps(asdict(s), indent=2, ensure_ascii=False)
        QMessageBox.information(self, "Review Settings", msg)

    def _start(self):
        if not self.ctrl:
            QMessageBox.warning(self, "Not Connected", "Please connect first.")
            return
        s = self._collect()
        if s.start_x is None or s.end_x is None:
            QMessageBox.warning(self, "Missing Parameters", "Please set start/end points.")
            return
        self.calib_map = parse_calib_um_per_px(s.calib_txt) if s.calib_txt else {}
        self._set_running(True); self.log.clear(); self.pb.setValue(0)
        self.worker = ScanWorker(s, self.ctrl, self.calib_map)
        self.worker.log.connect(lambda t: self.log.appendPlainText(t))
        self.worker.progress.connect(self._on_prog)
        self.worker.error.connect(self._on_err)
        self.worker.finished_ok.connect(self._on_ok)
        self.worker.start()

    def _stop(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
        self._set_running(False)

    def _on_prog(self, done, total):
        self.pb.setMaximum(total); self.pb.setValue(done)

    def _on_err(self, msg):
        QMessageBox.critical(self, "Error", msg)
        self._set_running(False)

    def _on_ok(self):
        QMessageBox.information(self, "Done", "Scan finished.")
        self._set_running(False)

    def _set_running(self, running: bool):
        enabled = not running
        for w in (self.btn_connect, self.btn_disconnect, self.btn_confirm, self.btn_start, self.btn_save, self.btn_load):
            w.setEnabled(enabled)
        self.btn_stop.setEnabled(running)

    # ==== save/load config ====
    def _save_cfg(self):
        s = self._collect()
        p, _ = QFileDialog.getSaveFileName(self, "Save Config", "", "JSON (*.json)")
        if not p: return
        Path(p).write_text(json.dumps(asdict(s), indent=2, ensure_ascii=False), encoding="utf-8")
        QMessageBox.information(self, "Saved", "Configuration saved.")

    def _load_cfg(self):
        p, _ = QFileDialog.getOpenFileName(self, "Load Config", "", "JSON (*.json)")
        if not p: return
        data = json.loads(Path(p).read_text(encoding="utf-8"))
        s = Settings(**data); self.s = s
        self.e_weights.setText(s.weights); self.e_calib.setText(s.calib_txt)
        self.e_out.setText(s.out_root); self.e_name.setText(s.scan_name)
        self.sp_led.setValue(s.led_percent); self.sp_safez.setValue(s.safe_z)
        self.sp_sx.setValue(s.start_x or 0); self.sp_sy.setValue(s.start_y or 0)
        self.sp_ex.setValue(s.end_x or 0); self.sp_ey.setValue(s.end_y or 0)
        self.sp_dx.setValue(s.step_x_mm); self.sp_dy.setValue(s.step_y_mm)
        self.sp_settle.setValue(s.settle_ms)
        self.sp_afn.setValue(s.af_10x_every_n); self.cb_af_after.setChecked(s.af_after_mag_change)
        self.sp_af10.setValue(s.af_distance_10x); self.sp_af50.setValue(s.af_distance_50x)
        self.sp_conf.setValue(s.conf); self.sp_iou.setValue(s.iou)
        self.e_allow.setText(s.allow_classes)
        self.cbo_mag.setCurrentText(str(int(s.target_mag)))
        self._update_est()
        QMessageBox.information(self, "Loaded", "Configuration loaded.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = Main(); w.resize(1000, 740); w.show()
    sys.exit(app.exec())
