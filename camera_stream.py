from flask import Flask, Response, render_template_string, request, jsonify
from picamera2 import Picamera2
from PIL import Image
from libcamera import controls
import cv2
import numpy as np
import io
import json
import time
import threading
import sys
import urllib.request
sys.path.insert(0, '/home/jerica')
from undistort import Undistorter

# ── LED strip (WS2815, 53 LEDs, GRB, SPI at 2.4 MHz) ─────────
# Encoding: 3 SPI bits per WS2815 bit → period = 3/2.4MHz = 1.25µs ✓
#   1-bit: 1 1 0  → T1H=833ns, T1L=417ns
#   0-bit: 1 0 0  → T0H=417ns, T0L=833ns
# 8 WS bits packed into 3 SPI bytes → 9 SPI bytes per 24-bit pixel
LED_OFFSET = 0   # new WS2815 strip: drive from the first LED
LED_COUNT  = 30   # WS2811 diffuse strip (~44 in); 30 addressable pixels, measured via /count_ruler
led_color  = (0, 0, 0)   # last colour set via /set_led; restored after the win sequence

def _encode_color_byte(v):
    """Encode one 8-bit color channel into 3 SPI bytes (3 SPI bits per WS bit)."""
    # Bit layout (MSB first): [1 b7 0][1 b6 0][1 b5 0][1 b4 0][1 b3 0][1 b2 0][1 b1 0][1 b0 0]
    # = 24 bits packed into 3 bytes
    return [
        0x92 | (((v >> 7) & 1) << 6) | (((v >> 6) & 1) << 3) | ((v >> 5) & 1),
        0x49 | (((v >> 4) & 1) << 5) | (((v >> 3) & 1) << 2),
        0x24 | (((v >> 2) & 1) << 7) | (((v >> 1) & 1) << 4) | ((v & 1) << 1),
    ]

def _encode_pixel(r, g, b):
    """Return 9 SPI bytes for one GRB WS2815 pixel."""
    return _encode_color_byte(g) + _encode_color_byte(r) + _encode_color_byte(b)

# LED strip driver — Pi 5 RP1 PIO (hardware-timed WS281x), NOT SPI.
# Data on board.D13 (GPIO13 / physical pin 33). Byte order is RGB.
# The function names below keep the historical _spi_* prefix so the rest of
# the app (/set_led, _win_animation) is untouched, but there is no SPI here.
try:
    import board
    from adafruit_raspberry_pi5_neopixel_write import neopixel_write
    _LED_PIN = board.D13
    LED_AVAILABLE = True

    def _led_write(buf):
        try:
            neopixel_write(_LED_PIN, buf)
        except Exception as e:
            print(f"LED write failed: {e}")

    def _spi_show(r, g, b):
        # Solid fill, RGB order; the first LED_OFFSET pixels are forced off.
        px = bytearray(LED_COUNT * 3)
        for i in range(LED_OFFSET, LED_COUNT):
            o = i * 3
            px[o] = r; px[o + 1] = g; px[o + 2] = b
        _led_write(bytes(px))

    _spi_show(0, 0, 0)   # clear on startup
except Exception as e:
    print(f"LED init failed: {e}")
    LED_AVAILABLE = False
    def _spi_show(r, g, b): pass

# ── IR LED (GPIO 18) ──────────────────────────────────
try:
    from gpiozero import LED as GPIOLED
    ir_led = GPIOLED(18)
    IR_AVAILABLE = True
except Exception as e:
    print(f"IR LED init failed: {e}")
    IR_AVAILABLE = False

app = Flask(__name__)

SENSOR_W, SENSOR_H = 4608, 2592
IMG_W, IMG_H = 2304, 1296
INPUT_HFOV   = 75.0

camera = Picamera2()
config = camera.create_video_configuration(main={"size": (IMG_W, IMG_H), "format": "RGB888"})
camera.configure(config)
camera.start()
time.sleep(1)

fov_pct      = 100
output_hfov  = 60.0
undistort_on = True
focus_pos    = 0.0    # 0 = autofocus; >0 = manual lens position (dioptres)
undistorter  = Undistorter(width=IMG_W, height=IMG_H,
                           input_hfov_deg=INPUT_HFOV,
                           output_hfov_deg=output_hfov)

state_lock         = threading.Lock()
frame_lock         = threading.Lock()
latest_frame       = None
filter_on          = False
filter_brightness  = 60     # luminance threshold 0-255

# ── Isometric grid state ──────────────────────────────
grid_holes         = []   # refined hole coords in original image: [{px, py}, ...]
ir_state           = False  # server-side IR LED state, kept in sync with /set_ir
SETTINGS_FILE      = "/home/jerica/settings.json"

def _load_settings():
    try:
        with open(SETTINGS_FILE) as f:
            return json.load(f)
    except Exception:
        return {}

def _save_settings(key, value):
    s = _load_settings()
    s[key] = value
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(s, f)
    except Exception as e:
        print(f"Settings save failed: {e}")

_settings = _load_settings()
diff_threshold     = _settings.get("diff_threshold", 20)   # absolute thr, used by /scan_holes diagnostic
adapt_margin       = _settings.get("adapt_margin", 30)     # live detection: margin above local background = filled
last_scan_results  = []   # most recent scan_holes result list

# ── Template store ────────────────────────────────────
TEMPLATES_FILE = "/home/jerica/templates.json"

def _load_templates():
    try:
        with open(TEMPLATES_FILE) as f:
            return json.load(f)
    except Exception:
        return []

def _atomic_write_json(path, obj):
    """Write JSON atomically: temp file in the same dir, then os.replace().
    A rename only needs write permission on the directory, so this can
    overwrite a target owned by another user (e.g. a stale root-owned file)."""
    import os, tempfile
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_calib_")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f)
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except Exception:
            pass
        raise


def _save_templates_to_disk(templates):
    try:
        with open(TEMPLATES_FILE, "w") as f:
            json.dump(templates, f)
    except Exception as e:
        print(f"Templates save failed: {e}")

# ── Pattern detection state ───────────────────────────
target_pattern               = set()   # hole indices that must be filled
current_template             = None    # name of loaded template (identity for win trigger)
last_trigger                 = None    # summary dict of last win-trigger send (for status)
consecutive_matches          = 0
puzzle_solved                = False
detection_running            = False
detection_strict             = False   # True = background nodes must also be empty
detection_live_state         = {}      # idx -> bool (last frame's per-hole result)
detection_last_diffs         = {}      # idx -> float (last frame's raw diffs, for debug)
REQUIRED_CONSECUTIVE_MATCHES = 10
NODE_DEBOUNCE_FRAMES         = 2     # consecutive confirming frames required to flip a node
HYST_FRACTION                = 0.15  # hysteresis half-gap as a fraction of diff_threshold
HYST_MIN_MARGIN              = 6     # floor for the half-gap, in luminance counts
ADAPT_BASELINE_PCTL          = 30    # local background = this percentile of nearby-hole diffs
ADAPT_RADIUS_FACTOR          = 2.2   # baseline window radius, in units of col_dx
ADAPT_MIN_DIFF               = 8    # absolute floor a peg must clear regardless of local baseline
NEIGHBOR_FILL_FRAC           = 0.6   # fraction of in-pattern neighbors filled to infer a missed peg
NEIGHBOR_MIN_FILLED          = 2     # ...and at least this many (so edge holes are not filled on one)

# Fixed camera exposure during detection. Auto-exposure lets the IR-on frame
# (plus bright LED backlight) clip the sensor at the top of the board, which
# collapses the IR differential to ~0 so filled pegs read as empty. A fixed,
# lower exposure keeps headroom so peg-vs-empty contrast survives everywhere.
DETECT_EXPOSURE_US           = 12000  # microseconds (~1/83s); no clipping, strong IR signal
DETECT_GAIN                  = 1.0    # lowest analogue gain = least sensor noise
detection_inferred           = []    # idx list: pattern nodes filled by neighbor inference (for UI)

def apply_crop(pct):
    w = int(SENSOR_W * pct / 100)
    h = int(SENSOR_H * pct / 100)
    x = (SENSOR_W - w) // 2
    y = (SENSOR_H - h) // 2
    camera.set_controls({"ScalerCrop": (x, y, w, h)})

def _apply_calibration(data):
    """Apply a loaded calibration dict (or legacy bare list) to live state.
    Sets grid_holes + camera globals, applies crop/undistorter/focus.
    Returns the applied values for JSON responses."""
    global grid_holes, fov_pct, output_hfov, undistort_on, focus_pos, undistorter
    cam   = data if isinstance(data, dict) else {}
    holes = data.get("holes", []) if isinstance(data, dict) else data  # legacy list
    with state_lock:
        grid_holes = holes
    if "fov_pct" in cam:
        fov_pct = int(cam["fov_pct"]); apply_crop(fov_pct)
    if "output_hfov" in cam:
        output_hfov = float(cam["output_hfov"])
        new_und = Undistorter(width=IMG_W, height=IMG_H,
                              input_hfov_deg=INPUT_HFOV, output_hfov_deg=output_hfov)
        with state_lock:
            undistorter = new_und
    if "undistort_on" in cam:
        with state_lock:
            undistort_on = bool(cam["undistort_on"])
    if "focus" in cam:
        focus_pos = float(cam["focus"])
        if focus_pos <= 0:
            camera.set_controls({"AfMode": controls.AfModeEnum.Continuous,
                                 "AfSpeed": controls.AfSpeedEnum.Fast,
                                 "AfRange": controls.AfRangeEnum.Macro})
        else:
            camera.set_controls({"AfMode": controls.AfModeEnum.Manual,
                                 "LensPosition": focus_pos})
    return {"count": len(holes), "holes": holes, "fov_pct": fov_pct,
            "output_hfov": output_hfov, "undistort_on": undistort_on, "focus": focus_pos}


apply_crop(fov_pct)
camera.set_controls({"AfMode": controls.AfModeEnum.Continuous,
                     "AfSpeed": controls.AfSpeedEnum.Fast})

# Auto-restore saved calibration on startup (survives stream restarts)
try:
    with open("/home/jerica/grid_holes.json") as f:
        _apply_calibration(json.load(f))
    print(f"Calibration restored: {len(grid_holes)} holes")
except FileNotFoundError:
    print("No saved calibration found at boot")
except Exception as e:
    print(f"Boot calibration load failed: {e}")


def capture_loop():
    global latest_frame
    while True:
        buf = io.BytesIO()
        camera.capture_file(buf, format="jpeg")
        buf.seek(0)
        img = Image.open(buf).rotate(180, expand=True)
        with state_lock:
            do_undistort = undistort_on
            und          = undistorter
            do_filter    = filter_on
            filt_thresh  = filter_brightness
        if do_undistort:
            img_np = np.array(img)
            img_np = und(img_np)
            img = Image.fromarray(img_np)
        if do_filter:
            img_np = np.array(img)
            lum    = (0.299 * img_np[:,:,0] +
                      0.587 * img_np[:,:,1] +
                      0.114 * img_np[:,:,2])
            mask         = lum < filt_thresh
            img_np[mask] = 0
            img = Image.fromarray(img_np)
        out = io.BytesIO()
        img.save(out, format="jpeg", quality=85)
        with frame_lock:
            latest_frame = out.getvalue()

threading.Thread(target=capture_loop, daemon=True).start()


def _hole_lum(img, px, py, half=4):
    h, w = img.shape[:2]
    x1, y1 = max(0, px - half), max(0, py - half)
    x2, y2 = min(w, px + half), min(h, py + half)
    crop = img[y1:y2, x1:x2]
    return float(np.mean(0.299*crop[:,:,0] + 0.587*crop[:,:,1]
                         + 0.114*crop[:,:,2])) if crop.size else 0.0


def _wheel(pos):
    """Rainbow colour wheel: pos 0-255 → (r, g, b)."""
    pos = pos % 256
    if pos < 85:
        return (255 - pos * 3, pos * 3, 0)
    if pos < 170:
        pos -= 85
        return (0, 255 - pos * 3, pos * 3)
    pos -= 170
    return (pos * 3, 0, 255 - pos * 3)

def _spi_show_pixels(pixels):
    """Send an arbitrary per-LED colour list (length LED_COUNT, (r,g,b) tuples).
    RGB byte order via the PIO driver; first LED_OFFSET pixels forced off."""
    if not LED_AVAILABLE:
        return
    px = bytearray(LED_COUNT * 3)
    for i, (r, g, b) in enumerate(pixels):
        if i < LED_OFFSET:
            continue
        o = i * 3
        px[o] = r; px[o + 1] = g; px[o + 2] = b
    _led_write(bytes(px))

def _win_animation():
    """Celebration sequence: strobe → rainbow wipe → colour chase → breathe green."""
    if not LED_AVAILABLE:
        return

    n = LED_COUNT - LED_OFFSET   # number of active LEDs

    # 1. Rainbow wipe — LEDs light up one-by-one along the strip
    pixels = [(0, 0, 0)] * LED_COUNT
    for i in range(n):
        pixels[i + LED_OFFSET] = _wheel(i * 256 // n)
        _spi_show_pixels(pixels)
        time.sleep(0.04)
    time.sleep(0.4)

    # 2. Colour chase — whole strip cycles through rainbow together
    for frame in range(120):
        pixels = [(0, 0, 0)] * LED_COUNT
        for i in range(n):
            pixels[i + LED_OFFSET] = _wheel((i * 256 // n + frame * 4) % 256)
        _spi_show_pixels(pixels)
        time.sleep(0.04)

    # 3. Breathe green — six slow pulses then off
    for _ in range(6):
        for v in list(range(0, 256, 6)) + list(range(255, -1, -6)):
            _spi_show(0, v, 0)
            time.sleep(0.008)

    # Return to the colour the LED was set at before the win
    _spi_show(*led_color)


def _set_current_template(name):
    """Record which template/pattern is currently loaded (identity source for the win trigger)."""
    global current_template
    with state_lock:
        current_template = name


def _send_win_trigger(code, url=None, path=None):
    """POST a single-letter win code to the external receiver. Non-blocking; never raises.

    URL resolution (first match wins):
      1. explicit `url`            — full URL override (used by /test_trigger)
      2. win_trigger.base_url + `path` — on-site you set base_url once; each
                                         template supplies its own path.
      3. win_trigger.url           — legacy single fixed URL fallback."""
    global last_trigger
    cfg = (_load_settings().get("win_trigger") or {})
    if not url:
        base = (cfg.get("base_url") or "").rstrip("/")
        if path and base:
            url = base + "/" + str(path).lstrip("/")
        else:
            url = cfg.get("url")
    if not cfg.get("enabled", False) or not url:
        last_trigger = {"time": time.strftime("%H:%M:%S"), "code": code,
                        "ok": False, "err": "disabled or no url"}
        print(f"[win_trigger] skipped (disabled/no url) code={code!r}")
        return
    timeout = float(cfg.get("timeout_s", 2.0))
    retries = int(cfg.get("retries", 1))
    payload = { cfg.get("field_code", "code"): code }
    body = json.dumps(payload).encode()

    def _worker():
        global last_trigger
        err = None
        for _ in range(max(1, retries + 1)):
            try:
                req = urllib.request.Request(
                    url, data=body,
                    headers={"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    rc = resp.getcode()
                last_trigger = {"time": time.strftime("%H:%M:%S"), "code": code,
                                "ok": True, "status": rc, "url": url}
                print(f"[win_trigger] sent code={code!r} -> {url} ({rc})")
                return
            except Exception as e:
                err = str(e)
                time.sleep(0.2)
        last_trigger = {"time": time.strftime("%H:%M:%S"), "code": code,
                        "ok": False, "err": err, "url": url}
        print(f"[win_trigger] FAILED code={code!r} -> {url}: {err}")

    threading.Thread(target=_worker, daemon=True).start()


def _resolve_pattern_entry(entry):
    """A pattern_map value is either a bare code string or a {code, path, url} dict.
    -> (code, url|None, path|None). `path` joins onto win_trigger.base_url;
    `url` is a full-URL override."""
    if isinstance(entry, dict):
        return entry.get("code"), entry.get("url"), entry.get("path")
    return entry, None, None


def _fire_win_trigger_for_current():
    """Look up the loaded template's identity in pattern_map and fire the trigger once."""
    with state_lock:
        tmpl = current_template
    entry = (_load_settings().get("pattern_map") or {}).get(tmpl)
    code, url, path = _resolve_pattern_entry(entry)
    if code:
        _send_win_trigger(code, url, path)
    else:
        print(f"[win_trigger] win but no pattern_map entry for template={tmpl!r}; skipped")


def _win_then_reset():
    """Play the win animation, THEN fire the trigger, then auto-reset for next participant."""
    _win_animation()
    _fire_win_trigger_for_current()
    global consecutive_matches, puzzle_solved
    with state_lock:
        consecutive_matches = 0
        puzzle_solved       = False


def detection_loop():
    """
    Continuous debounced pattern-match loop.
    Toggles IR on/off, computes per-hole differential luminance, checks
    target vs background nodes, and increments a consecutive-match counter.
    Triggers win state when REQUIRED_CONSECUTIVE_MATCHES frames all match.
    """
    global consecutive_matches, puzzle_solved, detection_running, detection_last_diffs, detection_live_state, detection_inferred

    with state_lock:
        holes   = list(grid_holes)
        thresh  = diff_threshold
        pattern = set(target_pattern)

    if not holes:
        with state_lock:
            detection_running = False
        return

    # Lock AE + fix exposure low enough to avoid sensor clipping (see DETECT_EXPOSURE_US).
    camera.set_controls({"AeEnable": False,
                         "ExposureTime": DETECT_EXPOSURE_US,
                         "AnalogueGain": DETECT_GAIN})
    time.sleep(0.5)   # exposure change needs several frames to settle

    # Persistent per-node state for hysteresis + temporal debounce
    node_state   = {}   # idx -> bool: committed (debounced) filled/empty
    node_pending = {}   # idx -> int:  consecutive frames the pending flip has held

    # ── Precompute lattice geometry once (holes fixed for this session) ──
    _bins = {}
    for _h in holes:
        _bins.setdefault(int(round(_h["py"] / 20)), []).append(_h["px"])
    _dxs = []
    for _xs in _bins.values():
        _xs_s = sorted(_xs)
        for _j in range(len(_xs_s) - 1):
            _d = _xs_s[_j + 1] - _xs_s[_j]
            if _d > 5:
                _dxs.append(_d)
    col_dx = float(np.median(_dxs)) if _dxs else 50.0

    _pts      = [(int(round(h["px"])), int(round(h["py"]))) for h in holes]
    _n        = len(_pts)
    nb_lo     = (0.55 * col_dx) ** 2
    nb_hi     = (1.45 * col_dx) ** 2
    base_r2   = (ADAPT_RADIUS_FACTOR * col_dx) ** 2
    neighbors = [[] for _ in range(_n)]
    base_win  = [[] for _ in range(_n)]
    for _a in range(_n):
        _ax, _ay = _pts[_a]
        for _b in range(_a + 1, _n):
            _bx, _by = _pts[_b]
            _d2 = (_bx - _ax) ** 2 + (_by - _ay) ** 2
            if _d2 <= base_r2:
                base_win[_a].append(_b)
                base_win[_b].append(_a)
                if nb_lo <= _d2 <= nb_hi:
                    neighbors[_a].append(_b)
                    neighbors[_b].append(_a)

    try:
        while True:
            with state_lock:
                if not detection_running:
                    break
                thresh  = diff_threshold
                pattern = set(target_pattern)

            # ── Capture IR-on frame ──────────────────────────
            if IR_AVAILABLE: ir_led.on()
            time.sleep(0.20)          # ~6 frames at 30 fps to flush IR change through
            with frame_lock: frame_on = latest_frame

            # ── Capture IR-off frame ─────────────────────────
            if IR_AVAILABLE: ir_led.off()
            time.sleep(0.20)
            with frame_lock: frame_off = latest_frame

            if not frame_on or not frame_off or frame_on is frame_off:
                continue

            img_on  = np.array(Image.open(io.BytesIO(frame_on)))
            img_off = np.array(Image.open(io.BytesIO(frame_off)))

            # ── Compute per-hole IR diffs (parallax-aware) ───
            # Off-axis pegs sit off the calibrated hole centre (camera looks up
            # at a board with depth). Instead of sampling only the centre patch,
            # compute the whole-frame smoothed IR differential once and take the
            # MAX within a small search window around each hole, so the peg is
            # captured wherever parallax has shifted it. Radius is capped below
            # half the lattice step so a window can never reach an adjacent hole.
            _lon  = (0.299 * img_on[..., 0]  + 0.587 * img_on[..., 1]  + 0.114 * img_on[..., 2]).astype(np.float32)
            _loff = (0.299 * img_off[..., 0] + 0.587 * img_off[..., 1] + 0.114 * img_off[..., 2]).astype(np.float32)
            dimg  = cv2.blur(_lon - _loff, (9, 9))   # ~8x8 patch mean, matches _hole_lum
            _H, _W = dimg.shape
            SEARCH_R = int(max(3, min(0.35 * col_dx, 0.45 * col_dx)))
            diffs = {}
            for i, hole in enumerate(holes):
                px = int(round(hole["px"]))
                py = int(round(hole["py"]))
                x1 = max(0, px - SEARCH_R); x2 = min(_W, px + SEARCH_R + 1)
                y1 = max(0, py - SEARCH_R); y2 = min(_H, py + SEARCH_R + 1)
                if x2 <= x1 or y2 <= y1:
                    diffs[i] = 0.0
                else:
                    diffs[i] = float(dimg[y1:y2, x1:x2].max())

            # ── Evaluate pattern match (adaptive + neighbor fill-in) ──
            with state_lock:
                strict   = detection_strict
                pattern  = set(target_pattern)
                a_margin = adapt_margin

            all_idx    = set(range(len(holes)))
            background = all_idx - pattern

            def _adapt_band(i):
                # Per-hole threshold: a peg must stand out from its LOCAL background.
                win = [diffs[j] for j in base_win[i]] or [diffs.get(i, 0)]
                baseline = float(np.percentile(win, ADAPT_BASELINE_PCTL))
                on_i = max(ADAPT_MIN_DIFF, baseline + a_margin)
                hyst = max(HYST_MIN_MARGIN, on_i * HYST_FRACTION)
                return on_i, on_i - hyst

            def _step_node(i, d):
                # Debounced hysteresis against the hole's own adaptive band.
                on_i, off_i = _adapt_band(i)
                cur  = node_state.get(i, False)
                want = (d > on_i) if not cur else (d > off_i)
                if want != cur:
                    node_pending[i] = node_pending.get(i, 0) + 1
                    if node_pending[i] >= NODE_DEBOUNCE_FRAMES:
                        node_state[i]   = want
                        node_pending[i] = 0
                else:
                    node_pending[i] = 0
                return node_state.get(i, False)

            # Step every target node (debounced, adaptive)
            measured = {i: _step_node(i, diffs.get(i, 0)) for i in pattern}

            # ── Neighbor fill-in: recover pegs missed INSIDE a filled cluster ──
            inferred = set()
            for i in pattern:
                if measured[i]:
                    continue
                pn = [j for j in neighbors[i] if j in pattern]
                fn = sum(1 for j in pn if measured[j])
                if pn and fn >= NEIGHBOR_MIN_FILLED and fn >= len(pn) * NEIGHBOR_FILL_FRAC:
                    inferred.add(i)

            live     = {i: (measured[i] or (i in inferred)) for i in pattern}
            is_match = len(pattern) > 0 and all(live.values())

            # Background nodes must be EMPTY (debounced) — only in strict mode
            if strict:
                for i in background:
                    if _step_node(i, diffs.get(i, 0)) and is_match:
                        is_match = False

            with state_lock:
                detection_live_state = live
                detection_last_diffs = dict(diffs)
                detection_inferred   = list(inferred)

            # ── Debounce counter ─────────────────────────────
            with state_lock:
                if is_match:
                    consecutive_matches = min(consecutive_matches + 1,
                                              REQUIRED_CONSECUTIVE_MATCHES)
                    if (consecutive_matches >= REQUIRED_CONSECUTIVE_MATCHES
                            and not puzzle_solved):
                        puzzle_solved = True
                        threading.Thread(target=_win_then_reset, daemon=True).start()
                else:
                    consecutive_matches = 0
                    if puzzle_solved:
                        puzzle_solved = False

    finally:
        camera.set_controls({"AeEnable": True})
        if IR_AVAILABLE: ir_led.off()
        with state_lock:
            detection_running = False


def generate():
    while True:
        with frame_lock:
            frame = latest_frame
        if frame:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.033)


PAGE = """
<!DOCTYPE html>
<html>
<head>
  <title>Pi Camera Stream</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: #111; color: #fff; font-family: sans-serif;
           display: flex; flex-direction: column; height: 100vh; overflow: hidden; }
    header { padding: 8px 16px; background: #1a1a1a; border-bottom: 1px solid #333;
             display: flex; align-items: center; gap: 14px; flex-wrap: wrap; flex-shrink: 0; }
    h2 { font-size: 15px; letter-spacing: .5px; white-space: nowrap; }
    .sep { width: 1px; height: 24px; background: #444; }
    label { font-size: 12px; color: #aaa; white-space: nowrap; }
    input[type=range] { width: 180px; accent-color: #4af; vertical-align: middle; }
    button { padding: 5px 12px; border: none; border-radius: 5px;
             font-size: 12px; cursor: pointer; font-weight: bold; }
    .stream-wrap { flex: 1; display: flex; align-items: center;
                   justify-content: center; overflow: hidden; position: relative; }

    /* Container that matches the displayed image size */
    #imgWrap { position: relative; display: inline-block; line-height: 0; }
    #stream  { display: block; max-width: 100%; max-height: calc(100vh - 50px); }
    #overlay { position: absolute; top: 0; left: 0; pointer-events: none; }
    #imgWrap.calib-mode { cursor: crosshair; }
  </style>
</head>
<body>
<header>
  <h2>Pi Camera Live Stream</h2>
  <div class="sep"></div>
  <label>Field of View
    <input type="range" id="fov" min="25" max="100" value="100"
           oninput="setFov(this.value)"> <span id="fovVal">100%</span>
  </label>
  <div class="sep"></div>
  <button id="undistBtn" onclick="toggleUndist()"
          style="background:#363;color:#afa">🔭 Undistort: ON</button>
  <div class="sep"></div>
  <label>Focus
    <input type="range" id="focusSlider" min="0" max="30" step="0.1" value="0"
           style="width:110px;accent-color:#4af;vertical-align:middle"
           oninput="setFocus(this.value)">
    <span id="focusVal">auto</span>
  </label>
  <div class="sep"></div>
  <label>Brightness
    <input type="range" id="threshSlider" min="0" max="255" value="60"
           style="width:100px;accent-color:#f70;vertical-align:middle"
           oninput="onThreshChange(this.value)">
    <span id="threshVal">60</span>
  </label>
  <button id="btnFilter" onclick="toggleFilter()" style="background:#444;color:#fff">🔆 Filter: OFF</button>
  <div class="sep"></div>
  <button id="btnCalib" onclick="startCalib()" style="background:#444;color:#fff">📐 Calibrate</button>
  <button onclick="loadSavedGrid()" style="background:#444;color:#fff">📂 Load Last</button>
  <button id="btnSaveCalib" onclick="saveCalibration()" style="background:#444;color:#fff;display:none">💾 Save Calibration</button>
  <button id="btnDone" onclick="phaseDone()" style="display:none;background:#363;color:#afa">✓ Done</button>
  <span id="calibStatus" style="font-size:12px;color:#fa0;"></span>
  <button id="btnScan" onclick="scanHoles()" style="display:none;background:#26a;color:#fff">🔍 Scan</button>
  <button id="btnDetectTmpl" onclick="detectTemplate()" style="display:none;background:#0a5;color:#fff;font-weight:bold">🔍 Detect Template</button>
  <div id="tmplDropWrap" style="position:relative;display:none">
    <button onclick="toggleTmplDropdown()" style="background:#444;color:#fff">📋 Templates ▾</button>
    <div id="tmplDropdown" style="display:none;position:absolute;top:100%;left:0;background:#222;
         border:1px solid #555;border-radius:5px;min-width:180px;z-index:200;padding:4px 0;">
      <div id="tmplList"></div>
      <hr style="border-color:#444;margin:4px 0">
      <div style="padding:4px 10px;cursor:pointer;font-size:12px;color:#4af"
           onclick="promptSaveTemplate()">➕ Save current scan as template…</div>
    </div>
  </div>
  <button id="btnSelectHoles" onclick="toggleSelectMode()" style="display:none;background:#444;color:#fff">🎯 Select Holes</button>
  <button id="btnDetect" onclick="toggleDetection()" style="display:none;background:#444;color:#fff">▶ Detect</button>
  <span id="detectBar" style="font-size:12px;color:#afa;font-family:monospace;white-space:nowrap;"></span>
  <button id="btnClearGrid" onclick="clearGrid()" style="display:none;background:#444;color:#fff">✕</button>
  <div class="sep"></div>
  <label>IR Sensitivity
    <input type="range" id="diffSlider" min="0" max="120" value="30"
           style="width:90px;accent-color:#0f0;vertical-align:middle"
           oninput="onDiffChange(this.value)">
    <span id="diffVal">30</span>
  </label>
  <div class="sep"></div>
  <label>LED
    <input type="color" id="ledColor" value="#ff0000"
           oninput="setLed()" style="width:36px;height:24px;border:none;cursor:pointer;vertical-align:middle">
  </label>
  <input type="range" id="ledBright" min="0" max="255" value="38"
         style="width:90px;accent-color:#ff0;vertical-align:middle" oninput="setLed()">
  <button onclick="setLedOff()" style="background:#222;color:#aaa;border:1px solid #555">Off</button>
  <button id="btnIR" onclick="toggleIR()" style="background:#444;color:#fff">💡 IR: OFF</button>
  <button onclick="testIR()" style="background:#555;color:#fff">🔦 Test IR</button>
</header>

<div class="stream-wrap">
  <div id="imgWrap">
    <img id="stream" src="/stream" />
    <canvas id="overlay"></canvas>
  </div>
</div>

<!-- Win overlay -->
<div id="winOverlay" style="display:none;position:fixed;top:0;left:0;width:100%;height:100%;
     background:rgba(0,0,0,0.88);z-index:100;align-items:center;justify-content:center;
     flex-direction:column;gap:20px;">
  <div style="font-size:72px">🎉</div>
  <div style="font-size:44px;color:#ffe000;font-weight:bold;letter-spacing:2px">PUZZLE SOLVED!</div>
  <div id="winResetMsg" style="font-size:18px;color:#aaa;display:none">Resetting for next player…</div>
</div>

<script>
  // ── Stream controls ────────────────────────────────
  function setFov(v) {
    document.getElementById('fovVal').textContent = v + '%';
    fetch('/set_fov?pct=' + v);
  }
  function toggleUndist() {
    fetch('/toggle_undistort').then(r => r.json()).then(d => {
      const btn = document.getElementById('undistBtn');
      btn.textContent      = '🔭 Undistort: ' + (d.on ? 'ON' : 'OFF');
      btn.style.background = d.on ? '#363' : '#444';
      btn.style.color      = d.on ? '#afa' : '#fff';
    });
  }
  function setLed() {
    const hex    = document.getElementById('ledColor').value;
    const bright = parseInt(document.getElementById('ledBright').value) / 255;
    const r = Math.round(parseInt(hex.slice(1,3),16) * bright);
    const g = Math.round(parseInt(hex.slice(3,5),16) * bright);
    const b = Math.round(parseInt(hex.slice(5,7),16) * bright);
    fetch(`/set_led?r=${r}&g=${g}&b=${b}`);
  }
  function setLedOff() {
    document.getElementById('ledBright').value = 0;
    fetch('/set_led?r=0&g=0&b=0');
  }
  function setFocus(v) {
    const f = parseFloat(v);
    document.getElementById('focusVal').textContent = f === 0 ? 'auto' : f.toFixed(1) + ' D';
    fetch('/set_focus?pos=' + f);
  }
  function onThreshChange(v) {
    document.getElementById('threshVal').textContent = v;
    fetch('/set_brightness_threshold?val=' + v);
  }
  let irOn = false;
  function toggleIR() {
    irOn = !irOn;
    fetch('/set_ir?on=' + (irOn ? '1' : '0')).then(r => r.json()).then(d => {
      const btn = document.getElementById('btnIR');
      btn.textContent      = '💡 IR: ' + (irOn ? 'ON' : 'OFF');
      btn.style.background = irOn ? '#a70' : '#444';
    });
  }
  function testIR() {
    fetch('/test_ir').then(r => r.json()).then(d => {
      if (d.error) alert('IR test failed: ' + d.error);
    });
  }
  function toggleFilter() {
    fetch('/toggle_filter').then(r => r.json()).then(d => {
      const btn = document.getElementById('btnFilter');
      btn.textContent      = '🔆 Filter: ' + (d.on ? 'ON' : 'OFF');
      btn.style.background = d.on ? '#f70' : '#444';
      btn.style.color      = d.on ? '#000' : '#fff';
    });
  }

  // ── Canvas overlay ─────────────────────────────────
  const IMG_W = 2304, IMG_H = 1296;
  const streamImg = document.getElementById('stream');
  // Auto-load saved calibration into the overlay once on initial page load
  let _autoLoadedGrid = false;
  function _autoLoadGridOnce() {
    if (_autoLoadedGrid) return;
    _autoLoadedGrid = true;
    loadSavedGrid();   // 404 path sets a harmless status; safe if no calibration exists
  }
  streamImg.addEventListener('load', _autoLoadGridOnce);
  // If the MJPEG frame already arrived before this listener attached, fire now
  if (streamImg.complete && streamImg.naturalWidth > 0) _autoLoadGridOnce();
  const wrap      = document.getElementById('imgWrap');
  const overlay   = document.getElementById('overlay');

  function resizeOverlay() {
    const w = streamImg.offsetWidth, h = streamImg.offsetHeight;
    if (w > 0 && h > 0 && (overlay.width !== w || overlay.height !== h)) {
      overlay.width  = w;
      overlay.height = h;
    }
    redrawOverlay();
  }
  // Only resize when layout actually changes — NOT on every MJPEG frame
  new ResizeObserver(resizeOverlay).observe(streamImg);
  window.addEventListener('resize', resizeOverlay);

  // ── Calibration: line-drawing tool ────────────────
  // Phase 1: draw ROW lines  (one per row of holes, ~horizontal)
  // Phase 2: draw COLUMN lines (one per column, ~vertical)
  // Intersections of row × column lines become hole positions.
  // Odd rows get a half-step rightward shift for the isometric stagger.
  const PHASE_NONE = 0, PHASE_ROW_LINES = 1, PHASE_COL_LINES = 2;
  let calibPhase   = PHASE_NONE;
  let rowLines     = [];   // [{x1,y1,x2,y2}, …]
  let colLines     = [];   // [{x1,y1,x2,y2}, …]
  let lineStart    = null; // first click of the line being drawn
  let mousePos     = null; // current cursor for live preview
  let holeState    = [];   // [{px,py,occupied,idx}, …]
  let triangleData = [];   // [{orientation,corners,pegs}, …] from last detect

  // ── Pattern / Detection state ──────────────────────
  let patternMode   = false;
  let patternSet    = new Set();   // hole indices marked as targets
  let detectRunning = false;
  let detectPollId  = null;

  function setStatus(msg) { document.getElementById('calibStatus').textContent = msg; }

  function startCalib() {
    calibPhase = PHASE_ROW_LINES;
    rowLines = []; colLines = []; lineStart = null; holeState = []; triangleData = [];
    wrap.classList.add('calib-mode');
    document.getElementById('btnDone').style.display         = '';
    document.getElementById('btnScan').style.display         = 'none';
    document.getElementById('btnTriangle').style.display     = 'none';
    document.getElementById('btnClearGrid').style.display    = 'none';
    setStatus('① Draw ROW lines — click two points along each row of holes, then ✓ Done');
    redrawOverlay();
  }

  function phaseDone() {
    if (calibPhase === PHASE_ROW_LINES) {
      if (rowLines.length < 2) { setStatus('⚠ Need at least 2 row lines'); return; }
      calibPhase = PHASE_COL_LINES;
      lineStart = null;
      setStatus('② Draw COLUMN lines — click two points along each column of holes, then ✓ Done');
      redrawOverlay();
    } else if (calibPhase === PHASE_COL_LINES) {
      if (colLines.length < 2) { setStatus('⚠ Need at least 2 column lines'); return; }
      calibPhase = PHASE_NONE;
      lineStart = null;
      wrap.classList.remove('calib-mode');
      document.getElementById('btnDone').style.display = 'none';
      sendLines();
    }
  }

  function clearGrid() {
    if (detectRunning) { fetch('/stop_detection'); detectRunning = false; }
    if (detectPollId)  { clearInterval(detectPollId); detectPollId = null; }
    calibPhase = PHASE_NONE;
    rowLines = []; colLines = []; lineStart = null; holeState = []; triangleData = [];
    patternSet = new Set(); patternMode = false;
    loadedTemplateName = null; liveState = {}; wasSolved = false;
    wrap.classList.remove('calib-mode');
    document.getElementById('btnDone').style.display           = 'none';
    document.getElementById('btnScan').style.display           = 'none';
    document.getElementById('btnDetectTmpl').style.display     = 'none';
    document.getElementById('tmplDropWrap').style.display      = 'none';
    document.getElementById('btnSelectHoles').style.display    = 'none';
    document.getElementById('btnDetect').style.display         = 'none';
    document.getElementById('detectBar').textContent           = '';
    document.getElementById('btnClearGrid').style.display      = 'none';
    document.getElementById('winOverlay').style.display        = 'none';
    setStatus('');
    redrawOverlay();
  }

  // ── Line drawing: first click = start, second click = commit ──
  wrap.addEventListener('click', e => {
    // Select mode: click a hole to toggle it as a target
    if ((patternMode || selectMode) && holeState.length > 0) {
      const rect = wrap.getBoundingClientRect();
      const cx = e.clientX - rect.left, cy = e.clientY - rect.top;
      const sx = streamImg.offsetWidth / IMG_W, sy = streamImg.offsetHeight / IMG_H;
      let best = -1, bestDist = 22;
      holeState.forEach((h, i) => {
        const d = Math.hypot(h.px * sx - cx, h.py * sy - cy);
        if (d < bestDist) { bestDist = d; best = i; }
      });
      if (best >= 0) {
        const idx = holeState[best].idx;
        if (selectMode) {
          // In select mode: toggle locally, no backend call needed yet
          if (patternSet.has(idx)) patternSet.delete(idx); else patternSet.add(idx);
          setStatus('🎯 ' + patternSet.size + ' holes selected — click ✓ Done Selecting when finished');
          redrawOverlay();
        } else {
          fetch('/toggle_pattern_node?idx=' + idx).then(r => r.json()).then(d => {
            if (d.state === 'added') patternSet.add(idx); else patternSet.delete(idx);
            const btn = document.getElementById('btnDetect');
            btn.style.display = patternSet.size > 0 ? '' : 'none';
            setStatus('🎯 Pattern: ' + patternSet.size + ' target node' + (patternSet.size !== 1 ? 's' : '') + ' — click ▶ Detect when ready');
            redrawOverlay();
          });
        }
      }
      return;
    }
    if (calibPhase === PHASE_NONE) return;
    if (e.target !== wrap && e.target !== streamImg) return;
    const rect = wrap.getBoundingClientRect();
    const pt = { x: e.clientX - rect.left, y: e.clientY - rect.top };
    if (lineStart === null) {
      lineStart = pt;
    } else {
      const line = { x1: lineStart.x, y1: lineStart.y, x2: pt.x, y2: pt.y };
      if (calibPhase === PHASE_ROW_LINES) {
        rowLines.push(line);
        setStatus('① Row lines: ' + rowLines.length + ' drawn — ✓ Done when finished');
      } else {
        colLines.push(line);
        setStatus('② Column lines: ' + colLines.length + ' drawn — ✓ Done when finished');
      }
      lineStart = null;
    }
    redrawOverlay();
  });

  // Live preview while drawing
  wrap.addEventListener('mousemove', e => {
    if (calibPhase === PHASE_NONE || lineStart === null) return;
    const rect = wrap.getBoundingClientRect();
    mousePos = { x: e.clientX - rect.left, y: e.clientY - rect.top };
    redrawOverlay();
  });

  // Ctrl+Z: undo last calibration action
  document.addEventListener('keydown', e => {
    if (!(e.ctrlKey && e.key === 'z') && !(e.metaKey && e.key === 'z')) return;
    if (calibPhase === PHASE_NONE) return;
    e.preventDefault();
    if (lineStart !== null) {
      // Cancel pending first click
      lineStart = null;
      mousePos = null;
    } else if (calibPhase === PHASE_ROW_LINES && rowLines.length > 0) {
      rowLines.pop();
      setStatus('① Row lines: ' + rowLines.length + ' drawn — ✓ Done when finished');
    } else if (calibPhase === PHASE_COL_LINES && colLines.length > 0) {
      colLines.pop();
      setStatus('② Column lines: ' + colLines.length + ' drawn — ✓ Done when finished');
    }
    redrawOverlay();
  });

  function sendLines() {
    setStatus('⏳ Computing grid…');
    const sx = IMG_W / streamImg.offsetWidth, sy = IMG_H / streamImg.offsetHeight;
    const toImg = l => [[Math.round(l.x1*sx), Math.round(l.y1*sy)],
                        [Math.round(l.x2*sx), Math.round(l.y2*sy)]];
    fetch('/set_grid_from_lines', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ row_lines: rowLines.map(toImg), col_lines: colLines.map(toImg) })
    }).then(r => r.json()).then(d => {
      holeState = d.holes.map((h, i) => ({ px: h.px, py: h.py, occupied: null, idx: i }));
      triangleData = []; patternSet.clear();
      setStatus('✓ ' + d.count + ' holes mapped — click 🔍 Detect Template');
      document.getElementById('btnScan').style.display         = 'none';
      document.getElementById('btnDetectTmpl').style.display   = '';
      document.getElementById('tmplDropWrap').style.display    = '';
      document.getElementById('btnSelectHoles').style.display  = '';
      document.getElementById('btnDetect').style.display       = 'none';
      document.getElementById('btnClearGrid').style.display    = '';
      redrawOverlay();
    }).catch(() => { setStatus('✗ Failed'); });
  }

  function loadSavedGrid() {
    setStatus('⏳ Loading saved calibration…');
    fetch('/load_saved_grid').then(r => r.json()).then(d => {
      if (d.error) { setStatus('✗ ' + d.error); return; }
      holeState = d.holes.map((h, i) => ({ px: h.px, py: h.py, occupied: null, idx: i }));
      triangleData = []; patternSet.clear();
      // ── Sync stream-control UI to the restored camera settings ──
      if (d.fov_pct !== undefined) {
        document.getElementById('fov').value = d.fov_pct;
        document.getElementById('fovVal').textContent = d.fov_pct + '%';
      }
      if (d.undistort_on !== undefined) {
        const ub = document.getElementById('undistBtn');
        ub.textContent      = '🔭 Undistort: ' + (d.undistort_on ? 'ON' : 'OFF');
        ub.style.background = d.undistort_on ? '#363' : '#444';
        ub.style.color      = d.undistort_on ? '#afa' : '#fff';
      }
      if (d.focus !== undefined) {
        document.getElementById('focusSlider').value = d.focus;
        document.getElementById('focusVal').textContent =
          d.focus === 0 ? 'auto' : d.focus.toFixed(1) + ' D';
      }
      document.getElementById('btnSaveCalib').style.display = '';
      setStatus('✓ ' + d.count + ' holes loaded — click 🔍 Detect Template');
      document.getElementById('btnScan').style.display         = 'none';
      document.getElementById('btnDetectTmpl').style.display   = '';
      document.getElementById('tmplDropWrap').style.display    = '';
      document.getElementById('btnSelectHoles').style.display  = '';
      document.getElementById('btnDetect').style.display       = 'none';
      document.getElementById('btnClearGrid').style.display    = '';
      redrawOverlay();
    }).catch(() => { setStatus('✗ Load failed'); });
  }

  function saveCalibration() {
    setStatus('⏳ Saving calibration + camera settings…');
    fetch('/save_calibration', {method:'POST'}).then(r => r.json()).then(d => {
      if (d.error) { setStatus('✗ ' + d.error); return; }
      const f = d.focus === 0 ? 'auto' : d.focus.toFixed(1) + ' D';
      setStatus('💾 Saved ' + d.count + ' holes  ·  FOV ' + d.fov_pct + '%  ·  focus ' + f +
                '  ·  undistort ' + (d.undistort_on ? 'ON' : 'OFF'));
    }).catch(() => { setStatus('✗ Save failed'); });
  }

  function onDiffChange(v) {
    document.getElementById('diffVal').textContent = v;
    fetch('/set_adapt_margin?val=' + v);
  }

  // Sync the IR Sensitivity slider to the persisted backend value on page load
  window.addEventListener('load', () => {
    fetch('/detection_status').then(r => r.json()).then(d => {
      if (d.adapt_margin !== undefined) {
        document.getElementById('diffSlider').value = d.adapt_margin;
        document.getElementById('diffVal').textContent = d.adapt_margin;
      }
    }).catch(() => {});
  });

  function scanHoles() {
    setStatus('⏳ Scanning…');
    fetch('/scan_holes').then(r => r.json()).then(d => {
      holeState = d.holes.map(h => ({ px: h.pixel_x, py: h.pixel_y, occupied: h.occupied, idx: h.index }));
      triangleData = [];
      const occ = d.holes.filter(h => h.occupied).length;
      setStatus('✓ ' + occ + ' / ' + d.count + ' occupied  (avg IR diff: ' + d.avg_diff + ')');
      document.getElementById('btnTriangle').style.display = '';
      redrawOverlay();
    }).catch(() => { setStatus('✗ Scan failed'); });
  }

  function detectTriangles() {
    setStatus('⏳ Detecting triangles…');
    fetch('/detect_triangles').then(r => r.json()).then(d => {
      if (d.error) { setStatus('✗ ' + d.error); return; }
      triangleData = d.triangles;
      if (d.count === 0) {
        setStatus('No triangles found');
      } else {
        const ups   = d.triangles.filter(t => t.orientation === 'up').length;
        const downs = d.triangles.filter(t => t.orientation === 'down').length;
        setStatus('🔺 ' + d.count + ' triangle' + (d.count > 1 ? 's' : '') +
                  ' found  (' + ups + '▲  ' + downs + '▽)');
      }
      redrawOverlay();
    }).catch(() => { setStatus('✗ Triangle detection failed'); });
  }

  // ── Template detection & hole selection ─────────
  let loadedTemplateName = null;
  let selectMode = false;
  let editingTemplate = null;   // name of template currently being edited (for overwrite-save)

  function toggleSelectMode() {
    selectMode = !selectMode;
    const btn = document.getElementById('btnSelectHoles');
    btn.textContent      = selectMode ? '✓ Done Selecting' : '🎯 Select Holes';
    btn.style.background = selectMode ? '#363' : '#444';
    btn.style.color      = selectMode ? '#afa' : '#fff';
    if (selectMode) {
      patternSet = new Set();
      editingTemplate = null;
      setStatus('🎯 Click holes to mark template — cyan = selected. Click ✓ Done Selecting when finished, then save via 📋 Templates ▾');
    } else {
      if (patternSet.size > 0) {
        setStatus('✓ ' + patternSet.size + ' holes selected — save via 📋 Templates ▾ or click ▶ Detect');
        document.getElementById('btnDetect').style.display = '';
      } else {
        setStatus('');
      }
    }
    redrawOverlay();
  }

  function detectTemplate() {
    setStatus('⏳ Scanning for template…');
    fetch('/detect_template').then(r => r.json()).then(d => {
      if (d.error) { setStatus('✗ ' + d.error); return; }
      if (!d.hole_count) {
        setStatus('⚠ No template detected — place template on board and try again');
        return;
      }
      loadedTemplateName = d.match ? d.match.name : null;
      // Update patternSet for overlay rendering
      patternSet = new Set(d.open_holes);
      holeState.forEach(h => { h.occupied = null; });
      if (d.match) {
        setStatus('✓ ' + d.match.name + ' detected (' + Math.round(d.match.score*100) + '% match) — ' + d.hole_count + ' holes  →  click ▶ Detect');
      } else {
        setStatus('✓ Template detected — ' + d.hole_count + ' holes (no saved match)  →  click ▶ Detect');
      }
      document.getElementById('btnDetect').style.display = '';
      redrawOverlay();
    }).catch(() => { setStatus('✗ Detect failed'); });
  }

  function toggleTmplDropdown() {
    const dd = document.getElementById('tmplDropdown');
    if (dd.style.display === 'none') {
      refreshTmplList();
      dd.style.display = '';
      // close on outside click
      setTimeout(() => document.addEventListener('click', closeTmplDropdown, {once:true}), 50);
    } else {
      dd.style.display = 'none';
    }
  }

  function closeTmplDropdown() {
    document.getElementById('tmplDropdown').style.display = 'none';
  }

  function refreshTmplList() {
    fetch('/templates').then(r => r.json()).then(list => {
      const el = document.getElementById('tmplList');
      if (!list.length) {
        el.innerHTML = '<div style="padding:6px 10px;font-size:12px;color:#888">No templates saved yet</div>';
        return;
      }
      el.innerHTML = list.map(t => {
        const n = t.name.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/"/g,'&quot;');
        return '<div style="display:flex;align-items:center;padding:4px 10px;gap:8px;">' +
          '<span style="flex:1;cursor:pointer;font-size:12px" data-tmpl="' + n + '" onclick="loadTmpl(this.dataset.tmpl)">' +
            t.name + ' <span style="color:#888">(' + t.hole_count + ' holes)</span></span>' +
          '<span style="cursor:pointer;color:#4af;font-size:13px" data-tmpl="' + n + '" onclick="editTmpl(this.dataset.tmpl)">&#x270E;</span>' +
          '<span style="cursor:pointer;color:#a33;font-size:11px" data-tmpl="' + n + '" onclick="deleteTmpl(this.dataset.tmpl)">&#x1F5D1;</span>' +
        '</div>';
      }).join('');
    });
  }

  function loadTmpl(name) {
    document.getElementById('tmplDropdown').style.display = 'none';
    fetch('/load_template', {method:'POST', headers:{'Content-Type':'application/json'},
          body: JSON.stringify({name})}).then(r => r.json()).then(d => {
      if (d.error) { setStatus('✗ ' + d.error); return; }
      loadedTemplateName = name;
      setStatus('✓ Loaded: ' + name + ' (' + d.hole_count + ' holes)  →  click ▶ Detect');
      patternSet = new Set(d.holes || []);
      document.getElementById('btnDetect').style.display = '';
      redrawOverlay();
    });
  }

  function editTmpl(name) {
    document.getElementById('tmplDropdown').style.display = 'none';
    fetch('/load_template', {method:'POST', headers:{'Content-Type':'application/json'},
          body: JSON.stringify({name})}).then(r => r.json()).then(d => {
      if (d.error) { setStatus('✗ ' + d.error); return; }
      patternSet      = new Set(d.holes || []);
      editingTemplate = name;
      loadedTemplateName = name;
      selectMode      = true;                       // enter select mode WITHOUT clearing
      const btn = document.getElementById('btnSelectHoles');
      btn.style.display   = '';
      btn.textContent     = '✓ Done Selecting';
      btn.style.background = '#363'; btn.style.color = '#afa';
      setStatus('✎ Editing "' + name + '" (' + patternSet.size + ' holes) — click holes to add/remove, then save via 📋 Templates ▾ (overwrites "' + name + '")');
      redrawOverlay();
    }).catch(() => { setStatus('✗ Edit load failed'); });
  }

  function deleteTmpl(name) {
    if (!confirm('Delete template "' + name + '"?')) return;
    fetch('/delete_template', {method:'POST', headers:{'Content-Type':'application/json'},
          body: JSON.stringify({name})}).then(() => refreshTmplList());
  }

  function promptSaveTemplate() {
    document.getElementById('tmplDropdown').style.display = 'none';
    const _msg = patternSet.size > 0
      ? 'Save ' + patternSet.size + ' selected holes as template — enter name' + (editingTemplate ? ' (overwrites "' + editingTemplate + '")' : '') + ':'
      : 'Template name (place template on board with no pegs, then enter name):';
    const name = prompt(_msg, editingTemplate || '');
    if (!name || !name.trim()) return;

    if (patternSet.size > 0) {
      // Save currently selected holes directly — no scan needed
      setStatus('⏳ Saving template…');
      fetch('/save_template_holes', {method:'POST', headers:{'Content-Type':'application/json'},
            body: JSON.stringify({name: name.trim(), holes: Array.from(patternSet)})}).then(r => r.json()).then(d => {
        if (d.error) { setStatus('✗ ' + d.error); return; }
        setStatus('✓ Saved template "' + d.name + '" with ' + d.hole_count + ' holes');
        selectMode = false;
        const btn = document.getElementById('btnSelectHoles');
        btn.textContent = '🎯 Select Holes'; btn.style.background = '#444'; btn.style.color = '#fff';
      }).catch(() => { setStatus('✗ Save failed'); });
    } else {
      // Fall back to ambient scan
      setStatus('⏳ Scanning and saving template…');
      fetch('/save_template', {method:'POST', headers:{'Content-Type':'application/json'},
            body: JSON.stringify({name: name.trim()})}).then(r => r.json()).then(d => {
        if (d.error) { setStatus('✗ ' + d.error); return; }
        setStatus('✓ Saved template "' + d.name + '" with ' + d.hole_count + ' holes');
      }).catch(() => { setStatus('✗ Save failed'); });
    }
  }

  // ── Detection ──────────────────────────────────────
  function toggleDetection() {
    if (detectRunning) {
      fetch('/stop_detection').then(() => {
        detectRunning = false;
        clearInterval(detectPollId); detectPollId = null;
        const btn = document.getElementById('btnDetect');
        btn.textContent = '▶ Detect'; btn.style.background = '#444'; btn.style.color = '#fff';
        document.getElementById('detectBar').textContent = '';
      });
    } else {
      if (patternSet.size === 0) {
        setStatus('✗ No target pattern — 🎯 Select Holes or load a template first');
        return;
      }
      // Push the browser-side target pattern to the backend, THEN start
      fetch('/set_target_pattern', {method:'POST', headers:{'Content-Type':'application/json'},
            body: JSON.stringify({holes: Array.from(patternSet), name: loadedTemplateName})})
        .then(r => r.json()).then(() => {
        fetch('/start_detection').then(r => r.json()).then(d => {
          if (d.error) { setStatus('✗ ' + d.error); return; }
          detectRunning = true;
          const btn = document.getElementById('btnDetect');
          btn.textContent = '⏹ Stop'; btn.style.background = '#a30'; btn.style.color = '#fff';
          setStatus('🔍 Detecting ' + patternSet.size + ' target holes…');
          detectPollId = setInterval(pollDetection, 350);
        });
      });
    }
  }

  let liveState  = {};     // idx -> bool from last detection poll
  let inferredSet = new Set();  // idx of pattern nodes filled by neighbor inference
  let wasSolved  = false;  // track solved state transitions

  function pollDetection() {
    fetch('/detection_status').then(r => r.json()).then(d => {
      if (!d.running && detectRunning && !d.solved) {
        detectRunning = false;
        clearInterval(detectPollId); detectPollId = null;
        const btn = document.getElementById('btnDetect');
        btn.textContent = '▶ Detect'; btn.style.background = '#444'; btn.style.color = '#fff';
      }
      // Update per-hole live state for overlay
      liveState = {};
      for (const [k, v] of Object.entries(d.live_state || {})) liveState[parseInt(k)] = v;
      inferredSet = new Set((d.inferred || []).map(Number));

      const total   = d.pattern_count;
      const matched = d.live_matched || 0;
      const filled  = '█'.repeat(d.consecutive_matches);
      const empty   = '░'.repeat(Math.max(0, d.required - d.consecutive_matches));
      const diffInfo = d.running ? '  [avg:' + d.avg_diff + ' max:' + d.max_diff + ' mgn:' + d.adapt_margin + ']' : '';
      const nInf     = (d.inferred || []).length;
      const infInfo  = (d.running && nInf > 0) ? '  (+' + nInf + ' inferred)' : '';
      document.getElementById('detectBar').textContent =
        (d.running && !d.solved) ? matched + '/' + total + ' filled' + infInfo + '  ' + filled + empty + ' ' + d.consecutive_matches + '/' + d.required + diffInfo
                                 : (d.solved ? '🎉 SOLVED!' : '');

      // Win overlay: show on solved, auto-dismiss when backend resets
      if (d.solved && !wasSolved) {
        wasSolved = true;
        document.getElementById('winOverlay').style.display = 'flex';
      } else if (!d.solved && wasSolved) {
        // Backend reset after win animation — auto-dismiss
        wasSolved = false;
        document.getElementById('winOverlay').style.display = 'none';
        document.getElementById('winResetMsg').style.display = 'none';
        setStatus('🎮 Ready — place template for next player');
        document.getElementById('detectBar').textContent = '';
        // patternSet was auto-loaded, detection still running — clear overlay to show cyan targets
        liveState = {};
        redrawOverlay();
      } else if (d.solved && wasSolved) {
        // Show "resetting" message partway through animation
        document.getElementById('winResetMsg').style.display = '';
      }

      redrawOverlay();
    });
  }

  function setStrict(on) {
    fetch('/set_detection_strict?on=' + (on ? '1' : '0'));
  }

  function resetPuzzle() {
    fetch('/reset_puzzle').then(() => {
      document.getElementById('winOverlay').style.display = 'none';
      wasSolved = false;
      document.getElementById('detectBar').textContent = '';
    });
  }

  // ── Drawing helpers ────────────────────────────────
  function drawDot(ctx, x, y, color, r) {
    ctx.beginPath(); ctx.arc(x, y, r + 2, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(0,0,0,0.75)'; ctx.fill();
    ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = color; ctx.fill();
  }

  // Extend a line defined by two points to the canvas edges
  function extendedLine(x1, y1, x2, y2) {
    const W = overlay.width, H = overlay.height;
    const dx = x2 - x1, dy = y2 - y1;
    if (Math.abs(dx) < 0.001 && Math.abs(dy) < 0.001) return null;
    const pts = [];
    const candidates = [];
    if (Math.abs(dx) > 0.001) {
      let t, y;
      t = -x1/dx; y = y1 + t*dy; if (y >= 0 && y <= H) candidates.push({x:0, y});
      t = (W-x1)/dx; y = y1 + t*dy; if (y >= 0 && y <= H) candidates.push({x:W, y});
    }
    if (Math.abs(dy) > 0.001) {
      let t, x;
      t = -y1/dy; x = x1 + t*dx; if (x >= 0 && x <= W) candidates.push({x, y:0});
      t = (H-y1)/dy; x = x1 + t*dx; if (x >= 0 && x <= W) candidates.push({x, y:H});
    }
    // deduplicate and pick two
    for (const c of candidates) {
      if (!pts.some(p => Math.abs(p.x-c.x)<1 && Math.abs(p.y-c.y)<1)) pts.push(c);
      if (pts.length === 2) break;
    }
    return pts.length === 2 ? pts : null;
  }

  function redrawOverlay() {
    const ctx = overlay.getContext('2d');
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    // Draw committed row lines (cyan), extended to canvas edges
    ctx.lineWidth = 1.5;
    rowLines.forEach(l => {
      const e = extendedLine(l.x1, l.y1, l.x2, l.y2);
      if (!e) return;
      ctx.strokeStyle = 'rgba(0,220,255,0.7)';
      ctx.setLineDash([]);
      ctx.beginPath(); ctx.moveTo(e[0].x, e[0].y); ctx.lineTo(e[1].x, e[1].y); ctx.stroke();
    });

    // Draw committed column lines (orange), extended
    colLines.forEach(l => {
      const e = extendedLine(l.x1, l.y1, l.x2, l.y2);
      if (!e) return;
      ctx.strokeStyle = 'rgba(255,160,0,0.7)';
      ctx.setLineDash([]);
      ctx.beginPath(); ctx.moveTo(e[0].x, e[0].y); ctx.lineTo(e[1].x, e[1].y); ctx.stroke();
    });

    // Live preview line (dashed) while second click is pending
    if (lineStart !== null && mousePos !== null) {
      const color = calibPhase === PHASE_ROW_LINES ? 'rgba(0,220,255,0.55)' : 'rgba(255,160,0,0.55)';
      ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.setLineDash([6, 4]);
      ctx.beginPath(); ctx.moveTo(lineStart.x, lineStart.y); ctx.lineTo(mousePos.x, mousePos.y); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = color;
      ctx.beginPath(); ctx.arc(lineStart.x, lineStart.y, 5, 0, Math.PI*2); ctx.fill();
    } else if (lineStart !== null) {
      const color = calibPhase === PHASE_ROW_LINES ? 'rgba(0,220,255,0.8)' : 'rgba(255,160,0,0.8)';
      ctx.fillStyle = color;
      ctx.beginPath(); ctx.arc(lineStart.x, lineStart.y, 5, 0, Math.PI*2); ctx.fill();
    }

    // Hole dots
    if (holeState.length > 0) {
      const sx = streamImg.offsetWidth / IMG_W, sy = streamImg.offsetHeight / IMG_H;
      holeState.forEach(h => {
        const x = h.px * sx, y = h.py * sy;
        const isTarget = patternSet.has(h.idx);
        const hasLive  = (h.idx in liveState);
        let color, r = isTarget ? 7 : 5;
        if (hasLive) {
          // Live detection state — drives colour for ANY hole the detector reports
          color = liveState[h.idx] ? (inferredSet.has(h.idx) ? '#aaff00' : '#00ff88') : '#ff2020';  // green=measured, chartreuse=inferred, red=missing
          if (liveState[h.idx]) r = isTarget ? 8 : 6;         // pop filled holes
        } else if (isTarget) {
          if (h.occupied !== null && !patternMode) {
            color = h.occupied ? '#00ff88' : '#ff2020';
          } else {
            color = '#00e5ff';  // cyan = template hole waiting for peg
          }
        } else if (patternMode || patternSet.size > 0) {
          // Background node: dimmed
          color = h.occupied === null ? 'rgba(255,255,255,0.25)'
                : h.occupied          ? 'rgba(255,60,60,0.5)' : 'rgba(0,255,136,0.2)';
        } else {
          color = h.occupied === null ? '#ffe000'
                : h.occupied          ? '#ff2020' : '#00ff88';
        }
        drawDot(ctx, x, y, color, r);
      });

      // Triangle outlines
      if (triangleData.length > 0) {
        ctx.setLineDash([]);
        triangleData.forEach(tri => {
          const c  = tri.corners;
          const c0 = [c[0][0] * sx, c[0][1] * sy];
          const c1 = [c[1][0] * sx, c[1][1] * sy];
          const c2 = [c[2][0] * sx, c[2][1] * sy];
          // Glow / shadow
          ctx.lineWidth   = 5;
          ctx.strokeStyle = 'rgba(0,0,0,0.5)';
          ctx.beginPath();
          ctx.moveTo(c0[0], c0[1]); ctx.lineTo(c1[0], c1[1]);
          ctx.lineTo(c2[0], c2[1]); ctx.closePath(); ctx.stroke();
          // Bright outline
          ctx.lineWidth   = 2.5;
          ctx.strokeStyle = tri.orientation === 'up'
                          ? 'rgba(255,220,0,0.95)'    // yellow for ▲
                          : 'rgba(0,220,255,0.95)';   // cyan for ▽
          ctx.beginPath();
          ctx.moveTo(c0[0], c0[1]); ctx.lineTo(c1[0], c1[1]);
          ctx.lineTo(c2[0], c2[1]); ctx.closePath(); ctx.stroke();
        });
      }
    }
  }
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(PAGE)

@app.route("/stream")
def stream():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/set_fov")
def set_fov():
    global fov_pct
    pct = max(25, min(100, int(request.args.get("pct", 100))))
    fov_pct = pct
    apply_crop(pct)
    return jsonify({"fov": pct})

@app.route("/toggle_undistort")
def toggle_undistort():
    global undistort_on
    with state_lock:
        undistort_on = not undistort_on
    return jsonify({"on": undistort_on})

@app.route("/set_grid_from_lines", methods=["POST"])
def set_grid_from_lines():
    global grid_holes
    data      = request.get_json()
    row_lines = data["row_lines"]   # list of [[x1,y1],[x2,y2]] in image pixels
    col_lines = data["col_lines"]

    def intersect(p1, p2, p3, p4):
        """Intersection of infinite lines through p1-p2 and p3-p4."""
        x1,y1 = p1;  x2,y2 = p2
        x3,y3 = p3;  x4,y4 = p4
        denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if abs(denom) < 1e-9:
            return None   # parallel
        t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
        return (x1 + t*(x2-x1), y1 + t*(y2-y1))

    # Find every (row_line, col_line) intersection, grouped by row line
    row_groups = []
    for rl in row_lines:
        pts = []
        for cl in col_lines:
            pt = intersect(rl[0], rl[1], cl[0], cl[1])
            if pt is not None:
                pts.append(pt)
        pts.sort(key=lambda p: p[0])   # left → right
        if pts:
            row_groups.append(pts)

    # Sort groups top → bottom by mean y
    row_groups.sort(key=lambda pts: sum(p[1] for p in pts) / len(pts))

    # Output all intersections directly — no stagger correction.
    # The user draws diagonal column lines that follow the isometric lattice,
    # so each (row_line × col_line) intersection already lands on a real hole.
    holes_out = []
    for pts in row_groups:
        for p in pts:
            holes_out.append({"px": int(round(p[0])), "py": int(round(p[1]))})

    # Sub-pixel snapping via 7×7 centre-of-mass moments
    with frame_lock:
        frame_data = latest_frame
    if frame_data:
        img_np  = np.array(Image.open(io.BytesIO(frame_data)))
        gray    = cv2.cvtColor(img_np[:,:,::-1].copy(), cv2.COLOR_BGR2GRAY)
        h_img, w_img = gray.shape
        refined = []
        for hole in holes_out:
            hx, hy = hole["px"], hole["py"]
            if 3 <= hx < w_img-4 and 3 <= hy < h_img-4:
                M = cv2.moments(gray[hy-3:hy+4, hx-3:hx+4])
                if M["m00"] != 0:
                    refined.append({"px": int(round(hx-3+M["m10"]/M["m00"])),
                                    "py": int(round(hy-3+M["m01"]/M["m00"]))})
                    continue
            refined.append(hole)
        holes_out = refined

    with state_lock:
        grid_holes = holes_out
    # Auto-save calibration to disk
    try:
        calib = {
            "holes":        holes_out,
            "fov_pct":      fov_pct,
            "output_hfov":  output_hfov,
            "undistort_on": undistort_on,
            "focus":        focus_pos,
        }
        _atomic_write_json("/home/jerica/grid_holes.json", calib)
    except Exception as e:
        print(f"Warning: could not save grid: {e}")
    return jsonify({"ok": True, "count": len(holes_out), "holes": holes_out})


@app.route("/save_calibration", methods=["POST"])
def save_calibration():
    """Re-save the currently-loaded grid plus current camera settings (new format)."""
    global grid_holes
    with state_lock:
        holes = list(grid_holes)
    if not holes:
        return jsonify({"error": "No grid loaded — calibrate or Load Last first"}), 400
    calib = {
        "holes":        holes,
        "fov_pct":      fov_pct,
        "output_hfov":  output_hfov,
        "undistort_on": undistort_on,
        "focus":        focus_pos,
    }
    try:
        _atomic_write_json("/home/jerica/grid_holes.json", calib)
    except Exception as e:
        return jsonify({"error": f"Save failed: {e}"}), 500
    return jsonify({"ok": True, "count": len(holes), "fov_pct": fov_pct,
                    "output_hfov": output_hfov, "undistort_on": undistort_on,
                    "focus": focus_pos})


@app.route("/load_saved_grid")
def load_saved_grid():
    try:
        with open("/home/jerica/grid_holes.json", "r") as f:
            data = json.load(f)
        return jsonify({"ok": True, **_apply_calibration(data)})
    except FileNotFoundError:
        return jsonify({"error": "No saved calibration found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/scan_holes")
def scan_holes():
    global last_scan_results
    with state_lock:
        holes     = list(grid_holes)
        thresh    = diff_threshold
        was_ir_on = ir_state
    if not holes:
        return jsonify({"error": "Not calibrated"}), 400

    half = 4
    try:
        _exp = int(request.args.get("exp", 0))
    except (TypeError, ValueError):
        _exp = 0
    try:
        _gain = float(request.args.get("gain", 0))
    except (TypeError, ValueError):
        _gain = 0.0

    # ── Lock AEC so IR toggle doesn't cause re-expose ────
    if _exp > 0:
        _ctrls = {"AeEnable": False, "ExposureTime": _exp}
        if _gain > 0:
            _ctrls["AnalogueGain"] = _gain
        camera.set_controls(_ctrls)
        time.sleep(0.5)    # exposure change needs several frames to settle
    else:
        camera.set_controls({"AeEnable": False})
        time.sleep(0.15)   # one frame for lock to take effect

    # ── IR ON frame ──────────────────────────────────────
    if IR_AVAILABLE:
        ir_led.on()
    time.sleep(0.20)   # allow ~6 frames at 30 fps to flush through
    with frame_lock:
        frame_on = latest_frame

    # ── IR OFF frame ─────────────────────────────────────
    if IR_AVAILABLE:
        ir_led.off()
    time.sleep(0.20)
    with frame_lock:
        frame_off = latest_frame

    # ── Restore IR state & re-enable AEC ─────────────────
    if IR_AVAILABLE:
        if was_ir_on:
            ir_led.on()
        else:
            ir_led.off()
    camera.set_controls({"AeEnable": True})

    if not frame_on or not frame_off:
        return jsonify({"error": "No frame"}), 503

    img_on  = np.array(Image.open(io.BytesIO(frame_on)))
    img_off = np.array(Image.open(io.BytesIO(frame_off)))
    h, w    = img_on.shape[:2]

    results = []
    for i, hole in enumerate(holes):
        px, py = int(round(hole["px"])), int(round(hole["py"]))
        x1, y1 = max(0, px-half), max(0, py-half)
        x2, y2 = min(w, px+half), min(h, py+half)

        def lum(crop):
            return float(np.mean(0.299*crop[:,:,0] + 0.587*crop[:,:,1]
                                 + 0.114*crop[:,:,2])) if crop.size else 0.0

        l_on  = lum(img_on[y1:y2, x1:x2])
        l_off = lum(img_off[y1:y2, x1:x2])
        diff  = l_on - l_off
        # Front-lit IR: peg reflects IR -> high diff = occupied
        # Empty hole absorbs IR -> low diff = unoccupied
        occupied = diff > thresh
        results.append({"index": i, "pixel_x": px, "pixel_y": py,
                        "lum": round(diff, 1), "on": round(l_on, 1),
                        "off": round(l_off, 1), "occupied": occupied})

    avg_diff = round(sum(r["lum"] for r in results) / len(results), 1) if results else 0.0
    sat = sum(1 for r in results if r["on"] >= 250)
    max_on = max((r["on"] for r in results), default=0.0)
    with state_lock:
        last_scan_results = list(results)
    return jsonify({"holes": results, "count": len(results), "avg_diff": avg_diff,
                    "saturated": sat, "max_on": max_on, "exp": _exp, "gain": _gain})


def find_triangles_in_grid(holes, results, side_len=2):
    """
    Find equilateral triangles of 3 occupied pegs on the isometric grid.

    Uses a pure-geometry approach: for every pair of occupied pegs that are
    approximately one lattice step apart, compute the two candidate equilateral
    third vertices (±60° rotation) and check whether an occupied peg sits there.
    No lattice-direction estimation needed — just distance + rotation.

    Returns a list of dicts:
        { "orientation": "up"|"down",
          "corners": [[px,py],[px,py],[px,py]],   # image-pixel coords
          "pegs":    [list of hole indices] }
    """
    if not results or not holes:
        return []

    occupied = [(r["pixel_x"], r["pixel_y"], r["index"])
                for r in results if r["occupied"]]
    if len(occupied) < 3:
        return []

    # ── Estimate lattice step (col_dx) from calibration holes ─────────────────
    # Bin holes by y, find median x-spacing within each row bin
    bins = {}
    for h in holes:
        k = int(round(h["py"] / 20))
        bins.setdefault(k, []).append(h["px"])
    col_dxs = []
    for xs in bins.values():
        xs_s = sorted(xs)
        for j in range(len(xs_s) - 1):
            d = xs_s[j+1] - xs_s[j]
            if d > 5:
                col_dxs.append(d)
    col_dx = float(np.median(col_dxs)) if col_dxs else 50.0

    SNAP = col_dx * 0.38   # ±38% tolerance on third-vertex position

    # ── Spatial hash for fast nearest-occupied lookup ─────────────────────────
    cell_sz = max(int(SNAP / 2), 1)

    def cell_key(x, y):
        return (int(x / cell_sz), int(y / cell_sz))

    occ_cells = {}
    for px, py, idx in occupied:
        occ_cells[cell_key(px, py)] = (px, py, idx)

    def find_near(x, y, exclude_a=-1, exclude_b=-1):
        """Occupied hole nearest (x,y) within SNAP, excluding two indices."""
        cx, cy = cell_key(x, y)
        best_idx, best_dist = -1, SNAP
        for ddx in range(-2, 3):
            for ddy in range(-2, 3):
                entry = occ_cells.get((cx + ddx, cy + ddy))
                if not entry:
                    continue
                hx, hy, hidx = entry
                if hidx == exclude_a or hidx == exclude_b:
                    continue
                d = ((hx - x) ** 2 + (hy - y) ** 2) ** 0.5
                if d < best_dist:
                    best_dist = d
                    best_idx = hidx
        return best_idx

    # Build a quick index: result_index -> (px, py)
    idx_to_pt = {r["index"]: (r["pixel_x"], r["pixel_y"])
                 for r in results if r["occupied"]}

    # ── Check every pair of occupied pegs ~one lattice step apart ─────────────
    triangles, seen = [], set()

    for i in range(len(occupied)):
        ax, ay, ai = occupied[i]
        for j in range(i + 1, len(occupied)):
            bx, by, bi = occupied[j]
            dx, dy = bx - ax, by - ay
            dist = (dx * dx + dy * dy) ** 0.5

            # Only consider nearest-neighbour pairs (within ±45% of col_dx)
            if dist < col_dx * 0.55 or dist > col_dx * 1.45:
                continue

            # Two candidate equilateral third vertices (±60° rotation of AB)
            for sign in (1, -1):
                cx = ax + 0.5 * dx - sign * 0.866 * dy
                cy = ay + sign * 0.866 * dx + 0.5 * dy
                ci = find_near(cx, cy, exclude_a=ai, exclude_b=bi)
                if ci < 0:
                    continue

                key = frozenset([ai, bi, ci])
                if key in seen:
                    continue
                seen.add(key)

                cpx, cpy = idx_to_pt[ci]

                # Orientation: apex above midpoint → "up" (▲), else "down" (▽)
                mid_y = (ay + by) / 2.0
                orient = "up" if cpy < mid_y else "down"

                triangles.append({
                    "orientation": orient,
                    "corners": [[ax, ay], [bx, by], [cpx, cpy]],
                    "pegs":    sorted([ai, bi, ci]),
                })

    return triangles


@app.route("/detect_triangles")
def detect_triangles():
    global last_scan_results
    with state_lock:
        holes   = list(grid_holes)
        results = list(last_scan_results)
    if not holes:
        return jsonify({"error": "Not calibrated"}), 400
    if not results:
        return jsonify({"error": "No scan data — run Scan first"}), 400
    tris = find_triangles_in_grid(holes, results, side_len=2)
    return jsonify({"triangles": tris, "count": len(tris)})


@app.route("/set_led")
def set_led():
    if not LED_AVAILABLE:
        return jsonify({"error": "LED not available"}), 503
    r = max(0, min(255, int(request.args.get("r", 0))))
    g = max(0, min(255, int(request.args.get("g", 0))))
    b = max(0, min(255, int(request.args.get("b", 0))))
    global led_color
    led_color = (r, g, b)
    _spi_show(r, g, b)
    return jsonify({"r": r, "g": g, "b": b})

@app.route("/set_focus")
def set_focus():
    global focus_pos
    pos = max(0.0, min(30.0, float(request.args.get("pos", 0))))
    focus_pos = pos
    if pos <= 0:
        camera.set_controls({
            "AfMode":  controls.AfModeEnum.Continuous,
            "AfSpeed": controls.AfSpeedEnum.Fast,
            "AfRange": controls.AfRangeEnum.Macro,   # prioritise close distances
        })
    else:
        camera.set_controls({
            "AfMode":       controls.AfModeEnum.Manual,
            "LensPosition": pos,
        })
    return jsonify({"lens_position": pos})

@app.route("/set_diff_threshold")
def set_diff_threshold():
    global diff_threshold
    val = max(0, min(255, int(request.args.get("val", 20))))
    with state_lock:
        diff_threshold = val
    _save_settings("diff_threshold", val)
    return jsonify({"diff_threshold": val})

@app.route("/set_adapt_margin")
def set_adapt_margin():
    global adapt_margin
    val = max(0, min(150, int(request.args.get("val", 30))))
    with state_lock:
        adapt_margin = val
    _save_settings("adapt_margin", val)
    return jsonify({"adapt_margin": val})

@app.route("/set_brightness_threshold")
def set_brightness_threshold():
    global filter_brightness
    val = max(0, min(255, int(request.args.get("val", 60))))
    with state_lock:
        filter_brightness = val
    return jsonify({"brightness_threshold": val})

@app.route("/test_ir")
def test_ir():
    if not IR_AVAILABLE:
        return jsonify({"error": "IR not available"}), 503
    ir_led.on()
    time.sleep(3)
    ir_led.off()
    return jsonify({"ok": True})

@app.route("/set_ir")
def set_ir():
    global ir_state
    if not IR_AVAILABLE:
        return jsonify({"error": "IR LED not available"}), 503
    state = request.args.get("on", "1")
    if state == "1":
        ir_led.on()
    else:
        ir_led.off()
    ir_state = (state == "1")
    return jsonify({"ir": ir_state})

@app.route("/toggle_pattern_node")
def toggle_pattern_node():
    global target_pattern
    idx = int(request.args.get("idx", -1))
    if idx < 0:
        return jsonify({"error": "invalid idx"}), 400
    with state_lock:
        if idx in target_pattern:
            target_pattern.discard(idx)
            state = "removed"
        else:
            target_pattern.add(idx)
            state = "added"
    return jsonify({"idx": idx, "state": state, "count": len(target_pattern)})

@app.route("/clear_pattern")
def clear_pattern():
    global target_pattern, consecutive_matches, puzzle_solved
    with state_lock:
        target_pattern     = set()
        consecutive_matches = 0
        puzzle_solved       = False
    _set_current_template(None)
    return jsonify({"ok": True})

@app.route("/set_target_pattern", methods=["POST"])
def set_target_pattern():
    """Set the active target pattern from a list of hole indices.
    Body: {"holes": [3, 7, 12, ...]}"""
    global target_pattern, consecutive_matches, puzzle_solved
    data  = request.get_json() or {}
    holes = data.get("holes", [])
    name  = (data.get("name") or "").strip() or None
    with state_lock:
        target_pattern      = set(int(i) for i in holes)
        consecutive_matches = 0
        puzzle_solved       = False
    _set_current_template(name)
    return jsonify({"ok": True, "hole_count": len(target_pattern), "template": name})


@app.route("/start_detection")
def start_detection():
    global detection_running, consecutive_matches, puzzle_solved
    with state_lock:
        if detection_running:
            return jsonify({"ok": True, "already_running": True})
        if not grid_holes:
            return jsonify({"error": "Not calibrated"}), 400
        if not target_pattern:
            return jsonify({"error": "No target pattern — select holes or load a template first"}), 400
        detection_running   = True
        consecutive_matches = 0
        puzzle_solved       = False
    threading.Thread(target=detection_loop, daemon=True).start()
    return jsonify({"ok": True})

@app.route("/stop_detection")
def stop_detection():
    global detection_running
    with state_lock:
        detection_running = False
    if IR_AVAILABLE:
        ir_led.off()   # cut IR immediately; loop's finally block will also call this
    camera.set_controls({"AeEnable": True})   # restore auto-exposure immediately
    return jsonify({"ok": True})

@app.route("/detection_status")
def detection_status():
    with state_lock:
        cm    = consecutive_matches
        ps    = puzzle_solved
        dr    = detection_running
        pc    = len(target_pattern)
        live  = dict(detection_live_state)
        st    = detection_strict
        diffs = dict(detection_last_diffs)
        thresh = diff_threshold
        am     = adapt_margin
        inf    = list(detection_inferred)
        lt     = last_trigger
    matched = sum(1 for v in live.values() if v)
    # debug: avg/max diff across all computed holes
    diff_vals = list(diffs.values())
    avg_diff = round(sum(diff_vals) / len(diff_vals), 1) if diff_vals else 0.0
    max_diff = round(max(diff_vals), 1) if diff_vals else 0.0
    return jsonify({
        "running":             dr,
        "consecutive_matches": cm,
        "required":            REQUIRED_CONSECUTIVE_MATCHES,
        "solved":              ps,
        "progress":            round(cm / REQUIRED_CONSECUTIVE_MATCHES, 2),
        "pattern_count":       pc,
        "live_matched":        matched,
        "live_state":          {str(k): v for k, v in live.items()},
        "strict":              st,
        "avg_diff":            avg_diff,
        "max_diff":            max_diff,
        "threshold":           thresh,
        "adapt_margin":        am,
        "inferred":            list(inf),
        "last_trigger":        lt,
    })

@app.route("/set_detection_strict")
def set_detection_strict():
    global detection_strict
    val = request.args.get("on", "0") == "1"
    with state_lock:
        detection_strict = val
    return jsonify({"strict": val})

@app.route("/reset_puzzle")
def reset_puzzle():
    global consecutive_matches, puzzle_solved
    with state_lock:
        consecutive_matches = 0
        puzzle_solved       = False
    return jsonify({"ok": True})

@app.route("/debug_state")
def debug_state():
    with state_lock:
        pattern = sorted(target_pattern)
        n_holes = len(grid_holes)
        diffs   = dict(detection_last_diffs)
        live    = dict(detection_live_state)
    pattern_diffs = {i: round(diffs.get(i, -1), 1) for i in pattern}
    return jsonify({
        "pattern_indices": pattern,
        "pattern_count":   len(pattern),
        "grid_hole_count": n_holes,
        "pattern_diffs":   pattern_diffs,
        "live_state":      live,
        "sample_diffs":    {k: round(v,1) for k,v in sorted(diffs.items())[:10]},
    })

@app.route("/toggle_filter")
def toggle_filter():
    global filter_on
    with state_lock:
        filter_on = not filter_on
        state = filter_on
    return jsonify({"on": state})


# ─────────────────────────────────────────────────────────────────────────────
# Template management routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/templates")
def get_templates():
    """List all stored templates."""
    templates = _load_templates()
    return jsonify([{
        "name":       t["name"],
        "hole_count": len(t["holes"]),
        "created":    t.get("created", ""),
    } for t in templates])


def _scan_ambient():
    """
    Take an ambient (IR-off) frame and return per-hole luminance list.
    Returns (lum_list, error_str). error_str is None on success.
    """
    with state_lock:
        holes = list(grid_holes)
    if not holes:
        return None, "Not calibrated"
    if IR_AVAILABLE:
        ir_led.off()
    time.sleep(0.15)
    with frame_lock:
        frame = latest_frame
    if not frame:
        return None, "No frame"
    img = np.array(Image.open(io.BytesIO(frame)))
    lums = [_hole_lum(img, int(round(h["px"])), int(round(h["py"]))) for h in holes]
    return lums, None


def _classify_open_holes(lum_list):
    """
    Use Otsu thresholding on hole luminance to find open (dark) holes.
    Returns (open_hole_indices, threshold_value).
    Returns ([], 0) if variance is too low (no template present).
    """
    arr = np.array(lum_list, dtype=np.float32)
    if arr.std() < 6.0:
        return [], 0.0
    # Normalise to 0-255 for Otsu
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return [], 0.0
    u8 = ((arr - mn) / (mx - mn) * 255).astype(np.uint8).reshape(1, -1)
    otsu_u8, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = float(otsu_u8) / 255.0 * (mx - mn) + mn
    open_holes = [i for i, l in enumerate(lum_list) if l < thresh]
    return open_holes, round(float(thresh), 1)


@app.route("/detect_template")
def detect_template_route():
    """
    Scan ambient frame, detect open holes (template cutouts) via Otsu,
    match against stored templates, auto-load best match as target pattern.
    """
    global target_pattern, consecutive_matches, puzzle_solved
    lums, err = _scan_ambient()
    if err:
        return jsonify({"error": err}), (400 if err == "Not calibrated" else 503)

    open_holes, thresh = _classify_open_holes(lums)
    if not open_holes:
        return jsonify({
            "open_holes": [], "hole_count": 0,
            "match": None, "all_matches": [],
            "msg": "No template detected — board appears uniform",
        })

    # Match against stored templates
    templates = _load_templates()
    matches = []
    open_set = set(open_holes)
    for tmpl in templates:
        tmpl_set = set(tmpl["holes"])
        if not tmpl_set and not open_set:
            score = 1.0
        elif not tmpl_set or not open_set:
            score = 0.0
        else:
            intersection = len(tmpl_set & open_set)
            union        = len(tmpl_set | open_set)
            score = intersection / union if union > 0 else 0.0
        matches.append({"name": tmpl["name"], "score": round(score, 3)})

    matches.sort(key=lambda x: -x["score"])
    best_match = matches[0] if matches and matches[0]["score"] >= 0.7 else None

    # Load matched template's holes (or use detected open holes if no match)
    if best_match:
        tmpl_holes = next(
            (t["holes"] for t in templates if t["name"] == best_match["name"]),
            open_holes,
        )
    else:
        tmpl_holes = open_holes

    with state_lock:
        target_pattern      = set(tmpl_holes)
        consecutive_matches = 0
        puzzle_solved       = False
    _set_current_template(best_match["name"] if best_match else None)

    return jsonify({
        "open_holes":  open_holes,
        "hole_count":  len(open_holes),
        "match":       best_match,
        "all_matches": matches,
        "lum_thresh":  thresh,
    })


@app.route("/save_template", methods=["POST"])
def save_template():
    """
    Save current ambient scan as a named template.
    Body: {"name": "Butterfly"}
    """
    data = request.get_json() or {}
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"error": "Name required"}), 400

    lums, err = _scan_ambient()
    if err:
        return jsonify({"error": err}), (400 if err == "Not calibrated" else 503)

    open_holes, thresh = _classify_open_holes(lums)
    if not open_holes:
        return jsonify({"error": "No template detected — place template on board first"}), 400

    templates = _load_templates()
    templates = [t for t in templates if t["name"] != name]  # replace if exists
    templates.append({
        "name":    name,
        "holes":   open_holes,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    _save_templates_to_disk(templates)

    return jsonify({"ok": True, "name": name, "hole_count": len(open_holes)})


@app.route("/save_template_holes", methods=["POST"])
def save_template_holes():
    """Save a manually-selected set of hole indices as a named template."""
    data = request.get_json() or {}
    name  = data.get("name", "").strip()
    holes = data.get("holes", [])
    if not name:
        return jsonify({"error": "Name required"}), 400
    if not holes:
        return jsonify({"error": "No holes provided"}), 400
    templates = _load_templates()
    templates = [t for t in templates if t["name"] != name]
    templates.append({
        "name":    name,
        "holes":   [int(i) for i in holes],
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    _save_templates_to_disk(templates)
    return jsonify({"ok": True, "name": name, "hole_count": len(holes)})


@app.route("/delete_template", methods=["POST"])
def delete_template():
    """Delete a named template. Body: {"name": "Butterfly"}"""
    data = request.get_json() or {}
    name = data.get("name", "").strip()
    templates = _load_templates()
    templates = [t for t in templates if t["name"] != name]
    _save_templates_to_disk(templates)
    return jsonify({"ok": True})


@app.route("/load_template", methods=["POST"])
def load_template():
    """Manually load a template by name. Body: {"name": "Butterfly"}"""
    global target_pattern, consecutive_matches, puzzle_solved
    data = request.get_json() or {}
    name = data.get("name", "").strip()
    templates = _load_templates()
    tmpl = next((t for t in templates if t["name"] == name), None)
    if not tmpl:
        return jsonify({"error": "Template not found"}), 404
    with state_lock:
        target_pattern      = set(tmpl["holes"])
        consecutive_matches = 0
        puzzle_solved       = False
    _set_current_template(name)
    return jsonify({"ok": True, "name": name, "hole_count": len(tmpl["holes"]),
                    "holes": list(tmpl["holes"])})


@app.route("/test_trigger")
def test_trigger():
    """Manually fire the win trigger for wiring tests. /test_trigger?code=S"""
    code = request.args.get("code", "S")
    url  = request.args.get("url")   # optional full-URL override
    path = None
    if not url:
        for _e in (_load_settings().get("pattern_map") or {}).values():
            _c, _u, _p = _resolve_pattern_entry(_e)
            if _c == code and (_u or _p):
                url, path = _u, _p
                break
    _send_win_trigger(code, url, path)
    return jsonify({"ok": True, "sent": {"code": code, "url": url, "path": path}})


@app.route("/test_win_animation")
def test_win_animation():
    """Run just the LED win animation (no trigger POST, no puzzle reset) for testing."""
    if not LED_AVAILABLE:
        return jsonify({"error": "LED not available"}), 503
    threading.Thread(target=_win_animation, daemon=True).start()
    return jsonify({"ok": True, "running": "win_animation"})


@app.route("/count_ruler")
def count_ruler():
    """Light the strip in coloured bands to gauge/recount its addressable length.
    /count_ruler?band=25  → each colour spans <band> pixels, in order:
    red, orange, yellow, green, cyan, blue, purple, magenta, white, dim-red."""
    if not LED_AVAILABLE:
        return jsonify({"error": "LED not available"}), 503
    band = max(1, int(request.args.get("band", 25)))
    names  = ["red", "orange", "yellow", "green", "cyan", "blue",
              "purple", "magenta", "white", "dim-red"]
    colors = [(255, 0, 0), (255, 80, 0), (255, 255, 0), (0, 255, 0), (0, 255, 255),
              (0, 0, 255), (160, 0, 255), (255, 0, 255), (255, 255, 255), (60, 0, 0)]
    pixels = [(0, 0, 0)] * LED_COUNT
    for i in range(LED_COUNT):
        bi = i // band
        if bi < len(colors):
            pixels[i] = colors[bi]
    _spi_show_pixels(pixels)
    return jsonify({"ok": True, "band": band, "order": names})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
