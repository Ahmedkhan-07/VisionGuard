"""
╔══════════════════════════════════════════════════════════╗
║         SMART INTRUDER ALERT SYSTEM  v1.0                ║
║   Motion + Human Detection → Email Alert with Photo      ║
╚══════════════════════════════════════════════════════════╝

INSTALL:
  pip install opencv-python mediapipe numpy pandas

RUN:
  python intruder_alert.py

USES SAME config.py AS TRACKER PROJECT
"""

import cv2
import mediapipe as mp
import numpy as np
import smtplib
import threading
import time
import csv
import os
import sys
import atexit
import urllib.request
from email.mime.multipart import MIMEMultipart
from email.mime.text      import MIMEText
from email.mime.image     import MIMEImage
from datetime             import datetime

# ── Config ─────────────────────────────────────────────────
try:
    import config
except ImportError:
    print("[ERROR] config.py not found.")
    sys.exit(1)

# ── Settings ────────────────────────────────────────────────
LOCATION_NAME      = "Restricted Area — Main Entrance"
RECEIVER_EMAIL     = "txaffankhan@gmail.com"   # change if needed
ALERT_COOLDOWN_SEC = 60       # min gap between alerts (no spam)
CONFIRM_FRAMES     = 8        # human must be detected N frames before alert
MOTION_THRESHOLD   = 2500     # min contour area to count as real motion
MEDIAPIPE_EVERY    = 3        # run body detection every N frames

# ── Colors (BGR) ────────────────────────────────────────────
CLR_GREEN  = (0,  210,  80)
CLR_RED    = (30,  50, 220)
CLR_YELLOW = (20, 200, 230)
CLR_WHITE  = (255, 255, 255)
CLR_BLACK  = (0,    0,   0)
CLR_DARK   = (18,  18,  25)
CLR_GOLD   = (255, 175,   0)
CLR_GRAY   = (140, 140, 140)
CLR_ORANGE = (0,  140, 255)

# ══════════════════════════════════════════════════════════
#  MEDIAPIPE POSE (body detection)
# ══════════════════════════════════════════════════════════
_POSE_MODEL = "pose_landmarker.task"
if not os.path.exists(_POSE_MODEL):
    print("[MEDIAPIPE] Downloading pose model (~5MB)...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
        _POSE_MODEL
    )
    print("[MEDIAPIPE] Done!")

_pose_landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(
    mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=_POSE_MODEL),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_poses=3,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
)

def detect_humans(frame_bgr):
    """
    Returns (count, bounding_boxes)
    count        : number of humans detected
    bounding_boxes: list of (x,y,w,h) in pixel coords
    """
    h, w = frame_bgr.shape[:2]
    rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res  = _pose_landmarker.detect(img)

    if not res.pose_landmarks:
        return 0, []

    boxes = []
    for person_lms in res.pose_landmarks:
        # Only count if enough landmarks are visible
        visible = sum(1 for lm in person_lms if lm.visibility > 0.4)
        if visible < 6:
            continue
        xs = [lm.x * w for lm in person_lms if lm.visibility > 0.3]
        ys = [lm.y * h for lm in person_lms if lm.visibility > 0.3]
        if not xs or not ys:
            continue
        x1 = max(0, int(min(xs)) - 20)
        y1 = max(0, int(min(ys)) - 20)
        x2 = min(w, int(max(xs)) + 20)
        y2 = min(h, int(max(ys)) + 20)
        boxes.append((x1, y1, x2 - x1, y2 - y1))

    return len(boxes), boxes

# ══════════════════════════════════════════════════════════
#  MOTION DETECTOR
# ══════════════════════════════════════════════════════════
class MotionDetector:
    def __init__(self):
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=40, detectShadows=False
        )

    def detect(self, frame):
        """
        Returns (has_motion, motion_mask, contours)
        Only counts significant motion (filters small noise).
        """
        # Blur to reduce noise
        blurred = cv2.GaussianBlur(frame, (15, 15), 0)
        mask    = self.bg_sub.apply(blurred)

        # Morphological cleanup — remove tiny specks
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask   = cv2.dilate(mask, kernel, iterations=2)

        # Find contours
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        # Filter — only large contours count
        big_cnts   = [c for c in cnts if cv2.contourArea(c) > MOTION_THRESHOLD]
        has_motion = len(big_cnts) > 0
        return has_motion, mask, big_cnts

# ══════════════════════════════════════════════════════════
#  EMAIL ALERT
# ══════════════════════════════════════════════════════════
_email_lock = threading.Lock()

def send_alert_email(snapshot, alert_number, detection_time, human_count):
    def _worker():
        with _email_lock:
            try:
                now_str  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                det_str  = datetime.fromtimestamp(detection_time).strftime("%H:%M:%S")
                duration = int(time.time() - detection_time)

                msg             = MIMEMultipart()
                msg["From"]     = config.SENDER_EMAIL
                msg["To"]       = RECEIVER_EMAIL
                msg["Subject"]  = (f"INTRUDER ALERT #{alert_number} "
                                   f"— Restricted Area Breach Detected")

                body = (
                    f"Dear Security Team,\n\n"
                    f"An unauthorized person has been detected "
                    f"in the restricted area.\n\n"
                    f"{'='*48}\n"
                    f"  ALERT DETAILS\n"
                    f"{'='*48}\n"
                    f"  Alert Number   : #{alert_number}\n"
                    f"  Location       : {LOCATION_NAME}\n"
                    f"  Detected At    : {det_str}\n"
                    f"  Alert Sent     : {now_str}\n"
                    f"  Persons Seen   : {human_count}\n"
                    f"{'='*48}\n\n"
                    f"A snapshot from the security camera is attached.\n"
                    f"Please take immediate action.\n\n"
                    f"DO NOT REPLY to this automated message.\n\n"
                    f"{'='*48}\n"
                    f"  Smart Security Monitoring System\n"
                    f"  Automated Alert — Do Not Reply\n"
                    f"{'='*48}"
                )
                msg.attach(MIMEText(body, "plain"))

                # Attach snapshot
                if snapshot is not None:
                    # Stamp the snapshot with alert info
                    stamped = snapshot.copy()
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    cv2.putText(stamped, f"ALERT #{alert_number}",
                                (10, 30), cv2.FONT_HERSHEY_DUPLEX,
                                0.9, (0, 0, 255), 2)
                    cv2.putText(stamped, ts,
                                (10, 58), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, (0, 0, 255), 1)
                    cv2.putText(stamped, LOCATION_NAME,
                                (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 255), 1)
                    ok, buf = cv2.imencode(".jpg", stamped,
                                          [cv2.IMWRITE_JPEG_QUALITY, 90])
                    if ok:
                        img = MIMEImage(buf.tobytes(),
                                        name=f"intruder_alert_{alert_number}.jpg")
                        msg.attach(img)

                with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=15) as srv:
                    srv.login(config.SENDER_EMAIL, config.SENDER_PASSWORD)
                    srv.send_message(msg)

                print(f"[EMAIL] Alert #{alert_number} sent to {RECEIVER_EMAIL}")

            except Exception as e:
                print(f"[EMAIL ERROR] {e}")

    threading.Thread(target=_worker, daemon=True).start()

# ══════════════════════════════════════════════════════════
#  LOGGING
# ══════════════════════════════════════════════════════════
LOG_FILE = "intruder_log.csv"

def log_alert(alert_num, human_count):
    exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["Timestamp","Alert#","Location","Persons"])
        w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    alert_num, LOCATION_NAME, human_count])

# ══════════════════════════════════════════════════════════
#  DRAWING HELPERS
# ══════════════════════════════════════════════════════════
_FONT_D = cv2.FONT_HERSHEY_DUPLEX
_FONT_S = cv2.FONT_HERSHEY_SIMPLEX

def blend_rect(frame, x1, y1, x2, y2, alpha=0.75, color=None):
    if color is None:
        color = CLR_DARK
    x1,y1 = max(0,x1), max(0,y1)
    x2,y2 = min(frame.shape[1],x2), min(frame.shape[0],y2)
    if x2<=x1 or y2<=y1:
        return
    roi  = frame[y1:y2, x1:x2]
    fill = np.full_like(roi, color, dtype=np.uint8)
    cv2.addWeighted(fill, alpha, roi, 1-alpha, 0, roi)
    frame[y1:y2, x1:x2] = roi
    cv2.rectangle(frame, (x1,y1), (x2,y2), CLR_GOLD, 1)

def draw_text(frame, text, pos, scale=0.6, color=CLR_WHITE,
              thick=1, shadow=True, font=None):
    f = font or _FONT_D
    if shadow:
        cv2.putText(frame, text, (pos[0]+1, pos[1]+1),
                    f, scale, CLR_BLACK, thick+1, cv2.LINE_AA)
    cv2.putText(frame, text, pos, f, scale, color, thick, cv2.LINE_AA)

def draw_bar(frame, x, y, w, h, pct, fg=CLR_GREEN, bg=(40,40,40)):
    pct = max(0.0, min(1.0, pct))
    cv2.rectangle(frame, (x,y), (x+w, y+h), bg, -1)
    filled = int(w*pct)
    if filled > 0:
        cv2.rectangle(frame, (x,y), (x+filled, y+h), fg, -1)
    cv2.rectangle(frame, (x,y), (x+w, y+h), CLR_GRAY, 1)

def draw_hud(frame, status, alert_count, human_count,
             confirm_progress, last_alert_ago, session_start):
    fh, fw = frame.shape[:2]
    now    = time.time()

    # Header bar
    blend_rect(frame, 0, 0, fw, 62, alpha=0.82)

    # Status
    if status == "CLEAR":
        s_txt, s_col = "AREA CLEAR", CLR_GREEN
    elif status == "MOTION":
        s_txt, s_col = "MOTION DETECTED", CLR_YELLOW
    elif status == "HUMAN":
        s_txt, s_col = "HUMAN DETECTED", CLR_ORANGE
    else:
        s_txt, s_col = "!! INTRUDER ALERT !!", CLR_RED

    draw_text(frame, s_txt, (12, 42),
              scale=1.0, color=s_col, thick=2)

    # Session timer
    elapsed = int(now - session_start)
    mm, ss  = divmod(elapsed, 60)
    hh, mm2 = divmod(mm, 60)
    draw_text(frame, f"Runtime: {hh:02d}:{mm2:02d}:{ss:02d}",
              (fw - 230, 42), scale=0.58, color=CLR_GRAY)

    # Bottom-left panel
    blend_rect(frame, 0, fh-145, 310, fh, alpha=0.82)

    draw_text(frame, f"Alerts sent   : {alert_count}",
              (10, fh-118), scale=0.55, color=CLR_WHITE)
    draw_text(frame, f"Persons seen  : {human_count}",
              (10, fh-90),  scale=0.55, color=CLR_WHITE)

    if last_alert_ago is not None:
        ago = int(now - last_alert_ago)
        cd  = max(0, ALERT_COOLDOWN_SEC - ago)
        draw_text(frame, f"Next alert in : {cd}s",
                  (10, fh-62), scale=0.55,
                  color=CLR_GRAY if cd > 0 else CLR_GREEN)

    # Confirm progress bar
    if confirm_progress > 0:
        draw_text(frame, "Confirming...",
                  (10, fh-38), scale=0.5, color=CLR_YELLOW)
        draw_bar(frame, 10, fh-22, 280, 14,
                 confirm_progress / CONFIRM_FRAMES, fg=CLR_ORANGE)
    else:
        draw_text(frame, "Watching...",
                  (10, fh-28), scale=0.5, color=CLR_GRAY)

    # Bottom-right: location + controls
    blend_rect(frame, fw-340, fh-70, fw, fh, alpha=0.82)
    draw_text(frame, f"CAM: {LOCATION_NAME[:28]}",
              (fw-335, fh-48), scale=0.42, color=CLR_GOLD)
    draw_text(frame, "Q = Quit   R = Reset counter",
              (fw-335, fh-22), scale=0.42, color=CLR_GRAY)

def draw_alert_overlay(frame, alert_count, blink):
    """Full-screen red flash when alert fires."""
    if not blink:
        return
    fh, fw = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (fw,fh), (0,0,180), -1)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
    draw_text(frame, f"!! INTRUDER ALERT #{alert_count} !!",
              (fw//2 - 260, fh//2 - 20),
              scale=1.3, color=CLR_WHITE, thick=3)
    draw_text(frame, "Email alert sent to security",
              (fw//2 - 185, fh//2 + 40),
              scale=0.75, color=CLR_YELLOW)

def draw_human_boxes(frame, boxes, status):
    col = CLR_RED if status == "ALERT" else CLR_ORANGE
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x,y), (x+w, y+h), col, 2)
        # Corner accents
        L = 20
        for px,py,dx,dy in [(x,y,1,1),(x+w,y,-1,1),
                             (x,y+h,1,-1),(x+w,y+h,-1,-1)]:
            cv2.line(frame,(px,py),(px+dx*L,py),col,3)
            cv2.line(frame,(px,py),(px,py+dy*L),col,3)
        draw_text(frame, "HUMAN",
                  (x+4, y-8), scale=0.5, color=col)

def draw_motion_overlay(frame, contours):
    """Draw semi-transparent motion region highlight."""
    if not contours:
        return
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, contours, -1, 255, -1)
    highlight = frame.copy()
    highlight[mask > 0] = np.clip(
        highlight[mask > 0].astype(int) + [0, 30, 50], 0, 255
    ).astype(np.uint8)
    cv2.addWeighted(highlight, 0.5, frame, 0.5, 0, frame)

def draw_scanline(frame, tick):
    """Animated scan line for security camera feel."""
    fh, fw = frame.shape[:2]
    y = int((tick * 3) % fh)
    cv2.line(frame, (0,y), (fw,y), (0,200,0), 1)
    # Vignette corners
    for corner in [(0,0),(fw,0),(0,fh),(fw,fh)]:
        cv2.circle(frame, corner, 120, (0,0,0), -1)

def draw_timestamp(frame):
    ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    fh, fw = frame.shape[:2]
    draw_text(frame, ts, (fw//2 - 120, fh-8),
              scale=0.48, color=(0,200,0),
              font=cv2.FONT_HERSHEY_SIMPLEX)

# ══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════
def main():
    print("\n╔══════════════════════════════════════════╗")
    print("║   Smart Intruder Alert — Starting ...    ║")
    print("╚══════════════════════════════════════════╝\n")
    print(f"[INFO] Monitoring: {LOCATION_NAME}")
    print(f"[INFO] Alerts  -> {RECEIVER_EMAIL}")
    print(f"[INFO] Cooldown: {ALERT_COOLDOWN_SEC}s between alerts")
    print("[INFO] Press Q to quit, R to reset counter\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[CAM] {fw}x{fh}")

    cv2.namedWindow("Smart Intruder Alert", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Smart Intruder Alert", fw, fh)

    motion_det      = MotionDetector()
    session_start   = time.time()
    alert_count     = 0
    total_humans    = 0
    last_alert_time = None
    confirm_hits    = 0    # consecutive frames with human detected
    alert_flash     = False
    flash_until     = 0.0
    frame_idx       = 0
    mp_human_count  = 0
    mp_boxes        = []
    status          = "CLEAR"

    # Warmup background model (5 seconds, no alerts)
    print("[INFO] Warming up background model (5 seconds)...")
    warmup_end = time.time() + 5
    while time.time() < warmup_end:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        motion_det.detect(frame)
        blend_rect(frame, 0, 0, fw, fh, alpha=0.5, color=(0,0,0))
        remaining = int(warmup_end - time.time()) + 1
        draw_text(frame, "Initializing security system...",
                  (fw//2-220, fh//2-20), scale=0.9, color=CLR_GOLD, thick=2)
        draw_text(frame, f"Please stay out of frame. Ready in {remaining}s",
                  (fw//2-260, fh//2+30), scale=0.6, color=CLR_WHITE)
        cv2.imshow("Smart Intruder Alert", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return
    print("[INFO] System ready. Monitoring started.\n")
    session_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Camera read failed.")
            break

        frame     = cv2.flip(frame, 1)
        frame_idx += 1
        now        = time.time()
        key        = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):
            alert_count  = 0
            confirm_hits = 0
            status       = "CLEAR"
            print("[RESET] Alert counter reset.")

        # ── Motion detection ─────────────────────────
        has_motion, mask, contours = motion_det.detect(frame)

        # ── MediaPipe human detection (every N frames) ─
        if frame_idx % MEDIAPIPE_EVERY == 0:
            mp_human_count, mp_boxes = detect_humans(frame)

        # ── State machine ─────────────────────────────
        if not has_motion:
            status       = "CLEAR"
            confirm_hits = 0

        elif has_motion and mp_human_count == 0:
            # Motion but no human — could be animal or small object
            status = "MOTION"
            confirm_hits = max(0, confirm_hits - 1)

        elif has_motion and mp_human_count > 0:
            status        = "HUMAN"
            confirm_hits  = min(confirm_hits + 1, CONFIRM_FRAMES)
            total_humans  = max(total_humans, mp_human_count)

            # ── Alert trigger ─────────────────────────
            cooldown_ok = (last_alert_time is None or
                           now - last_alert_time >= ALERT_COOLDOWN_SEC)

            if confirm_hits >= CONFIRM_FRAMES and cooldown_ok:
                status         = "ALERT"
                alert_count   += 1
                last_alert_time= now
                alert_flash    = True
                flash_until    = now + 4.0
                snapshot       = frame.copy()

                print(f"[ALERT #{alert_count}] Human detected! "
                      f"Persons={mp_human_count}. Sending email...")
                log_alert(alert_count, mp_human_count)
                send_alert_email(snapshot, alert_count, now, mp_human_count)
                confirm_hits = 0   # reset after alert

        # ── Draw ──────────────────────────────────────
        # Motion highlight
        if has_motion and status in ("MOTION","HUMAN","ALERT"):
            draw_motion_overlay(frame, contours)

        # Human bounding boxes
        if mp_boxes and status in ("HUMAN","ALERT"):
            draw_human_boxes(frame, mp_boxes, status)

        # Security camera effects
        draw_scanline(frame, frame_idx)

        # Alert flash
        blink = alert_flash and int((now - (flash_until - 4.0)) * 4) % 2 == 0
        if alert_flash and now >= flash_until:
            alert_flash = False
        draw_alert_overlay(frame, alert_count, blink)

        # HUD
        draw_hud(frame, status, alert_count, total_humans,
                 confirm_hits, last_alert_time, session_start)
        draw_timestamp(frame)

        cv2.imshow("Smart Intruder Alert", frame)

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    elapsed = int(time.time() - session_start)
    mm, ss  = divmod(elapsed, 60)
    print(f"\n[EXIT] Session ended. Runtime: {mm:02d}:{ss:02d}")
    print(f"[EXIT] Total alerts: {alert_count}")
    print(f"[EXIT] Log saved to: {LOG_FILE}\n")

if __name__ == "__main__":
    main()